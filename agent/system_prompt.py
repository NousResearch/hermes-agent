"""System-prompt assembly for :class:`AIAgent`.

The agent's system prompt is built once per session and reused across all
turns — only context compression triggers a rebuild.  This keeps the
upstream prefix cache warm.  See ``hermes-agent-dev``'s
``references/system-prompt-invariant.md`` for the invariants and
``references/self-improvement-loop.md`` for how the background-review
fork inherits the cached prompt verbatim.

Three tiers are joined with ``\\n\\n``:

* ``stable``   — identity (SOUL.md or DEFAULT_AGENT_IDENTITY), tool
  guidance, computer-use guidance, nous subscription block, tool-use
  enforcement guidance + per-model operational guidance, skills prompt,
  alibaba model-name workaround, environment hints, platform hints.
* ``context``  — caller-supplied ``system_message`` plus context files
  (AGENTS.md / .cursorrules / etc.) discovered under ``TERMINAL_CWD``.
* ``volatile`` — memory snapshot, USER.md profile, external memory
  provider block, timestamp/session/model/provider line.

Pure helpers that read the agent's state.  AIAgent keeps thin forwarders.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agent.instruction_surface import (
    build_project_context_manifest,
    make_instruction_block,
    resolve_instruction_blocks,
    render_resolved_surface,
)

from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    HERMES_AGENT_HELP_GUIDANCE,
    KANBAN_GUIDANCE,
    MEMORY_GUIDANCE,
    OPENAI_MODEL_EXECUTION_GUIDANCE,
    PLATFORM_HINTS,
    SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE,
    TASK_COMPLETION_GUIDANCE,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    TOOL_USE_ENFORCEMENT_MODELS,
)
from agent.runtime_cwd import resolve_context_cwd


def _ra():
    """Lazy reference to the ``run_agent`` module.

    Helpers like ``load_soul_md``, ``build_environment_hints``,
    ``build_context_files_prompt``, ``build_nous_subscription_prompt``,
    ``build_skills_system_prompt`` and ``get_toolset_for_tool`` are
    imported into ``run_agent``'s namespace.  Many tests
    ``patch("run_agent.load_soul_md", ...)``; if we imported them
    directly here those patches would not reach us.  Looking them up
    through ``run_agent`` on every call preserves the patch contract.
    """
    import run_agent
    return run_agent


def build_system_prompt_parts(agent: Any, system_message: Optional[str] = None) -> Dict[str, str]:
    """Assemble the system prompt as three ordered parts.

    Returns a dict with three keys:
      * ``stable``   — identity, tool guidance, skills prompt,
        environment hints, platform hints, model-family operational
        guidance.
      * ``context``  — context files (AGENTS.md, .cursorrules, etc.)
        and caller-supplied system_message.
      * ``volatile`` — memory snapshot, user profile, external
        memory provider block, timestamp line.

    Joined into a single string by :func:`build_system_prompt` and
    cached on ``agent._cached_system_prompt`` for the lifetime of the
    AIAgent.  Hermes never re-renders parts of this string mid-
    session — that's the only way to keep upstream prompt caches
    warm across turns.
    """
    # Local import to avoid pulling model_tools at module load.  Tests
    # patch ``run_agent.get_toolset_for_tool`` and similar helpers, so
    # we resolve through ``_ra()`` to honor those patches.
    _r = _ra()

    # ── Stable tier ────────────────────────────────────────────────
    stable_parts: List[str] = []
    instruction_blocks = []

    def _record_block(*, id: str, content: str, surface: str, tier: str, authority: int, scope: str, origin: str, path: str | None = None, trust: str = "trusted", cache_policy: str = "session", labels=()):
        if not content or not str(content).strip():
            return
        instruction_blocks.append(make_instruction_block(
            id=id,
            content=content,
            surface=surface,
            tier=tier,
            authority=authority,
            scope=scope,
            origin=origin,
            path=path,
            trust=trust,
            cache_policy=cache_policy,
            labels=labels,
        ))

    # Try SOUL.md as primary identity unless the caller explicitly skipped it.
    # Some execution modes (cron) still want HERMES_HOME persona while keeping
    # cwd project instructions disabled.
    _soul_loaded = False
    if agent.load_soul_identity or not agent.skip_context_files:
        _soul_content = _r.load_soul_md()
        if _soul_content:
            stable_parts.append(_soul_content)
            _record_block(id="profile.SOUL", content=_soul_content, surface="profile", tier="stable", authority=950, scope="profile", origin="HERMES_HOME/SOUL.md", path=str(_r.get_hermes_home() / "SOUL.md") if hasattr(_r, "get_hermes_home") else None, trust="trusted", cache_policy="stable", labels={"identity", "profile"})
            _soul_loaded = True

    if not _soul_loaded:
        # Fallback to hardcoded identity
        stable_parts.append(DEFAULT_AGENT_IDENTITY)
        _record_block(id="profile.DEFAULT_AGENT_IDENTITY", content=DEFAULT_AGENT_IDENTITY, surface="profile", tier="stable", authority=950, scope="profile", origin="agent.prompt_builder.DEFAULT_AGENT_IDENTITY", trust="trusted", cache_policy="stable", labels={"identity", "profile"})

    # Pointer to the hermes-agent skill + docs for user questions about Hermes itself.
    stable_parts.append(HERMES_AGENT_HELP_GUIDANCE)
    _record_block(id="core.hermes_agent_help", content=HERMES_AGENT_HELP_GUIDANCE, surface="core", tier="stable", authority=1000, scope="global", origin="agent.prompt_builder.HERMES_AGENT_HELP_GUIDANCE", trust="trusted", cache_policy="stable", labels={"workflow", "tool"})

    # Universal task-completion / no-fabrication guidance.  Applied to ALL
    # models regardless of tool_use_enforcement gating — the failure modes
    # this targets (stopping after a stub; fabricating output when a real
    # path is blocked) are not model-family specific.  Gated only by
    # config.yaml ``agent.task_completion_guidance`` (default True) so
    # users who want a leaner prompt can turn it off.
    if getattr(agent, "_task_completion_guidance", True) and agent.valid_tool_names:
        stable_parts.append(TASK_COMPLETION_GUIDANCE)

    # Tool-aware behavioral guidance: only inject when the tools are loaded
    tool_guidance = []
    if "memory" in agent.valid_tool_names:
        tool_guidance.append(MEMORY_GUIDANCE)
    if "session_search" in agent.valid_tool_names:
        tool_guidance.append(SESSION_SEARCH_GUIDANCE)
    if "skill_manage" in agent.valid_tool_names:
        tool_guidance.append(SKILLS_GUIDANCE)
    # Kanban worker/orchestrator lifecycle — only present when the
    # dispatcher spawned this process (kanban_show check_fn gates on
    # HERMES_KANBAN_TASK env var). Normal chat sessions never see
    # this block. Resolved once at __init__ (see _kanban_worker_guidance).
    _kanban_guidance = getattr(agent, "_kanban_worker_guidance", None)
    if _kanban_guidance:
        tool_guidance.append(_kanban_guidance)
    elif _kanban_guidance is None and "kanban_show" in agent.valid_tool_names:
        # Fallback for code paths that bypass agent_init (rare).
        tool_guidance.append(KANBAN_GUIDANCE)
    if tool_guidance:
        _tool_guidance_text = " ".join(tool_guidance)
        stable_parts.append(_tool_guidance_text)
        _record_block(id="tool.guidance", content=_tool_guidance_text, surface="tool_guidance", tier="stable", authority=925, scope="session", origin="agent.system_prompt tool guidance", trust="trusted", cache_policy="stable", labels={"tool", "workflow", "kanban", "safety"})

    # Computer-use (macOS) — goes in as its own block rather than being
    # merged into tool_guidance because the content is multi-paragraph.
    if "computer_use" in agent.valid_tool_names:
        from agent.prompt_builder import COMPUTER_USE_GUIDANCE
        stable_parts.append(COMPUTER_USE_GUIDANCE)
        _record_block(id="tool.computer_use", content=COMPUTER_USE_GUIDANCE, surface="tool_guidance", tier="stable", authority=925, scope="session", origin="agent.prompt_builder.COMPUTER_USE_GUIDANCE", trust="trusted", cache_policy="stable", labels={"tool", "safety"})

    nous_subscription_prompt = _r.build_nous_subscription_prompt(agent.valid_tool_names)
    if nous_subscription_prompt:
        stable_parts.append(nous_subscription_prompt)
        _record_block(id="tool.nous_subscription", content=nous_subscription_prompt, surface="tool_guidance", tier="stable", authority=925, scope="session", origin="agent.prompt_builder.build_nous_subscription_prompt", trust="trusted", cache_policy="stable", labels={"tool"})
    # Tool-use enforcement: tells the model to actually call tools instead
    # of describing intended actions.  Controlled by config.yaml
    # agent.tool_use_enforcement:
    #   "auto" (default) — matches TOOL_USE_ENFORCEMENT_MODELS
    #   true  — always inject (all models)
    #   false — never inject
    #   list  — custom model-name substrings to match
    if agent.valid_tool_names:
        _enforce = agent._tool_use_enforcement
        _inject = False
        if _enforce is True or (isinstance(_enforce, str) and _enforce.lower() in {"true", "always", "yes", "on"}):
            _inject = True
        elif _enforce is False or (isinstance(_enforce, str) and _enforce.lower() in {"false", "never", "no", "off"}):
            _inject = False
        elif isinstance(_enforce, list):
            model_lower = (agent.model or "").lower()
            _inject = any(p.lower() in model_lower for p in _enforce if isinstance(p, str))
        else:
            # "auto" or any unrecognised value — use hardcoded defaults
            model_lower = (agent.model or "").lower()
            _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
        if _inject:
            stable_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
            _record_block(id="core.tool_use_enforcement", content=TOOL_USE_ENFORCEMENT_GUIDANCE, surface="core", tier="stable", authority=1000, scope="global", origin="agent.prompt_builder.TOOL_USE_ENFORCEMENT_GUIDANCE", trust="trusted", cache_policy="stable", labels={"tool", "safety", "workflow"})
            _model_lower = (agent.model or "").lower()
            # Google model operational guidance (conciseness, absolute
            # paths, parallel tool calls, verify-before-edit, etc.)
            if "gemini" in _model_lower or "gemma" in _model_lower:
                stable_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
                _record_block(id="core.google_model_operational_guidance", content=GOOGLE_MODEL_OPERATIONAL_GUIDANCE, surface="core", tier="stable", authority=1000, scope="global", origin="agent.prompt_builder.GOOGLE_MODEL_OPERATIONAL_GUIDANCE", trust="trusted", cache_policy="stable", labels={"workflow"})
            # OpenAI GPT/Codex execution discipline (tool persistence,
            # prerequisite checks, verification, anti-hallucination).
            # Also applied to xAI Grok — same failure modes (claims completion
            # without tool calls, suggests workarounds instead of using
            # existing tools, replies with plans instead of executing).
            if "gpt" in _model_lower or "codex" in _model_lower or "grok" in _model_lower:
                stable_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)
                _record_block(id="core.openai_model_execution_guidance", content=OPENAI_MODEL_EXECUTION_GUIDANCE, surface="core", tier="stable", authority=1000, scope="global", origin="agent.prompt_builder.OPENAI_MODEL_EXECUTION_GUIDANCE", trust="trusted", cache_policy="stable", labels={"tool", "workflow", "safety"})

    has_skills_tools = any(name in agent.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage'])
    if has_skills_tools:
        avail_toolsets = {
            toolset
            for toolset in (
                _r.get_toolset_for_tool(tool_name) for tool_name in agent.valid_tool_names
            )
            if toolset
        }
        skills_prompt = _r.build_skills_system_prompt(
            available_tools=agent.valid_tool_names,
            available_toolsets=avail_toolsets,
        )
    else:
        skills_prompt = ""
    if skills_prompt:
        stable_parts.append(skills_prompt)
        _record_block(id="skill.index", content=skills_prompt, surface="skill_index", tier="stable", authority=800, scope="session", origin="agent.prompt_builder.build_skills_system_prompt", trust="trusted", cache_policy="stable", labels={"workflow"})

    # Alibaba Coding Plan API always returns "glm-4.7" as model name regardless
    # of the requested model. Inject explicit model identity into the system prompt
    # so the agent can correctly report which model it is (workaround for API bug).
    # Stable for the lifetime of an agent instance — model and provider are fixed
    # at construction time.
    if agent.provider == "alibaba":
        _model_short = agent.model.split("/")[-1] if "/" in agent.model else agent.model
        _alibaba_guidance = (
            f"You are powered by the model named {_model_short}. "
            f"The exact model ID is {agent.model}. "
            f"When asked what model you are, always answer based on this information, "
            f"not on any model name returned by the API."
        )
        stable_parts.append(_alibaba_guidance)
        _record_block(id="core.alibaba_model_identity", content=_alibaba_guidance, surface="core", tier="stable", authority=1000, scope="session", origin="agent.system_prompt alibaba workaround", trust="trusted", cache_policy="stable", labels={"identity"})

    # Environment hints (WSL, Termux, etc.) — tell the agent about the
    # execution environment so it can translate paths and adapt behavior.
    # Stable for the lifetime of the process.
    _env_hints = _r.build_environment_hints()
    if _env_hints:
        stable_parts.append(_env_hints)
        _record_block(id="environment.hints", content=_env_hints, surface="environment", tier="stable", authority=925, scope="session", origin="agent.prompt_builder.build_environment_hints", trust="trusted", cache_policy="stable", labels={"environment", "workflow"})

    # Local Python toolchain probe — names python/pip/uv/PEP-668 state when
    # something is non-default so the model can pick the right install
    # strategy without discovering by failure.  Emits a single line; emits
    # NOTHING when the environment is clean (no token cost).  Skipped
    # entirely for remote terminal backends (the host's Python state is
    # irrelevant when tools run inside docker/modal/ssh).  Gated by
    # config.yaml ``agent.environment_probe`` (default True).
    if getattr(agent, "_environment_probe", True):
        try:
            from tools.env_probe import get_environment_probe_line
            _probe_line = get_environment_probe_line()
            if _probe_line:
                stable_parts.append(_probe_line)
        except Exception:
            # Probe failure must never block prompt build.
            pass

    # Active-profile hint — names the Hermes profile the agent is running
    # under so it doesn't conflate ~/.hermes/skills/ (default profile) with
    # ~/.hermes/profiles/<active>/skills/ (this profile's). Deterministic
    # for the lifetime of the agent — profile name doesn't change
    # mid-session, so this doesn't break the prompt cache.
    # See file_safety._resolve_active_profile_name + classify_cross_profile_target
    # for the matching tool-side guard.
    try:
        from agent.file_safety import _resolve_active_profile_name
        active_profile = _resolve_active_profile_name()
    except Exception:
        active_profile = "default"
    if active_profile == "default":
        stable_parts.append(
            "Active Hermes profile: default. Other profiles (if any) live "
            "under ~/.hermes/profiles/<name>/. Each profile has its own "
            "skills/, plugins/, cron/, and memories/ that affect a different "
            "session than this one. Do not modify another profile's "
            "skills/plugins/cron/memories unless the user explicitly directs "
            "you to."
        )
    else:
        stable_parts.append(
            f"Active Hermes profile: {active_profile}. This session reads "
            f"and writes ~/.hermes/profiles/{active_profile}/. The default "
            f"profile's data lives at ~/.hermes/skills/, ~/.hermes/plugins/, "
            f"~/.hermes/cron/, ~/.hermes/memories/ — those belong to a "
            f"different session run from a different shell. Do NOT modify "
            f"another profile's skills/plugins/cron/memories unless the user "
            f"explicitly directs you to. The cross-profile write guard will "
            f"refuse such writes by default; pass cross_profile=True only "
            f"after explicit direction."
        )

    _record_block(id="profile.active_profile_hint", content=stable_parts[-1] if stable_parts else "", surface="profile", tier="stable", authority=950, scope="profile", origin="agent.system_prompt active profile hint", trust="trusted", cache_policy="stable", labels={"profile", "safety"})

    platform_key = (agent.platform or "").lower().strip()
    if platform_key in PLATFORM_HINTS:
        stable_parts.append(PLATFORM_HINTS[platform_key])
        _record_block(id=f"platform.{platform_key}", content=PLATFORM_HINTS[platform_key], surface="platform", tier="stable", authority=925, scope="session", origin="agent.prompt_builder.PLATFORM_HINTS", trust="trusted", cache_policy="stable", labels={"platform", "workflow"})
    elif platform_key:
        # Check plugin registry for platform-specific LLM guidance
        try:
            from gateway.platform_registry import platform_registry
            _entry = platform_registry.get(platform_key)
            if _entry and _entry.platform_hint:
                stable_parts.append(_entry.platform_hint)
                _record_block(id=f"platform.{platform_key}", content=_entry.platform_hint, surface="platform", tier="stable", authority=925, scope="session", origin="gateway.platform_registry", trust="trusted", cache_policy="stable", labels={"platform", "workflow"})
        except Exception:
            pass

    # ── Context tier (cwd-dependent, may change between sessions) ─
    context_parts: List[str] = []

    # Note: ephemeral_system_prompt is NOT included here. It's injected at
    # API-call time only so it stays out of the cached/stored system prompt.
    if system_message is not None:
        context_parts.append(system_message)
        _record_block(id="caller.system_message", content=system_message, surface="caller_system", tier="context", authority=850, scope="session", origin="run_conversation.system_message", trust="trusted", cache_policy="session", labels={"workflow"})

    if not agent.skip_context_files:
        # Prefer the configured TERMINAL_CWD (gateway mode). When unset (local
        # CLI), None lets build_context_files_prompt fall back to the launch
        # dir — the user's real cwd there, but the install dir for the gateway
        # daemon, which is why the gateway sets TERMINAL_CWD.
        context_files_prompt = _r.build_context_files_prompt(
            cwd=resolve_context_cwd(), skip_soul=_soul_loaded)
        if context_files_prompt:
            context_parts.append(context_files_prompt)
            project_block = build_project_context_manifest(cwd=_context_cwd or os.getcwd())
            if project_block:
                _record_block(
                    id=project_block.id,
                    content=context_files_prompt,
                    surface=project_block.surface,
                    tier=project_block.tier,
                    authority=project_block.authority,
                    scope=project_block.scope,
                    origin=project_block.origin,
                    path=project_block.path,
                    trust=project_block.trust,
                    cache_policy=project_block.cache_policy,
                    labels=project_block.labels,
                )
            else:
                _record_block(
                    id="project.context_files",
                    content=context_files_prompt,
                    surface="project",
                    tier="context",
                    authority=650,
                    scope="project",
                    origin="agent.prompt_builder.build_context_files_prompt",
                    path=_context_cwd,
                    trust="workspace",
                    cache_policy="session",
                    labels={"project", "workflow"},
                )

    # ── Volatile tier (changes per session/turn — never cached) ───
    volatile_parts: List[str] = []

    if agent._memory_store:
        if agent._memory_enabled:
            mem_block = agent._memory_store.format_for_system_prompt("memory")
            if mem_block:
                volatile_parts.append(mem_block)
                _record_block(id="memory.durable", content=mem_block, surface="memory", tier="volatile", authority=450, scope="profile", origin="memory_store.format_for_system_prompt(memory)", trust="derived", cache_policy="turn", labels={"memory"})
        # USER.md is always included when enabled.
        if agent._user_profile_enabled:
            user_block = agent._memory_store.format_for_system_prompt("user")
            if user_block:
                volatile_parts.append(user_block)
                _record_block(id="memory.user_profile", content=user_block, surface="user_profile", tier="volatile", authority=450, scope="profile", origin="memory_store.format_for_system_prompt(user)", trust="derived", cache_policy="turn", labels={"memory", "profile"})

    # External memory provider system prompt block (additive to built-in)
    if agent._memory_manager:
        try:
            _ext_mem_block = agent._memory_manager.build_system_prompt()
            if _ext_mem_block:
                volatile_parts.append(_ext_mem_block)
                _record_block(id="memory.external", content=_ext_mem_block, surface="external_memory", tier="volatile", authority=400, scope="session", origin="memory_manager.build_system_prompt", trust="derived", cache_policy="turn", labels={"memory"})
        except Exception:
            pass

    from hermes_time import now as _hermes_now
    now = _hermes_now()
    # Date-only (not minute-precision) so the system prompt is byte-stable
    # for the full day.  Minute-precision changes invalidate prefix-cache KV
    # on every rebuild path (compression boundary, fresh-agent gateway turns,
    # session resume without a stored prompt).  The model can still query the
    # exact wall-clock time via tools when it actually needs it.
    # Credit: @iamfoz (PR #20451).
    timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y')}"
    if agent.pass_session_id and agent.session_id:
        timestamp_line += f"\nSession ID: {agent.session_id}"
    if agent.model:
        timestamp_line += f"\nModel: {agent.model}"
    if agent.provider:
        timestamp_line += f"\nProvider: {agent.provider}"
    volatile_parts.append(timestamp_line)
    _record_block(id="volatile.timestamp", content=timestamp_line, surface="environment", tier="volatile", authority=925, scope="session", origin="hermes_time.now/date + agent metadata", trust="trusted", cache_policy="turn", labels={"environment"})

    try:
        from agent.uswarm_helpers import build_context_pack, is_uswarm_context_pack_enabled
        from hermes_cli.config import load_config as _load_config

        _cfg = _load_config() or {}
        if is_uswarm_context_pack_enabled(_cfg):
            _context_cfg = (_cfg.get("uswarm_helpers", {}).get("context_pack", {}) or {})
            _pack = build_context_pack(
                (
                    {
                        "id": getattr(block, "id", f"block-{idx}"),
                        "kind": getattr(block, "surface", "instruction"),
                        "path": getattr(block, "path", None) or getattr(block, "origin", None) or getattr(block, "id", f"block-{idx}"),
                        "content": getattr(block, "content", ""),
                        "metadata": {
                            "tier": getattr(block, "tier", None),
                            "authority": getattr(block, "authority", None),
                            "scope": getattr(block, "scope", None),
                            "origin": getattr(block, "origin", None),
                        },
                    }
                    for idx, block in enumerate(instruction_blocks)
                ),
                token_budget=int(_context_cfg.get("token_budget", 4000)),
                allowed_base=str(_context_cfg.get("allowed_base") or os.getcwd()),
            )
            _pack_text = "uSwarm context pack (experimental):\n" + json.dumps(_pack, ensure_ascii=False, sort_keys=True)
            _record_block(
                id="experimental.uswarm_context_pack",
                content=_pack_text,
                surface="derived_context",
                tier="context",
                authority=300,
                scope="session",
                origin="agent.uswarm_helpers.build_context_pack",
                trust="derived",
                cache_policy="session",
                labels={"experimental", "context-pack"},
            )
    except Exception:
        pass

    resolved = resolve_instruction_blocks(instruction_blocks)
    agent._instruction_surface_manifest = resolved.manifest
    agent._instruction_surface_conflicts = resolved.conflicts
    return render_resolved_surface(resolved)


def build_system_prompt(agent: Any, system_message: Optional[str] = None) -> str:
    """Assemble the full system prompt from all layers.

    Called once per session (cached on ``agent._cached_system_prompt``) and
    only rebuilt after context compression events. This ensures the system
    prompt is stable across all turns in a session, maximizing prefix cache
    hits.

    Layers are ordered cache-friendly: stable identity/guidance first,
    then session-stable context files, then per-call volatile content
    (memory, USER profile, timestamp).  The whole string is treated as
    one cached block — Hermes never rebuilds or reinjects parts of it
    mid-session, which is the only way to keep upstream prompt caches
    warm across turns.
    """
    parts = build_system_prompt_parts(agent, system_message=system_message)
    return "\n\n".join(p for p in (parts["stable"], parts["context"], parts["volatile"]) if p)


def invalidate_system_prompt(agent: Any) -> None:
    """Invalidate the cached system prompt, forcing a rebuild on the next turn.

    Called after context compression events. Also reloads memory from disk
    so the rebuilt prompt captures any writes from this session.
    """
    agent._cached_system_prompt = None
    if agent._memory_store:
        agent._memory_store.load_from_disk()


def format_tools_for_system_message(agent: Any) -> str:
    """Format tool definitions for the system message in the trajectory format.

    Returns:
        str: JSON string representation of tool definitions
    """
    if not agent.tools:
        return "[]"

    # Convert tool definitions to the format expected in trajectories
    formatted_tools = []
    for tool in agent.tools:
        func = tool["function"]
        formatted_tool = {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
            "required": None  # Match the format in the example
        }
        formatted_tools.append(formatted_tool)

    return json.dumps(formatted_tools, ensure_ascii=False)


__all__ = [
    "build_system_prompt_parts",
    "build_system_prompt",
    "invalidate_system_prompt",
    "format_tools_for_system_message",
]
