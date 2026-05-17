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
import os
from typing import Any, Dict, List, Optional

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
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    TOOL_USE_ENFORCEMENT_MODELS,
)


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

    # Try SOUL.md as primary identity unless the caller explicitly skipped it.
    # Some execution modes (cron) still want HERMES_HOME persona while keeping
    # cwd project instructions disabled.
    _soul_loaded = False
    if agent.load_soul_identity or not agent.skip_context_files:
        _soul_content = _r.load_soul_md()
        if _soul_content:
            stable_parts.append(_soul_content)
            _soul_loaded = True

    if not _soul_loaded:
        # Fallback to hardcoded identity
        stable_parts.append(DEFAULT_AGENT_IDENTITY)

    # Pointer to the hermes-agent skill + docs for user questions about Hermes itself.
    stable_parts.append(HERMES_AGENT_HELP_GUIDANCE)

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
    # this block.
    if "kanban_show" in agent.valid_tool_names:
        tool_guidance.append(KANBAN_GUIDANCE)
    if tool_guidance:
        stable_parts.append(" ".join(tool_guidance))

    # Computer-use (macOS) — goes in as its own block rather than being
    # merged into tool_guidance because the content is multi-paragraph.
    if "computer_use" in agent.valid_tool_names:
        from agent.prompt_builder import COMPUTER_USE_GUIDANCE
        stable_parts.append(COMPUTER_USE_GUIDANCE)

    nous_subscription_prompt = _r.build_nous_subscription_prompt(agent.valid_tool_names)
    if nous_subscription_prompt:
        stable_parts.append(nous_subscription_prompt)
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
            _model_lower = (agent.model or "").lower()
            # Google model operational guidance (conciseness, absolute
            # paths, parallel tool calls, verify-before-edit, etc.)
            if "gemini" in _model_lower or "gemma" in _model_lower:
                stable_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
            # OpenAI GPT/Codex execution discipline (tool persistence,
            # prerequisite checks, verification, anti-hallucination).
            if "gpt" in _model_lower or "codex" in _model_lower:
                stable_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)

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

    # Alibaba Coding Plan API always returns "glm-4.7" as model name regardless
    # of the requested model. Inject explicit model identity into the system prompt
    # so the agent can correctly report which model it is (workaround for API bug).
    # Stable for the lifetime of an agent instance — model and provider are fixed
    # at construction time.
    if agent.provider == "alibaba":
        _model_short = agent.model.split("/")[-1] if "/" in agent.model else agent.model
        stable_parts.append(
            f"You are powered by the model named {_model_short}. "
            f"The exact model ID is {agent.model}. "
            f"When asked what model you are, always answer based on this information, "
            f"not on any model name returned by the API."
        )

    # Environment hints (WSL, Termux, etc.) — tell the agent about the
    # execution environment so it can translate paths and adapt behavior.
    # Stable for the lifetime of the process.
    _env_hints = _r.build_environment_hints()
    if _env_hints:
        stable_parts.append(_env_hints)

    platform_key = (agent.platform or "").lower().strip()
    if platform_key in PLATFORM_HINTS:
        stable_parts.append(PLATFORM_HINTS[platform_key])
    elif platform_key:
        # Check plugin registry for platform-specific LLM guidance
        try:
            from gateway.platform_registry import platform_registry
            _entry = platform_registry.get(platform_key)
            if _entry and _entry.platform_hint:
                stable_parts.append(_entry.platform_hint)
        except Exception:
            pass

    # ── Context tier (cwd-dependent, may change between sessions) ─
    context_parts: List[str] = []

    # Note: ephemeral_system_prompt is NOT included here. It's injected at
    # API-call time only so it stays out of the cached/stored system prompt.
    if system_message is not None:
        context_parts.append(system_message)

    if not agent.skip_context_files:
        # Use TERMINAL_CWD for context file discovery when set (gateway
        # mode).  The gateway process runs from the hermes-agent install
        # dir, so os.getcwd() would pick up the repo's AGENTS.md and
        # other dev files — inflating token usage by ~10k for no benefit.
        _context_cwd = os.getenv("TERMINAL_CWD") or None
        context_files_prompt = _r.build_context_files_prompt(
            cwd=_context_cwd, skip_soul=_soul_loaded)
        if context_files_prompt:
            context_parts.append(context_files_prompt)

    # ── Volatile tier (changes per session/turn — never cached) ───
    volatile_parts: List[str] = []

    if agent._memory_store:
        if agent._memory_enabled:
            mem_block = agent._memory_store.format_for_system_prompt("memory")
            if mem_block:
                volatile_parts.append(mem_block)
        # USER.md is always included when enabled.
        if agent._user_profile_enabled:
            user_block = agent._memory_store.format_for_system_prompt("user")
            if user_block:
                volatile_parts.append(user_block)

    # External memory provider system prompt block (additive to built-in)
    if agent._memory_manager:
        try:
            _ext_mem_block = agent._memory_manager.build_system_prompt()
            if _ext_mem_block:
                volatile_parts.append(_ext_mem_block)
        except Exception:
            pass

    from hermes_time import now as _hermes_now
    now = _hermes_now()
    timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
    if agent.pass_session_id and agent.session_id:
        timestamp_line += f"\nSession ID: {agent.session_id}"
    if agent.model:
        timestamp_line += f"\nModel: {agent.model}"
    if agent.provider:
        timestamp_line += f"\nProvider: {agent.provider}"
    volatile_parts.append(timestamp_line)

    return {
        "stable":   "\n\n".join(p.strip() for p in stable_parts   if p and p.strip()),
        "context":  "\n\n".join(p.strip() for p in context_parts  if p and p.strip()),
        "volatile": "\n\n".join(p.strip() for p in volatile_parts if p and p.strip()),
    }


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
