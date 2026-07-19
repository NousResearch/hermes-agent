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
  alibaba model-name workaround, environment hints, platform hints,
  plugin-owned frozen sections at constrained anchors.
* ``context``  — caller-supplied ``system_message`` plus context files
  (AGENTS.md / .cursorrules / etc.) discovered under ``TERMINAL_CWD``.
* ``volatile`` — memory snapshot, USER.md profile, external memory
  provider block, timestamp/session/model/provider line.

Pure helpers that read the agent's state.  AIAgent keeps thin forwarders.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    HERMES_AGENT_HELP_GUIDANCE,
    KANBAN_GUIDANCE,
    MEMORY_GUIDANCE,
    OPENAI_MODEL_EXECUTION_GUIDANCE,
    PARALLEL_TOOL_CALL_GUIDANCE,
    PLATFORM_HINTS,
    SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE,
    STEER_CHANNEL_NOTE,
    TASK_COMPLETION_GUIDANCE,
    TELEGRAM_RICH_MESSAGES_HINT,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    TOOL_USE_ENFORCEMENT_MODELS,
    drain_truncation_warnings,
)
from agent.runtime_cwd import resolve_context_cwd
from hermes_constants import get_hermes_home
from utils import is_truthy_value


logger = logging.getLogger(__name__)
_PLUGIN_PROMPT_STATE_VERSION = 1


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


def _resolve_platform_hint(agent: Any, platform_key: str, default_hint: str) -> str:
    """Apply a per-platform prompt-hint override to the default hint.

    Reads ``agent._platform_hint_overrides`` (populated from
    ``config.yaml`` ``platform_hints`` by ``agent_init``) and resolves the
    effective hint for *platform_key*:

      * ``replace`` — substitute the default hint entirely.
      * ``append``  — keep the default and append the extra text.
      * a bare string value — treated as ``append`` (convenience shorthand).

    Precedence: ``replace`` wins over ``append`` if both are present.
    Override text is added on top of (not instead of) the SOUL/context/
    memory tiers — it only affects the platform-hint segment, so other
    platforms are unaffected and general system instructions still apply.

    Defensive: any malformed entry falls back to the unmodified default so
    a bad config value can never break prompt assembly or leak across
    platforms.
    """
    if not platform_key:
        return default_hint
    overrides = getattr(agent, "_platform_hint_overrides", None)
    if not isinstance(overrides, dict) or not overrides:
        return default_hint
    spec = overrides.get(platform_key)
    if spec is None:
        return default_hint

    # Shorthand: a bare string is treated as append text.
    if isinstance(spec, str):
        extra = spec.strip()
        return f"{default_hint}\n\n{extra}".strip() if extra else default_hint

    if not isinstance(spec, dict):
        return default_hint

    replace_text = spec.get("replace")
    if isinstance(replace_text, str) and replace_text.strip():
        base = replace_text.strip()
    else:
        base = default_hint

    append_text = spec.get("append")
    if isinstance(append_text, str) and append_text.strip():
        return f"{base}\n\n{append_text.strip()}".strip()
    return base


_TUI_EMBEDDED_PANE_CLARIFIER = (
    " You're in its embedded terminal pane, beside the GUI chat — the user can "
    "select your output (Option-drag on macOS, Shift-drag elsewhere) and press "
    "Cmd/Ctrl+L to send it to the chat composer."
)


def _tui_embedded_pane_clarifier(hint: str) -> str:
    """Append the desktop-embedded-terminal-pane clarifier to a tui hint.

    Triggered by ``HERMES_DESKTOP_TERMINAL=1`` (set by ``main.cjs`` only on the
    shell env of the desktop's embedded TUI PTY — never on the chat backend).
    This is a runtime-surface qualifier, not a config override, so it lives at
    the resolution site rather than inside ``_resolve_platform_hint`` (which
    is purely the config-platform_hints override applier). Byte-stable for the
    cache: called once per session build, deterministically from env state.

    Idempotent and empty-safe: re-applying on an already-augmented hint is a
    no-op, and an empty input returns empty (we never synthesize the
    clarifier without its tui framing).
    """
    if not hint:
        return hint
    if _TUI_EMBEDDED_PANE_CLARIFIER in hint:
        return hint
    if not is_truthy_value(os.getenv("HERMES_DESKTOP_TERMINAL")):
        return hint
    return hint + _TUI_EMBEDDED_PANE_CLARIFIER


def _plugin_session_info(agent: Any) -> Dict[str, str]:
    """Return the immutable-at-render-time metadata exposed to plugins."""
    try:
        cwd = str(resolve_context_cwd() or "")
    except Exception:
        cwd = ""
    try:
        from hermes_cli.profiles import get_active_profile_name

        profile_name = str(get_active_profile_name() or "default")
    except Exception:
        profile_name = "default"
    return {
        "session_id": str(getattr(agent, "session_id", None) or ""),
        "model": str(getattr(agent, "model", None) or ""),
        "provider": str(getattr(agent, "provider", None) or ""),
        "platform": str(getattr(agent, "platform", None) or ""),
        "profile_name": profile_name,
        "cwd": cwd,
    }


def _frozen_plugin_prompt_sections(agent: Any) -> tuple:
    attr = "_plugin_system_prompt_sections_snapshot"
    if hasattr(agent, attr):
        return getattr(agent, attr)
    try:
        from hermes_cli.plugins import render_system_prompt_sections

        rendered = tuple(render_system_prompt_sections(_plugin_session_info(agent)))
    except Exception as exc:
        logger.warning("Plugin system prompt sections could not be rendered: %s", exc)
        rendered = ()
    setattr(agent, attr, rendered)
    return rendered


def _frozen_plugin_environment_hints(agent: Any, core_hints: str) -> tuple:
    attr = "_plugin_environment_hints_snapshot"
    if hasattr(agent, attr):
        return getattr(agent, attr)
    try:
        from hermes_cli.plugins import collect_environment_hints

        rows = tuple(
            collect_environment_hints(core_hints, _plugin_session_info(agent))
        )
    except Exception as exc:
        logger.warning("Plugin environment hints could not be rendered: %s", exc)
        rows = ()
    setattr(agent, attr, rows)
    return rows


def _plugin_section_blocks(sections: tuple, position: str) -> List[str]:
    return [
        f"## Plugin Context: {section.id}\n\n{section.content}"
        for section in sections
        if section.position == position
    ]


def serialize_plugin_prompt_state(agent: Any) -> Optional[str]:
    """Serialize frozen plugin prompt output for session persistence."""
    if not hasattr(agent, "_plugin_system_prompt_sections_snapshot"):
        return None
    sections = getattr(agent, "_plugin_system_prompt_sections_snapshot", ())
    environment_hints = getattr(agent, "_plugin_environment_hints_snapshot", ())
    payload = {
        "version": _PLUGIN_PROMPT_STATE_VERSION,
        "sections": [
            {
                "id": section.id,
                "content": section.content,
                "position": section.position,
                "max_chars": section.max_chars,
                "plugin": section.plugin,
            }
            for section in sections
        ],
        "environment_hints": [list(row) for row in environment_hints],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def restore_plugin_prompt_state(agent: Any, raw_state: Any) -> bool:
    """Restore validated plugin output before a resumed-session rebuild."""
    if not raw_state:
        return False
    try:
        payload = json.loads(raw_state) if isinstance(raw_state, str) else raw_state
        if not isinstance(payload, dict) or payload.get("version") != _PLUGIN_PROMPT_STATE_VERSION:
            raise ValueError("unsupported plugin prompt state version")

        from hermes_cli.plugins import (
            MAX_PLUGIN_ENVIRONMENT_HINT_ROW_CHARS,
            MAX_PLUGIN_ENVIRONMENT_HINT_ROWS,
            MAX_PLUGIN_ENVIRONMENT_HINTS_TOTAL_CHARS,
            MAX_SYSTEM_PROMPT_SECTION_CHARS,
            MAX_SYSTEM_PROMPT_SECTIONS,
            MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS,
            SYSTEM_PROMPT_SECTION_POSITIONS,
            RenderedPluginSystemPromptSection,
            is_valid_system_prompt_section_id,
            system_prompt_section_rendered_chars,
        )

        section_items = payload.get("sections", [])
        if not isinstance(section_items, list):
            raise ValueError("sections must be a list")
        if len(section_items) > MAX_SYSTEM_PROMPT_SECTIONS:
            raise ValueError("too many system prompt sections")
        sections = []
        section_chars = 0
        for item in section_items:
            if not isinstance(item, dict):
                raise ValueError("section entry must be an object")
            section_id = item.get("id")
            content = item.get("content")
            position = item.get("position")
            plugin = item.get("plugin")
            max_chars = item.get("max_chars")
            if not all(isinstance(value, str) for value in (section_id, content, plugin)):
                raise ValueError("section id, content, and plugin must be strings")
            if not is_valid_system_prompt_section_id(section_id) or not plugin:
                raise ValueError("invalid section identity")
            if position not in SYSTEM_PROMPT_SECTION_POSITIONS:
                raise ValueError("invalid section position")
            if (
                isinstance(max_chars, bool)
                or not isinstance(max_chars, int)
                or not 0 < max_chars <= MAX_SYSTEM_PROMPT_SECTION_CHARS
                or len(content) > max_chars
            ):
                raise ValueError("invalid section budget")
            section_chars += system_prompt_section_rendered_chars(section_id, content)
            if section_chars > MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS:
                raise ValueError("section total budget exceeded")
            sections.append(
                RenderedPluginSystemPromptSection(
                    id=section_id,
                    content=content,
                    position=position,
                    max_chars=max_chars,
                    plugin=plugin,
                )
            )

        hint_items = payload.get("environment_hints", [])
        if not isinstance(hint_items, list) or len(hint_items) > MAX_PLUGIN_ENVIRONMENT_HINT_ROWS:
            raise ValueError("invalid environment hints")
        rows = []
        hint_chars = 0
        for row in hint_items:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("invalid environment hint row")
            key, value = row
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("environment hint values must be strings")
            if not key.strip() or not value.strip() or "\n" in key or "\n" in value:
                raise ValueError("invalid environment hint value")
            row_chars = len(key) + len(value) + 2
            if row_chars > MAX_PLUGIN_ENVIRONMENT_HINT_ROW_CHARS:
                raise ValueError("environment hint row budget exceeded")
            hint_chars += row_chars
            if hint_chars > MAX_PLUGIN_ENVIRONMENT_HINTS_TOTAL_CHARS:
                raise ValueError("environment hints total budget exceeded")
            rows.append((key, value))
    except Exception as exc:
        logger.warning("Stored plugin prompt state was invalid and was ignored: %s", exc)
        return False

    agent._plugin_system_prompt_sections_snapshot = tuple(sections)
    agent._plugin_environment_hints_snapshot = tuple(rows)
    return True


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

    # Resolve the model's context window once so context-file caps can scale
    # to it (dynamic cap — see prompt_builder._dynamic_context_file_max_chars).
    # None falls back to the historical flat default. This value is stable for
    # the life of the conversation, so it does not threaten prompt caching.
    _ctx_len: Optional[int] = None
    _cc = getattr(agent, "context_compressor", None)
    if _cc is not None:
        _cc_len = getattr(_cc, "context_length", None)
        if isinstance(_cc_len, int) and _cc_len > 0:
            _ctx_len = _cc_len

    # Plugin sections are evaluated at most once for this AIAgent. The frozen
    # snapshot survives compression rebuilds and is persisted for process
    # restarts by ``serialize_plugin_prompt_state``.
    _plugin_sections = _frozen_plugin_prompt_sections(agent)

    # ── Stable tier ────────────────────────────────────────────────
    stable_parts: List[str] = []

    # Try SOUL.md as primary identity unless the caller explicitly skipped it.
    # Some execution modes (cron) still want HERMES_HOME persona while keeping
    # cwd project instructions disabled.
    _soul_loaded = False
    if agent.load_soul_identity or not agent.skip_context_files:
        _soul_content = _r.load_soul_md(_ctx_len)
        if _soul_content:
            stable_parts.append(_soul_content)
            _soul_loaded = True

    if not _soul_loaded:
        # Fallback to hardcoded identity
        stable_parts.append(DEFAULT_AGENT_IDENTITY)

    # Pointer to the hermes-agent skill + docs for user questions about Hermes itself.
    stable_parts.append(HERMES_AGENT_HELP_GUIDANCE)
    stable_parts.extend(_plugin_section_blocks(_plugin_sections, "after_identity"))

    # Universal task-completion / no-fabrication guidance.  Applied to ALL
    # models regardless of tool_use_enforcement gating — the failure modes
    # this targets (stopping after a stub; fabricating output when a real
    # path is blocked) are not model-family specific.  Gated only by
    # config.yaml ``agent.task_completion_guidance`` (default True) so
    # users who want a leaner prompt can turn it off.
    if getattr(agent, "_task_completion_guidance", True) and agent.valid_tool_names:
        stable_parts.append(TASK_COMPLETION_GUIDANCE)

    # Universal parallel-tool-call guidance.  Tells the model to batch
    # independent tool calls into one assistant turn rather than emitting one
    # call per turn — the runtime already runs independent calls concurrently
    # (read-only tools always; non-overlapping path-scoped file ops), so the
    # only thing missing was steering the model to produce the batch.  Cuts
    # round-trips and the resent-context cost that compounds over a long
    # conversation.  Gated by config.yaml ``agent.parallel_tool_call_guidance``
    # (default True) and only injected when tools are actually loaded.
    if getattr(agent, "_parallel_tool_call_guidance", True) and agent.valid_tool_names:
        stable_parts.append(PARALLEL_TOOL_CALL_GUIDANCE)

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
        stable_parts.append(" ".join(tool_guidance))

    # Steering only lands inside tool results, so it's only reachable when the
    # agent has tools. Static text → byte-stable prompt (no cache hit).
    if agent.valid_tool_names:
        stable_parts.append(STEER_CHANNEL_NOTE)

    # Computer-use — goes in as its own block rather than being merged into
    # tool_guidance because the content is multi-paragraph. The guidance is
    # rendered for the host platform so Windows/Linux hosts don't see
    # macOS-only wording (Mac, Space, cmd+s).
    if "computer_use" in agent.valid_tool_names:
        from agent.prompt_builder import computer_use_guidance
        stable_parts.append(computer_use_guidance())

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
            # Also applied to xAI Grok — same failure modes (claims completion
            # without tool calls, suggests workarounds instead of using
            # existing tools, replies with plans instead of executing).
            if "gpt" in _model_lower or "codex" in _model_lower or "grok" in _model_lower:
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
        # Focus mode (opt-in) demotes non-coding skill categories to
        # names-only in the index (never hidden — skill_view/skills_list
        # reach everything, and every name stays visible for recall). The
        # default coding posture leaves the index untouched.
        _compact_cats = frozenset()
        try:
            from agent.coding_context import coding_compact_skill_categories

            _compact_cats = coding_compact_skill_categories(
                platform=agent.platform, cwd=resolve_context_cwd()
            )
        except Exception:
            _compact_cats = frozenset()
        skills_prompt = _r.build_skills_system_prompt(
            available_tools=agent.valid_tool_names,
            available_toolsets=avail_toolsets,
            compact_categories=_compact_cats or None,
        )
    else:
        skills_prompt = ""
    if skills_prompt:
        stable_parts.append(skills_prompt)
    stable_parts.extend(_plugin_section_blocks(_plugin_sections, "after_tools"))

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
    _plugin_env_rows = _frozen_plugin_environment_hints(agent, _env_hints)
    if _plugin_env_rows:
        _plugin_env_block = "\n".join(
            f"{key}: {value}" for key, value in _plugin_env_rows
        )
        _env_hints = "\n\n".join(
            part for part in (_env_hints, _plugin_env_block) if part
        )
    if _env_hints:
        stable_parts.append(_env_hints)

    # Coding posture (base Hermes, any interactive coding surface in a code
    # workspace — see agent/coding_context.py). The operating brief + the live
    # git/workspace snapshot are built once here and cached for the session;
    # the snapshot is never re-probed per turn (that would break the prompt
    # cache), so the brief tells the model to re-check git before relying on it.
    if agent.valid_tool_names:
        try:
            from agent.coding_context import coding_system_blocks

            stable_parts.extend(
                coding_system_blocks(
                    platform=agent.platform,
                    cwd=resolve_context_cwd(),
                    model=agent.model,
                )
            )
        except Exception:
            # Coding-context probing must never block prompt build.
            pass

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
            "under " + str(get_hermes_home()) + "/profiles/<name>/. Each profile has its own "
            "skills/, plugins/, cron/, and memories/ that affect a different "
            "session than this one. Do not modify another profile's "
            "skills/plugins/cron/memories unless the user explicitly directs "
            "you to."
        )
    else:
        stable_parts.append(
            f"Active Hermes profile: {active_profile}. This session reads "
            f"and writes {get_hermes_home()}/profiles/{active_profile}/. The default "
            f"profile's data lives at {get_hermes_home()}/skills/, {get_hermes_home()}/plugins/, "
            f"{get_hermes_home()}/cron/, {get_hermes_home()}/memories/ — those belong to a "
            f"different session run from a different shell. Do NOT modify "
            f"another profile's skills/plugins/cron/memories unless the user "
            f"explicitly directs you to. The cross-profile write guard will "
            f"refuse such writes by default; pass cross_profile=True only "
            f"after explicit direction."
        )

    platform_key = (agent.platform or "").lower().strip()
    # Resolve the built-in/plugin default hint for this platform, then apply
    # any per-platform override from config (platform_hints.<platform>).
    _default_hint = ""
    if platform_key in PLATFORM_HINTS:
        _default_hint = PLATFORM_HINTS[platform_key]
    elif platform_key:
        # Check plugin registry for platform-specific LLM guidance
        try:
            from gateway.platform_registry import platform_registry
            _entry = platform_registry.get(platform_key)
            if _entry and _entry.platform_hint:
                _default_hint = _entry.platform_hint
        except Exception:
            pass

    # For Telegram: append the rich-messages extension only when the user has
    # opted in to ``platforms.telegram.extra.rich_messages: true``.  The base
    # hint covers MarkdownV2-compatible constructs; the extension adds Bot API
    # 10.1 guidance (tables, task lists, math, collapsible details, etc.).
    if platform_key == "telegram" and _default_hint:
        try:
            from hermes_cli.config import load_config_readonly
            _cfg = load_config_readonly()
            _tg_extra = ((_cfg.get("platforms") or {}).get("telegram") or {}).get("extra") or {}
            if _tg_extra.get("rich_messages"):
                _default_hint = _default_hint.rstrip() + " " + TELEGRAM_RICH_MESSAGES_HINT
        except Exception:
            pass  # Config read failure — fall back to base hint only

    _effective_hint = _resolve_platform_hint(agent, platform_key, _default_hint)
    if platform_key == "tui" and _effective_hint:
        _effective_hint = _tui_embedded_pane_clarifier(_effective_hint)
    if _effective_hint:
        stable_parts.append(_effective_hint)

    # ── Context tier (cwd-dependent, may change between sessions) ─
    context_parts: List[str] = []

    # Note: ephemeral_system_prompt is NOT included here. It's injected at
    # API-call time only so it stays out of the cached/stored system prompt.
    if system_message is not None:
        context_parts.append(system_message)

    if not agent.skip_context_files:
        # Prefer the configured TERMINAL_CWD (gateway mode). When unset (local
        # CLI), None lets build_context_files_prompt fall back to the launch
        # dir — the user's real cwd there, but the install dir for the gateway
        # daemon, which is why the gateway sets TERMINAL_CWD.
        #
        # allow_install_tree_fallback: for cli/tui the launch dir IS the
        # user's shell cwd, so an in-tree fallback is a deliberate choice
        # (developing Hermes). Every other surface (desktop chat panel,
        # gateway daemons) self-spawns into the install tree, where the
        # fallback would inject this repo's contributor AGENTS.md (#64590).
        context_files_prompt = _r.build_context_files_prompt(
            cwd=resolve_context_cwd(), skip_soul=_soul_loaded,
            context_length=_ctx_len,
            allow_install_tree_fallback=agent.platform in ("cli", "tui"))
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

    volatile_parts.extend(_plugin_section_blocks(_plugin_sections, "after_memory"))

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
    joined = "\n\n".join(p for p in (parts["stable"], parts["context"], parts["volatile"]) if p)

    # Surface context-file truncation warnings through the normal agent status
    # channel so gateway/CLI users see them in chat instead of only in logs.
    for warning in drain_truncation_warnings():
        agent._emit_status(warning)

    return joined


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
    "serialize_plugin_prompt_state",
    "restore_plugin_prompt_state",
]
