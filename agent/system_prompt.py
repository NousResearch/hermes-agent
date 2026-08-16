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
  enforcement guidance + per-model operational guidance,
  alibaba model-name workaround, environment hints, coding guidance,
  platform hints.
* ``context``  — caller-supplied ``system_message`` plus context files
  (AGENTS.md / .cursorrules / etc.) discovered under ``TERMINAL_CWD``,
  plus the session's coding-workspace snapshot.
* ``volatile`` — skills index, memory snapshot, USER.md profile, external
  memory provider block, timestamp/session/model/provider line.

Pure helpers that read the agent's state.  AIAgent keeps thin forwarders.
"""

from __future__ import annotations

import json
import logging
import os
import re
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
from hermes_constants import get_default_hermes_root, get_hermes_home
from pathlib import Path
from utils import is_truthy_value

logger = logging.getLogger(__name__)
_PLUGIN_SECTION_FRAME_RE = re.compile(
    r"^## Plugin Context: (?P<id>[a-z0-9][a-z0-9._-]{0,127})\n"
    r"<!-- hermes-plugin-section-chars:(?P<chars>[0-9]{1,4}) -->\n\n",
    re.MULTILINE,
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
    """Return immutable-at-render-time metadata exposed to prompt sections."""
    try:
        cwd = str(resolve_context_cwd() or "")
    except Exception:
        cwd = ""
    try:
        # Prefer the agent's own home (override-aware, session_db fallback) —
        # ambient get_active_profile_name() misreports on threads that lost
        # the HERMES_HOME ContextVar (#86313 class; plugin half per @helix4u).
        _home = _agent_home(agent)
        if _home is not None:
            profile_name = _profile_name_for_home(_home)
        else:
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
    """Render once on a new session; never re-evaluate a restored prompt.

    Compression rebuilds reuse the per-agent tuple. A fresh process restores
    ``_cached_system_prompt`` before reconstructing its static cache prefix;
    because plugin sections live after memory in the volatile tail, that
    reconstruction can safely omit them and must not call plugin code again.
    """
    attr = "_plugin_system_prompt_sections_snapshot"
    if hasattr(agent, attr):
        return getattr(agent, attr)
    stored_prompt = getattr(agent, "_cached_system_prompt", None)
    if isinstance(stored_prompt, str) and stored_prompt:
        rendered = _restore_plugin_prompt_sections(stored_prompt)
        setattr(agent, attr, rendered)
        return rendered
    try:
        from hermes_cli.plugins import render_system_prompt_sections

        rendered = tuple(render_system_prompt_sections(_plugin_session_info(agent)))
    except Exception as exc:
        logger.warning("Plugin system prompt sections could not be rendered: %s", exc)
        rendered = ()
    setattr(agent, attr, rendered)
    return rendered


def _restore_plugin_prompt_sections(prompt: str) -> tuple:
    """Recover frozen section bytes from the already-persisted full prompt."""
    from hermes_cli.plugins import (
        MAX_SYSTEM_PROMPT_SECTION_CHARS,
        PLUGIN_SECTIONS_END,
        PLUGIN_SECTIONS_START,
        RenderedPluginSystemPromptSection,
        format_system_prompt_sections,
    )

    start = prompt.rfind(PLUGIN_SECTIONS_START)
    if start < 0:
        return ()
    end = prompt.find(PLUGIN_SECTIONS_END, start + len(PLUGIN_SECTIONS_START))
    if end < 0:
        return ()
    after_end = end + len(PLUGIN_SECTIONS_END)
    if not prompt[after_end:].startswith("\n\nConversation started:"):
        return ()
    framed = prompt[start:after_end]

    restored = []
    for match in _PLUGIN_SECTION_FRAME_RE.finditer(framed):
        content_len = int(match.group("chars"))
        if content_len > MAX_SYSTEM_PROMPT_SECTION_CHARS:
            continue
        content_start = match.end()
        content = framed[content_start : content_start + content_len]
        if len(content) != content_len:
            continue
        restored.append(
            RenderedPluginSystemPromptSection(
                id=match.group("id"),
                content=content,
                position="after_memory",
                plugin="persisted-prompt",
            )
        )
    # User/project text may resemble a frame. Accept only the exact canonical
    # container emitted by core, never a partial or malformed lookalike.
    if format_system_prompt_sections(restored) != framed:
        return ()
    return tuple(restored)


def restore_plugin_prompt_sections(agent: Any, prompt: str) -> None:
    """Seed a resumed agent's frozen snapshot from persisted prompt bytes."""
    agent._plugin_system_prompt_sections_snapshot = _restore_plugin_prompt_sections(prompt)


def _plugin_section_blocks(sections: tuple, position: str) -> List[str]:
    from hermes_cli.plugins import format_system_prompt_sections

    selected = [section for section in sections if section.position == position]
    block = format_system_prompt_sections(selected)
    return [block] if block else []


def _agent_home(agent: Any) -> Optional[Path]:
    """The agent's OWN profile home.

    Resolution order:

    1. A bound HERMES_HOME ContextVar override wins. Surfaces that multiplex
       several profiles over ONE shared session DB (the messaging gateway:
       ``gateway/run.py`` hands every agent the launch-home ``state.db`` and
       binds the profile home per turn via ``_profile_runtime_scope`` +
       ``copy_context``) would otherwise have the db-derived launch home
       STOMP the correctly-bound profile — inverting the leak this helper
       exists to fix (found by @kshitijk4poor's post-merge probe on #86313).
    2. Fallback: the home containing the agent's ``_session_db.db_path``
       (``<home>/state.db``) — ground truth on threads that lost the
       ContextVar (ContextVars don't propagate into ``threading.Thread``),
       where the unbound build previously fell back to the launch home and
       leaked the default profile's skills/identity into a bot prompt.

    Returns None when neither resolves so callers fall back to ambient.
    """
    try:
        from hermes_constants import get_hermes_home_override

        override = get_hermes_home_override()
        if override:
            return Path(override)
    except Exception:
        pass
    try:
        db = getattr(agent, "_session_db", None)
        db_path = getattr(db, "db_path", None)
        if db_path:
            return Path(db_path).parent
    except Exception:
        pass
    return None


def _agent_skills_dir(agent: Any) -> Optional[Path]:
    """The agent's own ``<home>/skills`` dir, or None to use ambient home."""
    home = _agent_home(agent)
    return (home / "skills") if home is not None else None


def _profile_name_for_home(home: Path) -> str:
    """Derive the profile name for an explicit agent home.

    ``<root>/profiles/X`` -> ``"X"``; anything else -> ``"default"``.

    Uses :func:`get_default_hermes_root` (NOT ``get_hermes_home()``): on a
    correctly bound profile session the ambient home IS the profile dir, so
    ``get_hermes_home()/profiles`` would never contain ``home`` and every
    profile would misreport as "default".
    """
    try:
        from hermes_constants import get_default_hermes_root

        root = get_default_hermes_root()
        rel = home.resolve().relative_to((root / "profiles").resolve())
        return rel.parts[0] if rel.parts else "default"
    except (ValueError, OSError):
        # Home IS the root (default profile) or unrelatable -> default.
        return "default"


def build_system_prompt_parts(agent: Any, system_message: Optional[str] = None) -> Dict[str, str]:
    """Assemble the system prompt as three ordered cache tiers.

    Returns a dict with three keys:
      * ``stable``   — the cross-session-stable prefix, through the coding
        operating brief when a workspace snapshot follows.
      * ``context``  — the workspace snapshot followed by the remaining
        session-stable guidance, context files, and caller-supplied
        system_message.
      * ``volatile`` — skills index, memory snapshot, user profile,
        external memory provider block, timestamp line.

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

    # ── Stable tier ────────────────────────────────────────────────
    stable_parts: List[str] = []

    # Try SOUL.md as primary identity unless the caller explicitly skipped it.
    # Some execution modes (cron) still want HERMES_HOME persona while keeping
    # cwd project instructions disabled.
    _soul_loaded = False
    if agent.load_soul_identity or not agent.skip_context_files:
        # Scope the SOUL.md read to the agent's OWN home (see _agent_home) —
        # ambient resolution on a thread that lost the HERMES_HOME ContextVar
        # reads the launch profile's SOUL.md instead (#50233).
        _soul_content = _r.load_soul_md(_ctx_len, home_override=_agent_home(agent))
        if _soul_content:
            stable_parts.append(_soul_content)
            _soul_loaded = True

    if not _soul_loaded:
        # Fallback to hardcoded identity
        stable_parts.append(DEFAULT_AGENT_IDENTITY)

    # Pointer to the hermes-agent skill + docs for user questions about Hermes itself.
    stable_parts.append(HERMES_AGENT_HELP_GUIDANCE)

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
            skills_dir_override=_agent_skills_dir(agent),
        )
    else:
        skills_prompt = ""

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

    # Coding posture (base Hermes, any interactive coding surface in a code
    # workspace — see agent/coding_context.py). Keep the operating brief in
    # the cross-session-stable prefix, while placing the live git/workspace
    # snapshot behind its own cache boundary. The post-snapshot blocks must
    # stay in their historical position after the workspace snapshot.
    coding_workspace_parts: List[str] = []
    coding_trailing_parts: List[str] = []
    if agent.valid_tool_names:
        try:
            from agent.coding_context import coding_system_prompt_parts

            coding_prefix_parts, coding_workspace_parts, coding_trailing_parts = coding_system_prompt_parts(
                platform=agent.platform,
                cwd=resolve_context_cwd(),
                model=agent.model,
            )
            stable_parts.extend(coding_prefix_parts)
        except Exception:
            # Coding-context probing must never block prompt build.
            pass

    # Guidance assembled after the coding posture historically followed the
    # workspace snapshot. With no snapshot, the coding tail instead remains
    # directly after the coding prefix in the cacheable prefix.
    if coding_workspace_parts:
        post_workspace_parts: List[str] = []
    else:
        stable_parts.extend(coding_trailing_parts)
        post_workspace_parts = stable_parts

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
                post_workspace_parts.append(_probe_line)
        except Exception:
            # Probe failure must never block prompt build.
            pass

    # Bot Mode teammate protocol — when the desktop's Bot Mode manages this
    # install (any profile carries ui_meta['hermes-bots']), every session of
    # every profile learns the bot-to-bot messaging protocol, including
    # headless `hermes -p <bot> chat` sessions started by a teammate. Silent
    # on non-Bot-Mode installs and when the profile's SOUL.md already carries
    # the section (legacy plugin-side append). Deterministic per (process,
    # home) — cache-safe. Gated by config.yaml ``agent.bot_mode_protocol``
    # (default True).
    if getattr(agent, "_bot_mode_protocol", True):
        try:
            from tools.bot_mode_probe import get_bot_mode_protocol_section
            _bot_section = get_bot_mode_protocol_section(_agent_home(agent))
            if _bot_section:
                post_workspace_parts.append(_bot_section)
        except Exception:
            pass

    # Active-profile hint — names the Hermes profile the agent is running
    # under so it doesn't conflate ~/.hermes/skills/ (default profile) with
    # ~/.hermes/profiles/<active>/skills/ (this profile's). Deterministic
    # for the lifetime of the agent — profile name doesn't change
    # mid-session, so this doesn't break the prompt cache.
    # See file_safety._resolve_active_profile_name + classify_cross_profile_target
    # for the matching tool-side guard.
    #
    # Resolve from the agent's OWN home first (its session_db path), not the
    # ambient HERMES_HOME: on a build thread that lost the ContextVar this
    # line would otherwise print "default" for a bot profile — the same
    # thread-fallback bug that leaked default's skills index.
    _agent_home_path = _agent_home(agent)
    active_profile = "default"
    try:
        if _agent_home_path is not None:
            active_profile = _profile_name_for_home(_agent_home_path)
        else:
            from agent.file_safety import _resolve_active_profile_name
            active_profile = _resolve_active_profile_name()
    except Exception:
        active_profile = "default"
    # Home string for the message text: prefer the agent's own home so the
    # paths named match the profile just resolved. When we have an explicit
    # agent home, the root (where the default profile's data lives) comes
    # from get_default_hermes_root(): get_hermes_home() on a bound profile
    # session is the PROFILE dir, which would misname the default profile's
    # paths. Without an agent home, keep the ambient resolution byte-identical
    # to the legacy behavior (and patchable via this module's get_hermes_home).
    if _agent_home_path is not None:
        _home_str = str(_agent_home_path)
        _root_str = str(get_default_hermes_root())
    else:
        _home_str = _root_str = str(get_hermes_home())
    if active_profile == "default":
        post_workspace_parts.append(
            "Active Hermes profile: default. Other profiles (if any) live "
            "under " + _root_str + "/profiles/<name>/. Each profile has its own "
            "skills/, plugins/, cron/, and memories/ that affect a different "
            "session than this one. Do not modify another profile's "
            "skills/plugins/cron/memories unless the user explicitly directs "
            "you to."
        )
    else:
        # A non-default name is only ever returned when the resolved home is
        # ALREADY <root>/profiles/<name> — that is exactly how both
        # _profile_name_for_home() and _resolve_active_profile_name() derive
        # it. So the profile home is the session home itself; appending
        # /profiles/<name> again doubled it (#72894). The default profile's
        # data sits at the ROOT (get_default_hermes_root()), which in ambient
        # profile mode is NOT get_hermes_home().
        profile_home = _home_str
        default_root = get_default_hermes_root()
        post_workspace_parts.append(
            f"Active Hermes profile: {active_profile}. This session reads "
            f"and writes {profile_home}/. The default "
            f"profile's data lives at {default_root}/skills/, {default_root}/plugins/, "
            f"{default_root}/cron/, {default_root}/memories/ — those belong to a "
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
    # opted in to ``gateway.platforms.telegram.extra.rich_messages: true``
    # (the canonical location the adapter reads from).  Merge with the
    # top-level ``platforms.telegram.extra`` so config-wizard writes and
    # dashboard-setup keys are also visible — same precedence the adapter
    # uses: top-level platform overrides gateway.platforms at the leaf.
    if platform_key == "telegram" and _default_hint:
        try:
            from hermes_cli.config import load_config_readonly
            _cfg = load_config_readonly()
            _gw_tg_extra = (((_cfg.get("gateway") or {}).get("platforms") or {}).get("telegram") or {}).get("extra")
            _top_tg_extra = ((_cfg.get("platforms") or {}).get("telegram") or {}).get("extra")
            if not isinstance(_gw_tg_extra, dict):
                _gw_tg_extra = {}
            if not isinstance(_top_tg_extra, dict):
                _top_tg_extra = {}
            _tg_extra = {**_gw_tg_extra, **_top_tg_extra}
            if _tg_extra.get("rich_messages"):
                _default_hint = _default_hint.rstrip() + " " + TELEGRAM_RICH_MESSAGES_HINT
        except Exception:
            pass  # Config read failure — fall back to base hint only

    _effective_hint = _resolve_platform_hint(agent, platform_key, _default_hint)
    if platform_key == "tui" and _effective_hint:
        _effective_hint = _tui_embedded_pane_clarifier(_effective_hint)
    if _effective_hint:
        post_workspace_parts.append(_effective_hint)

    # ── Context tier (cwd-dependent, may change between sessions) ─
    context_parts: List[str] = []

    if coding_workspace_parts:
        context_parts.extend(coding_workspace_parts)
        context_parts.extend(coding_trailing_parts)
        context_parts.extend(post_workspace_parts)

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
            allow_install_tree_fallback=agent.platform in ("cli", "tui"),
            home_override=_agent_home(agent))
        if context_files_prompt:
            context_parts.append(context_files_prompt)

    # ── Volatile tier (most likely to differ on a rebuild; kept last so the stable prefix stays reusable) ──
    volatile_parts: List[str] = []
    # Skills are runtime-mutable: the agent adds and patches them across a
    # session (SKILLS_GUIDANCE tells it to patch a skill the moment it goes
    # stale). The built prompt is cached per session and only rebuilt on
    # compaction/restore (see build_system_prompt), so a skill change is not
    # byte-stable across rebuilds. With the index in the stable band, a rebuild
    # that picked up a skill change would bust the cached prefix from the index
    # down, taking the whole scaffold with it. Render it at the FRONT of the
    # volatile band instead, ahead of the turn-varying memory/timestamp tail:
    # on an implicit longest-prefix backend an unchanged index still falls
    # inside the reused prefix, and a changed one only re-prefills from here on.
    # (No effect for single-block cache_control backends, where the whole
    # system message is one cache unit regardless of internal order.)
    if skills_prompt:
        volatile_parts.append(skills_prompt)

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

    # Plugin sections are intentionally confined to one coarse anchor in the
    # volatile tail. This preserves deterministic ordering and lets a resumed
    # process reconstruct the stable cache prefix without re-running plugins.
    volatile_parts.extend(
        _plugin_section_blocks(_frozen_plugin_prompt_sections(agent), "after_memory")
    )

    from hermes_time import get_timezone as _hermes_tz, now as _hermes_now
    now = _hermes_now()
    # Date-only (not minute-precision) so the system prompt is byte-stable
    # for the full day.  Minute-precision changes invalidate prefix-cache KV
    # on every rebuild path (compression boundary, fresh-agent gateway turns,
    # session resume without a stored prompt).  The model can still query the
    # exact wall-clock time via tools when it actually needs it.
    # Credit: @iamfoz (PR #20451).
    #
    # Zone and UTC offset ARE included: tools that accept instants reject naive
    # datetimes and require an explicit offset, and with the bare date the model
    # has to infer EST vs EDT on its own (a coin-flip near a DST boundary, and a
    # wrong guess silently writes the record onto the wrong day).  Both values
    # are constant for the whole day -- they shift only at a DST transition --
    # so the byte-stability the comment above depends on is preserved.
    # ``get_timezone()`` returns None when no timezone is configured, in which
    # case we fall back to the abbreviation of the server-local (still tz-aware)
    # time.
    _tz = _hermes_tz()
    _zone_bits = []
    _iana = getattr(_tz, "key", None)
    if _iana:
        _zone_bits.append(_iana)
    _abbrev = now.strftime("%Z")
    if _abbrev and _abbrev != _iana:
        _zone_bits.append(_abbrev)
    _offset = now.strftime("%z")
    if _offset:  # '-0400' -> 'UTC-04:00'
        _zone_bits.append(f"UTC{_offset[:3]}:{_offset[3:]}")
    _zone_suffix = f" ({', '.join(_zone_bits)})" if _zone_bits else ""
    timestamp_line = (
        f"Conversation started: {now.strftime('%A, %B %d, %Y')}{_zone_suffix}"
    )
    if agent.pass_session_id and agent.session_id:
        timestamp_line += f"\nSession ID: {agent.session_id}"
    if agent.model:
        timestamp_line += f"\nModel: {agent.model}"
    if agent.provider:
        timestamp_line += f"\nProvider: {agent.provider}"
    if agent.platform:
        timestamp_line += f"\nPlatform: {agent.platform}"
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
    (skills index, memory, USER profile, timestamp). For explicit
    cache_control backends the whole string is one cached block. For
    implicit longest-prefix backends the order is what matters: the
    content most likely to change is rendered last, so when the prompt is
    rebuilt (on compaction/restore) the unchanged stable scaffold ahead of
    the change stays in the reused prefix.
    """
    parts = build_system_prompt_parts(agent, system_message=system_message)
    joined = "\n\n".join(p for p in (parts["stable"], parts["context"], parts["volatile"]) if p)
    agent._cached_system_prompt_static = parts["stable"]

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
    agent._cached_system_prompt_static = None
    if agent._memory_store:
        agent._memory_store.load_from_disk()


def reconstruct_static_prefix(
    agent: Any,
    system_message: Optional[str] = None,
    *,
    log_label: str = "restore",
) -> None:
    """Reconstruct ``_cached_system_prompt_static`` for a stored prompt.

    The static prefix is not persisted (only the full prompt is), so any
    path that adopts a stored/kept ``_cached_system_prompt`` — session
    restore, the compression keep-prompt path, or a failover to a cache-on
    provider mid-turn (#72626) — must rebuild the stable tier to regain the
    two-block ``[static, volatile]`` system layout.

    Safety: the rebuilt stable tier is used ONLY when the stored prompt
    literally starts with it (checked here AND re-checked by
    ``_apply_system_cache_markers``'s ``startswith`` gate). If any
    stable-tier input changed since the prompt was persisted (identity
    changed, SOUL.md edited), the prefix mismatches, the static stays
    None, and requests fall back to the legacy layout with the stored
    prompt bytes untouched — never a rewritten prompt.

    A failed reconstruction is memoized per stored prompt
    (``_static_rebuild_failed_for``): ``build_system_prompt_parts`` does
    real file I/O (SOUL.md, context files, memory), and callers on the
    retry-loop hot path must not re-run it every attempt when the inputs
    haven't changed. A legitimately changed stored prompt retries once.
    """
    if not getattr(agent, "_use_prompt_caching", False):
        return
    stored = getattr(agent, "_cached_system_prompt", None)
    if not isinstance(stored, str) or not stored:
        return
    existing = getattr(agent, "_cached_system_prompt_static", None)
    if isinstance(existing, str) and existing and stored.startswith(existing):
        return
    if getattr(agent, "_static_rebuild_failed_for", None) == stored:
        return
    try:
        static = build_system_prompt_parts(agent, system_message=system_message)["stable"]
        if static and stored.startswith(static):
            agent._cached_system_prompt_static = static
            agent._static_rebuild_failed_for = None
            return
    except Exception:
        logger.debug(
            "static system-prefix reconstruction failed on %s",
            log_label,
            exc_info=True,
        )
    agent._cached_system_prompt_static = None
    agent._static_rebuild_failed_for = stored


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
    "restore_plugin_prompt_sections",
    "format_tools_for_system_message",
]
