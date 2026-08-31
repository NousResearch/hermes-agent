"""System-prompt assembly for :class:`AIAgent`.

Built once per session and reused across turns (only context compression
triggers a rebuild) so the upstream prefix cache stays warm.  Three tiers are
joined with ``\\n\\n``: ``stable`` (identity, guidance, env hints, coding brief,
platform hints), ``context`` (workspace snapshot, caller ``system_message``,
context files) and ``volatile`` (skills index, memory, USER.md, external memory
provider, timestamp line).  See ``references/system-prompt-invariant.md``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    EXECUTION_GUIDANCE_MODELS,
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    HERMES_AGENT_HELP_GUIDANCE,
    HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS,
    KANBAN_GUIDANCE,
    MEMORY_GUIDANCE,
    USER_PROFILE_GUIDANCE,
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
from agent import prompt_builder as _pb
from agent.runtime_cwd import resolve_context_cwd
from hermes_constants import get_default_hermes_root, get_hermes_home
from utils import is_truthy_value

logger = logging.getLogger(__name__)
_PLUGIN_SECTION_FRAME_RE = re.compile(
    r"^## Plugin Context: (?P<id>[a-z0-9][a-z0-9._-]{0,127})\n<!-- hermes-plugin-section-chars:(?P<chars>[0-9]{1,4}) -->\n\n",
    re.MULTILINE,
)
_GATE_WORDS = {**dict.fromkeys(("true", "always", "yes", "on"), True), **dict.fromkeys(("false", "never", "no", "off"), False)}


def _ra():
    """Lazy reference to the ``run_agent`` module.

    Helpers like ``load_soul_md``, ``build_environment_hints``,
    ``build_context_files_prompt``,
    ``build_skills_system_prompt`` and ``get_toolset_for_tool`` are
    imported into ``run_agent``'s namespace.  Many tests
    ``patch("run_agent.load_soul_md", ...)``; if we imported them
    directly here those patches would not reach us.  Looking them up
    through ``run_agent`` on every call preserves the patch contract.
    """
    import run_agent
    return run_agent


def _resolve_platform_hint(agent: Any, platform_key: str, default_hint: str) -> str:
    """Apply the ``platform_hints.<platform>`` config override: ``replace``
    substitutes the default, ``append`` adds text (a bare string is shorthand
    for append). Malformed entries fall back to the unmodified default so bad
    config can never break prompt assembly or leak across platforms."""
    overrides = getattr(agent, "_platform_hint_overrides", None)
    spec = overrides.get(platform_key) if platform_key and isinstance(overrides, dict) else None
    if isinstance(spec, str):
        spec = {"append": spec}
    if not isinstance(spec, dict):
        return default_hint
    replace_text, append_text = (v.strip() if isinstance(v, str) else "" for v in (spec.get("replace"), spec.get("append")))
    base = replace_text or default_hint
    return f"{base}\n\n{append_text}".strip() if append_text else base


_TUI_EMBEDDED_PANE_CLARIFIER = (
    " You're in its embedded terminal pane, beside the GUI chat — the user can "
    "select your output (Option-drag on macOS, Shift-drag elsewhere) and press "
    "Cmd/Ctrl+L to send it to the chat composer."
)


def _tui_embedded_pane_clarifier(hint: str) -> str:
    """Append the desktop embedded-terminal clarifier when ``HERMES_DESKTOP_TERMINAL``
    is set (only the desktop's TUI PTY, never the chat backend). Idempotent."""
    if not hint or _TUI_EMBEDDED_PANE_CLARIFIER in hint or not is_truthy_value(os.getenv("HERMES_DESKTOP_TERMINAL")):
        return hint
    return hint + _TUI_EMBEDDED_PANE_CLARIFIER


def _plugin_session_info(agent: Any) -> Dict[str, str]:
    """Return immutable-at-render-time metadata exposed to prompt sections."""
    try:
        cwd = str(resolve_context_cwd() or "")
    except Exception:
        cwd = ""
    info = {k: str(getattr(agent, k, None) or "") for k in ("session_id", "model", "provider", "platform")}
    info.update(profile_name=_active_profile_name(agent, _ambient_plugin_profile_name), cwd=cwd)
    return info


def _ambient_plugin_profile_name() -> str:
    from hermes_cli.profiles import get_active_profile_name
    return str(get_active_profile_name() or "default")


def _active_profile_name(agent: Any, ambient) -> str:
    """Profile name from the agent's OWN home, else *ambient()*; "default" on any
    failure. Ambient resolution misreports on threads that lost the HERMES_HOME
    ContextVar, which is why the agent's home is preferred."""
    try:
        home = _agent_home(agent)
        return _profile_name_for_home(home) if home is not None else ambient()
    except Exception:
        return "default"


def _frozen_plugin_prompt_sections(agent: Any) -> tuple:
    """Render plugin sections once per session and freeze them on the agent.
    A restored ``_cached_system_prompt`` is parsed instead of re-running plugin
    code; a render that raises at a rebuild boundary keeps the previous bytes
    (stashed by ``invalidate_system_prompt``) instead of silently vanishing."""
    if hasattr(agent, "_plugin_system_prompt_sections_snapshot"):
        return agent._plugin_system_prompt_sections_snapshot
    stored_prompt = getattr(agent, "_cached_system_prompt", None)
    if isinstance(stored_prompt, str) and stored_prompt:
        rendered = _restore_plugin_prompt_sections(stored_prompt)
        setattr(agent, attr, rendered)
        return rendered
    try:
        from hermes_cli.plugins import render_system_prompt_sections

        rendered = tuple(render_system_prompt_sections(_plugin_session_info(agent)))
    except Exception as exc:
        # Fail-open: a plugin whose render raises at a rebuild boundary
        # keeps its last good bytes (stashed by invalidate_system_prompt)
        # instead of silently vanishing from the prompt.
        previous = getattr(agent, "_plugin_system_prompt_sections_previous", None)
        if previous:
            logger.warning(
                "Plugin system prompt sections failed to re-render (%s); "
                "keeping the previous frozen sections", exc,
            )
            rendered = previous
        else:
            logger.warning("Plugin system prompt sections could not be rendered: %s", exc)
            rendered = ()
    setattr(agent, attr, rendered)
    return rendered


def _restore_plugin_prompt_sections(prompt: str) -> tuple:
    """Recover frozen section bytes from the persisted full prompt.  Only the
    exact canonical container emitted by core is accepted — user/project text
    may resemble a frame."""
    from hermes_cli.plugins import (
        MAX_SYSTEM_PROMPT_SECTION_CHARS, PLUGIN_SECTIONS_END, PLUGIN_SECTIONS_START,
        RenderedPluginSystemPromptSection, format_system_prompt_sections,
    )
    start = prompt.rfind(PLUGIN_SECTIONS_START)
    end = prompt.find(PLUGIN_SECTIONS_END, start + len(PLUGIN_SECTIONS_START)) if start >= 0 else -1
    if end < 0:
        return ()
    after_end = end + len(PLUGIN_SECTIONS_END)
    if not prompt[after_end:].startswith("\n\nConversation started:"):
        return ()
    framed = prompt[start:after_end]
    restored = []
    for match in _PLUGIN_SECTION_FRAME_RE.finditer(framed):
        content_len = int(match.group("chars"))
        content = framed[match.end() : match.end() + content_len]
        if content_len > MAX_SYSTEM_PROMPT_SECTION_CHARS or len(content) != content_len:
            continue
        restored.append(RenderedPluginSystemPromptSection(id=match.group("id"), content=content,
                                                          position="after_memory", plugin="persisted-prompt"))
    return tuple(restored) if format_system_prompt_sections(restored) == framed else ()


def restore_plugin_prompt_sections(agent: Any, prompt: str) -> None:
    """Seed a resumed agent's frozen snapshot from persisted prompt bytes."""
    agent._plugin_system_prompt_sections_snapshot = _restore_plugin_prompt_sections(prompt)


def _plugin_section_blocks(sections: tuple, position: str) -> List[str]:
    from hermes_cli.plugins import format_system_prompt_sections
    block = format_system_prompt_sections([s for s in sections if s.position == position])
    return [block] if block else []


def _session_start_like(agent: Any, now: Any) -> Any:
    """Best-known conversation start time, or ``now`` as a fallback.

    ``Conversation started:`` must reference when the conversation actually
    began, not when the system prompt was last (re)built.  The prompt is
    rebuilt on compression, fresh-agent gateway turns, and resume paths, and
    stamping build time made the date drift forward across midnight (a chat
    that started on Wednesday read as "Conversation started: Thursday" after
    a Thursday-morning resume), contradicting the fresh per-turn time hint.
    Prefer, in order:

    0. the LINEAGE-ROOT session id's embedded timestamp — compaction can
       rotate the session id, and each rotated id embeds its OWN mint time,
       so after months of compactions rung 1 alone would quietly re-birth
       the conversation at its latest rotation. Walking to the lineage root
       (same walk as ``_conversation_root_id``) recovers the ORIGINAL
       birth stamp — a Bot Mode forever-chat keeps knowing when it was
       first born, across every compaction (maintainer-directed, #98426);
    1. the timestamp embedded in ``session_id`` (``YYYYMMDD_HHMMSS_...``) —
       immutable for the life of the session, so the line is byte-stable
       across every rebuild boundary (preserving prefix-cache KV);
    2. ``agent.session_start`` (session-creation stamp);
    3. ``now`` (initial/legacy build without either).

    Session-id and ``session_start`` stamps are recorded in the box's local
    wall-clock; attach that zone first, then convert to the configured /
    rendered zone (``now``'s tzinfo) so the displayed date is consistent with
    the per-turn clock even when the box's TZ differs from the configured one.
    """
    from datetime import datetime

    try:
        machine_local_tz = datetime.now().astimezone().tzinfo
    except (ValueError, OSError):
        machine_local_tz = None

    def _to_display_tz(dt: Any) -> Any:
        if machine_local_tz is not None and dt.tzinfo is None:
            try:
                dt = dt.replace(tzinfo=machine_local_tz)
            except ValueError:
                pass
        if getattr(now, "tzinfo", None) is not None and dt.tzinfo is not None:
            try:
                dt = dt.astimezone(now.tzinfo)
            except (ValueError, OSError):
                pass
        return dt

    # 0. Lineage root: compaction rotation mints NEW ids with NEW embedded
    # stamps. Walk to the root id (cached on the agent — the lineage only
    # grows at compaction, and this function runs at that exact boundary,
    # so one walk per rebuild is fresh enough) and prefer ITS embedded
    # timestamp: the conversation's true birth. Fail-open to rung 1.
    session_id = getattr(agent, "session_id", None)
    root_id = None
    try:
        db = getattr(agent, "_session_db", None)
        if db is not None and isinstance(session_id, str) and session_id:
            root_id = db.get_conversation_root(session_id)
    except Exception:
        root_id = None
    for candidate in (root_id, session_id):
        if isinstance(candidate, str) and candidate:
            m = re.match(r"^(\d{8})_(\d{6})", candidate)
            if m:
                try:
                    embedded = datetime.strptime(
                        f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S"
                    )
                    return _to_display_tz(embedded)
                except ValueError:
                    pass

    # 2. Session-creation stamp set by the runner.
    session_start = getattr(agent, "session_start", None)
    if hasattr(session_start, "astimezone"):
        return _to_display_tz(session_start)

    # 3. Fallback: build time.
    return now


def _agent_home(agent: Any) -> Optional[Path]:
    """The agent's OWN profile home, or None to use ambient resolution.
    A bound HERMES_HOME ContextVar override wins (the gateway multiplexes
    profiles over one shared session DB and binds the home per turn); else the
    parent of ``_session_db.db_path`` — ground truth on threads that lost the
    ContextVar, where ambient resolution would leak the launch profile.

    1. Surfaces that multiplex several profiles over ONE shared session DB (the messaging gateway:
    ``gateway/run.py`` hands every agent the launch-home ``state.db`` and binds the profile home per turn
    via ``_profile_runtime_scope`` + ``copy_context``) would otherwise have the db-derived launch home STOMP
    the correctly-bound profile — inverting the leak this helper exists to fix (found by @kshitijk4poor's
    post-merge probe on #86313). 2. Fallback: the home containing the agent's ``_session_db.db_path``
    (``<home>/state.db``) — ground truth on threads that lost the ContextVar (ContextVars don't propagate
    into ``threading.Thread``), where the unbound build previously fell back to the launch home and leaked
    the default profile's skills/identity into a bot prompt.
    """
    try:
        from hermes_constants import get_hermes_home_override
        override = get_hermes_home_override()
        if override:
            return Path(override)
    except Exception:
        pass
    try:
        db_path = getattr(getattr(agent, "_session_db", None), "db_path", None)
        return Path(db_path).parent if db_path else None
    except Exception:
        return None


def _agent_skills_dir(agent: Any) -> Optional[Path]:
    """The agent's own ``<home>/skills`` dir, or None to use ambient home."""
    home = _agent_home(agent)
    return home / "skills" if home is not None else None


def _profile_name_for_home(home: Path) -> str:
    """``<root>/profiles/X`` -> ``"X"``; anything else -> ``"default"``.
    Uses ``get_default_hermes_root()`` (NOT ``get_hermes_home()``): on a bound
    profile session the ambient home IS the profile dir, so every profile
    would misreport as "default"."""
    try:
        from hermes_constants import get_default_hermes_root
        rel = home.resolve().relative_to((get_default_hermes_root() / "profiles").resolve())
        return rel.parts[0] if rel.parts else "default"
    except (ValueError, OSError):
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

    # Pointer to the docs (and, when it exists, the hermes-agent skill) for
    # user questions about Hermes itself. The skill_view() pointer is a
    # dangling reference in two cases — no skill tools in the toolset
    # (Blank Slate) OR the hermes-agent skill not installed — so the
    # variant is chosen AFTER the skills index is built (see below) and
    # this slot holds its position. Toolset and skill set are fixed
    # per-session, so cache-safe either way.
    _has_skill_view = "skill_view" in (agent.valid_tool_names or set())
    _help_guidance_slot = len(stable_parts)
    stable_parts.append(HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS)

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
    # MEMORY_GUIDANCE instructs the model to save facts to the built-in
    # MEMORY.md/USER.md stores. With both disabled in config no store is built,
    # so the guidance would steer the model at a tool whose every call returns
    # "Memory is not available". Defaults to True for the rare code paths that
    # build an agent view without going through agent_init.
    # When only the user profile store is enabled, the narrower
    # USER_PROFILE_GUIDANCE is injected instead — the full block instructs the
    # model to write notes to a MEMORY.md store that does not exist.
    _mem_enabled = getattr(agent, "_memory_enabled", True)
    _profile_enabled = getattr(agent, "_user_profile_enabled", True)
    if "memory" in agent.valid_tool_names:
        if _mem_enabled:
            tool_guidance.append(MEMORY_GUIDANCE)
        elif _profile_enabled:
            tool_guidance.append(USER_PROFILE_GUIDANCE)
    if "session_search" in agent.valid_tool_names:
        tool_guidance.append(SESSION_SEARCH_GUIDANCE)
    if "skill_manage" in agent.valid_tool_names:
        tool_guidance.append(SKILLS_GUIDANCE)
    # Kanban worker/orchestrator lifecycle — only present when the
    # dispatcher spawned this process (kanban_show check_fn gates on
    # HERMES_KANBAN_TASK env var). Normal chat sessions never see
    # this block. Resolved once at __init__ (see _kanban_worker_guidance).
    _kanban_guidance = getattr(agent, "_kanban_worker_guidance", None)
    if _kanban_guidance is None and "kanban_show" in names:
        _kanban_guidance = KANBAN_GUIDANCE
    tool_guidance = [
        memory_guidance,
        SESSION_SEARCH_GUIDANCE if "session_search" in names else None,
        SKILLS_GUIDANCE if "skill_manage" in names else None,
        _kanban_guidance,
    ]
    return " ".join(g for g in tool_guidance if g) or None


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

    # Execution-discipline guidance (tool persistence, mandatory tool use
    # for arithmetic, external-write read-back, count reconciliation,
    # literal preservation, verification-gated completion).  Historically
    # nested inside the tool-use-enforcement branch and fenced to
    # gpt/codex/grok; now an independent gate so DeepSeek/Kimi/Qwen-class
    # models receive it even when tool_use_enforcement is off.  Controlled
    # by config.yaml agent.execution_guidance:
    #   "auto" (default) — matches EXECUTION_GUIDANCE_MODELS
    #   true  — always inject (all models)
    #   false — never inject
    #   list  — custom model-name substrings to match
    # Resolved once at session start keyed on the (fixed) model name, so
    # the system prompt stays byte-stable for the life of the conversation.
    if agent.valid_tool_names:
        _exec_guidance = getattr(agent, "_execution_guidance", "auto")
        _exec_inject = False
        if _exec_guidance is True or (isinstance(_exec_guidance, str) and _exec_guidance.lower() in {"true", "always", "yes", "on"}):
            _exec_inject = True
        elif _exec_guidance is False or (isinstance(_exec_guidance, str) and _exec_guidance.lower() in {"false", "never", "no", "off"}):
            _exec_inject = False
        elif isinstance(_exec_guidance, list):
            model_lower = (agent.model or "").lower()
            _exec_inject = any(p.lower() in model_lower for p in _exec_guidance if isinstance(p, str))
        else:
            # "auto" or any unrecognised value — use hardcoded defaults
            model_lower = (agent.model or "").lower()
            _exec_inject = any(p in model_lower for p in EXECUTION_GUIDANCE_MODELS)
        if _exec_inject:
            from agent.prompt_builder import execution_guidance_text
            stable_parts.append(execution_guidance_text(agent.valid_tool_names))

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

    # Resolve the help-guidance variant now that the skills index exists:
    # the skill-pointer variant requires BOTH skill_view in the toolset AND
    # the hermes-agent skill actually present in the index (gating on the
    # rendered index line keeps this a pure string check — no second
    # filesystem scan, and it inherits the index cache's stability).
    if _has_skill_view and "- hermes-agent:" in skills_prompt:
        stable_parts[_help_guidance_slot] = HERMES_AGENT_HELP_GUIDANCE

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
                valid_tool_names=agent.valid_tool_names,
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

    # Bot Mode teammate protocol — injected ONLY into a bot's canonical
    # "Bot Chat" session (the conversation teammate bots message into via
    # `hermes -p <bot> chat --in ~ -c "Bot Chat"` and the desktop pins), on
    # installs where Bot Mode manages profiles (ui_meta['hermes-bots']).
    # Regular sessions never carry it — the desktop's composer middleware
    # owns the @mention send path. Title is read once at first build and the
    # rendered prompt is cached + DB-restored, so this is cache-safe.
    # Gated by config.yaml ``agent.bot_mode_protocol`` (default True).
    if getattr(agent, "_bot_mode_protocol", True):
        try:
            from tools.bot_mode_probe import (
                BOT_CHAT_TITLE,
                epoch_line,
                get_bot_mode_protocol_section,
            )
            _title = str(getattr(agent, "_session_title_hint", "") or "").strip()
            if not _title:
                _sdb = getattr(agent, "_session_db", None)
                _sid = getattr(agent, "session_id", None)
                _title = str((_sdb.get_session_title(_sid) if (_sdb and _sid) else None) or "").strip()
            if _title == BOT_CHAT_TITLE:
                _bot_section = get_bot_mode_protocol_section(_agent_home(agent))
                if _bot_section:
                    post_workspace_parts.append(_bot_section)
                    # Eternal-session support: stamp the capability epoch so
                    # the restore path can detect user-initiated capability
                    # changes (skills/toolsets/MCP/SOUL/roster) and rebuild
                    # ONCE per change instead of waiting for /new or
                    # compression. Also marks this prompt as timeless — the
                    # volatile timestamp line is omitted (see below), since a
                    # birth date pinned in a session that lives for months is
                    # misinformation.
                    post_workspace_parts.append(epoch_line(_agent_home(agent)))
                    agent._bot_chat_timeless_prompt = True
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
        from agent.coding_context import coding_compact_skill_categories
        _compact_cats = coding_compact_skill_categories(platform=agent.platform, cwd=resolve_context_cwd())
    except Exception:
        _compact_cats = frozenset()
    return _pb.build_skills_system_prompt(available_tools=agent.valid_tool_names, available_toolsets=avail_toolsets,
                                         compact_categories=_compact_cats or None, skills_dir_override=_agent_skills_dir(agent))


def _auto_load_parts(agent: Any) -> List[str]:
    """``skills.auto_load`` blocks, resolved once per agent lifecycle (config, skill files and
    HERMES_IGNORE_RULES are read on the first build only) so the prompt stays byte-stable
    across model switches, compression and static-prefix restoration.

    Same gate as ``_skills_prompt``: nothing without the skills toolset, and nothing for agents that skip
    context files (delegate children, curator/review forks, gateway hygiene agents) — pinned skills are
    operator guidance for the user's session, not payload for every internal fork."""
    if getattr(agent, "skip_context_files", False) or not any(
            name in agent.valid_tool_names for name in ("skills_list", "skill_view", "skill_manage")):
        return []
    if not getattr(agent, "_auto_load_skills_resolved", False):
        result: Tuple[str, List[str], List[str]] = ("", [], [])
        try:
            if not is_truthy_value(os.environ.get("HERMES_IGNORE_RULES")):
                from agent.skill_commands import build_auto_load_prompt
                result = build_auto_load_prompt(task_id=getattr(agent, "session_id", None), home_override=_agent_home(agent))
            if result[2]:
                logger.warning("skills.auto_load: skill(s) not found or disabled, skipped: %s", ", ".join(result[2]))
        except Exception:
            logger.debug("skills.auto_load: injection skipped", exc_info=True)  # config errors never block session start
        agent._auto_load_skills_result = result
        agent._auto_load_skills_resolved = True
    prompt = agent._auto_load_skills_result[0]
    return [prompt] if prompt else []


def _bot_mode_parts(agent: Any) -> List[str]:
    """Bot Mode teammate protocol — only in a bot's canonical "Bot Chat" session.
    Marks the prompt timeless (the volatile date line is dropped) since a birth
    date pinned in a months-long session is misinformation."""
    parts: List[str] = []
    try:
        from tools.bot_mode_probe import BOT_CHAT_TITLE, epoch_line, get_bot_mode_protocol_section
        _title = str(getattr(agent, "_session_title_hint", "") or "").strip()
        if not _title:
            _sdb = getattr(agent, "_session_db", None)
            _sid = getattr(agent, "session_id", None)
            _title = str((_sdb.get_session_title(_sid) if (_sdb and _sid) else None) or "").strip()
        _bot_section = get_bot_mode_protocol_section(_agent_home(agent)) if _title == BOT_CHAT_TITLE else None
        if _bot_section:
            parts.append(_bot_section)
            # Capability epoch lets the restore path rebuild ONCE per
            # user-initiated capability change in an eternal session.
            parts.append(epoch_line(_agent_home(agent)))
            agent._bot_chat_timeless_prompt = True
    except Exception:
        pass
    return parts


def _ambient_file_safety_profile_name() -> str:
    from agent.file_safety import _resolve_active_profile_name
    return _resolve_active_profile_name()


def _active_profile_line(agent: Any) -> str:
    """Name the running profile so the agent doesn't conflate ``~/.hermes/skills``
    (default) with ``~/.hermes/profiles/<active>/skills``.  Resolved from the
    agent's OWN home first (a build thread that lost the ContextVar would
    otherwise print "default" for a bot profile)."""
    _agent_home_path = _agent_home(agent)
    active_profile = _active_profile_name(agent, _ambient_file_safety_profile_name)
    if active_profile == "default":
        # With an explicit agent home, the default profile's data lives at the
        # ROOT (get_hermes_home() on a bound profile session is the PROFILE dir).
        # Without one, keep the ambient (patchable) resolution byte-identical.
        _root_str = str(get_default_hermes_root() if _agent_home_path is not None else get_hermes_home())
        return (
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
            f"explicitly directs you to."
        )


def platform_hint(agent: Any) -> str:
    """Built-in/plugin platform hint + Telegram rich-messages opt-in + config
    override + desktop TUI clarifier."""
    platform_key = (agent.platform or "").lower().strip()
    _default_hint = PLATFORM_HINTS.get(platform_key, "")
    if not _default_hint and platform_key:
        try:
            from gateway.platform_registry import platform_registry
            _entry = platform_registry.get(platform_key)
            _default_hint = (_entry and _entry.platform_hint) or ""
        except Exception:
            pass
    if platform_key == "telegram" and _default_hint and _telegram_rich_messages_enabled():
        _default_hint = _default_hint.rstrip() + " " + TELEGRAM_RICH_MESSAGES_HINT
    _effective_hint = _resolve_platform_hint(agent, platform_key, _default_hint)
    if platform_key == "tui" and _effective_hint:
        _effective_hint = _tui_embedded_pane_clarifier(_effective_hint)
    return _effective_hint


def _telegram_rich_messages_enabled() -> bool:
    """``rich_messages`` from the Telegram ``extra`` config; same precedence the
    adapter uses (top-level ``platforms.telegram.extra`` overrides
    ``gateway.platforms.telegram.extra`` at the leaf). False on any read failure."""
    try:
        from hermes_cli.config import load_config_readonly
        _cfg = load_config_readonly()
        _gw = (((_cfg.get("gateway") or {}).get("platforms") or {}).get("telegram") or {}).get("extra")
        _top = ((_cfg.get("platforms") or {}).get("telegram") or {}).get("extra")
        merged = {**(_gw if isinstance(_gw, dict) else {}), **(_top if isinstance(_top, dict) else {})}
        return bool(merged.get("rich_messages"))
    except Exception:
        return False


def _zone_bits(now: Any, tz: Any) -> List[str]:
    """IANA key, abbreviation (if different) and UTC offset — all constant for
    the day, so the byte-stable date line stays cacheable."""
    _iana = getattr(tz, "key", None)
    _abbrev = now.strftime("%Z")
    _offset = now.strftime("%z")  # '-0400' -> 'UTC-04:00'
    bits = [_iana] if _iana else []
    if _abbrev and _abbrev != _iana:
        bits.append(_abbrev)
    if _offset:
        bits.append(f"UTC{_offset[:3]}:{_offset[3:]}")
    return bits

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

    # External memory provider system prompt block (additive to built-in).
    # Gated on the same check ``inject_memory_provider_tools`` uses so we
    # never advertise provider tools that the agent's toolset configuration
    # has already gated off (#81014).
    if agent._memory_manager:
        try:
            from agent.memory_manager import memory_provider_tools_exposed as _mem_exposed
        except Exception:
            _mem_exposed = None
        if _mem_exposed is None or _mem_exposed(agent):
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

def _timestamp_line(agent: Any) -> str:
    """Date-only so the prompt is byte-stable for the day; zone + offset so
    tools needn't guess EST vs EDT. Long-lived sessions get an "as of" line on
    rebuild days (the cache prefix is already invalidated at that boundary)."""
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
    _start = _session_start_like(agent, now)
    timestamp_line = (
        f"Conversation started: {_start.strftime('%A, %B %d, %Y')}{_zone_suffix}"
    )
    # Second line (maintainer design, salvaging #96224's anchor): long-lived
    # sessions — Bot Mode forever-chats, messenger channels people never
    # close — span many days and many compactions. A lone birth date leads
    # the model to believe it is still living in that old day. The prompt is
    # rebuilt at every compaction boundary, so stamp the rebuild day too:
    # 'started' stays anchored and byte-stable, 'as of' refreshes exactly
    # when the cache prefix is already being invalidated (compaction), so
    # the added line costs no extra cache churn. Same-day sessions skip the
    # second line entirely — nothing to correct, and the single-line shape
    # stays byte-identical for the day (prefix-cache safe).
    if now.strftime("%Y%m%d") != _start.strftime("%Y%m%d"):
        timestamp_line += (
            f"\nToday's date (as of the last context rebuild): "
            f"{now.strftime('%A, %B %d, %Y')} — trust this over the start "
            f"date for what day it is now; query tools for exact time."
        )
    # Bot Chat sessions are effectively eternal — a birth date frozen in the
    # prompt becomes confidently-wrong misinformation within days. Timeless
    # prompts keep the identity lines but drop the date (the timezone still
    # rides workspace context; live time comes from the terminal tool).
    if getattr(agent, "_bot_chat_timeless_prompt", False):
        timestamp_line = f"Timezone: {', '.join(_bits)}" if _bits else ""
    trailer = (("Session ID", agent.session_id if agent.pass_session_id else None), ("Model", agent.model),
               ("Provider", agent.provider), ("Platform", agent.platform))
    return timestamp_line + "".join(f"\n{label}: {value}" for label, value in trailer if value)


def _memory_parts(agent: Any) -> List[str]:
    """Built-in memory/USER.md blocks plus the external provider block (gated on
    the same check ``inject_memory_provider_tools`` uses, so we never advertise
    tools the toolset config gated off)."""
    parts: List[str] = []
    if agent._memory_store:
        for enabled, kind in ((agent._memory_enabled, "memory"), (agent._user_profile_enabled, "user")):
            block = agent._memory_store.format_for_system_prompt(kind) if enabled else None
            if block:
                parts.append(block)
    # External memory provider system prompt block (additive to built-in). Gated on the same check
    # ``inject_memory_provider_tools`` uses so we never advertise provider tools that the agent's toolset
    # configuration has already gated off (#81014).
    if agent._memory_manager:
        try:
            from agent.memory_manager import memory_provider_tools_exposed as _mem_exposed
        except Exception:
            _mem_exposed = None
        if _mem_exposed is None or _mem_exposed(agent):
            try:
                _ext_mem_block = agent._memory_manager.build_system_prompt()
            except Exception:
                _ext_mem_block = None
            if _ext_mem_block:
                parts.append(_ext_mem_block)
    return parts


def _identity_parts(agent: Any, ctx_len: Optional[int]) -> Tuple[List[str], bool]:
    """SOUL.md (primary identity; cron keeps the persona while skipping cwd
    instructions, scoped to the agent's OWN home) or the default identity.
    Returns ``(parts, soul_loaded)``."""
    wants_soul = agent.load_soul_identity or not agent.skip_context_files
    _soul_content = _pb.load_soul_md(ctx_len, home_override=_agent_home(agent)) if wants_soul else None
    return ([_soul_content], True) if _soul_content else ([DEFAULT_AGENT_IDENTITY], False)


def _guidance_parts(agent: Any) -> List[str]:
    """Universal + tool-aware + model-gated guidance blocks, each gated by its config.yaml key."""
    parts: List[str] = []
    if agent.valid_tool_names:
        parts += [
            text for flag, text in (
                ("_task_completion_guidance", TASK_COMPLETION_GUIDANCE),
                ("_parallel_tool_call_guidance", PARALLEL_TOOL_CALL_GUIDANCE),
            ) if getattr(agent, flag, True)
        ]
    parts.append(_tool_guidance_block(agent))  # None/empty entries are dropped by _join_tier
    if not agent.valid_tool_names:
        return parts
    # Steering only lands inside tool results, so only reachable with tools.
    parts.append(STEER_CHANNEL_NOTE)
    # agent.tool_use_enforcement / agent.execution_guidance: "auto" (default)
    # matches the hardcoded model lists; true/false force; a list gives custom
    # model-name substrings.  Execution guidance is an independent gate so
    # DeepSeek/Kimi/Qwen-class models get it even with enforcement off.
    if _model_gate(agent._tool_use_enforcement, agent.model, TOOL_USE_ENFORCEMENT_MODELS):
        parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
        if any(g in (agent.model or "").lower() for g in ("gemini", "gemma")):
            parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
    if _model_gate(getattr(agent, "_execution_guidance", "auto"), agent.model, EXECUTION_GUIDANCE_MODELS):
        from agent.prompt_builder import execution_guidance_text
        parts.append(execution_guidance_text())
    return parts


def _alibaba_identity_part(agent: Any) -> List[str]:
    """Alibaba Coding Plan always reports "glm-4.7" as the model name; inject
    the real identity so the agent can answer correctly."""
    if agent.provider != "alibaba":
        return []
    _model_short = agent.model.rsplit("/", 1)[-1]
    return [
        f"You are powered by the model named {_model_short}. "
        f"The exact model ID is {agent.model}. "
        f"When asked what model you are, always answer based on this information, "
        f"not on any model name returned by the API."
    ]


def _coding_parts(agent: Any) -> Tuple[List[str], List[str], List[str]]:
    """``(prefix, workspace, trailing)`` coding-posture blocks; all empty
    without tools or when probing fails (it must never block prompt build).

    The workspace block is a live git probe after project context, ahead of the whole
    volatile band; re-probing at the compaction rebuild re-emits different bytes for any
    repo that moved and defeats the keep-prompt fast path.  So the bytes are pinned per
    session on the agent, keyed by the resolved cwd (a gateway serves many cwds), and
    replayed on rebuilds; ``reset_session_state`` drops the pin at a session boundary.
    """
    try:
        from agent.coding_context import coding_system_prompt_parts
        if not agent.valid_tool_names:
            return [], [], []
        cwd = resolve_context_cwd()
        cwd_key = str(cwd) if cwd is not None else ""
        pinned = getattr(agent, "_frozen_workspace_snapshot", None)
        # "" is a real pinned value (no workspace here) — only a cwd mismatch re-probes.
        replay = pinned[1] if pinned is not None and pinned[0] == cwd_key else None
        parts = coding_system_prompt_parts(platform=agent.platform, cwd=cwd, model=agent.model,
                                           valid_tool_names=agent.valid_tool_names, workspace_block=replay)
        if replay is None:
            agent._frozen_workspace_snapshot = (cwd_key, parts[1][0] if parts[1] else "")
        return parts
    except Exception:
        pass
    return [], [], []


def _post_workspace_parts(agent: Any) -> List[str]:
    """Blocks that follow the worktree-specific context: environment probe
    (config.yaml agent.environment_probe; one line, nothing when clean, skipped
    for remote backends), bot-mode protocol, profile line, platform hint."""
    parts: List[str] = []
    if getattr(agent, "_environment_probe", True):
        try:
            from tools.env_probe import get_environment_probe_line
            parts.append(get_environment_probe_line())
        except Exception:
            pass  # Probe failure must never block prompt build.
    if getattr(agent, "_bot_mode_protocol", True):
        parts.extend(_bot_mode_parts(agent))
    parts += [_active_profile_line(agent), platform_hint(agent)]
    return parts


def _context_files_part(agent: Any, ctx_len: Optional[int], soul_loaded: bool) -> List[str]:
    """Project context files (AGENTS.md etc.) for the context tier. TERMINAL_CWD
    when set (gateway); None lets discovery fall back to the launch dir.  The
    install-tree fallback is only legitimate for cli/tui where the launch dir
    IS the user's shell cwd; desktop-pinned launch dirs are treated as the
    fallback they really are so the guard can reject Hermes's bundled AGENTS.md."""
    if agent.skip_context_files:
        return []
    launch_artifact = getattr(agent, "_context_cwd_is_launch_artifact", False)
    return [_pb.build_context_files_prompt(
        cwd=None if launch_artifact else resolve_context_cwd(), skip_soul=soul_loaded, context_length=ctx_len,
        allow_install_tree_fallback=agent.platform in ("cli", "tui"), home_override=_agent_home(agent))]


def _join_tier(parts: List[Optional[str]]) -> str:
    """Join non-empty parts; None/blank entries are dropped."""
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def build_system_prompt_parts(agent: Any, system_message: Optional[str] = None) -> Dict[str, str]:
    """Assemble the system prompt as three ordered cache tiers: ``stable`` (identity,
    guidance and the coding brief), ``context`` (caller ``system_message``, project
    context files, workspace snapshot and remaining workspace guidance) and
    ``volatile`` (skills index, memory, user profile, external memory block,
    timestamp line, runtime environment hints).  Worktree-dependent blocks follow project context so a
    shared context file can remain in the longest common prefix across worktrees.
    Never re-rendered mid-session."""
    # Model context window scales the context-file caps; stable per conversation.
    _cc_len = getattr(getattr(agent, "context_compressor", None), "context_length", None)
    _ctx_len = _cc_len if isinstance(_cc_len, int) and _cc_len > 0 else None
    # ── Stable tier ────────────────────────────────────────────────
    stable_parts, _soul_loaded = _identity_parts(agent, _ctx_len)
    # The skill_view() pointer dangles without skill tools OR without the
    # hermes-agent skill installed, so the variant is chosen after the skills
    # index is built; this slot holds its position.
    _help_guidance_slot = len(stable_parts)
    stable_parts.append(HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS)
    stable_parts.extend(_guidance_parts(agent))
    skills_prompt = _skills_prompt(agent)
    # Skill-pointer variant requires BOTH skill_view AND the hermes-agent skill
    # in the rendered index (pure string check — inherits the index's stability).
    if "skill_view" in (agent.valid_tool_names or set()) and "- hermes-agent:" in skills_prompt:
        stable_parts[_help_guidance_slot] = HERMES_AGENT_HELP_GUIDANCE
    stable_parts.extend(_alibaba_identity_part(agent))
    # Pinned skills are per-agent constants (resolved once), so they live in the stable prefix.
    stable_parts.extend(_auto_load_parts(agent))
    # Coding posture: the operating brief stays in the stable prefix. The
    # environment block contains the current cwd/backend and belongs after
    # project context, not ahead of a large shared AGENTS.md block.
    environment_hints = _pb.build_environment_hints()
    coding_prefix_parts, coding_workspace_parts, coding_trailing_parts = _coding_parts(agent)
    stable_parts.extend(coding_prefix_parts)
    post_workspace_parts = _post_workspace_parts(agent)
    # ── Context tier (project/worktree-dependent, may change between sessions) ──
    context_parts: List[str] = []
    # ephemeral_system_prompt is injected at API-call time only, never cached.
    if system_message is not None:
        context_parts.append(system_message)
    context_parts.extend(_context_files_part(agent, _ctx_len, _soul_loaded))
    if coding_workspace_parts:
        context_parts.extend([*coding_workspace_parts, *coding_trailing_parts, *post_workspace_parts])
    else:
        # Preserve the stable placement for non-workspace sessions; there is no
        # worktree snapshot whose later position would improve their prefix.
        stable_parts.extend([*coding_trailing_parts, *post_workspace_parts])
    # ── Volatile tier (most likely to differ on a rebuild; kept last so the stable prefix stays reusable) ──
    # Skills are runtime-mutable, so the index leads the volatile band: on a longest-prefix
    # backend an unchanged index stays inside the reused prefix; a changed one re-prefills from here.
    volatile_parts: List[str] = [skills_prompt, *_memory_parts(agent)]
    # Plugin sections are confined to one coarse anchor in the volatile tail so
    # a resumed process can reconstruct the stable prefix without re-running plugins.
    volatile_parts.extend(_plugin_section_blocks(_frozen_plugin_prompt_sections(agent), "after_memory"))
    volatile_parts.append(_timestamp_line(agent))
    # Keep the renderer-owned runtime anchor after all user/plugin prose so quoted
    # host examples cannot shadow it during persisted-prompt validation.
    if environment_hints:
        # Embedder hints are prose too; reserve the delimiter for the renderer.
        environment_hints = environment_hints.replace(_pb.RUNTIME_ENVIRONMENT_HEADING, "> " + _pb.RUNTIME_ENVIRONMENT_HEADING)
        volatile_parts.append(f"{_pb.RUNTIME_ENVIRONMENT_HEADING}\n\n{environment_hints}\n\n{_pb.RUNTIME_ENVIRONMENT_END}")
    return {"stable": _join_tier(stable_parts), "context": _join_tier(context_parts), "volatile": _join_tier(volatile_parts)}


def build_system_prompt(agent: Any, system_message: Optional[str] = None) -> str:
    """Assemble the full prompt; cached on ``agent._cached_system_prompt`` and
    only rebuilt after compression.  Tiers are ordered stable -> context ->
    volatile so implicit longest-prefix caches keep the unchanged scaffold."""
    parts = build_system_prompt_parts(agent, system_message=system_message)
    agent._cached_system_prompt_static = parts["stable"]
    # Surface context-file truncation warnings in chat, not only in logs.
    for warning in drain_truncation_warnings():
        agent._emit_status(warning)
    return "\n\n".join(p for p in (parts["stable"], parts["context"], parts["volatile"]) if p)


def invalidate_system_prompt(agent: Any) -> None:
    """Force a rebuild on the next turn (after compression): reload memory from
    disk and clear the frozen plugin snapshot (previous bytes stashed as the
    fail-open fallback) so plugins re-render at the same boundary.

    Called after context compression events. Also reloads memory from disk
    so the rebuilt prompt captures any writes from this session, and clears
    the frozen plugin-section snapshot so plugins re-render at the same
    boundary (maintainer-directed, #95681 arc): a plugin section is just
    another prompt block carrying state — freezing it while memory, skills,
    and guidance refresh would recreate the stale-block disease inside
    plugin-land. The previous bytes are stashed so a plugin whose render
    RAISES falls back to its last good section instead of vanishing
    (fail-open guard, not a freeze).
    """
    agent._cached_system_prompt = None
    agent._cached_system_prompt_static = None
    _snapshot_attr = "_plugin_system_prompt_sections_snapshot"
    if hasattr(agent, _snapshot_attr):
        agent._plugin_system_prompt_sections_previous = getattr(agent, _snapshot_attr)
        delattr(agent, _snapshot_attr)
    if agent._memory_store:
        agent._memory_store.load_from_disk()


def reconstruct_static_prefix(agent: Any, system_message: Optional[str] = None, *, log_label: str = "restore") -> None:
    """Reconstruct ``_cached_system_prompt_static`` for a stored prompt.
    Only the full prompt is persisted, so restore / keep-prompt compression /
    mid-turn failover to a cache-on provider must rebuild the stable tier to
    regain the ``[static, volatile]`` layout.  The rebuilt tier is used ONLY
    when the stored prompt literally starts with it; otherwise static stays
    None and the stored bytes are sent untouched.  A failed rebuild is memoized
    per stored prompt so the retry-loop hot path doesn't redo the file I/O."""
    stored = getattr(agent, "_cached_system_prompt", None)
    existing = getattr(agent, "_cached_system_prompt_static", None)
    if (
        not getattr(agent, "_use_prompt_caching", False)
        or not isinstance(stored, str) or not stored
        or (isinstance(existing, str) and existing and stored.startswith(existing))
        or getattr(agent, "_static_rebuild_failed_for", None) == stored
    ):
        return
    try:
        static = build_system_prompt_parts(agent, system_message=system_message)["stable"]
        if static and stored.startswith(static):
            agent._cached_system_prompt_static = static
            agent._static_rebuild_failed_for = None
            return
    except Exception:
        logger.debug("static system-prefix reconstruction failed on %s", log_label, exc_info=True)
    agent._cached_system_prompt_static = None
    agent._static_rebuild_failed_for = stored


def format_tools_for_system_message(agent: Any) -> str:
    """JSON tool definitions in the trajectory format."""
    if not agent.tools:
        return "[]"
    return json.dumps([{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                        "parameters": t["function"].get("parameters", {}),
                        "required": None}  # Match the format in the example
                       for t in agent.tools], ensure_ascii=False)


__all__ = ["build_system_prompt_parts", "build_system_prompt", "invalidate_system_prompt",
           "platform_hint", "restore_plugin_prompt_sections", "format_tools_for_system_message"]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'OPENAI_MODEL_EXECUTION_GUIDANCE': ('agent.prompt_builder', 'OPENAI_MODEL_EXECUTION_GUIDANCE'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
