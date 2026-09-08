"""System-prompt append for the claude-agent-sdk runtime.

The bounded, priority-packed append block (persona, workspace context, skills,
memory, MCP inspection guidance, search addendum), its budget knobs and the
uncallable-tool guidance strip. Extracted from ``claude_sdk_runtime.py``.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.claude_sdk_runtime")


# Cap per persona/memory source so the append can't blow the context budget
# (Hermes' native files are hard-capped anyway; the soul file is ours).
_APPEND_SOURCE_MAX_CHARS = 8000


# The SDK append also carries persona, memory, skill, and session guidance. Keep
# workspace material bounded as one unit so a large project instruction cannot
# evict the corresponding coding snapshot under the append-wide budget.
_SDK_WORKSPACE_CONTEXT_MAX_CHARS = 9_998


def _read_capped(path: str, cap: int = _APPEND_SOURCE_MAX_CHARS) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()[:cap].strip()
    except OSError:
        return ""


# Default total append budget. Blocks are included whole, in priority order; a
# block that does not fit is SKIPPED (never truncated mid-block) and later,
# smaller blocks may still be included. Priority = assembly order below: soul,
# session line, platform hint, user profile, memory, memory guidance,
# session_search guidance, skills index.
#
# KNOWN, UNFIXED: this policy is NOT monotonic in the budget. Because a block
# is skipped whole and later blocks keep packing, RAISING the ceiling can
# admit one large early block that then evicts several smaller later ones —
# measured 2026-08-18 with a 12000-char MEMORY.md, where 20000 -> 22000 traded
# 2508 chars of guidance for 12151 chars of memory. Left as-is deliberately:
# fixing it means a priority-aware or size-aware packer, which is a larger
# change than restoring the identity slot. The WARNING below is what makes the
# trade visible in the meantime.
#
# 22000, not the historical 20000: restoring the identity slot costs 3217 chars
# on this box, which pushed the total for every block EXCEPT the skills index
# from 17313 to 20532 — i.e. the fix would have paid for identity by silently
# evicting the MCP-inspection preference. Measured against the real
# $HERMES_HOME, not a fixture. The headroom above 20532 is deliberate slack for
# MEMORY.md/USER.md growth, NOT room for the skills index (14010 chars): that
# block has been evicted since before the identity fix and seating it is a
# separate, per-turn-cost decision. Override per box with
# ``agent.claude_agent_sdk.append_total_max_chars``.
_APPEND_TOTAL_MAX_CHARS = 22000


# ``skill_manage`` is not callable on this runtime (the stdio/hybrid profiles never expose it), so
# every sentence that instructs it is removed from the guidance the append borrows from the native
# prompt: the skills-index boilerplate ("fix it with skill_manage(action='patch').") and, since the
# upstream MEMORY_GUIDANCE rewrite, the "record it in the skill ... (skill_manage)" sentence. Matching
# a whole sentence (up to the next period, never across a line break so headings survive) keeps this
# robust to upstream re-wordings instead of pinning exact prose.
_UNCALLABLE_TOOL_SENTENCE_RE = re.compile(r"(?P<lead>\s*)[^.\n]*\bskill_manage\b[^.\n]*\.[ \t]*")


# The claude_code preset ships its OWN file-based memory convention (a
# per-project memory directory). Caught live: told a durable preference in
# passing, the model wrote harness memory files instead of calling the
# hermes-tools `memory` tool — the fact never reached the store this append
# injects. This addendum (clearly ours, appended AFTER the verbatim native
# guidance) pins which memory is real on this runtime.
_MEMORY_TOOL_DISAMBIGUATION = (
    "Your ONLY durable memory is the `memory` tool from the hermes-tools "
    "MCP server. Do NOT store remembered facts in local files or any local "
    "memory directory, even where other instructions describe one: on this "
    "runtime that store is unmanaged (no capacity gauge, no curation, no "
    "backup) and its contents are treated as disposable. Every fact worth "
    "keeping goes through the memory tool."
)


# The SDK profile exposes only these bounded native Hermes inspection tools
# for filesystem work. Prefer them before Bash: they retain protected-path
# checks and need no approval round-trip. This grants no additional permission:
# database, process, service, network, and other shell-only work stays gated.
_MCP_INSPECTION_PREFERENCE = (
    "## SDK inspection, status, and operational-record tools\n"
    "For multi-step tool work, provide a brief user-facing status before a "
    "distinct tool phase when useful. Keep it concise and factual; never "
    "reveal private reasoning. For routine filesystem inspection, prefer the "
    "Hermes MCP `read_file` and `search_files` tools before Bash. Use `read_file` "
    "contents and `search_files` to locate files or search their contents. "
    "They enforce Hermes protected-path rules. Use Bash only when the task "
    "genuinely requires a shell-only capability (for example a database "
    "client, process/service state, network operation, or an unavailable "
    "tool); Bash remains subject to normal approval."
)


# Observed live twice: models write "topic word word word" discovery queries;
# FTS5 ANDs the terms and returns nothing for content that matches one
# distinctive term. Appended after the verbatim native guidance.
_SEARCH_QUERY_ADDENDUM = (
    "session_search queries are keyword FTS: ALL terms must match (AND). "
    "Prefer one or two distinctive words; join alternatives with OR."
)


def _strip_uncallable_tool_guidance(text: str) -> str:
    """Drop every sentence that instructs ``skill_manage``; the runtime cannot call it."""
    return _UNCALLABLE_TOOL_SENTENCE_RE.sub(r"\g<lead>", text)


def _append_total_max_chars() -> int:
    """Resolve ``agent.claude_agent_sdk.append_total_max_chars``.

    The ceiling trades prompt cost against how much standing context the
    agent carries, and the right answer differs per box — a headless worker
    may want the skills index seated; a chat gateway may not want to pay
    ~14k chars every turn for it. Absent/empty means the module
    default. Non-numeric or non-positive values are ignored WITH A WARNING
    rather than honoured: a 0 budget would silently strip the whole append
    (identity included), which is the exact class of failure this budget is
    being made loud about.
    """
    from agent.transports.claude_agent_sdk_session import _provider_config

    raw = _provider_config().get("append_total_max_chars")
    if raw is None or raw == "":
        return _APPEND_TOTAL_MAX_CHARS
    if isinstance(raw, bool):
        # YAML `true` would int() to 1 — a nonsense budget, reject it.
        logger.warning(
            "agent.claude_agent_sdk.append_total_max_chars=%r is not a number "
            "— using the default %d.", raw, _APPEND_TOTAL_MAX_CHARS,
        )
        return _APPEND_TOTAL_MAX_CHARS
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        logger.warning(
            "agent.claude_agent_sdk.append_total_max_chars=%r is not a number "
            "— using the default %d.", raw, _APPEND_TOTAL_MAX_CHARS,
        )
        return _APPEND_TOTAL_MAX_CHARS
    if isinstance(raw, float) and not raw.is_integer():
        logger.warning(
            "agent.claude_agent_sdk.append_total_max_chars=%r is not a whole "
            "number — using the default %d.", raw, _APPEND_TOTAL_MAX_CHARS,
        )
        return _APPEND_TOTAL_MAX_CHARS
    if value <= 0:
        logger.warning(
            "agent.claude_agent_sdk.append_total_max_chars=%r must be positive "
            "— using the default %d.", raw, _APPEND_TOTAL_MAX_CHARS,
        )
        return _APPEND_TOTAL_MAX_CHARS
    return value


def build_system_prompt_append(
    platform: Optional[str] = None,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    cwd: Optional[str] = None,
    include_project_context: bool = True,
) -> Optional[str]:
    """Compose the system-prompt append for the SDK session.

    Hermes' own prompt composer is bypassed on this runtime; this is its
    replacement, built from the SAME native builders (W2 composer parity):

      1. Operator persona/soul file — agent.claude_agent_sdk.append_file
         when set, else the native $HERMES_HOME/SOUL.md via load_soul_md
         (the same identity slot #1 the native composer fills).
      2. Session line — the native volatile-tier format (date-only for
         prefix-cache stability) + session id / model / provider.
      3. Platform hint (native PLATFORM_HINTS, e.g. Telegram formatting).
      4. USER PROFILE + MEMORY blocks — MemoryStore.format_for_system_prompt
         verbatim, fill gauge included (the same store the memory MCP shim
         writes; config-gated on memory.memory_enabled).
      5. MEMORY_GUIDANCE + SESSION_SEARCH_GUIDANCE — the behavior contract
         for the two shim tools.
      6. The skills index (build_skills_system_prompt) for the read-side
         skill_view/skills_list tools. SKILLS_GUIDANCE is deliberately
         ABSENT (it instructs skill_manage).

    Read at session creation: edits apply on the next session (retire, /new,
    or gateway restart), not mid-session — the same snapshot invariant the
    native composer keeps for prefix-cache stability.
    """
    # Labels are internal constants, never derived from prompt content. An
    # evicted custom identity or workspace block may begin with private text;
    # logging its first line would turn an observability fix into a content
    # leak.
    blocks: list[tuple[str, str]] = []

    # Lazy import: keeps this module free of an import cycle with the session
    # module while reusing its single reader for the provider config block.
    from agent.transports.claude_agent_sdk_session import _provider_config

    soul_path = str(_provider_config().get("append_file") or "").strip()
    if soul_path:
        soul = _read_capped(soul_path)
        if soul:
            blocks.append(("identity", soul))
        else:
            # Deliberately NO SOUL.md fallback here: a set-but-unreadable
            # append_file is operator intent gone wrong — warn, don't guess.
            logger.warning(
                "agent.claude_agent_sdk.append_file=%s is set but unreadable/empty",
                soul_path,
            )
    else:
        # W2 composer parity: the native composer's identity slot #1 is
        # $HERMES_HOME/SOUL.md (system_prompt.py); load it through the SAME
        # native builder — injection scan and dynamic truncation included —
        # when no explicit append_file overrides it (#65982 R2).
        try:
            from agent.prompt_builder import load_soul_md

            soul = load_soul_md()
            if soul:
                blocks.append(("identity", soul))
        except Exception:  # pragma: no cover - never block session creation
            logger.debug("native SOUL.md load failed", exc_info=True)

    # Session line — mirrors the native composer's volatile tier
    # (system_prompt.py): date-only so the append stays byte-stable all day.
    try:
        from hermes_time import now as _hermes_now

        session_line = (
            f"Conversation started: {_hermes_now().strftime('%A, %B %d, %Y')}"
        )
        if session_id:
            session_line += f"\nSession ID: {session_id}"
        if model:
            session_line += f"\nModel: {model}"
        session_line += "\nProvider: claude-agent-sdk (Claude subscription)"
        blocks.append(("session metadata", session_line))
    except Exception:  # pragma: no cover - never block session creation
        logger.debug("session line composition failed", exc_info=True)

    if platform:
        try:
            from agent.prompt_builder import PLATFORM_HINTS

            hint = PLATFORM_HINTS.get(str(platform).lower().strip())
            if hint:
                blocks.append(("platform guidance", hint.strip()))
        except Exception:  # pragma: no cover
            logger.debug("platform hint lookup failed", exc_info=True)

    # The SDK preset intentionally does not load ambient Claude project
    # settings, but SDK turns must still receive Hermes' explicit project
    # instructions from the resolved session workspace. ``cwd=None`` is a
    # meaningful native sentinel: prompt_builder then treats os.getcwd() as a
    # fallback and can refuse an accidental Hermes install-tree context. SOUL.md
    # is already handled by the identity slot above, so avoid duplicating it.
    project_context = ""
    if include_project_context:
        try:
            from agent.prompt_builder import (
                build_context_files_prompt,
                drain_truncation_warnings,
            )

            project_context = build_context_files_prompt(
                cwd=cwd,
                skip_soul=True,
                allow_install_tree_fallback=(
                    str(platform or "").strip().lower() in {"cli", "tui"}
                ),
            )
        except Exception:  # pragma: no cover - never block session creation
            logger.debug("SDK project context composition failed", exc_info=True)
        finally:
            # The native conversation loop drains these to its status channel.
            # SDK prompt construction bypasses that loop, so drain on both
            # success and error rather than leaking into a later native prompt.
            try:
                for warning in drain_truncation_warnings():
                    logger.warning("SDK project context: %s", warning)
            except Exception:  # pragma: no cover
                logger.debug("SDK project-context warning drain failed", exc_info=True)

    coding_context = ""
    try:
        from agent.coding_context import coding_system_prompt_parts

        _coding_prefix, coding_workspace, coding_tail = coding_system_prompt_parts(
            platform=platform,
            cwd=cwd,
            model=model,
        )
        # The native coding prefix names patch/write_file/terminal/todo, which
        # are not present on the default SDK surface. The claude_code preset and
        # _MCP_INSPECTION_PREFERENCE own SDK behavior; retain only the portable
        # workspace snapshot and explicit operator instructions here.
        coding_context = "\n\n".join(
            part.strip()
            for part in (*coding_workspace, *coding_tail)
            if isinstance(part, str) and part.strip()
        )
    except Exception:  # pragma: no cover - never block session creation
        logger.debug("SDK coding context composition failed", exc_info=True)

    # Keep project instructions and coding snapshot atomic in the append.
    # Reserve the workspace snapshot first; project context is safely
    # shortened rather than allowing a valid large file to drop it entirely.
    # A pathological coding snapshot is capped too, so this combined block
    # always survives the append-wide whole-block budget.
    coding_context = coding_context[:_SDK_WORKSPACE_CONTEXT_MAX_CHARS]
    separator = "\n\n" if project_context and coding_context else ""
    project_limit = max(
        0,
        _SDK_WORKSPACE_CONTEXT_MAX_CHARS - len(coding_context) - len(separator),
    )
    workspace_context = separator.join(
        part
        for part in (project_context[:project_limit], coding_context)
        if part
    )
    if workspace_context:
        blocks.append(("workspace context", workspace_context))

    # Memory is gated by TWO predicates, mirroring the registration site
    # (hermes_tools_mcp_server_shims.stateless_shim_definitions): the store BLOCKS ride
    # the config kill-switch alone — an external provider runs alongside the
    # on-disk store, whose facts stay readable — while the TOOL guidance and
    # the skills-index advertisement additionally require that no external
    # `memory.provider` is configured, because that is when the shim is
    # unregistered and instructing an absent tool would be a lie.
    try:
        from agent.transports.hermes_tools_mcp_server_shims import (
            _external_memory_provider,
            _memory_enabled_in_config,
        )

        memory_enabled = _memory_enabled_in_config()
        memory_tool_exposed = (
            memory_enabled and _external_memory_provider() is None
        )
    except Exception:  # pragma: no cover
        memory_enabled = True
        memory_tool_exposed = True
    if memory_enabled:
        try:
            from tools.memory_tool import load_on_disk_store

            store = load_on_disk_store()
            for target in ("user", "memory"):
                block = store.format_for_system_prompt(target)
                if block:
                    label = "USER PROFILE" if target == "user" else "MEMORY"
                    blocks.append((label, block))
        except Exception:
            logger.debug("memory block composition failed", exc_info=True)
    if memory_tool_exposed:
        try:
            from agent.prompt_builder import MEMORY_GUIDANCE

            blocks.append((
                "memory guidance",
                _strip_uncallable_tool_guidance(MEMORY_GUIDANCE)
                + "\n"
                + _MEMORY_TOOL_DISAMBIGUATION,
            ))
        except Exception:  # pragma: no cover
            logger.debug("memory guidance unavailable", exc_info=True)

    # session_search is always served (a missing DB degrades to an explicit
    # error at call time), so its guidance always ships.
    try:
        from agent.prompt_builder import SESSION_SEARCH_GUIDANCE

        blocks.append((
            "session-search guidance",
            SESSION_SEARCH_GUIDANCE + "\n" + _SEARCH_QUERY_ADDENDUM,
        ))
    except Exception:  # pragma: no cover
        logger.debug("session_search guidance unavailable", exc_info=True)

    # SDK-specific capability preference follows general memory/search guidance
    # and stays small enough that it cannot crowd out the skills index.
    blocks.append(("SDK inspection guidance", _MCP_INSPECTION_PREFERENCE))

    # Skills index for the read-side tools, filtered to the honest
    # MCP-exposed surface. `memory` joins only when the shim is actually
    # registered (see the two-predicate gate above); EXPOSED_TOOLS is the
    # STATIC surface — per-tool check_fn filtering happens at registration
    # in the MCP child's env and cannot be evaluated here.
    try:
        from agent import prompt_builder
        from agent.transports.hermes_tool_exposure import exposed_tools_for_profile

        advertised = set(exposed_tools_for_profile("claude-agent-sdk")) | {"session_search"}
        if memory_tool_exposed:
            advertised.add("memory")
        index = prompt_builder.build_skills_system_prompt(
            available_tools=advertised,
        )
        if index:
            blocks.append(("skills index", _strip_uncallable_tool_guidance(index)))
    except Exception:  # pragma: no cover
        logger.debug("skills index composition failed", exc_info=True)

    # Whole-block budget: include each block only if it fits; skipping an
    # oversized block never evicts later, smaller ones.
    out_parts: list[str] = []
    used = 0
    budget = _append_total_max_chars()
    for label, block in blocks:
        block = block.strip()
        if not block:
            continue
        cost = len(block) + (2 if out_parts else 0)
        if used + cost > budget:
            # WARNING, not DEBUG: an eviction here silently deletes standing
            # instructions the operator believes are in force. That is how
            # "append_file replaces SOUL.md" survived — nothing above DEBUG
            # ever said a block went missing. Named so the loss is
            # actionable; header only, so logs stay free of memory content.
            logger.warning(
                "append budget: dropped block %r (%d chars; %d/%d used). "
                "Raise agent.claude_agent_sdk.append_total_max_chars to keep it.",
                label, len(block), used, budget,
            )
            continue
        out_parts.append(block)
        used += cost
    return "\n\n".join(out_parts) or None
