"""claude-agent-sdk runtime — the subscription-Claude agent-loop path.

The structural twin of ``agent/codex_runtime.py``'s app-server path: hands the
entire turn to Anthropic's official ``claude-agent-sdk`` (which drives the
Claude Code CLI's own agent loop under subscription OAuth by default, with
known metered lanes refused unless explicitly enabled) and projects its typed
message stream back into Hermes'
messages list so transcript persistence and recall keep working. GitHub
issue #25267.

* ``run_claude_agent_sdk_turn`` — drives one turn through a lazily-created
  ``ClaudeAgentSdkSession`` (used when ``agent.api_mode == "claude_agent_sdk"``).
"""

from __future__ import annotations

import copy
import logging
import math
import os
import re
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaudeSdkTurnEffects:
    """Replay-safety ledger for one SDK provider attempt."""

    tool: bool = False
    streamed: bool = False
    projected: bool = False
    interrupted: bool = False
    mutated: bool = False

    @property
    def replay_safe(self) -> bool:
        return not any(asdict(self).values())

    def as_result_dict(self) -> Dict[str, bool]:
        return asdict(self)


_SDK_PROVIDER_FAILOVER_REASONS = frozenset(
    {
        "auth",
        "auth_permanent",
        "rate_limit",
        "upstream_rate_limit",
        "overloaded",
        "server_error",
        "timeout",
    }
)


def _sdk_provider_failover_reason(agent, error: str, fatal_reason: Optional[str]):
    """Return one canonical provider handoff reason, or ``None``.

    SDK failures are serialized into result text, so recover a status code when
    present and delegate classification to the shared API error taxonomy.  The
    explicit allowlist is intentionally narrower than ``should_fallback``:
    local/configuration errors, request-shape failures, policy refusals, and
    billing-safety guards must remain terminal on this runtime.
    """
    from agent.error_classifier import FailoverReason, classify_api_error

    if fatal_reason:
        try:
            reason = FailoverReason(str(fatal_reason))
        except ValueError:
            return None
        return reason if reason.value in _SDK_PROVIDER_FAILOVER_REASONS else None

    text = str(error or "").strip()
    if not text:
        return None

    class _SerializedSdkProviderError(RuntimeError):
        status_code: Optional[int] = None

    exc = _SerializedSdkProviderError(text)
    match = re.search(
        r"(?:api\s+error|status(?:_code)?|http)\s*[:=]?\s*(\d{3})\b",
        text,
        re.IGNORECASE,
    )
    if match:
        exc.status_code = int(match.group(1))
    classified = classify_api_error(
        exc,
        provider=str(getattr(agent, "provider", "") or ""),
        model=str(getattr(agent, "model", "") or ""),
    )
    return (
        classified.reason
        if classified.reason.value in _SDK_PROVIDER_FAILOVER_REASONS
        else None
    )

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


def _hybrid_bridge_enabled() -> bool:
    """agent.claude_agent_sdk.hybrid_mcp_bridge from config.yaml.

    Off by default — the wide bridge exposes agent-level tools whose
    enablement is a security choice. Operators opt in via config.
    """
    from agent.transports.claude_agent_sdk_session import _provider_flag

    return _provider_flag("hybrid_mcp_bridge", default=False)


def _snapshot_agent_tools_with_mcp_refresh(agent) -> Optional[List[Dict[str, Any]]]:
    """Return ``agent.tools`` after making sure late-registered MCP tools are
    included. See ``tools/mcp_tool.py::refresh_agent_mcp_tools`` for context:
    the agent snapshots its tool list once at build time and never re-reads
    the registry, so MCP servers whose initial connect finishes AFTER that
    snapshot (slow HTTP handshake, OAuth-gated servers, ``/reload-mcp``) are
    invisible until the snapshot is rebuilt. ``turn_context.py`` already does
    this between turns; we do it at session-creation time too so the hybrid
    MCP bridge (which is built ONCE per session) doesn't freeze a stale
    snapshot into the SDK's ``mcp_servers`` for the entire session.

    Never raises; a refresh failure falls back to the raw snapshot.
    """
    try:
        # Same import-cost gate as turn_context.py's between-turns refresh:
        # ``tools.mcp_tool`` is heavy (~0.4s) and only worth importing when
        # something else already did (which means MCPs may be registered).
        import sys as _sys
        if "tools.mcp_tool" in _sys.modules:
            from tools.mcp_tool_agent import refresh_agent_mcp_tools
            from tools.mcp_tool_discovery import has_registered_mcp_tools
            if has_registered_mcp_tools() and not getattr(
                agent, "_skip_mcp_refresh", False
            ):
                refresh_agent_mcp_tools(agent, quiet_mode=True)
    except Exception:
        logger.debug("MCP refresh before hybrid build failed", exc_info=True)
    return getattr(agent, "tools", None)


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


def _sdk_context_prompt_tokens(usage: dict, fallback: int) -> int:
    """Prompt size of the LAST SDK iteration — i.e. real context pressure.

    LOCAL DIVERGENCE (2026-08-14). ``ResultMessage.usage`` AGGREGATES across
    every internal iteration of a turn. Each iteration re-reads the whole
    prompt from cache, so on a tool-heavy turn the aggregate answers "how many
    tokens did this turn bill?" — NOT "how big is the prompt?".

    Feeding the aggregate to the context compressor over-reported this session
    by ~25x (3,601,066 reported vs ~145k real), which pinned the runtime footer
    to a clamped 100% and made gateway hygiene fire every ~5 minutes forever
    against a threshold it could never satisfy.

    The aggregate remains correct for BILLING and is still used for session
    totals / cost persistence; only context-pressure consumers use this value.
    Falls back to the aggregate when per-iteration data is absent, so a future
    SDK that drops ``iterations`` degrades to today's behaviour rather than
    reporting zero (which would disable compression entirely).
    """
    iterations = usage.get("iterations")
    if not isinstance(iterations, (list, tuple)) or not iterations:
        return fallback
    last = iterations[-1]
    if not isinstance(last, dict):
        return fallback
    total = (
        _coerce_usage_int(last.get("input_tokens"))
        + _coerce_usage_int(last.get("cache_read_input_tokens"))
        + _coerce_usage_int(last.get("cache_creation_input_tokens"))
    )
    return total or fallback


def _usable_cli_context_window(value: Any) -> Optional[int]:
    """Return a CLI window only when it is safe for Hermes tool workflows.

    ``context_usage()`` is an SDK message boundary, not model metadata. Keep
    malformed or sub-minimum reports from collapsing the parent's footer,
    compression, and hygiene sizing; the existing metadata value is safer.
    """
    from agent.model_metadata import MINIMUM_CONTEXT_LENGTH

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        candidate = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or not text.isdecimal():
            return None
        candidate = int(text)
    else:
        # In particular, reject floats: silently truncating 63_999.9 turns an
        # invalid report into a plausible but unsafe context window.
        return None
    return candidate if candidate >= MINIMUM_CONTEXT_LENGTH else None


def _sync_context_length_from_cli(agent: Any) -> None:
    """Point the compressor's context_length at the CLI's REAL window.

    Hermes sizes context from model metadata -- 1,000,000 for claude-opus-5 --
    but on this lane the window is whatever the spawned CLI is actually running
    with, and ``agent.claude_agent_sdk.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW`` can
    cut that to a fraction. Measured 2026-08-16 with the window at 300,000: the
    runtime footer read 16% while the CLI sat at 53% of its real window, one
    turn from autocompacting. The gauge could never have read above ~27%,
    because compaction fires at 267k of a denominator of 1,000,000 -- so the
    whole scale was squashed into its bottom quarter. Gateway session hygiene
    sizes off the same value.

    ``maxTokens`` is fixed when the CLI is spawned, so this is queried once per
    session and cached -- ``context_usage()`` is a real round-trip to the child.
    The attempt is cached on failure too: retrying every turn would spend up to
    the query timeout on the turn path to re-learn the same unavailability.
    """
    compressor = getattr(agent, "context_compressor", None)
    session = getattr(agent, "_claude_sdk_session", None)
    if compressor is None or session is None:
        return
    if getattr(agent, "_sdk_ctxlen_synced_for", None) is session:
        return
    agent._sdk_ctxlen_synced_for = session

    usage = None
    try:
        usage = session.context_usage()
    except Exception:
        logger.debug("claude-sdk context-length sync failed", exc_info=True)
    max_tokens = _usable_cli_context_window(
        usage.get("maxTokens") if isinstance(usage, dict) else None
    )
    if max_tokens is None:
        logger.debug(
            "claude-sdk context length: CLI did not report maxTokens; keeping "
            "the model-metadata value (%s)",
            getattr(compressor, "context_length", None),
        )
        return

    previous = getattr(compressor, "context_length", 0) or 0
    if previous == max_tokens:
        return
    try:
        compressor.context_length = max_tokens
    except Exception:
        logger.debug("claude-sdk context-length assignment failed", exc_info=True)
        return
    logger.info(
        "claude-sdk context length: %d (CLI maxTokens) replaces %d "
        "(model metadata) for footer and hygiene sizing",
        max_tokens, previous,
    )


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def _claude_sdk_billing_accounting(turn) -> tuple[str, str, str, Optional[float]]:
    """Return (mode, status, source, actual_cost) from child evidence.

    ``ResultMessage.total_cost_usd`` is a usage value, not proof of which lane
    paid it. It becomes an actual cost only when the CLI's init/rate-limit
    messages identified a metered lane. The default fail-closed session guard
    labels safe turns ``subscription_included``; an explicit metered opt-in
    with incomplete evidence stays unknown rather than being called included.
    """
    mode = str(getattr(turn, "billing_mode", None) or "unknown")
    actual_cost: Optional[float] = None
    if mode == "sdk_reported_metered":
        raw_cost = getattr(turn, "total_cost_usd", None)
        if not isinstance(raw_cost, bool):
            try:
                candidate = float(raw_cost)
            except (TypeError, ValueError):
                candidate = -1.0
            if candidate >= 0 and math.isfinite(candidate):
                actual_cost = candidate
        return (
            mode,
            "reported" if actual_cost is not None else "unknown",
            "claude-agent-sdk",
            actual_cost,
        )
    if mode == "subscription_included":
        return mode, "included", "claude-subscription", None
    return "unknown", "unknown", "claude-agent-sdk-unverified", None


def _record_claude_sdk_usage(agent, turn) -> dict[str, Any]:
    """Translate SDK ResultMessage usage into Hermes accounting.

    The SDK reports Anthropic-shaped usage: input_tokens, output_tokens,
    cache_read_input_tokens, cache_creation_input_tokens. Billing labels come
    from the child evidence captured by ``ClaudeAgentSdkSession``; never call
    an explicitly allowed API-key/Extra-Usage turn subscription-included."""
    agent.session_api_calls += 1

    billing_mode, cost_status, cost_source, actual_cost = (
        _claude_sdk_billing_accounting(turn)
    )
    agent.session_cost_status = cost_status
    agent.session_cost_source = cost_source

    # Attribution: the configured model wins; when it is unset (documented
    # default — the CLI picks), back-fill from the model id the SDK itself
    # reported so usage rows stop reading model='unknown'.
    resolved_model = agent.model or getattr(turn, "model_last", None) or ""

    usage = getattr(turn, "token_usage_last", None)
    if not isinstance(usage, dict) or not usage:
        if agent._session_db and agent.session_id:
            try:
                if not agent._session_db_created:
                    agent._ensure_db_session()
                agent._session_db.update_token_counts(
                    agent.session_id,
                    model=resolved_model,
                    actual_cost_usd=actual_cost,
                    cost_status=cost_status,
                    cost_source=cost_source,
                    billing_provider=agent.provider,
                    billing_base_url=agent.base_url,
                    billing_mode=billing_mode,
                    api_call_count=1,
                )
            except Exception as exc:
                logger.debug(
                    "claude-sdk api-call persistence failed (session=%s): %s",
                    agent.session_id, exc,
                )
        if billing_mode == "subscription_included":
            return {}
        return {
            "estimated_cost_usd": None,
            "actual_cost_usd": actual_cost,
            "cost_status": cost_status,
            "cost_source": cost_source,
        }

    from agent.usage_pricing import CanonicalUsage

    input_tokens = _coerce_usage_int(usage.get("input_tokens"))
    output_tokens = _coerce_usage_int(usage.get("output_tokens"))
    cache_read_tokens = _coerce_usage_int(usage.get("cache_read_input_tokens"))
    cache_write_tokens = _coerce_usage_int(
        usage.get("cache_creation_input_tokens")
    )

    canonical_usage = CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=0,
        raw_usage=usage,
    )
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = canonical_usage.total_tokens
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": 0,
    }

    # Context pressure != tokens billed. See _sdk_context_prompt_tokens.
    context_prompt_tokens = _sdk_context_prompt_tokens(usage, prompt_tokens)
    if context_prompt_tokens != prompt_tokens:
        logger.debug(
            "claude-sdk context tokens: using last-iteration %d (turn aggregate "
            "was %d) for context pressure; aggregate retained for billing.",
            context_prompt_tokens, prompt_tokens,
        )

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        # BEFORE update_from_response, which compares last_prompt_tokens against
        # threshold_tokens: the setter invalidates the derived budgets, so
        # syncing first means even the first turn's compression decision is
        # taken against the CLI's real window rather than the metadata one.
        _sync_context_length_from_cli(agent)
        try:
            compressor_usage = dict(usage_dict)
            compressor_usage["prompt_tokens"] = context_prompt_tokens
            compressor_usage["total_tokens"] = (
                context_prompt_tokens + completion_tokens
            )
            compressor.update_from_response(compressor_usage)
        except Exception:
            logger.debug("claude-sdk usage update failed", exc_info=True)

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens

    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.update_token_counts(
                agent.session_id,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=0,
                actual_cost_usd=actual_cost,
                cost_status=cost_status,
                cost_source=cost_source,
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode=billing_mode,
                model=resolved_model,
                api_call_count=1,
            )
        except Exception as exc:
            logger.debug(
                "claude-sdk token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, exc,
            )

    return {
        **usage_dict,
        # Context-pressure value: drives the runtime footer's context_pct and
        # gateway session hygiene. usage_dict still carries the BILLING
        # aggregate under "prompt_tokens".
        "last_prompt_tokens": context_prompt_tokens,
        "estimated_cost_usd": None,
        "actual_cost_usd": actual_cost,
        "cost_status": cost_status,
        "cost_source": cost_source,
    }


def _persisted_sdk_session_id(agent) -> Optional[str]:
    """The SDK session id stored on the Hermes session row (or None)."""
    if getattr(agent, "_persist_disabled", False):
        return None
    if not (getattr(agent, "_session_db", None) and getattr(agent, "session_id", None)):
        return None
    try:
        row = agent._session_db.get_session(agent.session_id) or {}
        return row.get("claude_sdk_session_id") or None
    except Exception:
        logger.debug("resume-id read failed", exc_info=True)
        return None


def _store_sdk_session_id(agent, value: Optional[str]) -> None:
    """Persist (or clear, with None) the SDK session id on the session row."""
    if getattr(agent, "_persist_disabled", False):
        # A review/curator fork shares the parent's session_id — it must
        # never write its own resume id onto the parent's row.
        return
    if not (getattr(agent, "_session_db", None) and getattr(agent, "session_id", None)):
        return
    try:
        agent._session_db.update_claude_sdk_session_id(agent.session_id, value)
    except Exception:
        logger.debug("resume-id write failed", exc_info=True)


_CONTINUITY_DIGEST_MAX_CHARS = 4000


def _render_continuity_digest(prior_messages: List[Dict[str, Any]]) -> str:
    """Bounded text preamble for a FRESH SDK session that has prior Hermes
    history (resume impossible: no stored id, or the stored one went stale).
    Reuses _digest_history's compaction, then flattens to capped text."""
    # Projected background results are the agent's OWN answers, already
    # delivered outbound; re-presenting them here is the double-presentation
    # pathology the background lane exists to kill. Filter before the
    # compaction pass — _digest_history may rebuild dicts and drop the mark.
    prior_messages = [
        m for m in (prior_messages or [])
        if not (
            isinstance(m, dict)
            and m.get("display_kind") == "sdk_background_result"
        )
    ]
    try:
        from agent.background_review import _digest_history

        msgs = _digest_history(list(prior_messages or []), tail=8)
    except Exception:  # pragma: no cover - compaction is best-effort
        msgs = list(prior_messages or [])[-8:]
    lines: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not content:
            continue
        text = str(content).replace("\n", " ").strip()
        if text:
            lines.append(f"{role.upper()}: {text[:400]}")
    if not lines:
        return ""
    body = "\n".join(lines)
    if len(body) > _CONTINUITY_DIGEST_MAX_CHARS:
        body = body[-_CONTINUITY_DIGEST_MAX_CHARS:]
    return (
        "[Continuity digest — the runtime restarted and the live model "
        "context was lost; recent turns from the stored transcript, oldest "
        "first:]\n" + body + "\n[End digest. The user's new message follows.]\n\n"
    )


def _configured_max_budget_usd() -> Optional[float]:
    """agent.claude_agent_sdk.max_budget_usd from config.yaml.

    Forwarded to the SDK's ``max_budget_usd`` option: the query stops with an
    ``error_max_budget_usd`` result once exceeded (which run_turn already
    surfaces as "SDK turn ended: error_max_budget_usd"). None/absent — the
    canonical default — means no budget, i.e. current behavior. Non-numeric
    or non-positive values are ignored with a warning rather than passed
    through: a 0 cap would fail every turn instantly, and a typo must never
    become a silent behavior change."""
    from agent.transports.claude_agent_sdk_session import _provider_config

    raw = _provider_config().get("max_budget_usd")
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        # YAML `true` would float() to 1.0 — a nonsense budget, reject it.
        logger.warning(
            "agent.claude_agent_sdk.max_budget_usd=%r is not a number — "
            "ignoring (no budget cap).", raw,
        )
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "agent.claude_agent_sdk.max_budget_usd=%r is not a number — "
            "ignoring (no budget cap).", raw,
        )
        return None
    if value <= 0:
        logger.warning(
            "agent.claude_agent_sdk.max_budget_usd=%r must be positive — "
            "ignoring (no budget cap).", raw,
        )
        return None
    return value


def run_claude_agent_sdk_turn(
    agent,
    *,
    user_message: Any,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """claude-agent-sdk runtime path. Hands the entire turn to the SDK's
    agent loop and projects its messages back into Hermes' list.

    Called from run_conversation() when agent.api_mode == "claude_agent_sdk".
    Returns the same dict shape as the chat_completions path.

    Continuity retire matrix (#25267):
      /new, session expiry      → NEW Hermes session row → no persisted id → fresh
      gateway restart/eviction  → same row, id persisted  → RESUME
      error/timeout retire      → id CLEARED → next turn fresh + digest
      stale/failed resume       → retire → clear → ONE fresh retry with digest
    """
    from agent.transports.claude_agent_sdk_session import (
        ClaudeAgentSdkSession,
        _coerce_turn_input,
    )

    user_input = _coerce_turn_input(user_message)
    if isinstance(user_input, str) and not user_input.strip():
        agent._interrupt_requested = False
        live_session = getattr(agent, "_claude_sdk_session", None)
        if live_session is not None:
            try:
                live_session.consume_interrupt()
            except Exception:
                logger.debug("consume_interrupt failed", exc_info=True)
        rejection = (
            "This Claude Agent SDK route can't process an empty message. "
            "Please send text or a supported image."
        )
        return {
            "final_response": rejection,
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": rejection,
            "interrupted": False,
            "agent_persisted": True,
        }

    # P1.b: refresh the approval-context snapshot EVERY turn (including
    # session-reuse turns). This runs on the agent turn thread, where the
    # session contextvars are visible; the SDK invokes the approval callback
    # from its own loop thread, where they never are — the callback reads
    # this holder instead. A cron turn writes gateway=False (honest deny, no
    # block); a later interactive turn on the SAME SDK session rewrites it
    # and un-freezes the cron-born session.
    try:
        from tools.approval_sdk_gateway import current_approval_turn_context

        agent._sdk_approval_turn_ctx = current_approval_turn_context()
    except Exception:
        logger.debug("approval turn-context refresh failed", exc_info=True)

    def _make_visibility_callbacks():
        """Create visibility callbacks fenced to this exact Hermes turn."""
        visibility_turn_id = str(getattr(agent, "_current_turn_id", "") or "")
        lock = getattr(agent, "_sdk_visibility_lock", None)
        if lock is None:
            lock = threading.RLock()
            agent._sdk_visibility_lock = lock
        with lock:
            agent._sdk_visibility_epoch = getattr(agent, "_sdk_visibility_epoch", 0) + 1
            visibility_epoch = agent._sdk_visibility_epoch
            agent._sdk_visibility_turn_id = visibility_turn_id
            agent._sdk_visibility_iteration_count = 0

        def _visibility_is_current() -> bool:
            with lock:
                return (
                    getattr(agent, "_sdk_visibility_epoch", None) == visibility_epoch
                    and getattr(agent, "_sdk_visibility_turn_id", None) == visibility_turn_id
                    and getattr(agent, "_current_turn_id", None) == visibility_turn_id
                    and not getattr(agent, "_interrupt_requested", False)
                )

        def _on_tool_iteration() -> None:
            with lock:
                if not _visibility_is_current():
                    return
                agent._sdk_visibility_iteration_count += 1
            try:
                agent._touch_activity("completed SDK tool iteration")
            except Exception:
                logger.debug("claude-sdk iteration activity update failed", exc_info=True)

        def _relay_interim_assistant(text: str) -> None:
            if not _visibility_is_current() or not isinstance(text, str):
                return
            visible = agent._strip_think_blocks(text).strip()
            if visible:
                from agent.redact import redact_sensitive_text
                visible = redact_sensitive_text(visible)
            if not visible or visible == "(empty)" or agent._interim_text_was_delivered(visible):
                return
            callback = getattr(agent, "interim_assistant_callback", None)
            if callback is None:
                return
            try:
                callback(visible, already_streamed=False)
                agent._record_delivered_interim_text(visible)
            except Exception:
                logger.debug("interim assistant relay raised", exc_info=True)

        return _relay_interim_assistant, _on_tool_iteration

    def _create_session(resume_id: Optional[str]) -> None:
        from agent.runtime_cwd import resolve_agent_cwd, resolve_context_cwd

        cwd = str(resolve_agent_cwd())
        context_cwd = resolve_context_cwd()
        try:
            from tools.terminal_tool import _get_approval_callback
            approval_callback = _get_approval_callback()
        except Exception:
            approval_callback = None
        if approval_callback is None:
            # Gateway turns have no thread-local CLI callback — without this
            # bridge the SDK denies every un-allowlisted tool silently, no
            # prompt reaching the user, even though the gateway registers a
            # notify channel around every turn (production finding on a 24/7
            # telegram deployment). The builder returns None for surfaces
            # that are not gateway-shaped, so CLI posture is unchanged; the
            # context_provider hands it the per-turn snapshot refreshed
            # above, so cron-ness and the session key are resolved per CALL
            # (a cron-born session must not be frozen into forever-deny).
            try:
                from tools.approval_sdk_gateway import build_sdk_gateway_approval_callback
                approval_callback = build_sdk_gateway_approval_callback(
                    context_provider=lambda: (
                        getattr(agent, "_sdk_approval_turn_ctx", None) or {}
                    ),
                )
            except Exception:
                approval_callback = None

        def _on_tool_started(tool_name: str, preview: str, args: dict) -> None:
            # Claude SDK tool calls bypass the native tool executor, so mirror
            # its shared activity updates here. The gateway heartbeat reads
            # get_activity_summary(), which derives its useful current action
            # from these fields; without this, an active SDK turn remains
            # stuck at its initial "initializing" state.
            agent._sdk_issued_tool_effect = True
            agent._current_tool = tool_name
            try:
                agent._touch_activity(f"executing tool: {tool_name}")
            except Exception:
                logger.debug("claude-sdk activity update failed", exc_info=True)
            progress_callback = getattr(agent, "tool_progress_callback", None)
            if progress_callback is None:
                return
            try:
                progress_callback("tool.started", tool_name, preview, args)
            except Exception:
                logger.debug(
                    "claude-sdk tool-progress callback raised", exc_info=True
                )

        def _relay_stream_delta(text: str) -> None:
            # Late-bound: the gateway assigns stream_delta_callback per turn
            # AFTER the session exists (and clears it between turns).
            # Fan out to BOTH display sinks, mirroring the native runtimes
            # (run_agent.py: [self.stream_delta_callback, self._stream_callback]).
            # `stream_delta_callback` is the CLI/TUI sink. `_stream_callback` is
            # the one the JSON-RPC gateway installs via run_conversation's
            # `stream_callback=` kwarg, and that is the sink the DESKTOP listens
            # on (it feeds the `message.delta` notification). Relaying only to
            # the first meant the desktop never streamed on this runtime, no
            # matter how the operator set display.streaming.
            callbacks = [
                cb
                for cb in (
                    getattr(agent, "stream_delta_callback", None),
                    getattr(agent, "_stream_callback", None),
                )
                if cb is not None
            ]
            if not callbacks:
                return
            agent._record_streamed_assistant_text(text)
            for cb in callbacks:
                try:
                    cb(text)
                except Exception:
                    logger.debug("stream delta relay raised", exc_info=True)

        append = build_system_prompt_append(
            platform=getattr(agent, "platform", None),
            session_id=getattr(agent, "session_id", None),
            model=getattr(agent, "model", None),
            cwd=str(context_cwd) if context_cwd is not None else None,
            include_project_context=not bool(
                getattr(agent, "skip_context_files", False)
            ),
        )

        # Delivery half of the stream-ownership fix: when the CLI finishes a
        # background Agent task between turns, the session captures the
        # answer burst and this callback enqueues it as an
        # "sdk_background_result" completion event; the gateway watcher sends
        # it DIRECTLY on the platform outbound lane (completion_queue →
        # _async_delegation_watcher → adapter send). In-memory at-least-once,
        # same as the watcher's requeue semantics. Config-gated, default OFF
        # per the block's upstream-conservative contract (every default falsy
        # — pinned by test_canonical_defaults); gateway-bot deployments opt
        # in.
        on_unsolicited_result = None
        from agent.transports.claude_agent_sdk_session import _provider_flag

        if _provider_flag("deliver_background_results", default=False):
            # Creation-time snapshots survive as FALLBACKS only — the SDK
            # session outlives hermes session rotations, so anything read
            # here can be stale by the time a background completion fires.
            try:
                from tools.approval_context import get_current_session_key

                _bg_session_key = get_current_session_key() or ""
            except Exception:
                _bg_session_key = ""
            _bg_parent_session_id = getattr(agent, "session_id", None)
            _bg_model = getattr(agent, "model", None)

            def _deliver_background_result(texts: list[str]) -> None:
                # The completion is the AGENT'S OWN finished answer — it must
                # go straight to the platform outbound lane, never back into
                # the model as a synthetic delegation (2026-08-06 self-echo:
                # the model recognized its own text, refused to "relay" it,
                # and the report never left the box). The watcher delivers
                # each payload as its own outbound message, in order.
                #
                # Parent/route are resolved AT DELIVERY TIME: a completion
                # firing after a hermes session rotation must carry the LIVE
                # session id, not the creation-time snapshot — the gateway
                # classifies a rotated-away parent as permanently gone and
                # drops the delivery.
                try:
                    from tools.approval_context import (
                        get_current_session_key as _live_key_fn,
                    )

                    _live_key = _live_key_fn() or ""
                except Exception:
                    _live_key = ""
                # This callback fires on the SDK loop thread, where the
                # get_current_session_key contextvar may be unset — an empty
                # live read falls back to the creation-time snapshot rather
                # than losing the route.
                session_key = _live_key or _bg_session_key
                parent_session_id = (
                    getattr(agent, "session_id", None) or _bg_parent_session_id
                )
                model = getattr(agent, "model", None) or _bg_model
                try:
                    import time as _time

                    from tools.process_registry import process_registry

                    now = _time.time()
                    process_registry.completion_queue.put({
                        "type": "sdk_background_result",
                        "payloads": list(texts),
                        "session_key": session_key,
                        "parent_session_id": parent_session_id,
                        "model": model,
                        "dispatched_at": now,
                        "completed_at": now,
                    })
                except Exception:
                    logger.warning(
                        "claude-sdk background-result enqueue failed — "
                        "answer may be lost", exc_info=True,
                    )

            on_unsolicited_result = _deliver_background_result

        def _on_compaction(trigger: str) -> None:
            """CLI is compacting — surface it with the SHARED status wording.

            Reuses conversation_compression's constants rather than inventing a
            second vocabulary: the gateway's noise filter is built from those
            same templates (#69550), so a re-inlined string would be silently
            dropped on chat surfaces.

            A manual /compact is the user's own action and already has its own
            feedback, so only the automatic case is announced -- that is the one
            that stalls a turn with no explanation.
            """
            if str(trigger).strip().lower() == "manual":
                return
            try:
                from agent.conversation_compression import (
                    COMPACTION_STATUS,
                    COMPACTION_STATUS_KEY,
                )

                agent._sdk_compaction_pending = True
                emit = getattr(agent, "_emit_status_kind", None)
                if callable(emit):
                    emit(COMPACTION_STATUS_KEY, COMPACTION_STATUS, origin="claude_sdk_compaction")
                    logger.info("CLI compaction started (trigger=%s); status emitted", trigger)
                else:
                    logger.info(
                        "CLI compaction started (trigger=%s); no _emit_status_kind, "
                        "status not emitted",
                        trigger,
                    )
            except Exception:
                logger.debug("failed to emit CLI compaction status", exc_info=True)

        def _on_compact_boundary(trigger: str) -> None:
            """Compaction finished — close the status the PreCompact hook opened.

            This is the real terminal edge, mid-turn, ~1 minute before the turn
            ends. The end-of-turn emit below is now only a fallback for a CLI
            that stops streaming compact_boundary; whichever fires first clears
            the pending flag, so the notice is emitted exactly once.

            Guarded on the pending flag rather than emitted unconditionally: a
            manual /compact is never announced on the start side, and announcing
            only its completion would be a notice for an event the user was
            never told had begun.
            """
            if not getattr(agent, "_sdk_compaction_pending", False):
                return
            agent._sdk_compaction_pending = False
            try:
                from agent.conversation_compression import _emit_compaction_done

                logger.info("CLI compaction finished (trigger=%s)", trigger)
                _emit_compaction_done(agent)
            except Exception:
                logger.debug("failed to emit CLI compaction completion", exc_info=True)

        def _approval_bypass_active() -> bool:
            """Resolve live trusted bypass posture for the foreign SDK thread."""
            try:
                from tools.approval import is_approval_bypass_active_for_session

                ctx = getattr(agent, "_sdk_approval_turn_ctx", None)
                session_key = (
                    ctx.get("session_key", "") if type(ctx) is dict else ""
                )
                return is_approval_bypass_active_for_session(session_key)
            except Exception:
                return False

        agent._claude_sdk_session = ClaudeAgentSdkSession(
            cwd=cwd,
            model=getattr(agent, "model", None) or None,
            approval_callback=approval_callback,
            approval_bypass_provider=_approval_bypass_active,
            on_tool_started=_on_tool_started,
            system_prompt_append=append,
            hermes_session_id=getattr(agent, "session_id", None),
            resume_session_id=resume_id,
            on_stream_delta=_relay_stream_delta,
            on_interim_assistant=on_interim_assistant,
            on_tool_iteration=on_tool_iteration,
            on_unsolicited_result=on_unsolicited_result,
            on_compaction=_on_compaction,
            on_compact_boundary=_on_compact_boundary,
            # Operator budget cap (agent.claude_agent_sdk.max_budget_usd);
            # None = no budget. Read per session creation so a config edit
            # applies on the next session, same as the append snapshot.
            max_budget_usd=_configured_max_budget_usd(),
            # Hybrid MCP bridge inputs (ported from PR #56413). Passing the
            # live agent + its OpenAI-format tool list activates an in-process
            # MCP server that exposes the full Hermes tool registry — so
            # proxified third-party MCP servers become reachable from inside
            # the SDK loop, not just the ~25 curated stdio tools.
            #
            # Off by default (agent.claude_agent_sdk.hybrid_mcp_bridge:
            # false) so a green-field upgrade is byte-identical to fcava's
            # stdio-only behaviour — the wide bridge exposes agent-level
            # tools whose enablement is a security choice. Operators opt in
            # explicitly.
            #
            # agent.tools is a snapshot taken at agent build time and never
            # re-reads the registry (see tools/mcp_tool.py::refresh_agent_mcp_tools
            # docstring). If an HTTP MCP finished connecting AFTER that snapshot
            # (e.g. slow initial handshake, or /reload-mcp), its tools would be
            # invisible to the hybrid bridge. Force a refresh here so the bridge
            # sees the current registry — the same call turn_context.py does
            # between turns, but pulled forward so it also applies to the
            # session-creation build.
            agent=(agent if _hybrid_bridge_enabled() else None),
            tools=(
                _snapshot_agent_tools_with_mcp_refresh(agent)
                if _hybrid_bridge_enabled()
                else None
            ),
        )
        # The prologue persisted Hermes' native composed prompt — a prompt
        # this runtime never sends. Overwrite the snapshot with the
        # EFFECTIVE prompt so the audit trail tells the truth.
        try:
            if getattr(agent, "_session_db", None) and agent.session_id:
                agent._session_db.update_system_prompt(
                    agent.session_id, "[claude_code preset]\n\n" + (append or "")
                )
        except Exception:
            logger.debug("effective-prompt snapshot failed", exc_info=True)

    # NOTE: the user message is ALREADY appended to messages by the standard
    # run_conversation() flow before the early return reaches us. Do NOT
    # append again — that would duplicate. (Same contract as codex_runtime.)

    # An interrupt that landed before the SDK session exists (first turn, or
    # right after a retire) only set agent._interrupt_requested — honor it
    # here, mirroring the native loop's top-of-loop check, and consume the
    # flag so the NEXT turn runs normally.
    if getattr(agent, "_interrupt_requested", False):
        agent._interrupt_requested = False
        live_session = getattr(agent, "_claude_sdk_session", None)
        if live_session is not None:
            # interrupt() also set the live session's event; consume it here
            # or the NEXT legitimate message dies on the stale event with no
            # model call.
            try:
                live_session.consume_interrupt()
            except Exception:
                logger.debug("consume_interrupt failed", exc_info=True)
        return {
            "final_response": "",
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            # Without this key the gateway's empty-response normalizer has no
            # branch to take (interrupted absent, api_calls 0, partial True →
            # every arm defeated) and the user's message dies in SILENCE.
            # With it, api_calls==0 + interrupted surfaces the honest
            # "interrupted before processing — send it again" path.
            "interrupted": True,
            "error": None,
            "agent_persisted": True,
        }

    # Stream/replay state belongs to this SDK attempt, never to the cached
    # gateway agent.  Reset before session startup too: an authoritative auth
    # failure may happen before a model call and must not inherit prior output.
    agent._current_streamed_assistant_text = ""
    agent._sdk_issued_tool_effect = False
    _messages_before_sdk_attempt = copy.deepcopy(messages)

    on_interim_assistant, on_tool_iteration = _make_visibility_callbacks()
    live_session = getattr(agent, "_claude_sdk_session", None)
    if live_session is not None:
        try:
            live_session.set_turn_visibility_callbacks(
                on_interim_assistant=on_interim_assistant,
                on_tool_iteration=on_tool_iteration,
            )
        except Exception:
            logger.debug("claude-sdk visibility callback refresh failed", exc_info=True)

    turn = None
    resumed = False
    send_input = user_input
    for attempt in (0, 1):
        if not hasattr(agent, "_claude_sdk_session") or agent._claude_sdk_session is None:
            resume_id = _persisted_sdk_session_id(agent) if attempt == 0 else None
            resumed = bool(resume_id)
            send_input = user_input
            if not resume_id and len(messages) > 1:
                digest = _render_continuity_digest(messages[:-1])
                if digest:
                    if isinstance(user_input, list):
                        send_input = [
                            {"type": "text", "text": digest},
                            *user_input,
                        ]
                    else:
                        send_input = digest + user_input
            _create_session(resume_id)

        try:
            turn = agent._claude_sdk_session.run_turn(user_input=send_input)
        except Exception as exc:
            safe_exc = redact_sensitive_text(str(exc), force=True)
            # A PreCompact hook may have opened a transient user-visible status.
            # This exception bypasses the normal terminal edge below; clear it
            # here so a later unrelated turn cannot announce stale completion.
            if getattr(agent, "_sdk_compaction_pending", False):
                agent._sdk_compaction_pending = False
                try:
                    emit = getattr(agent, "_emit_status", None)
                    if callable(emit):
                        emit("⚠️ Context compaction interrupted")
                except Exception:
                    logger.debug("failed to close interrupted compaction status", exc_info=True)
            # Do not use logger.exception here: it appends the raw exception
            # string after the redacted message to the log record.
            logger.error("claude-agent-sdk turn failed: %s", safe_exc)
            try:
                agent._claude_sdk_session.close()
            except Exception:
                pass
            agent._claude_sdk_session = None
            if resumed and attempt == 0:
                # A raising RESUMED session is a suspect resume — clear the
                # id and give the turn one fresh chance (digest included).
                _store_sdk_session_id(agent, None)
                resumed = False
                continue
            return {
                "final_response": f"claude-agent-sdk turn failed: {safe_exc}",
                "messages": messages,
                "api_calls": 0,
                "completed": False,
                "partial": True,
                # run_turn consumes its own exceptions into TurnResult, so
                # anything RAISING here is a dead turn, not a recoverable
                # partial — mark it failed so one-shot runs exit nonzero
                # (mirrors conversation_loop's generic non-retryable return).
                "failed": True,
                "error": safe_exc,
            }

        if getattr(turn, "should_retire", False):
            logger.warning(
                "claude-agent-sdk session retired (turn error: %s)",
                redact_sensitive_text(str(turn.error or ""), force=True),
            )
            try:
                agent._claude_sdk_session.close()
            except Exception:
                pass
            agent._claude_sdk_session = None
            # Error/timeout retire always clears the persisted resume id —
            # never resume a conversation that just failed.
            _store_sdk_session_id(agent, None)
            if (
                resumed
                and attempt == 0
                and not getattr(turn, "interrupted", False)
            ):
                # Stale/failed resume: one fresh retry with digest. Never for
                # an INTERRUPTED retire (user /stop that killed the CLI, or a
                # hard watchdog trip) — re-running the stopped turn in full
                # would evaporate the stop and deliver the answer anyway.
                resumed = False
                continue
        break

    _sdk_effects = ClaudeSdkTurnEffects(
        tool=(
            bool(getattr(agent, "_sdk_issued_tool_effect", False))
            or int(getattr(turn, "tool_iterations", 0) or 0) > 0
        ),
        streamed=bool(getattr(agent, "_current_streamed_assistant_text", "")),
        projected=bool(getattr(turn, "projected_messages", None)),
        interrupted=bool(
            getattr(turn, "interrupted", False)
            or getattr(agent, "_interrupt_requested", False)
        ),
        mutated=messages != _messages_before_sdk_attempt,
    )
    _sdk_failover_reason = None
    if getattr(turn, "error", None) and _sdk_effects.replay_safe:
        _sdk_failover_reason = _sdk_provider_failover_reason(
            agent,
            str(turn.error),
            getattr(turn, "fatal_reason", None),
        )
        if _sdk_failover_reason is not None and agent._claude_sdk_session is not None:
            # A provider switch must never retain transport/session state from
            # the failed SDK backend.
            try:
                agent._claude_sdk_session.close()
            except Exception:
                pass
            agent._claude_sdk_session = None
            _store_sdk_session_id(agent, None)

    # FALLBACK ONLY. _on_compact_boundary above is the real terminal edge and
    # normally clears the flag mid-turn; reaching here means the CLI started a
    # compaction and never streamed its compact_boundary (older/newer CLI, or a
    # turn that died mid-compaction). A completed turn still proves the
    # compaction ended, so this closes the dangling status rather than leaving
    # "🗜️ Compacting..." as the user's last word from the turn.
    #
    # Do NOT promote this back to the primary path: end-of-turn is exactly where
    # end-of-turn progress cleanup deletes the message, so the notice is emitted
    # and destroyed in the same instant (see _handle_compact_boundary).
    if getattr(agent, "_sdk_compaction_pending", False):
        agent._sdk_compaction_pending = False
        try:
            from agent.conversation_compression import _emit_compaction_done

            logger.info(
                "CLI compaction: no compact_boundary seen; emitting completion "
                "at turn end (fallback)"
            )
            _emit_compaction_done(agent)
        except Exception:
            logger.debug("failed to emit CLI compaction completion", exc_info=True)

    # Interrupt handoff (codex_runtime parity, its ~739-746): capture BEFORE
    # the consume below zeroes the agent flag — the result dict needs it, and
    # without an "interrupted" key the gateway's queued-drain classifies an
    # interrupted turn as a plain partial error and DELIVERS the abandoned
    # turn's error text ("⚠️ Processing stopped… Try again", 2026-08-09
    # barge-in incident) instead of discarding it via its interrupted branch.
    _user_interrupted = bool(
        getattr(turn, "interrupted", False)
        and getattr(agent, "_interrupt_requested", False)
    )

    if getattr(turn, "interrupted", False):
        # The interrupt was honored by THIS turn — consume the agent-level
        # flag so the next turn is not short-circuited by it.
        agent._interrupt_requested = False
        if agent._claude_sdk_session is not None:
            # The abandoned stream may still hold the interrupted turn's
            # ResultMessage; a REUSED client would serve it as the NEXT
            # turn's answer. Retire the client — the persisted id below lets
            # the next turn RESUME the same SDK conversation cleanly.
            try:
                agent._claude_sdk_session.close()
            except Exception:
                pass
            agent._claude_sdk_session = None

    if turn.projected_messages:
        messages.extend(turn.projected_messages)
        # Early-return path bypasses conversation_loop's per-step persistence;
        # flush the new projected rows ourselves (idempotent via the intrinsic
        # _DB_PERSISTED_MARKER — the user turn was flushed at turn start).
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug(
                    "claude-sdk projected-message flush failed", exc_info=True
                )

    if not getattr(turn, "should_retire", False) and _sdk_failover_reason is None:
        # Persist the SDK session id for restart/eviction/interrupt resume.
        # AFTER the flush on purpose: the flush's _ensure_db_session retry is
        # what (re)creates the session row when turn-start persistence hit a
        # transient lock — storing first would silently discard the id.
        thread_id = getattr(turn, "thread_id", None)
        if thread_id:
            _store_sdk_session_id(agent, thread_id)

    # Counter ticks — _turns_since_memory/_user_turn_count are incremented by
    # run_conversation()'s pre-loop block; only _iters_since_skill is ours.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    usage_result = (
        _record_claude_sdk_usage(agent, turn)
        if getattr(turn, "api_call_made", True)
        else {}
    )

    should_review_skills = False
    # Skill-review cadence belongs to review policy, not foreground tool
    # availability. If routed, a distinct normal runtime owns optional writes.
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    if (
        turn.final_text
        and not turn.interrupted
        and not agent.skip_background_review
        and (should_review_memory or should_review_skills)
    ):
        # #25267 suppressed this spawn unconditionally: the fork inherits
        # api_mode="claude_agent_sdk" and early-returns into a fresh SDK
        # session whose tool surface has no `memory` / `skill_manage`, so it
        # burned a subscription turn and wrote nothing.
        #
        # That reasoning only holds when the fork stays on THIS runtime. With
        # auxiliary.background_review naming a concrete different
        # provider/model, _resolve_review_runtime reports routed=True and the
        # review runs over there — on a normal tool surface that can write.
        # Suppressing it in that case is what left this lane unable to record
        # anything durable across sessions.
        #
        # Unrouted, #25267 still applies: skip, and let the counters above
        # keep ticking for a bounded replacement pass.
        routed = False
        try:
            from agent.background_review import _resolve_review_runtime

            routed = bool(_resolve_review_runtime(agent).get("routed"))
        except Exception:
            logger.debug("review-runtime resolve failed", exc_info=True)

        if routed:
            try:
                agent._spawn_background_review(
                    messages_snapshot=list(messages),
                    review_memory=should_review_memory,
                    review_skills=should_review_skills,
                )
            except Exception:
                logger.debug("background review spawn raised", exc_info=True)
        else:
            logger.debug(
                "claude-sdk runtime: background review skipped "
                "(memory=%s, skills=%s) — the review fork cannot write on "
                "this runtime",
                should_review_memory,
                should_review_skills,
            )

    result = {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": int(getattr(turn, "api_call_made", True)),
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "failed": bool(turn.error) and not _sdk_effects.interrupted,
        "error": redact_sensitive_text(str(turn.error or ""), force=True) if turn.error else None,
        "interrupted": _user_interrupted,
        "sdk_effects": _sdk_effects.as_result_dict(),
        **(
            {"failover_reason": _sdk_failover_reason.value}
            if _sdk_failover_reason is not None
            else {}
        ),
        # Same persistence contract as the codex app-server path: we flushed
        # the projected rows ourselves, so the gateway must not re-write the
        # user turn (append_message has no dedup).
        "agent_persisted": True,
        "claude_sdk_session_id": turn.thread_id,
        **usage_result,
    }
    # Fatal startup/auth/billing refusals surface the same machine-readable
    # fields the chat_completions path sets (conversation_loop), so the -Q
    # exit contract and gateway consumers see a real failure instead of a
    # recoverable partial. getattr: test doubles build TurnResult-shaped
    # namespaces that may predate the field.
    _fatal = getattr(turn, "fatal_reason", None)
    if _fatal:
        result["failed"] = True
        result["failure_reason"] = _fatal
    return result


__all__ = ["run_claude_agent_sdk_turn"]
