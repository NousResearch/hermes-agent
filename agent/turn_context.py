"""Per-turn setup for ``run_conversation`` (the turn prologue).

``run_conversation`` opened with ~470 lines of straight-line setup before the
tool-calling loop ever started: stdio guarding, runtime-main wiring, retry-counter
resets, user-message sanitization, todo/nudge-counter hydration, system-prompt
restore-or-build, session-row creation (before compression, whose DB writes
reference the row), preflight context compression, the ``pre_llm_call`` plugin
hook, external-memory prefetch, and crash-resilience persistence (last, so the
user row is written once with its final ``api_content`` sidecar).

All of that is *prologue* — it runs once per turn, has no back-references into the
loop, and produces a fixed set of values the loop then consumes. ``TurnContext``
captures those produced values; ``build_turn_context`` performs the setup work and
returns one. ``run_conversation`` is left to unpack the context and run the loop,
shrinking the orchestrator by the full prologue.

The builder still mutates ``agent`` heavily (counters, thread id, cached prompt,
session DB) exactly as the inline code did — those side effects are the point. The
``TurnContext`` it returns carries only the *locals* the loop reads back.

Behavior is identical to the original inline prologue; this is a pure
move-and-name refactor with no semantic change.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from agent.conversation_compression import (
    IDLE_COMPACTION_STATUS_TEMPLATE,
    PREFLIGHT_COMPRESSION_STATUS_TEMPLATE,
    conversation_history_after_compression,
)
from agent.iteration_budget import IterationBudget
from agent.memory_manager import build_memory_context_block
from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
)

logger = logging.getLogger(__name__)


def compose_user_api_content(
    content: Any,
    ext_prefetch_cache: str,
    plugin_user_context: str,
) -> Optional[str]:
    """Compose the API-bound content of the current turn's user message.

    Sources: memory-manager prefetch + ``pre_llm_call`` plugin context with
    target="user_message" (the default). Both are appended to the *API copy*
    of the user message only — the stored content stays clean.

    This is the single source of that composition. The prologue stamps the
    result onto the live message as ``api_content`` (persisted alongside the
    clean content) and the ``api_messages`` build in ``conversation_loop``
    sends the same helper's output, so the persisted sidecar can never drift
    from the bytes on the wire — which is the whole prompt-cache invariant:
    what turn N sends must be what turn N+1 replays.

    Returns ``None`` when nothing is injected (multimodal/non-string content,
    or no ephemeral context), meaning the message is sent as-is.
    """
    if not isinstance(content, str):
        return None
    injections = []
    if ext_prefetch_cache:
        fenced = build_memory_context_block(ext_prefetch_cache)
        if fenced:
            injections.append(fenced)
    if plugin_user_context:
        injections.append(plugin_user_context)
    if not injections:
        return None
    return content + "\n\n" + "\n\n".join(injections)


def substitute_api_content(api_msg: Dict[str, Any]) -> Optional[str]:
    """Pop the ``api_content`` sidecar and substitute it into ``content``.

    Used at every API-bound message-build site (the ``api_messages`` build in
    ``conversation_loop``, the max-iterations summary in
    ``chat_completion_helpers``, the chat-completions transport). The sidecar
    carries the exact bytes previously sent to the API for this message when
    they differ from the clean stored content; substituting it here keeps the
    provider prompt-cache prefix byte-stable across turns.

    Returns the popped sidecar string (for callers that need the value for
    current-turn composition logic) or ``None`` when absent.
    """
    sidecar = api_msg.pop("api_content", None)
    if (
        isinstance(sidecar, str)
        and sidecar
        and api_msg.get("role") in ("user", "assistant")
    ):
        api_msg["content"] = sidecar
    return sidecar


def drop_stale_api_content(msg: Dict[str, Any]) -> None:
    """Drop the ``api_content`` sidecar from a message whose content was rewritten.

    Called from every content-rewrite path (historical image strip,
    merge-summary-into-tail, consecutive-user repair merge, stale-confirmation
    redaction). Replaying the pre-rewrite sidecar would resend exactly what
    the rewrite removed, so it must be dropped — the cost is one cache
    boundary miss, never wrong content.
    """
    msg.pop("api_content", None)


def extract_api_content_sidecar(msg: Mapping[str, Any]) -> Optional[str]:
    """Extract the ``api_content`` sidecar from a message dict for persistence.

    Shared by the gateway/branch forwarding sites that copy the sidecar into a
    new row. Returns the string sidecar or ``None`` when absent/non-string.
    """
    v = msg.get("api_content")
    return v if isinstance(v, str) else None


def consume_gateway_turn_context_notes(agent: Any) -> str:
    """Pop the gateway's per-turn must-deliver notes off the agent (one-shot).

    The gateway relocates volatile per-turn facts OUT of the ephemeral system
    prompt (auto-reset notes, the first-contact intro, voice-channel changes)
    and delivers them on the current user message via the api_content sidecar
    instead, so the composed system prompt stays byte-stable turn-over-turn.
    It stages the rendered notes on ``agent._gateway_turn_context_notes``
    right before ``run_conversation``; this consumes them so a cached agent
    can never replay a stale note on a later turn.
    """
    notes = getattr(agent, "_gateway_turn_context_notes", "") or ""
    if hasattr(agent, "_gateway_turn_context_notes"):
        try:
            agent._gateway_turn_context_notes = ""
        except Exception:
            pass
    return notes if isinstance(notes, str) else ""


def append_notes_to_multimodal_content(content: Any, notes: str) -> bool:
    """Deliver must-deliver notes on a multimodal (list) user message.

    ``compose_user_api_content`` returns ``None`` for non-string content, so
    sidecar-borne facts would silently drop on image/attachment turns.  For
    gateway must-deliver notes we instead append a text part to the content
    list in place — the part becomes durable message content (persisted and
    replayed as-is), which keeps the wire and the transcript byte-identical.

    Returns ``True`` when a part was appended.
    """
    if not notes or not isinstance(content, list):
        return False
    try:
        content.append({"type": "text", "text": notes})
        return True
    except Exception:
        return False


def reanchor_current_turn_user_idx(messages: List[Any], user_message: Any) -> int:
    """Locate this turn's user message after compaction rebuilt ``messages``.

    Compression replaces list entries with fresh copies (and may append a
    todo-snapshot user message or a restored user turn AFTER the surviving
    copy of the current turn's message), so a pre-compression index is
    meaningless. Prefer the LAST user message whose content exactly matches
    this turn's text — the surviving copy in the common case — so the
    injection stamp and the #48677 persist override can't land on a
    todo-snapshot or historical row. Fall back to the last user message when
    no exact match survives (merge-summary-into-tail rewrites the content but
    the trackers still need a live anchor). Returns -1 when the list has no
    user message at all.
    """
    fallback = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not (isinstance(msg, dict) and msg.get("role") == "user"):
            continue
        if fallback < 0:
            fallback = i
        if msg.get("content") == user_message:
            return i
    return fallback


def _compression_made_progress(
    orig_len: int, new_len: int, orig_tokens: int, new_tokens: int
) -> bool:
    """Return ``True`` if a compression pass materially reduced the request.

    Compression can succeed by summarising message contents — reducing the
    estimated request token count — without reducing the message row
    count.  Treating row count as the sole progress signal false-positives
    on size-only wins and surfaces a misleading "Cannot compress further"
    failure even when post-compression tokens are well below the model
    context window.  See issue #39548 for an observed case: 220 → 220
    messages, ~288k → ~183k tokens on a 1M-context model still triggered
    auto-reset.

    The token reduction must be *material* (>5%) to count as progress — the
    same floor the overflow-handler retry path uses (conversation_loop.py,
    #39550) — so a sub-5% wobble doesn't keep the multi-pass loop spinning.
    """
    if new_len < orig_len:
        return True
    return orig_tokens > 0 and new_tokens < orig_tokens * 0.95


def _compression_warrants_another_preflight_pass(
    orig_tokens: int, new_tokens: int, threshold_tokens: int
) -> bool:
    """Whether an over-threshold request merits another immediate summary.

    Row-count progress is enough to prove that a compression boundary was real,
    but not enough to justify another expensive pass before trying the provider.
    Continue only when the request remains over threshold *and* the previous pass
    materially reduced its estimated token pressure (>5%).
    """
    return (
        new_tokens >= threshold_tokens
        and orig_tokens > 0
        and new_tokens < orig_tokens * 0.95
    )


def _should_run_preflight_estimate(
    messages: List[Dict[str, Any]],
    protect_first_n: int,
    protect_last_n: int,
    threshold_tokens: int,
) -> bool:
    """Cheap gate for the (expensive) full preflight token estimate.

    Returns ``True`` when either:
      (a) message count exceeds the protected ranges (the historical gate), or
      (b) a cheap char-based estimate already crosses the configured threshold
          — the few-but-huge case from issue #27405 that the count-only gate
          would silently skip (a handful of very large messages never trips
          the count condition, so compression was never attempted and the
          turn hit a hard context-overflow error).

    Branch (b) uses ``estimate_messages_tokens_rough`` (the shared char-based
    estimator) so a single large base64 image isn't mistaken for ~250K tokens.
    It intentionally undercounts vs. the full request estimate — it omits the
    system prompt and tool schemas — because it is only a *hint* deciding
    whether to pay for the authoritative ``estimate_request_tokens_rough``,
    which (together with ``should_compress``) makes the real decision.
    """
    if len(messages) > protect_first_n + protect_last_n + 1:
        return True
    return estimate_messages_tokens_rough(messages) >= threshold_tokens


def _should_idle_compact(
    *,
    enabled: bool,
    idle_after_seconds: int,
    idle_gap_seconds: float,
    tokens: int,
    floor_tokens: int,
    cooldown_active: bool,
) -> bool:
    """Decide whether an idle-triggered compaction should run this turn.

    Idle compaction is opt-in (``idle_after_seconds <= 0`` disables it). It
    fires when a session resumes after a wall-clock gap of at least
    ``idle_after_seconds`` since its last activity, so a long-lived thread
    that is paused and later resumed compacts its accumulated history up
    front instead of re-reading it on every subsequent turn.

    It is orthogonal to the token-threshold trigger: it does NOT require the
    context to exceed ``threshold_tokens``. It still skips work when the
    context is at or below ``floor_tokens`` (the size compaction would reduce
    *to*), so a small idle thread never pays for a summarisation that saves
    nothing, and it defers to an active compression-failure cooldown.

    Pure predicate so the policy is unit-testable without a live agent.
    """
    if not enabled or idle_after_seconds <= 0:
        return False
    if idle_gap_seconds < idle_after_seconds:
        return False
    if cooldown_active:
        return False
    return tokens > floor_tokens


@dataclass
class TurnContext:
    """Values produced by the turn prologue and consumed by the turn loop."""

    # Sanitized inbound message (surrogates stripped).
    user_message: str
    # Clean message preserved for transcripts / memory queries (no nudge injection).
    original_user_message: Any
    # Working message list for this turn (loop appends to it).
    messages: List[Dict[str, Any]]
    # May be reset to None by preflight compression (new session created).
    conversation_history: Optional[List[Dict[str, Any]]]
    # Cached system prompt active for this turn (may be rebuilt by compression).
    active_system_prompt: Optional[str]
    # Task / turn identifiers.
    effective_task_id: str
    turn_id: str
    # Index of the current user turn within ``messages``.
    current_turn_user_idx: int
    # Whether the post-turn memory review should fire.
    should_review_memory: bool = False
    # Context contributed by ``pre_llm_call`` plugins (appended to user message).
    plugin_user_context: str = ""
    # External-memory prefetch result, reused across loop iterations.
    ext_prefetch_cache: str = ""
    # Turn-start preflight already proved an immediate retry ineffective.
    preflight_compression_blocked: bool = False


def build_turn_context(
    agent,
    user_message: Any,
    system_message: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]],
    task_id: Optional[str],
    stream_callback,
    persist_user_message: Optional[Any],
    persist_user_timestamp: Optional[float] = None,
    *,
    restore_or_build_system_prompt,
    install_safe_stdio,
    sanitize_surrogates,
    summarize_user_message_for_log,
    set_session_context,
    set_current_write_origin,
    ra,
    moa_active: bool = False,
) -> TurnContext:
    """Run the once-per-turn setup and return the loop's input context.

    The callables/helpers the original prologue referenced from the
    ``conversation_loop`` module are passed in explicitly to keep this module
    free of an import cycle with ``agent.conversation_loop``.
    """
    # Guard stdio against OSError from broken pipes (systemd/headless/daemon).
    install_safe_stdio()

    # NOTE: the DB session row is created later, AFTER the system prompt is
    # restored/built (see _ensure_db_session() below the system-prompt block).
    # Creating it here — before _cached_system_prompt is populated — inserts a
    # row with system_prompt=NULL on a fresh API/gateway agent that carries
    # client-managed history, which then trips the "stored system prompt is
    # null; rebuilding from scratch" warning and a needless first-turn prefix
    # cache miss. (Issue #45499.)

    # Tag log records on this thread with the session ID for ``hermes logs``.
    set_session_context(agent.session_id)

    # Bind the skill write-origin ContextVar for this thread.
    set_current_write_origin(getattr(agent, "_memory_write_origin", "assistant_tool"))

    # Restore the primary runtime if the previous turn activated fallback.
    agent._restore_primary_runtime()

    # Tell auxiliary_client what the live main provider/model are for this turn
    # after primary restoration has settled the runtime.
    try:
        from agent.auxiliary_client import set_runtime_main
        set_runtime_main(
            getattr(agent, "provider", "") or "",
            getattr(agent, "model", "") or "",
            base_url=getattr(agent, "base_url", "") or "",
            api_key=getattr(agent, "api_key", "") or "",
            api_mode=getattr(agent, "api_mode", "") or "",
            auth_mode=getattr(agent, "auth_mode", "") or "",
        )
    except Exception:
        pass

    # Between-turns MCP refresh: an MCP server that finished connecting since
    # the previous turn (slow HTTP/OAuth servers routinely take 2-6s on a cold
    # connect, missing the bounded startup wait) lands in THIS turn's tool
    # snapshot.  This is cache-safe by construction: it runs in the per-turn
    # prologue, before this turn's first API call assembles ``tools=``, so it
    # only ever extends a fresh request prefix — it never mutates the cached
    # prefix of an in-flight turn.  No-op when no MCP servers are registered
    # (the common case, gated by the cheap ``has_registered_mcp_tools`` check)
    # or when the tool set is unchanged (``refresh_agent_mcp_tools`` diffs by
    # name and leaves the snapshot untouched on no-change).
    try:
        if not getattr(agent, "_skip_mcp_refresh", False):
            # Import-cost gate: ``tools.mcp_tool`` pulls in the whole ``mcp``
            # package (~0.4s measured) even when the user has zero MCP servers
            # configured.  MCP tools can only be registered by code that has
            # already imported ``tools.mcp_tool`` (discovery, /reload-mcp,
            # late-binding refresh) — so if it isn't in sys.modules yet, there
            # is nothing to refresh and the import can be skipped outright.
            # This keeps the no-MCP first turn off the heavy import path
            # without changing behavior for MCP users.
            import sys as _sys
            if "tools.mcp_tool" in _sys.modules:
                from tools.mcp_tool import has_registered_mcp_tools, refresh_agent_mcp_tools
                if has_registered_mcp_tools():
                    refresh_agent_mcp_tools(agent, quiet_mode=True)
    except Exception:
        logger.debug("between-turns MCP tool refresh skipped", exc_info=True)

    # Sanitize surrogate characters from user input.
    if isinstance(user_message, str):
        user_message = sanitize_surrogates(user_message)
    if isinstance(persist_user_message, str):
        persist_user_message = sanitize_surrogates(persist_user_message)

    # Store stream callback for _interruptible_api_call to pick up.
    agent._stream_callback = stream_callback
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = persist_user_message
    agent._persist_user_message_timestamp = persist_user_timestamp
    # Generate unique task_id if not provided to isolate VMs between tasks.
    effective_task_id = task_id or str(uuid.uuid4())
    agent._current_task_id = effective_task_id
    turn_id = f"{agent.session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex[:8]}"
    agent._current_turn_id = turn_id
    agent._current_api_request_id = ""
    # Tripwire: warn (with both turn ids) when this turn starts before the
    # previous turn's turn-end persist — concurrent turns on one session
    # interleave transcript writes. Cleared in _persist_session.
    from agent.agent_runtime_helpers import note_turn_start
    note_turn_start(agent, turn_id)

    # Reset retry counters and iteration budget at the start of each turn.
    agent._invalid_tool_retries = 0
    agent._invalid_json_retries = 0
    agent._empty_content_retries = 0
    agent._incomplete_scratchpad_retries = 0
    agent._codex_incomplete_retries = 0
    agent._thinking_prefill_retries = 0
    agent._post_tool_empty_retried = False
    agent._last_content_with_tools = None
    agent._last_content_tools_all_housekeeping = False
    agent._mute_post_response = False
    agent._unicode_sanitization_passes = 0
    agent._tool_guardrails.reset_for_turn()
    agent._tool_guardrail_halt_decision = None
    _reset_consol = getattr(agent._memory_store, "reset_consolidation_failures", None)
    if callable(_reset_consol):
        _reset_consol()
    agent._vision_supported = True

    # Pre-turn connection health check: clean up dead TCP connections.
    if agent.api_mode != "anthropic_messages":
        try:
            if agent._cleanup_dead_connections():
                agent._emit_status(
                    "🔌 Detected stale connections from a previous provider "
                    "issue — cleaned up automatically. Proceeding with fresh "
                    "connection."
                )
        except Exception:
            pass
    # Replay compression warning through status_callback for gateway platforms.
    if agent._compression_warning:
        agent._replay_compression_warning()
        agent._compression_warning = None  # send once

    # NOTE: _turns_since_memory and _iters_since_skill are NOT reset here.
    agent.iteration_budget = IterationBudget(agent.max_iterations)

    # Log conversation turn start for debugging/observability.
    _preview_text = summarize_user_message_for_log(user_message)
    _msg_preview = (_preview_text[:80] + "...") if len(_preview_text) > 80 else _preview_text
    _msg_preview = _msg_preview.replace("\n", " ")
    logger.info(
        "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
        agent.session_id or "none", agent.model, agent.provider or "unknown",
        agent.platform or "unknown", len(conversation_history or []),
        _msg_preview,
    )

    # Initialize conversation (copy to avoid mutating the caller's list).
    messages = list(conversation_history) if conversation_history else []

    # The CLI may already have staged this input outside the history passed to
    # ``run_conversation``. Reuse it only when its clean transcript text matches
    # this turn; a stale handoff from a failed prior turn must not replace a
    # later, different user input. Voice turns compare against their explicit
    # clean persistence override rather than the API-only prefixed payload.
    pending_cli_message = getattr(agent, "_pending_cli_user_message", None)
    expected_persist_content = (
        persist_user_message if persist_user_message is not None else user_message
    )
    if (
        isinstance(pending_cli_message, dict)
        and pending_cli_message.get("content") == expected_persist_content
    ):
        user_msg = pending_cli_message
        # The CLI-staged value is the clean transcript text. Restore the
        # API-facing variant (for example, a voice-mode prefix) while retaining
        # the same dict and any close-path durable marker.
        user_msg["content"] = user_message
    else:
        user_msg = {"role": "user", "content": user_message}
        if isinstance(pending_cli_message, dict):
            agent._pending_cli_user_message = None

    # Hydrate todo store from conversation history.
    if conversation_history and not agent._todo_store.has_items():
        agent._hydrate_todo_store(conversation_history)

    # Hydrate per-session nudge counters from persisted history (issue #22357).
    if conversation_history and agent._user_turn_count == 0:
        prior_user_turns = sum(
            1 for m in conversation_history if m.get("role") == "user"
        )
        if prior_user_turns > 0:
            agent._user_turn_count = prior_user_turns
            if agent._memory_nudge_interval > 0 and agent._turns_since_memory == 0:
                agent._turns_since_memory = prior_user_turns % agent._memory_nudge_interval

    # Add the current user message after the prompt/session setup has made
    # close persistence safe. The handoff above preserves any marker already
    # stamped by an earlier close flush.
    messages.append(user_msg)
    current_turn_user_idx = len(messages) - 1
    agent._persist_user_message_idx = current_turn_user_idx

    # Track user turns for memory flush and periodic nudge logic.
    agent._user_turn_count += 1
    # Copilot x-initiator: the first API call of this user turn is
    # user-initiated; tool-loop follow-ups revert to "agent" (#3040).
    agent._is_user_initiated_turn = True

    # Reset the streaming context scrubber at the top of each turn.
    scrubber = getattr(agent, "_stream_context_scrubber", None)
    if scrubber is not None:
        scrubber.reset()
    # Reset the think scrubber for the same reason.
    think_scrubber = getattr(agent, "_stream_think_scrubber", None)
    if think_scrubber is not None:
        think_scrubber.reset()

    # Preserve the original user message (no nudge injection).
    original_user_message = persist_user_message if persist_user_message is not None else user_message

    # Track memory nudge trigger (turn-based, checked here).
    should_review_memory = False
    if (agent._memory_nudge_interval > 0
            and "memory" in agent.valid_tool_names
            and agent._memory_store):
        agent._turns_since_memory += 1
        if agent._turns_since_memory >= agent._memory_nudge_interval:
            should_review_memory = True
            agent._turns_since_memory = 0

    # Cosmetic side-signal: detect an affection "reaction" (ily / <3 / good bot)
    # and notify the host so it can play hearts. Token-free, never touches the
    # conversation, and never fatal — a purely optional UI beat.
    reaction_callback = getattr(agent, "reaction_callback", None)
    if reaction_callback is not None:
        try:
            from agent.reactions import detect_reaction

            kind = detect_reaction(original_user_message)
            if kind:
                reaction_callback(kind)
        except Exception:
            pass

    if not agent.quiet_mode:
        _print_preview = summarize_user_message_for_log(user_message)
        agent._safe_print(
            f"💬 Starting conversation: '{_print_preview[:60]}"
            f"{'...' if len(_print_preview) > 60 else ''}'"
        )

    # ── System prompt (cached per session for prefix caching) ──
    if agent._cached_system_prompt is None:
        restore_or_build_system_prompt(agent, system_message, conversation_history)

    active_system_prompt = agent._cached_system_prompt

    # Create the DB session row now that _cached_system_prompt is populated, so
    # the persisted snapshot is written non-NULL on the first turn (Issue
    # #45499). Idempotent: _ensure_db_session() no-ops once the row exists.
    # Must run BEFORE preflight compression: in-place compaction inserts
    # message rows referencing this session (archive_and_compact), and
    # rotation creates a child with parent_session_id pointing at it — with
    # PRAGMA foreign_keys=ON, a missing parent row fails both INSERTs on a
    # fresh oversized first turn. The user-turn crash persist itself runs
    # LATER (after memory prefetch / pre_llm_call), so the row is written
    # once with its final api_content — both steps take the same per-agent
    # persist lock as CLI close persistence.
    persist_lock = getattr(agent, "_session_persist_lock", None)
    try:
        if persist_lock is None:
            agent._ensure_db_session()
        else:
            with persist_lock:
                agent._ensure_db_session()
    except Exception:
        logger.warning(
            "Turn-start session row creation failed for session=%s",
            agent.session_id or "none",
            exc_info=True,
        )
    finally:
        # Clear the staged CLI input eagerly (as the pre-refactor code did)
        # so a crash in preflight compression — which runs between this row
        # create and the late crash-persist below — doesn't leave a stale
        # _pending_cli_user_message that the next turn would mistake for a
        # fresh staged input.
        if not isinstance(pending_cli_message, dict) or pending_cli_message.get("_db_persisted"):
            agent._pending_cli_user_message = None

    # ── Idle-triggered compaction (opt-in; ``idle_compact_after_seconds``) ──
    # When a session resumes after a long idle gap, compact the accumulated
    # history up front so the rest of the conversation does not keep re-reading
    # a large stale context on every turn. This fires on elapsed wall-clock time
    # rather than size, so it complements (does not replace) the token-threshold
    # preflight below. ``_last_activity_ts`` is the last time this turn loop did
    # work; nothing has touched it yet this turn, so it measures the gap since
    # the previous turn finished. The cheap gap pre-check gates the (more
    # expensive) token estimate, mirroring ``_should_run_preflight_estimate``.
    _idle_after = getattr(agent, "compression_idle_compact_after_seconds", 0)
    if agent.compression_enabled and _idle_after > 0 and messages:
        _idle_gap = time.time() - getattr(agent, "_last_activity_ts", time.time())
        if _idle_gap >= _idle_after:
            _compressor = agent.context_compressor
            _idle_tokens = estimate_request_tokens_rough(
                messages,
                system_prompt=active_system_prompt or "",
                tools=agent.tools or None,
            )
            # Post-compression target size: don't summarise a thread already
            # below what compaction would reduce it to.
            _idle_floor = int(
                _compressor.threshold_tokens * _compressor.summary_target_ratio
            )
            _idle_cooldown = getattr(
                _compressor, "get_active_compression_failure_cooldown", lambda: None
            )()
            if _should_idle_compact(
                enabled=agent.compression_enabled,
                idle_after_seconds=_idle_after,
                idle_gap_seconds=_idle_gap,
                tokens=_idle_tokens,
                floor_tokens=_idle_floor,
                cooldown_active=bool(_idle_cooldown),
            ):
                logger.info(
                    "Idle compaction: %ss idle >= %ss, ~%s tokens > %s floor "
                    "(session %s)",
                    int(_idle_gap),
                    _idle_after,
                    f"{_idle_tokens:,}",
                    f"{_idle_floor:,}",
                    agent.session_id or "none",
                )
                agent._emit_status(
                    IDLE_COMPACTION_STATUS_TEMPLATE.format(
                        idle_seconds=int(_idle_gap), tokens=_idle_tokens
                    )
                )
                _idle_input = messages
                messages, active_system_prompt = agent._compress_context(
                    messages, system_message, approx_tokens=_idle_tokens,
                    task_id=effective_task_id,
                )
                # ``_compress_context`` returns the INPUT list object when it
                # skips (per-session lock held by another path, failure
                # cooldown, anti-thrash breaker, codex-native routing). Only
                # re-baseline + re-anchor after a real compaction — a skip
                # must leave the turn's flush baseline and user-message index
                # untouched.
                if messages is not _idle_input:
                    conversation_history = conversation_history_after_compression(
                        agent, messages
                    )
                    # Compaction rebuilt the list, so the index of this turn's
                    # just-appended user message is stale — re-anchor it the
                    # same way the preflight path does below.
                    current_turn_user_idx = reanchor_current_turn_user_idx(
                        messages, user_message
                    )
                    agent._persist_user_message_idx = current_turn_user_idx

    # ── Preflight context compression ──
    # Gate the (expensive) full token estimate behind a cheap pre-check.
    # See ``_should_run_preflight_estimate`` for the OR semantics that fix
    # issue #27405 (a few very large messages slipping past the count gate).
    _preflight_compressed = False
    _preflight_compression_blocked = False
    if agent.compression_enabled and _should_run_preflight_estimate(
        messages,
        agent.context_compressor.protect_first_n,
        agent.context_compressor.protect_last_n,
        agent.context_compressor.threshold_tokens,
    ):
        _preflight_tokens = estimate_request_tokens_rough(
            messages,
            system_prompt=active_system_prompt or "",
            tools=agent.tools or None,
        )
        _compressor = agent.context_compressor
        _defer_preflight = getattr(
            _compressor,
            "should_defer_preflight_to_real_usage",
            lambda _tokens: False,
        )
        _preflight_deferred = _defer_preflight(_preflight_tokens)
        # Codex app-server threads are compacted by the codex agent itself;
        # Hermes only initiates compaction in "hermes" mode (#36801).
        _codex_native_auto = (
            getattr(agent, "api_mode", None) == "codex_app_server"
            and str(
                getattr(
                    agent,
                    "codex_app_server_auto_compaction",
                    "native",
                )
                or "native"
            ).lower()
            in {"native", "off"}
        )

        if not _preflight_deferred:
            _last = _compressor.last_prompt_tokens
            # Do NOT overwrite the -1 sentinel (#36718).
            if _last >= 0 and _preflight_tokens > _last:
                _compressor.last_prompt_tokens = _preflight_tokens

        _compression_cooldown = getattr(
            _compressor,
            "get_active_compression_failure_cooldown",
            lambda: None,
        )()

        if _preflight_deferred:
            logger.info(
                "Skipping preflight compression: rough estimate ~%s >= %s, "
                "but last real provider prompt was %s after compression",
                f"{_preflight_tokens:,}",
                f"{_compressor.threshold_tokens:,}",
                f"{_compressor.last_real_prompt_tokens:,}",
            )
        elif _compression_cooldown:
            logger.info(
                "Skipping preflight compression: same-session cooldown active "
                "(~%s seconds remaining, session %s)",
                int(_compression_cooldown.get("remaining_seconds", 0.0)),
                agent.session_id or "none",
            )
        elif _codex_native_auto:
            logger.info(
                "Skipping Hermes preflight compression for codex app-server "
                "(mode=%s); Hermes will not start thread compaction here.",
                getattr(agent, "codex_app_server_auto_compaction", "native"),
            )
        elif _compressor.should_compress(_preflight_tokens):
            _preflight_compressed = True
            logger.info(
                "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                f"{_preflight_tokens:,}",
                f"{_compressor.threshold_tokens:,}",
                agent.model,
                f"{_compressor.context_length:,}",
            )
            agent._emit_status(
                PREFLIGHT_COMPRESSION_STATUS_TEMPLATE.format(
                    tokens=_preflight_tokens,
                    threshold=_compressor.threshold_tokens,
                )
            )
            # Preflight passes honor the same configured per-turn cap
            # (compression.max_attempts) as the loop's compression sites;
            # default 3 preserves the prior hardcoded behavior.
            _max_preflight_passes = max(
                1, int(getattr(agent, "max_compression_attempts", 3) or 3)
            )
            for _pass in range(_max_preflight_passes):
                _orig_len = len(messages)
                _orig_tokens = _preflight_tokens
                messages, active_system_prompt = agent._compress_context(
                    messages, system_message, approx_tokens=_preflight_tokens,
                    task_id=effective_task_id,
                )
                # Re-estimate now so size-only compression (same row count,
                # lower token count — e.g. summarising tool outputs) is
                # recognised as progress instead of being misread as
                # "Cannot compress further". Fixes #39548.
                _preflight_tokens = estimate_request_tokens_rough(
                    messages,
                    system_prompt=active_system_prompt or "",
                    tools=agent.tools or None,
                )
                if not _compression_made_progress(
                    _orig_len, len(messages), _orig_tokens, _preflight_tokens
                ):
                    _preflight_compression_blocked = True
                    break  # Cannot compress further: neither rows nor tokens moved
                conversation_history = conversation_history_after_compression(
                    agent, messages
                )
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                agent._last_content_with_tools = None
                agent._last_content_tools_all_housekeeping = False
                agent._mute_post_response = False
                if not _compressor.should_compress(_preflight_tokens):
                    break
                if not _compression_warrants_another_preflight_pass(
                    _orig_tokens,
                    _preflight_tokens,
                    _compressor.threshold_tokens,
                ):
                    _preflight_compression_blocked = True
                    logger.warning(
                        "Preflight compression made insufficient progress: "
                        "~%s -> ~%s request tokens; skipping additional passes",
                        f"{_orig_tokens:,}",
                        f"{_preflight_tokens:,}",
                    )
                    break

    if _preflight_compressed:
        # Compression rebuilt the list (tail messages are fresh compaction
        # copies), so the pre-compression index of this turn's user message
        # is stale. Re-anchor both index trackers: the api_content stamp
        # below, the loop's injection site, and the flush's persist-override
        # row (#48677) must all target the surviving dict, not a stale
        # position. Exact-content match first so a todo-snapshot user message
        # appended after the tail can't steal the anchor.
        current_turn_user_idx = reanchor_current_turn_user_idx(
            messages, user_message
        )
        agent._persist_user_message_idx = current_turn_user_idx

    # Plugin hook: pre_llm_call (context injected into user message, not system prompt).
    plugin_user_context = ""
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _pre_results = _invoke_hook(
            "pre_llm_call",
            session_id=agent.session_id,
            task_id=effective_task_id,
            turn_id=turn_id,
            user_message=original_user_message,
            conversation_history=list(messages),
            is_first_turn=(not bool(conversation_history)),
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
            sender_id=getattr(agent, "_user_id", None) or "",
        )
        _ctx_parts: list[str] = []
        # Spill oversized per-hook context to disk so a runaway plugin
        # can't inflate every subsequent turn's prompt. Ported from
        # openai/codex PR #21069 ("Spill large hook outputs from context").
        try:
            from tools.hook_output_spill import (
                get_spill_config as _spill_cfg,
                spill_if_oversized as _spill_if_oversized,
            )
            _spill_config_cached = _spill_cfg()
        except Exception:
            _spill_if_oversized = None  # type: ignore[assignment]
            _spill_config_cached = None
        for r in _pre_results:
            _piece: str = ""
            if isinstance(r, dict) and r.get("context"):
                _piece = str(r["context"])
            elif isinstance(r, str) and r.strip():
                _piece = r
            else:
                continue
            if _spill_if_oversized is not None:
                try:
                    _piece = _spill_if_oversized(
                        _piece,
                        session_id=agent.session_id,
                        source="plugin hook",
                        config=_spill_config_cached,
                    )
                except Exception as _spill_exc:
                    logger.warning("hook context spill failed: %s", _spill_exc)
            _ctx_parts.append(_piece)
        if _ctx_parts:
            plugin_user_context = "\n\n".join(_ctx_parts)
    except Exception as exc:
        logger.warning("pre_llm_call hook failed: %s", exc)

    # Gateway must-deliver notes (auto-reset note, first-contact intro,
    # voice-channel change) ride the same user-message injection channel as
    # plugin context so the ephemeral system prompt can stay byte-stable.
    # One-shot: staged by the gateway right before this turn, consumed here.
    # Multimodal (list) content can't take the string sidecar — append a
    # durable text part instead of dropping the fact.
    _gateway_notes = consume_gateway_turn_context_notes(agent)
    if _gateway_notes:
        _gw_turn_content = (
            messages[current_turn_user_idx].get("content")
            if 0 <= current_turn_user_idx < len(messages)
            and isinstance(messages[current_turn_user_idx], dict)
            else None
        )
        if isinstance(_gw_turn_content, list):
            append_notes_to_multimodal_content(_gw_turn_content, _gateway_notes)
        else:
            plugin_user_context = (
                plugin_user_context + "\n\n" + _gateway_notes
                if plugin_user_context
                else _gateway_notes
            )

    # Per-turn file-mutation verifier state.
    agent._turn_failed_file_mutations = {}
    agent._turn_file_mutation_paths = set()
    agent._verification_stop_nudges = 0
    agent._pre_verify_nudges = 0

    # Record the execution thread so interrupt()/clear_interrupt() can scope
    # the tool-level interrupt signal to THIS agent's thread only.
    agent._execution_thread_id = threading.current_thread().ident

    # Clear stale per-thread interrupt state, preserving a pending interrupt.
    ra()._set_interrupt(False, agent._execution_thread_id)
    if agent._interrupt_requested:
        ra()._set_interrupt(True, agent._execution_thread_id)
        agent._interrupt_thread_signal_pending = False
    else:
        agent._interrupt_message = None
        agent._interrupt_thread_signal_pending = False

    # Notify memory providers of the new turn (BEFORE prefetch_all).
    if agent._memory_manager:
        try:
            _turn_msg = original_user_message if isinstance(original_user_message, str) else ""
            agent._memory_manager.on_turn_start(agent._user_turn_count, _turn_msg)
        except Exception:
            pass

    # External memory provider: prefetch once before the tool loop.
    ext_prefetch_cache = ""
    if agent._memory_manager:
        try:
            _query = original_user_message if isinstance(original_user_message, str) else ""
            ext_prefetch_cache = agent._memory_manager.prefetch_all(_query) or ""
        except Exception:
            pass

    # ── api_content sidecar: persist what you send ──
    # The prefetch/plugin context above is injected into the API copy of this
    # turn's user message, never into the stored content — so on the next
    # turn the message would replay WITHOUT the injection, diverging the
    # request prefix at this point and re-prefilling everything after it
    # (the whole previous turn's assistant/tool chain). Stamp the exact
    # API-bound bytes on the live dict, only when they differ from the clean
    # content, so the crash persist below writes both in the same row and
    # replay can reproduce the sent prefix byte-for-byte. Guarded by the
    # same predicate the api_messages build uses, so the stamped bytes are
    # exactly the bytes the loop sends. codex_app_server turns bypass the
    # api_messages build entirely (the codex thread gets the plain user
    # message), so stamping there would persist bytes that were never sent.
    # MoA turns append per-call aggregated reference context to the same API
    # copy AFTER this composition, so the stamped bytes would never match the
    # wire either — skip the stamp rather than persist provably wrong "exact
    # sent bytes" (MoA keeps its pre-sidecar cache behavior).
    if (
        not moa_active
        and getattr(agent, "api_mode", None) != "codex_app_server"
        and 0 <= current_turn_user_idx < len(messages)
        and messages[current_turn_user_idx].get("role") == "user"
    ):
        _turn_user_msg = messages[current_turn_user_idx]
        _api_content = compose_user_api_content(
            _turn_user_msg.get("content", ""), ext_prefetch_cache, plugin_user_context
        )
        if _api_content is not None and _api_content != _turn_user_msg.get("content"):
            _turn_user_msg["api_content"] = _api_content
            # In-place preflight compaction has ALREADY inserted this turn's
            # user row (archive_and_compact runs before prefetch/pre_llm_call
            # can compose the sidecar), and the crash persist below identity-
            # skips every compacted dict (they are all in the rebound
            # conversation_history) — so the stamp would never reach the DB.
            # Backfill it onto the freshly-inserted row directly. Rotation
            # mode needs nothing here: its compacted copies flush to the
            # child session after this stamp.
            if _preflight_compressed and bool(
                getattr(agent, "_last_compaction_in_place", False)
            ):
                _db = getattr(agent, "_session_db", None)
                if _db is not None:
                    try:
                        _db.set_latest_user_api_content(
                            agent.session_id,
                            _turn_user_msg.get("content"),
                            _api_content,
                        )
                    except Exception:
                        logger.warning(
                            "in-place compaction api_content backfill failed "
                            "for session=%s",
                            agent.session_id or "none",
                            exc_info=True,
                        )

    # Crash-resilience: persist the inbound user turn before the first LLM
    # call. Runs after preflight compression (which rewrites history anyway)
    # and after prefetch/pre_llm_call, so the user row is written once with
    # its final api_content instead of being re-written mid-turn.
    # Keep row creation and the marker-based append in the same per-agent
    # critical section as CLI close persistence, and retry the row create if
    # the pre-compression attempt above failed transiently.
    def _ensure_and_persist() -> None:
        agent._ensure_db_session()
        agent._persist_session(messages, conversation_history)

    try:
        if persist_lock is None:
            _ensure_and_persist()
        else:
            with persist_lock:
                _ensure_and_persist()
    except Exception:
        logger.warning(
            "Early turn-start session persistence failed for session=%s",
            agent.session_id or "none",
            exc_info=True,
        )
    finally:
        # Keep an unmarked staged input available to a later close retry if the
        # normal persistence attempt failed. Once the marker is present, the
        # close path must no longer treat it as a pre-worker UI input.
        if not isinstance(pending_cli_message, dict) or pending_cli_message.get("_db_persisted"):
            agent._pending_cli_user_message = None

    return TurnContext(
        user_message=user_message,
        original_user_message=original_user_message,
        messages=messages,
        conversation_history=conversation_history,
        active_system_prompt=active_system_prompt,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        current_turn_user_idx=current_turn_user_idx,
        should_review_memory=should_review_memory,
        plugin_user_context=plugin_user_context,
        ext_prefetch_cache=ext_prefetch_cache,
        preflight_compression_blocked=_preflight_compression_blocked,
    )
