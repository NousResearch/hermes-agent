"""Per-turn setup for ``run_conversation`` (the turn prologue).

``build_turn_context`` runs the once-per-turn setup (stdio guard, sanitization, prompt
restore-or-build, session row, idle/preflight compaction via ``turn_context_compaction``,
pre_llm_call hook, prefetch, persistence), mutating ``agent`` as the loop expects, and
returns a ``TurnContext`` with only the locals the loop reads back.
``build_api_messages`` builds the wire copy for one API call."""

from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from agent.conversation_compression import recover_rotated_compression_session
from agent.iteration_budget import IterationBudget
from agent.memory_manager import build_memory_context_block
from agent.memory_provider import is_trivial_prompt
from agent.message_metadata import append_message, stamp_message_timestamp
from agent.model_metadata import (
    anchored_context_tokens, estimate_messages_tokens_rough, estimate_request_tokens_rough
)

logger = logging.getLogger(__name__)


def _str_attr(agent: Any, name: str) -> str:
    """``getattr(agent, name, "") or ""`` — route facts read off partial agents/doubles."""
    return getattr(agent, name, "") or ""


def _preflight_request_tokens(
    agent: Any, messages: List[Dict[str, Any]], system_prompt: str
) -> int:
    """Token estimate for automatic preflight compression: a valid provider usage anchor,
    else the checkpoint-pruned native wire payload, else the generic estimator."""
    anchored = anchored_context_tokens(messages, getattr(agent, "_usage_anchor", None))
    if anchored is not None:
        return anchored
    tools = getattr(agent, "tools", None) or None
    try:
        from agent.codex_responses_adapter import estimate_native_responses_preflight_tokens

        native = estimate_native_responses_preflight_tokens(
            agent, messages, system_prompt=system_prompt or "", tools=tools
        )
        if isinstance(native, int) and not isinstance(native, bool) and native >= 0:
            return native
    except Exception:
        logger.debug(
            "native Responses preflight estimate unavailable; "
            "using generic transcript estimate",
            exc_info=True,
        )
    return estimate_request_tokens_rough(
        messages, system_prompt=system_prompt or "", tools=tools,
        charge_stale_thinking=_agent_stale_thinking_on_wire(agent),
    )


def _agent_stale_thinking_on_wire(agent: Any) -> bool:
    """Whether the active route replays stale thinking text; ``True`` (conservative full
    charge) when route facts are unavailable."""
    try:
        from agent.message_sanitization import stale_thinking_reaches_wire

        return stale_thinking_reaches_wire(
            *(_str_attr(agent, k) for k in ("api_mode", "provider", "model", "base_url"))
        )
    except Exception:
        return True


def compose_user_api_content(
    content: Any, ext_prefetch_cache: str, plugin_user_context: str
) -> Optional[str]:
    """Compose the API-bound content of the current turn's user message.

    Single source for the ``api_content`` sidecar and the wire bytes so they never drift
    (what turn N sends is what turn N+1 replays). ``None`` when nothing is injected."""
    if not isinstance(content, str):
        return None
    fenced = build_memory_context_block(ext_prefetch_cache) if ext_prefetch_cache else ""
    injections = [part for part in (fenced, plugin_user_context) if part]
    if not injections:
        return None
    return content + "\n\n" + "\n\n".join(injections)


def substitute_api_content(api_msg: Dict[str, Any]) -> Optional[str]:
    """Pop the ``api_content`` sidecar and substitute it into ``content`` (keeps the
    prompt-cache prefix byte-stable). Returns the popped sidecar, or ``None``."""
    sidecar = api_msg.pop("api_content", None)
    if isinstance(sidecar, str) and sidecar and api_msg.get("role") in ("user", "assistant"):
        api_msg["content"] = sidecar
    return sidecar


def drop_stale_api_content(msg: Dict[str, Any]) -> None:
    """Drop the ``api_content`` sidecar from a message whose content was rewritten
    (replaying it would resend what the rewrite removed; cost is one cache miss)."""
    msg.pop("api_content", None)


def extract_api_content_sidecar(msg: Mapping[str, Any]) -> Optional[str]:
    """Extract the ``api_content`` sidecar; ``None`` when absent/non-string."""
    v = msg.get("api_content")
    return v if isinstance(v, str) else None


def consume_gateway_turn_context_notes(agent: Any) -> str:
    """Pop the gateway's per-turn must-deliver notes off the agent (one-shot, so the
    system prompt stays byte-stable and a cached agent never replays a stale note)."""
    notes = getattr(agent, "_gateway_turn_context_notes", "") or ""
    if hasattr(agent, "_gateway_turn_context_notes"):
        with suppress(Exception):
            agent._gateway_turn_context_notes = ""
    return notes if isinstance(notes, str) else ""


def append_notes_to_multimodal_content(content: Any, notes: str) -> bool:
    """Append must-deliver notes as a durable text part on a multimodal (list) user
    message (the sidecar path returns ``None`` for non-string content)."""
    if not notes or not isinstance(content, list):
        return False
    with suppress(Exception):
        content.append({"type": "text", "text": notes})
        return True
    return False


# Surfaces whose sessions must not be auto-titled: cron names its own session and
# its opener is a delivery hint; subagent sessions are hidden from every picker.
_UNTITLED_PLATFORMS = frozenset({"cron", "subagent"})


def _maybe_title_session_at_turn_start(agent: Any, messages: List[Any]) -> None:
    """Kick off auto-titling for the session's first user message; never fatal."""
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if not session_db or not session_id:
        return
    if str(getattr(agent, "platform", "") or "").lower() in _UNTITLED_PLATFORMS:
        return
    try:
        from agent.message_content import flatten_message_text
        from agent.title_generator import maybe_auto_title

        # Turn's user message as text; image-only turns yield "" and are skipped.
        user_text = ""
        for msg in reversed(messages or []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_text = flatten_message_text(msg.get("content")).strip()
                break
        if not user_text:
            return
        # The session row is created lazily; force it now or the title write matches
        # zero rows.
        if not getattr(agent, "_session_db_created", False):
            ensure = getattr(agent, "_ensure_db_session", None)
            if callable(ensure):
                ensure()
            if not getattr(agent, "_session_db_created", False):
                return
        # Snapshot runtime identity so the background titler can skip if the user
        # switches models before it fires.
        main_runtime = {
            k: getattr(agent, k, None) for k in ("model", "provider", "base_url", "api_key", "api_mode")
        }
        # See #19027.
        maybe_auto_title(
            session_db,
            session_id,
            user_text,
            conversation_history=messages,
            failure_callback=(
                getattr(agent, "_title_failure_callback", None)
                or getattr(agent, "_emit_auxiliary_failure", None)
            ),
            main_runtime=main_runtime,
            title_callback=getattr(agent, "_on_session_title", None),
            runtime_validator=lambda: (
                getattr(agent, "model", None) == main_runtime["model"]
                and getattr(agent, "provider", None) == main_runtime["provider"]
            ),
        )
    except Exception:
        logger.debug("Turn-start auto-title dispatch failed", exc_info=True)


def reanchor_current_turn_user_idx(messages: List[Any], user_message: Any) -> int:
    """Locate this turn's user message after compaction rebuilt ``messages``.

    Prefers the LAST user message whose content exactly matches this turn's text, else
    the last user-originated turn; compaction handoffs are never the fallback.
    Returns -1 when there is no user-originated message.

    Compression replaces list entries with fresh copies (and may append a todo-snapshot user message or a
    restored user turn AFTER the surviving copy of the current turn's message), so a pre-compression index
    is meaningless. Prefer the LAST user message whose content exactly matches this turn's text — the
    surviving copy in the common case — so the injection stamp and the #48677 persist override can't land on
    a todo-snapshot or historical row. Fall back to the last *user-originated* turn when no exact match
    survives (merge-summary-into-tail rewrites the content but the trackers still need a live anchor).
    Compaction handoffs must never become the fallback anchor (#80622) — they are reference-only
    scaffolding, not the active ask.
    """
    from agent.context_compressor import user_originated_turn_view

    fallback = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not (isinstance(msg, dict) and msg.get("role") == "user"):
            continue
        # Typed synthetic current events keep their persistence anchor when raw
        # content is unchanged; not eligible for the human-only fallback below.
        if msg.get("content") == user_message:
            return i
        live_view = user_originated_turn_view(msg)
        if live_view is None:
            continue
        if live_view.get("content") == user_message:
            return i
        # Prefer a real human turn over a synthetic handoff / continuation marker
        # when the exact content was rewritten by merge-into-tail.
        if fallback < 0:
            fallback = i
    return fallback


def compression_made_progress(
    orig_len: int, new_len: int, orig_tokens: int, new_tokens: int
) -> bool:
    """``True`` if a compression pass materially reduced the request: fewer rows, or a
    >5% token cut with the same rows (same floor as the overflow-handler retry).

    Compression can succeed by summarising message contents — reducing the estimated request token count —
    without reducing the message row count. Treating row count as the sole progress signal false-positives
    on size-only wins and surfaces a misleading "Cannot compress further" failure even when post-compression
    tokens are well below the model context window. See issue #39548 for an observed case: 220 → 220
    messages, ~288k → ~183k tokens on a 1M-context model still triggered auto-reset.
    The token reduction must be *material* (>5%) to count as progress — the same floor the overflow-handler
    retry path uses (conversation_loop.py, 39550) — so a sub-5% wobble doesn't keep the multi-pass loop
    spinning. See #39550.
    """
    return new_len < orig_len or (orig_tokens > 0 and new_tokens < orig_tokens * 0.95)


class PreflightCompressionTimedOut(RuntimeError):
    """Raised when an oversized turn cannot safely finish preflight."""


def _fail_closed_after_preflight_timeout(agent, request_tokens: int) -> None:
    """Stop an oversized turn instead of sending its unchanged provider payload."""
    from agent.conversation_compression import context_compression_timed_out

    if not context_compression_timed_out(agent):
        return
    raise PreflightCompressionTimedOut(
        "Context compression timed out before it could commit while the request "
        f"was still approximately {request_tokens:,} tokens. The provider call "
        "was not sent. Run /compress and wait for it to finish, then retry."
    )


def _review_fork_first_request_pending(agent: Any) -> bool:
    """Whether a detached review fork has yet to send its first provider request: it
    replays the parent's FULL snapshot as a warm cache read, so compaction must wait
    for that first response. Dormant without the attribute."""
    return bool(
        getattr(agent, "_review_defer_compaction_before_first_response", False)
        and not getattr(agent, "_turn_received_provider_response", False)
    )


def _compression_warrants_another_preflight_pass(
    orig_tokens: int, new_tokens: int, threshold_tokens: int
) -> bool:
    """Another immediate summary only if still over threshold AND the previous pass cut
    tokens by >5%."""
    return new_tokens >= threshold_tokens and orig_tokens > 0 and new_tokens < orig_tokens * 0.95


def _should_run_preflight_estimate(
    messages: List[Dict[str, Any]], protect_first_n: int, protect_last_n: int, threshold_tokens: int
) -> bool:
    """Cheap gate for the (expensive) full preflight estimate: message count exceeds the
    protected ranges OR a rough char-based estimate crosses the threshold (few-but-huge
    case). The estimator undercounts by design (omits system/tools) so one large base64
    image is not mistaken for ~250K tokens."""
    return (
        len(messages) > protect_first_n + protect_last_n + 1
        or estimate_messages_tokens_rough(messages) >= threshold_tokens
    )


def _should_idle_compact(
    *, enabled: bool, idle_after_seconds: int, idle_gap_seconds: float, tokens: int,
    floor_tokens: int, cooldown_active: bool, last_compaction_tokens: int = 0,
) -> bool:
    """Pure predicate: idle compaction fires after a wall-clock gap of
    ``idle_after_seconds`` (opt-in, <= 0 disables), independent of ``threshold_tokens``;
    never at/below ``floor_tokens`` or during a compression-failure cooldown.

    ``floor_tokens`` (``threshold_tokens × summary_target_ratio``) is a theoretical target a
    real pass routinely misses (system prompt, tool schemas and protected head/tail are
    incompressible), so a session compacted to above it would re-summarise on every idle
    resume without growing. ``last_compaction_tokens`` — what the previous pass actually
    produced (``ContextCompressor.last_compression_rough_tokens``, same rough shape as
    ``tokens``) — raises the floor to ``last + floor_tokens`` so the transcript must gain a
    floor's worth of NEW content first. ``0`` (nothing compacted yet / counter reset) keeps
    the original semantics exactly.

    A session that compacted to well above that target therefore stays above it forever, so every later idle
    resume re-runs a full summarisation over a transcript that has not grown — minutes of silently blocked
    prompt on a slow route, reclaiming nothing (#97239).
    """
    if not enabled or idle_after_seconds <= 0 or idle_gap_seconds < idle_after_seconds or cooldown_active:
        return False
    effective_floor = floor_tokens
    if last_compaction_tokens > 0:
        effective_floor = max(effective_floor, last_compaction_tokens + floor_tokens)
    return tokens > effective_floor


@dataclass
class TurnContext:
    """Values produced by the turn prologue and consumed by the turn loop."""

    user_message: str  # sanitized inbound message (surrogates stripped)
    original_user_message: Any  # clean text for transcripts / memory queries (no nudges)
    messages: List[Dict[str, Any]]  # working list for this turn (loop appends to it)
    conversation_history: Optional[List[Dict[str, Any]]]  # None after rotation
    active_system_prompt: Optional[str]  # may be rebuilt by compression
    effective_task_id: str
    turn_id: str
    current_turn_user_idx: int  # index of the current user turn within ``messages``
    should_review_memory: bool = False  # post-turn memory review should fire
    plugin_user_context: str = ""  # ``pre_llm_call`` context (appended to user message)
    ext_prefetch_cache: str = ""  # external-memory prefetch, reused across iterations
    preflight_compression_blocked: bool = False  # immediate retry proved ineffective


def _persist_under_lock(agent: Any, fn, failure_msg: str, pending_cli_message: Any) -> None:
    """Run ``fn`` under the session persist lock (when the agent has one), log-and-swallow
    failures, then drop staged CLI input — unless it is an unmarked handoff kept for a
    close retry (once ``_db_persisted`` the close path must not treat it as pre-worker
    UI input). Eager clearing keeps a preflight crash from leaking stale input."""
    try:
        lock = getattr(agent, "_session_persist_lock", None)
        if lock is None:
            fn()
        else:
            with lock:
                fn()
    except Exception:
        logger.warning(failure_msg, agent.session_id or "none", exc_info=True)
    finally:
        if not isinstance(pending_cli_message, dict) or pending_cli_message.get("_db_persisted"):
            agent._pending_cli_user_message = None


def _publish_runtime_main(agent: Any) -> None:
    """Tell auxiliary_client the live main provider/model for this turn (after primary
    restoration settled the runtime). Never raises: failure loses only the scope."""
    with suppress(Exception):
        from agent.auxiliary_client import set_runtime_main
        from agent.prompt_cache_scope import resolve_prompt_cache_scope_safe
        # Rotation-stable prompt-cache scope (lineage root), memoized per segment; a new
        # session uses the physical id until build_api_kwargs re-resolves.
        # Memoized per segment on the agent, so this is a DB walk at most once per segment — except a
        # brand-new session whose row lands later in turn setup (_ensure_db_session); that first turn falls
        # back to the physical id here and the first build_api_kwargs re-resolves. Stays valid through a
        # mid-turn compression rotation because the lineage root is by definition rotation-invariant
        # (#79017). Resolved with the never-raising variant OUTSIDE the argument list, so a resolution
        # failure can only lose the scope — never the whole runtime binding.
        _cache_scope = resolve_prompt_cache_scope_safe(agent) or ""
        set_runtime_main(
            _str_attr(agent, "provider"), _str_attr(agent, "model"),
            **{k: _str_attr(agent, k) for k in (
                "requested_provider", "base_url", "api_key", "api_mode", "auth_mode", "session_id"
            )},
            cache_scope=_cache_scope,
        )


def _refresh_mcp_tools_between_turns(agent: Any) -> None:
    """Late-connecting MCP servers land in THIS turn's snapshot, before the first API
    call assembles ``tools=``. ``preserve_prefix`` keeps the tool array append-only so a
    flapping ``check_fn`` can't fork the cache."""
    try:
        # Import-cost gate: MCP tools are only registered by code that already imported
        # ``tools.mcp_tool`` (~0.4s); not in sys.modules => nothing to do.
        if not getattr(agent, "_skip_mcp_refresh", False) and "tools.mcp_tool" in sys.modules:
            from tools.mcp_tool_discovery import has_registered_mcp_tools
            from tools.mcp_tool_agent import refresh_agent_mcp_tools
            if has_registered_mcp_tools():
                refresh_agent_mcp_tools(agent, quiet_mode=True, preserve_prefix=True)
    except Exception:
        logger.debug("between-turns MCP tool refresh skipped", exc_info=True)


def _bind_turn_identity(
    agent: Any, task_id: Optional[str], stream_callback, persist_user_message: Any,
    persist_user_timestamp: Optional[float], persist_user_platform_id: Optional[str],
) -> Tuple[str, str]:
    """Stage callback/persist overrides on the agent and bind this turn's task and turn
    ids. Returns ``(effective_task_id, turn_id)``."""
    agent._stream_callback = stream_callback  # picked up by _interruptible_api_call
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = persist_user_message
    agent._persist_user_message_timestamp = persist_user_timestamp
    agent._persist_user_message_platform_id = persist_user_platform_id
    # Unique task_id when not provided isolates VMs between tasks.
    effective_task_id = task_id or str(uuid.uuid4())
    agent._current_task_id = effective_task_id
    turn_id = str(getattr(agent, "_relay_pending_turn_id", "") or "") or (
        f"{agent.session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex[:8]}"
    )
    agent._relay_pending_turn_id = None
    agent._current_turn_id = turn_id
    agent._current_api_request_id = ""
    # Tripwire: warn when this turn starts before the previous turn-end persist
    # (concurrent turns interleave transcript writes). Cleared in _persist_session.
    from agent.agent_runtime_helpers import note_turn_start
    note_turn_start(agent, turn_id)
    return effective_task_id, turn_id


# Per-turn agent state reset at turn start (retry counters, guardrail halt, file-mutation
# verifier). ``_turns_since_memory`` / ``_iters_since_skill`` are deliberately NOT reset.
_PER_TURN_RESET_STATE: Tuple[Tuple[str, Any], ...] = (
    ("_invalid_tool_retries", 0), ("_invalid_json_retries", 0), ("_empty_content_retries", 0),
    ("_incomplete_scratchpad_retries", 0), ("_codex_incomplete_retries", 0),
    ("_thinking_prefill_retries", 0), ("_post_tool_empty_retried", False),
    ("_last_content_with_tools", None), ("_last_content_tools_all_housekeeping", False),
    ("_mute_post_response", False), ("_unicode_sanitization_passes", 0),
    ("_tool_guardrail_halt_decision", None), ("_vision_supported", True),
    ("_run_budget_wrapup_injected", False), ("_verification_stop_nudges", 0),
    ("_pre_verify_nudges", 0),
)


def _reset_per_turn_agent_state(agent: Any) -> None:
    """Reset retry counters, guardrails, iteration and run budgets at turn start."""
    for name, value in _PER_TURN_RESET_STATE:
        setattr(agent, name, value)
    agent._turn_failed_file_mutations = {}
    agent._turn_file_mutation_paths = set()
    agent._tool_guardrails.reset_for_turn()
    _reset_consol = getattr(agent._memory_store, "reset_consolidation_failures", None)
    if callable(_reset_consol):
        _reset_consol()

    # Pre-turn connection health check: clean up dead TCP connections.
    if agent.api_mode != "anthropic_messages":
        with suppress(Exception):
            if agent._cleanup_dead_connections():
                agent._emit_status(
                    "🔌 Detected stale connections from a previous provider "
                    "issue — cleaned up automatically. Proceeding with fresh "
                    "connection."
                )
    # Replay compression warning through status_callback for gateway platforms.
    if agent._compression_warning:
        agent._replay_compression_warning()
        agent._compression_warning = None  # send once

    agent.iteration_budget = IterationBudget(agent.max_iterations)
    # Wall-clock run budget: stamped only when configured (one wrap-up notice per run).
    agent._run_budget_started_at = (
        time.time() if getattr(agent, "run_budget_seconds", None) else None
    )
    # Reset the streaming context / think scrubbers at the top of each turn.
    for name in ("_stream_context_scrubber", "_stream_think_scrubber"):
        scrubber = getattr(agent, name, None)
        if scrubber is not None:
            scrubber.reset()


def _stage_turn_user_message(
    agent: Any, user_message: Any, persist_user_message: Any,
    persist_user_timestamp: Optional[float], persist_user_platform_id: Optional[str],
    persist_user_display_kind: Optional[str],
    persist_user_display_metadata: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Any]:
    """Build this turn's user dict, reusing CLI-staged input only when its clean text
    matches this turn (a stale handoff must not replace later input; voice turns
    compare the clean override). Returns ``(user_msg, pending_cli_message)``."""
    pending_cli_message = getattr(agent, "_pending_cli_user_message", None)
    expected_persist_content = (
        persist_user_message if persist_user_message is not None else user_message
    )
    if (
        isinstance(pending_cli_message, dict)
        and pending_cli_message.get("content") == expected_persist_content
    ):
        user_msg = pending_cli_message
        # CLI-staged value is the clean text; restore the API-facing variant (e.g. voice
        # prefix) on the same dict, keeping any close-path durable marker.
        user_msg["content"] = user_message
    else:
        user_msg = {"role": "user", "content": user_message}
        if isinstance(pending_cli_message, dict):
            agent._pending_cli_user_message = None
    # CLI input is stamped when staged; gateway input may carry the platform event
    # time. Preserve either value and cover any legacy unstamped handoff.
    stamp_message_timestamp(user_msg, timestamp=persist_user_timestamp)

    # Synthesized turns stamp their transcript type so the crash persist writes a typed
    # row; the model still receives role/content unchanged (api_messages strips both).
    if persist_user_display_kind:
        user_msg["display_kind"] = persist_user_display_kind
        if persist_user_display_metadata:
            user_msg["display_metadata"] = persist_user_display_metadata
    # The platform message id survives the turn-start flush; restart drain-window
    # recovery dedups via ``has_platform_message_id`` against this row.
    if persist_user_platform_id is not None:
        user_msg["platform_message_id"] = persist_user_platform_id
    return user_msg, pending_cli_message


def _hydrate_from_history(agent: Any, conversation_history: Optional[List[Any]]) -> None:
    """Hydrate the todo store and per-session nudge counters from persisted history."""
    if not conversation_history:
        return
    if not agent._todo_store.has_items():
        agent._hydrate_todo_store(conversation_history)
    # Hydrate per-session nudge counters from persisted history.
    if agent._user_turn_count == 0:
        prior_user_turns = sum(1 for m in conversation_history if m.get("role") == "user")
        if prior_user_turns > 0:
            agent._user_turn_count = prior_user_turns
            if agent._memory_nudge_interval > 0 and agent._turns_since_memory == 0:
                agent._turns_since_memory = prior_user_turns % agent._memory_nudge_interval


def _tick_memory_nudge(agent: Any) -> bool:
    """Advance the turn-based memory nudge counter; ``True`` when the review should fire."""
    if (agent._memory_nudge_interval > 0
            and "memory" in agent.valid_tool_names
            and agent._memory_store):
        agent._turns_since_memory += 1
        if agent._turns_since_memory >= agent._memory_nudge_interval:
            agent._turns_since_memory = 0
            return True
    return False


def _emit_reaction(agent: Any, original_user_message: Any) -> None:
    """Cosmetic side-signal: detect an affection reaction so the host can play hearts.
    Token-free, never touches the conversation, never fatal."""
    reaction_callback = getattr(agent, "reaction_callback", None)
    if reaction_callback is None:
        return
    with suppress(Exception):
        from agent.reactions import detect_reaction

        kind = detect_reaction(original_user_message)
        if kind:
            reaction_callback(kind)


def _ensure_session_row(agent: Any, pending_cli_message: Any) -> None:
    """Create the DB row now (system prompt populated => non-NULL) and BEFORE preflight
    compression: compaction/rotation INSERTs reference this row under PRAGMA
    foreign_keys=ON. Idempotent; the user-turn crash persist runs later."""
    _persist_under_lock(
        agent, agent._ensure_db_session,
        "Turn-start session row creation failed for session=%s", pending_cli_message,
    )


def _collect_pre_llm_call_context(
    agent: Any, *, effective_task_id: str, turn_id: str, original_user_message: Any,
    messages: List[Any], conversation_history: Optional[List[Any]],
) -> str:
    """Run ``pre_llm_call`` plugins; their context is injected into the user message
    (never the system prompt). Oversized per-hook context is spilled to disk so a
    runaway plugin can't inflate every subsequent turn's prompt."""
    try:
        from hermes_cli.lifecycle import invoke_hook as _invoke_hook
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
            parent_session_id=getattr(agent, "_parent_session_id", None) or "",
            sender_id=getattr(agent, "_user_id", None) or "",
        )
        try:
            # Spill oversized per-hook context to disk so a runaway plugin can't inflate every subsequent
            # turn's prompt. Ported from openai/codex PR #21069 ("Spill large hook outputs from context").
            from tools.hook_output_spill import (
                get_spill_config as _spill_cfg, spill_if_oversized as _spill_if_oversized
            )
            _spill_config_cached = _spill_cfg()
        except Exception:
            _spill_if_oversized = None  # type: ignore[assignment]
            _spill_config_cached = None
        _ctx_parts: list[str] = []
        for r in _pre_results:
            if isinstance(r, dict) and r.get("context"):
                _piece = str(r["context"])
            elif isinstance(r, str) and r.strip():
                _piece = r
            else:
                continue
            if _spill_if_oversized is not None:
                try:
                    _piece = _spill_if_oversized(
                        _piece, session_id=agent.session_id, source="plugin hook",
                        config=_spill_config_cached,
                    )
                except Exception as _spill_exc:
                    logger.warning("hook context spill failed: %s", _spill_exc)
            _ctx_parts.append(_piece)
        return "\n\n".join(_ctx_parts)
    except Exception as exc:
        logger.warning("pre_llm_call hook failed: %s", exc)
    return ""


def _merge_gateway_notes(
    agent: Any, messages: List[Any], current_turn_user_idx: int, plugin_user_context: str
) -> str:
    """Gateway must-deliver notes ride the user-message injection channel (one-shot,
    gateway-staged) so the ephemeral system prompt stays byte-stable. Multimodal (list)
    content can't take the string sidecar — append a durable text part instead."""
    _gateway_notes = consume_gateway_turn_context_notes(agent)
    if not _gateway_notes:
        return plugin_user_context
    _gw_turn_content = (
        messages[current_turn_user_idx].get("content")
        if 0 <= current_turn_user_idx < len(messages)
        and isinstance(messages[current_turn_user_idx], dict)
        else None
    )
    if isinstance(_gw_turn_content, list):
        append_notes_to_multimodal_content(_gw_turn_content, _gateway_notes)
        return plugin_user_context
    return (
        plugin_user_context + "\n\n" + _gateway_notes if plugin_user_context else _gateway_notes
    )


def _bind_interrupt_scope(agent: Any, ra) -> None:
    """Record the execution thread so interrupt()/clear_interrupt() scope the tool-level
    signal to THIS agent's thread; clear stale state, preserving a pending interrupt."""
    agent._execution_thread_id = threading.current_thread().ident
    ra()._set_interrupt(False, agent._execution_thread_id)
    if agent._interrupt_requested:
        ra()._set_interrupt(
            True, agent._execution_thread_id, reason=getattr(agent, "_tool_interrupt_reason", None)
        )
    else:
        agent._interrupt_message = None
        agent._tool_interrupt_reason = None
    agent._interrupt_thread_signal_pending = False


def _memory_turn_start_and_prefetch(agent: Any, original_user_message: Any) -> str:
    """Notify memory providers of the new turn, then prefetch external memory once
    before the tool loop (skipped on trivial prompts with no semantic signal).
    Returns the prefetch text (``""`` when nothing was injected)."""
    if not agent._memory_manager:
        return ""
    _query = original_user_message if isinstance(original_user_message, str) else ""
    with suppress(Exception):
        agent._memory_manager.on_turn_start(agent._user_turn_count, _query)
    ext_prefetch_cache = ""
    with suppress(Exception):
        if not is_trivial_prompt(_query):
            ext_prefetch_cache = agent._memory_manager.prefetch_all(_query) or ""
    # Deterministic recall indicator via _emit_status so the model can't silently
    # drop injected memory.
    if ext_prefetch_cache:
        with suppress(Exception):
            _recall_indicator = agent._memory_manager.describe_recall()
            if _recall_indicator:
                agent._emit_status(_recall_indicator)
    return ext_prefetch_cache


def _stamp_api_content_sidecar(
    agent: Any, messages: List[Any], current_turn_user_idx: int, ext_prefetch_cache: str,
    plugin_user_context: str, *, preflight_compressed: bool,
) -> None:
    """api_content sidecar — persist what you send: injected context lives only in the
    API copy, so stamp the exact sent bytes on the live dict for replay."""
    _turn_user_msg = messages[current_turn_user_idx]
    _api_content = compose_user_api_content(
        _turn_user_msg.get("content", ""), ext_prefetch_cache, plugin_user_context
    )
    if _api_content is None or _api_content == _turn_user_msg.get("content"):
        return
    _turn_user_msg["api_content"] = _api_content
    # In-place preflight compaction already inserted this turn's user row and the
    # crash persist identity-skips compacted dicts, so backfill the stamp onto the row
    # directly. Rotation mode flushes to the child session later.
    if not (preflight_compressed and getattr(agent, "_last_compaction_in_place", False)):
        return
    _db = getattr(agent, "_session_db", None)
    if _db is not None:
        try:
            _db.set_latest_user_api_content(
                agent.session_id, _turn_user_msg.get("content"), _api_content
            )
        except Exception:
            logger.warning(
                "in-place compaction api_content backfill failed "
                "for session=%s",
                agent.session_id or "none",
                exc_info=True,
            )


def _persist_turn_start(
    agent: Any, messages: List[Any], conversation_history: Optional[List[Any]],
    pending_cli_message: Any,
) -> None:
    """Crash-resilience: persist the inbound user turn once, with final api_content,
    before the first LLM call. Same critical section as CLI close persistence; retries
    the row create if the pre-compression attempt failed transiently."""
    def _ensure_and_persist() -> None:
        agent._ensure_db_session()
        agent._persist_session(messages, conversation_history)

    _persist_under_lock(
        agent, _ensure_and_persist,
        "Early turn-start session persistence failed for session=%s", pending_cli_message,
    )


def build_turn_context(
    agent, user_message: Any, system_message: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]], task_id: Optional[str], stream_callback,
    persist_user_message: Optional[Any], persist_user_timestamp: Optional[float]=None,
    persist_user_platform_id: Optional[str]=None, *, persist_user_display_kind: Optional[str]=None,
    persist_user_display_metadata: Optional[Dict[str, Any]]=None, restore_or_build_system_prompt,
    install_safe_stdio, sanitize_surrogates, summarize_user_message_for_log, set_session_context,
    set_current_write_origin, ra, moa_active: bool=False,
) -> TurnContext:
    """Run the once-per-turn setup and return the loop's input context.

    Helpers are passed in to avoid an import cycle with ``agent.conversation_loop``.
    Order matters: the DB session row is created only AFTER the system prompt is built
    (else it persists system_prompt=NULL and costs a cache miss) and BEFORE preflight
    compression."""
    from agent.turn_context_compaction import run_turn_start_compaction

    # Guard stdio against OSError from broken pipes (systemd/headless/daemon).
    install_safe_stdio()

    # Recover a rotated session before binding log/turn ids or copying client history so
    # everything in this turn belongs to the canonical child.
    recovered_history = recover_rotated_compression_session(agent)
    if recovered_history is not None:
        conversation_history = recovered_history

    # Tag log records on this thread with the session ID for ``hermes logs``; bind the
    # skill write-origin ContextVar; restore the primary runtime after a fallback turn.
    # NOTE: the DB session row is created later, AFTER the system prompt is restored/built (see
    # _ensure_db_session() below the system-prompt block). Creating it here — before _cached_system_prompt
    # is populated — inserts a row with system_prompt=NULL on a fresh API/gateway agent that carries
    # client-managed history, which then trips the "stored system prompt is null; rebuilding from scratch"
    # warning and a needless first-turn prefix cache miss. (Issue #45499.)
    set_session_context(agent.session_id)
    set_current_write_origin(getattr(agent, "_memory_write_origin", "assistant_tool"))
    agent._restore_primary_runtime()
    _publish_runtime_main(agent)
    _refresh_mcp_tools_between_turns(agent)

    if isinstance(user_message, str):
        user_message = sanitize_surrogates(user_message)
    if isinstance(persist_user_message, str):
        persist_user_message = sanitize_surrogates(persist_user_message)

    effective_task_id, turn_id = _bind_turn_identity(
        agent, task_id, stream_callback, persist_user_message,
        persist_user_timestamp, persist_user_platform_id,
    )
    _reset_per_turn_agent_state(agent)

    _preview_text = summarize_user_message_for_log(user_message)
    _msg_preview = _preview_text[:80] + ("..." if len(_preview_text) > 80 else "")
    logger.info(
        "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
        agent.session_id or "none", agent.model, agent.provider or "unknown",
        agent.platform or "unknown", len(conversation_history or []),
        _msg_preview.replace("\n", " "),
    )

    # Copy so the caller's list is never mutated.
    messages = list(conversation_history) if conversation_history else []
    user_msg, pending_cli_message = _stage_turn_user_message(
        agent, user_message, persist_user_message, persist_user_timestamp,
        persist_user_platform_id, persist_user_display_kind, persist_user_display_metadata,
    )
    _hydrate_from_history(agent, conversation_history)
    # Append the user message now that close persistence is safe.
    append_message(messages, user_msg)
    current_turn_user_idx = len(messages) - 1
    agent._persist_user_message_idx = current_turn_user_idx

    agent._user_turn_count += 1
    # Copilot x-initiator: the first API call of this user turn is user-initiated;
    # tool-loop follow-ups revert to "agent".
    agent._is_user_initiated_turn = True

    # Preserve the original user message (no nudge injection).
    original_user_message = persist_user_message if persist_user_message is not None else user_message
    should_review_memory = _tick_memory_nudge(agent)
    _emit_reaction(agent, original_user_message)

    if not agent.quiet_mode:
        agent._safe_print(
            f"💬 Starting conversation: '{_preview_text[:60]}"
            f"{'...' if len(_preview_text) > 60 else ''}'"
        )

    # System prompt is cached per session for prefix caching.
    if agent._cached_system_prompt is None:
        restore_or_build_system_prompt(agent, system_message, conversation_history)
    active_system_prompt = agent._cached_system_prompt

    # Bot Mode DM tool — injected only into routed Bot Mode sessions (canonical
    # Bot Chat or a classified human messaging chat bound to a real profile).
    # The shared gate is frozen per session/source, so the tool list is cache-safe.
    try:
        from tools.bot_mode_dm import ensure_message_agent_tool

        ensure_message_agent_tool(agent)
    except Exception:
        logger.debug("message_agent injection skipped", exc_info=True)

    _ensure_session_row(agent, pending_cli_message)

    compaction = run_turn_start_compaction(
        agent, messages=messages, system_message=system_message,
        active_system_prompt=active_system_prompt, conversation_history=conversation_history,
        current_turn_user_idx=current_turn_user_idx, user_message=user_message,
        effective_task_id=effective_task_id,
    )
    messages = compaction.messages
    active_system_prompt = compaction.active_system_prompt
    conversation_history = compaction.conversation_history
    current_turn_user_idx = compaction.current_turn_user_idx

    plugin_user_context = _collect_pre_llm_call_context(
        agent, effective_task_id=effective_task_id, turn_id=turn_id,
        original_user_message=original_user_message, messages=messages,
        conversation_history=conversation_history,
    )
    plugin_user_context = _merge_gateway_notes(
        agent, messages, current_turn_user_idx, plugin_user_context
    )

    _bind_interrupt_scope(agent, ra)
    ext_prefetch_cache = _memory_turn_start_and_prefetch(agent, original_user_message)

    # Sidecar skipped for codex_app_server/MoA.
    if (
        not moa_active
        and getattr(agent, "api_mode", None) != "codex_app_server"
        and 0 <= current_turn_user_idx < len(messages)
        and messages[current_turn_user_idx].get("role") == "user"
    ):
        _stamp_api_content_sidecar(
            agent, messages, current_turn_user_idx, ext_prefetch_cache,
            plugin_user_context, preflight_compressed=compaction.compressed,
        )

    _persist_turn_start(agent, messages, conversation_history, pending_cli_message)

    # Title the session now: the row exists and titling depends only on the user's ask,
    # so it runs concurrently with the turn. Daemon thread, no-op once titled.
    _maybe_title_session_at_turn_start(agent, messages)

    return TurnContext(
        user_message=user_message, original_user_message=original_user_message, messages=messages,
        conversation_history=conversation_history, active_system_prompt=active_system_prompt,
        effective_task_id=effective_task_id, turn_id=turn_id,
        current_turn_user_idx=current_turn_user_idx, should_review_memory=should_review_memory,
        plugin_user_context=plugin_user_context, ext_prefetch_cache=ext_prefetch_cache,
        preflight_compression_blocked=compaction.blocked,
    )


def _sanitize_model_for(agent: Any, moa_config: Any) -> Any:
    """Model name for strict-API tool-call sanitization. In MoA mode ``agent.model`` is
    the virtual preset name; use the resolved aggregator so Gemini keeps
    thought_signature (extra_content)."""
    _sanitize_model = agent.model
    if agent.provider == "moa":
        if moa_config:
            _agg = moa_config.get("aggregator") or {}
            if _agg.get("model"):
                _sanitize_model = _agg["model"]
        if _sanitize_model == agent.model:
            # Virtual-provider mode: no moa_config is threaded through; ask the facade
            # for the aggregator slot from the previous create().
            _agg_slot = getattr(getattr(agent, "client", None), "last_aggregator_slot", None)
            if _agg_slot and _agg_slot.get("model"):
                _sanitize_model = _agg_slot["model"]
    return _sanitize_model


def build_api_messages(
    agent: Any, messages: List[Dict[str, Any]], *, current_turn_user_idx: Any,
    ext_prefetch_cache: Any, plugin_user_context: Any, moa_config: Any, active_system_prompt: Any,
) -> Tuple[List[Dict[str, Any]], str]:
    """Build the wire copy of ``messages`` for one API call plus the effective system
    message. Returns ``(api_messages, effective_system)``.

    Prompt-cache invariant: historical user/assistant rows replay their ``api_content``
    sidecar (the exact bytes sent live) so the prefix stays byte-stable; the current
    user turn reuses the prologue's stamp (or composes live when a caller bypassed the
    prologue). Ephemeral context (prefetch, ``pre_llm_call`` hooks,
    ``ephemeral_system_prompt``) is added at API time only — ``messages`` stays untouched
    beyond the sidecar stamp, and the system prompt is built ONCE per session and
    replayed verbatim."""
    from agent.agent_runtime_helpers import fill_empty_non_final_wire_payload
    from agent.conversation_loop import _clone_message_for_send

    api_messages = []
    for idx, msg in enumerate(messages):
        # Structural clone, NOT msg.copy(): in-place transforms below must not reach
        # persisted history via nested containers; see _clone_message_for_send.
        api_msg = _clone_message_for_send(msg)
        # api_content is bookkeeping (exact bytes sent), never a provider field — pop
        # it from EVERY outgoing copy. display_* is display-only timeline metadata
        # (strict OpenAI backends reject unknown keys); _row_id is the durable row id
        # from _rows_to_conversation and only chat-completions strips underscore keys.
        _api_content = api_msg.pop("api_content", None)
        for key in ("display_kind", "display_metadata", "_row_id"):
            api_msg.pop(key, None)

        # Inject ephemeral context (memory prefetch + pre_llm_call user hooks)
        # at API time only; `messages` is untouched beyond the api_content stamp.
        if idx == current_turn_user_idx and msg.get("role") == "user":
            if isinstance(_api_content, str) and _api_content:
                # Reuse the prologue's stamp so sidecar and wire cannot drift
                # and every pass this turn sends identical bytes.
                api_msg["content"] = _api_content
            else:
                # Callers that bypass the prologue stamping: compose live.
                _composed = compose_user_api_content(
                    api_msg.get("content", ""), ext_prefetch_cache, plugin_user_context
                )
                if _composed is not None:
                    api_msg["content"] = _composed
        elif (
            isinstance(_api_content, str) and _api_content
            and msg.get("role") in ("user", "assistant")
        ):
            # Historical row: replay the exact bytes sent live so the prompt-cache
            # prefix stays byte-stable. User rows carry the injection sidecar; user
            # and assistant rows may carry a sanitize-divergence sidecar.
            api_msg["content"] = _api_content

        # Pass reasoning back to the API for ALL assistant messages so multi-turn
        # reasoning context is preserved.
        agent._copy_reasoning_content_for_api(msg, api_msg)
        # 'reasoning' is trajectory-only (copied to 'reasoning_content' above);
        # finish_reason is rejected by strict APIs (e.g. Mistral).
        api_msg.pop("reasoning", None)
        api_msg.pop("finish_reason", None)
        # Fill empty non-final user/assistant wire copies so the pre-call sanitizer
        # stops re-healing and flooding errors.log; durable history is untouched.
        # After the reasoning copy so thinking-only turns keep payload.
        fill_empty_non_final_wire_payload(api_msg, is_final=(idx == len(messages) - 1))
        # _thinking_prefill survives intentionally: the drop pass below needs it.
        # Strip length-continuation marks; some transports keep underscore keys.
        api_msg.pop("_length_continuation_fragment", None)
        api_msg.pop("_length_continuation_nudge", None)
        # Strip Codex Responses fields (call_id, response_item_id): strict providers
        # reject unknown fields. New dicts keep the internal list intact for Codex.
        if agent._should_sanitize_tool_calls():
            agent._sanitize_tool_calls_for_strict_api(
                api_msg, model=_sanitize_model_for(agent, moa_config)
            )
        # 'reasoning_details' is kept: OpenRouter uses it for multi-turn reasoning
        # continuity.
        api_messages.append(api_msg)

    # Final system message = cached prompt + ephemeral additions (API-time only).
    # Plugin/recall context goes into the user message, never the system prompt: the
    # prompt is built ONCE per session and replayed verbatim (stable cache prefix).
    effective_system = active_system_prompt or ""
    if agent.ephemeral_system_prompt:
        effective_system = (effective_system + "\n\n" + agent.ephemeral_system_prompt).strip()
    if effective_system:
        api_messages = [{"role": "system", "content": effective_system}] + api_messages
    return api_messages, effective_system


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'IDLE_COMPACTION_STATUS_TEMPLATE': ('agent.conversation_compression', 'IDLE_COMPACTION_STATUS_TEMPLATE'),
    'PREFLIGHT_COMPRESSION_STATUS_TEMPLATE': ('agent.conversation_compression', 'PREFLIGHT_COMPRESSION_STATUS_TEMPLATE'),
    'automatic_compaction_status_message': ('agent.context_engine', 'automatic_compaction_status_message'),
    'compression_skipped_due_to_lock': ('agent.conversation_compression', 'compression_skipped_due_to_lock'),
    'conversation_history_after_compression': ('agent.conversation_compression', 'conversation_history_after_compression'),
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
