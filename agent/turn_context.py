"""Per-turn setup for ``run_conversation`` (the turn prologue).

``run_conversation`` opened with ~470 lines of straight-line setup before the
tool-calling loop ever started: stdio guarding, runtime-main wiring, retry-counter
resets, user-message sanitization, todo/nudge-counter hydration, system-prompt
restore-or-build, crash-resilience persistence, preflight context compression, the
``pre_llm_call`` plugin hook, and external-memory prefetch.

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
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.conversation_compression import conversation_history_after_compression
from agent.iteration_budget import IterationBudget
from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
)

logger = logging.getLogger(__name__)


def _apply_pre_llm_call_model_override(agent, new_model: str, new_provider: str) -> None:
    """Apply a per-turn model override requested by a ``pre_llm_call`` plugin.

    A ``pre_llm_call`` callback may return ``{"model": ..., "provider": ...}``
    to route THIS turn to a different model (e.g. a coding-tuned model for a
    coding task). Credentials are resolved through the same pipeline as the
    ``/model`` command, then the runtime swap is applied.

    Any failure is logged and swallowed so a bad override can never abort the
    turn — the agent simply keeps its current model. The swap persists like
    ``/model`` (not turn-scoped), so routing plugins should return a choice on
    every turn.
    """
    try:
        from hermes_cli.model_switch import switch_model as _resolve_switch
        from hermes_cli.config import load_config, get_compatible_custom_providers
        from agent.agent_runtime_helpers import switch_model as _runtime_switch

        _cfg = load_config()
        result = _resolve_switch(
            raw_input=new_model,
            current_provider=agent.provider or "",
            current_model=agent.model or "",
            current_base_url=getattr(agent, "base_url", "") or "",
            current_api_key=getattr(agent, "api_key", "") or "",
            explicit_provider=new_provider,
            custom_providers=get_compatible_custom_providers(_cfg),
            is_global=False,
        )
        if not result.success:
            logger.warning(
                "pre_llm_call model override to %s/%s failed: %s; keeping %s/%s",
                new_provider, new_model, result.error_message,
                agent.provider, agent.model,
            )
            return
        _runtime_switch(
            agent,
            result.new_model,
            result.target_provider,
            api_key=result.api_key,
            base_url=result.base_url,
            api_mode=result.api_mode,
        )
        logger.info(
            "pre_llm_call: routed this turn to %s/%s",
            result.target_provider, result.new_model,
        )
    except Exception as exc:  # never let routing break a turn
        logger.warning(
            "pre_llm_call model override error (%s/%s): %s; keeping %s/%s",
            new_provider, new_model, exc, agent.provider, agent.model,
        )


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


def build_turn_context(
    agent,
    user_message: str,
    system_message: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]],
    task_id: Optional[str],
    stream_callback,
    persist_user_message: Optional[str],
    persist_user_timestamp: Optional[float] = None,
    *,
    restore_or_build_system_prompt,
    install_safe_stdio,
    sanitize_surrogates,
    summarize_user_message_for_log,
    set_session_context,
    set_current_write_origin,
    ra,
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

    # Tell auxiliary_client what the live main provider/model are for this turn.
    try:
        from agent.auxiliary_client import set_runtime_main
        set_runtime_main(
            getattr(agent, "provider", "") or "",
            getattr(agent, "model", "") or "",
            base_url=getattr(agent, "base_url", "") or "",
            api_key=getattr(agent, "api_key", "") or "",
            api_mode=getattr(agent, "api_mode", "") or "",
        )
    except Exception:
        pass

    # Tag log records on this thread with the session ID for ``hermes logs``.
    set_session_context(agent.session_id)

    # Bind the skill write-origin ContextVar for this thread.
    set_current_write_origin(getattr(agent, "_memory_write_origin", "assistant_tool"))

    # Restore the primary runtime if the previous turn activated fallback.
    agent._restore_primary_runtime()

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

    # Track user turns for memory flush and periodic nudge logic.
    agent._user_turn_count += 1

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

    # Add user message.
    user_msg = {"role": "user", "content": user_message}
    messages.append(user_msg)
    current_turn_user_idx = len(messages) - 1
    agent._persist_user_message_idx = current_turn_user_idx

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
    agent._ensure_db_session()

    # Crash-resilience: persist the inbound user turn as soon as the session row exists.
    try:
        agent._persist_session(messages, conversation_history)
    except Exception:
        logger.warning(
            "Early turn-start session persistence failed for session=%s",
            agent.session_id or "none",
            exc_info=True,
        )

    # ── Preflight context compression ──
    # Gate the (expensive) full token estimate behind a cheap pre-check.
    # See ``_should_run_preflight_estimate`` for the OR semantics that fix
    # issue #27405 (a few very large messages slipping past the count gate).
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
        elif _compressor.should_compress(_preflight_tokens):
            logger.info(
                "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                f"{_preflight_tokens:,}",
                f"{_compressor.threshold_tokens:,}",
                agent.model,
                f"{_compressor.context_length:,}",
            )
            agent._emit_status(
                f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                f">= {_compressor.threshold_tokens:,} threshold. "
                "This may take a moment."
            )
            for _pass in range(3):
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
        for r in _pre_results:
            if isinstance(r, dict) and r.get("context"):
                _ctx_parts.append(str(r["context"]))
            elif isinstance(r, str) and r.strip():
                _ctx_parts.append(r)
        if _ctx_parts:
            plugin_user_context = "\n\n".join(_ctx_parts)

        # Per-turn model routing: a pre_llm_call plugin may return
        # {"model": "...", "provider": "..."} to route THIS turn to a
        # different model (e.g. a coding model for a coding task). The first
        # such result wins. This runs after context injection so a plugin can
        # do both. See _apply_pre_llm_call_model_override for semantics.
        _routed = next(
            (r for r in _pre_results if isinstance(r, dict) and r.get("model")),
            None,
        )
        if _routed:
            _rm = str(_routed.get("model") or "").strip()
            _rp = str(_routed.get("provider") or agent.provider or "").strip()
            if _rm and (_rm != agent.model or _rp != (agent.provider or "")):
                _apply_pre_llm_call_model_override(agent, _rm, _rp)
    except Exception as exc:
        logger.warning("pre_llm_call hook failed: %s", exc)

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
    )
