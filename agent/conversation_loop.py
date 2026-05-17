"""The agent conversation loop — extracted from ``run_agent.AIAgent``.

This is the biggest single chunk pulled out of ``run_agent.py``: the
roughly 3,900-line :func:`run_conversation` body that drives one user
turn through the agent (model call, tool dispatch, retries, fallbacks,
compression, post-turn hooks, background memory/skill review nudges).

The function takes the parent ``AIAgent`` instance as its first
argument (``agent``) and accesses its state via attribute lookup.
``_ra().AIAgent.run_conversation`` is now a thin forwarder.

Symbols that production code or tests patch on ``run_agent`` directly
(``handle_function_call``, ``_set_interrupt``, ``OpenAI``, ...) are
resolved through :func:`_ra` so those patches keep working.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import ssl
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from agent.anthropic_adapter import _is_oauth_token
from agent.auxiliary_client import set_runtime_main
from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.display import KawaiiSpinner
from agent.error_classifier import FailoverReason, classify_api_error
from agent.iteration_budget import IterationBudget
from agent.memory_manager import build_memory_context_block
from agent.message_sanitization import (
    _repair_tool_call_arguments,
    _sanitize_messages_non_ascii,
    _sanitize_messages_surrogates,
    _sanitize_structure_non_ascii,
    _sanitize_structure_surrogates,
    _sanitize_surrogates,
    _sanitize_tools_non_ascii,
    _strip_images_from_messages,
    _strip_non_ascii,
)
from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
    get_next_probe_tier,
    parse_available_output_tokens_from_error,
    parse_context_limit_from_error,
    save_context_length,
)
from agent.nous_rate_guard import (
    clear_nous_rate_limit,
    is_genuine_nous_rate_limit,
    nous_rate_limit_remaining,
    record_nous_rate_limit,
)
from agent.process_bootstrap import _install_safe_stdio
from agent.prompt_caching import apply_anthropic_cache_control
from agent.retry_utils import jittered_backoff
from agent.trajectory import has_incomplete_scratchpad
from agent.usage_pricing import estimate_usage_cost, normalize_usage
from hermes_constants import display_hermes_home as _dhh_fn
from hermes_logging import set_session_context
from tools.schema_sanitizer import strip_pattern_and_format
from tools.skill_provenance import set_current_write_origin
from utils import base_url_host_matches, env_var_enabled

logger = logging.getLogger(__name__)


def _ra():
    """Lazy reference to ``run_agent`` so callers can patch
    ``run_agent.handle_function_call`` / ``run_agent._set_interrupt`` /
    ``run_agent.OpenAI`` and have those patches reach this code path.
    """
    import run_agent
    return run_agent


def run_conversation(
    agent,
    user_message: str,
    system_message: str = None,
    conversation_history: List[Dict[str, Any]] = None,
    task_id: str = None,
    stream_callback: Optional[callable] = None,
    persist_user_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a complete conversation with tool calling until completion.

    Args:
        user_message (str): The user's message/question
        system_message (str): Custom system message (optional, overrides ephemeral_system_prompt if provided)
        conversation_history (List[Dict]): Previous conversation messages (optional)
        task_id (str): Unique identifier for this task to isolate VMs between concurrent tasks (optional, auto-generated if not provided)
        stream_callback: Optional callback invoked with each text delta during streaming.
            Used by the TTS pipeline to start audio generation before the full response.
            When None (default), API calls use the standard non-streaming path.
        persist_user_message: Optional clean user message to store in
            transcripts/history when user_message contains API-only
            synthetic prefixes.
                or queuing follow-up prefetch work.

    Returns:
        Dict: Complete conversation result with final response and message history
    """
    # Guard stdio against OSError from broken pipes (systemd/headless/daemon).
    # Installed once, transparent when streams are healthy, prevents crash on write.
    _install_safe_stdio()

    agent._ensure_db_session()

    # Tell auxiliary_client what the live main provider/model are for
    # this turn. Used by tools whose behaviour depends on the active
    # main model (e.g. vision_analyze's native fast path) so they see
    # the CLI/gateway override instead of the stale config.yaml
    # default. Idempotent — fine to call every turn.
    try:
        from agent.auxiliary_client import set_runtime_main
        set_runtime_main(
            getattr(agent, "provider", "") or "",
            getattr(agent, "model", "") or "",
        )
    except Exception:
        pass

    # Tag all log records on this thread with the session ID so
    # ``hermes logs --session <id>`` can filter a single conversation.
    from hermes_logging import set_session_context
    set_session_context(agent.session_id)

    # Bind the skill write-origin ContextVar for this thread so tool
    # handlers (e.g. skill_manage create) can tell whether they are
    # running inside the background agent-improvement review fork vs.
    # a foreground user-directed turn. Set at the top of each call;
    # the review fork runs on its own thread with a fresh context,
    # so the foreground value here does not leak into it.
    from tools.skill_provenance import set_current_write_origin
    set_current_write_origin(getattr(agent, "_memory_write_origin", "assistant_tool"))

    # If the previous turn activated fallback, restore the primary
    # runtime so this turn gets a fresh attempt with the preferred model.
    # No-op when _fallback_activated is False (gateway, first turn, etc.).
    agent._restore_primary_runtime()

    # Sanitize surrogate characters from user input.  Clipboard paste from
    # rich-text editors (Google Docs, Word, etc.) can inject lone surrogates
    # that are invalid UTF-8 and crash JSON serialization in the OpenAI SDK.
    if isinstance(user_message, str):
        user_message = _sanitize_surrogates(user_message)
    if isinstance(persist_user_message, str):
        persist_user_message = _sanitize_surrogates(persist_user_message)

    # Store stream callback for _interruptible_api_call to pick up
    agent._stream_callback = stream_callback
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = persist_user_message
    # Generate unique task_id if not provided to isolate VMs between concurrent tasks
    effective_task_id = task_id or str(uuid.uuid4())
    # Expose the active task_id so tools running mid-turn (e.g. delegate_task
    # in delegate_tool.py) can identify this agent for the cross-agent file
    # state registry.  Set BEFORE any tool dispatch so snapshots taken at
    # child-launch time see the parent's real id, not None.
    agent._current_task_id = effective_task_id
    
    # Reset retry counters and iteration budget at the start of each turn
    # so subagent usage from a previous turn doesn't eat into the next one.
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
    # True until the server rejects an image_url content part with an error
    # like "Only 'text' content type is supported."  Set to False on first
    # rejection and kept False for the rest of the session so we never re-send
    # images to a text-only endpoint.  Scoped per `_run()` call, not per instance.
    agent._vision_supported = True

    # Pre-turn connection health check: detect and clean up dead TCP
    # connections left over from provider outages or dropped streams.
    # This prevents the next API call from hanging on a zombie socket.
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
    # Replay compression warning through status_callback for gateway
    # platforms (the callback was not wired during __init__).
    if agent._compression_warning:
        agent._replay_compression_warning()
        agent._compression_warning = None  # send once

    # NOTE: _turns_since_memory and _iters_since_skill are NOT reset here.
    # They are initialized in __init__ and must persist across run_conversation
    # calls so that nudge logic accumulates correctly in CLI mode.
    agent.iteration_budget = IterationBudget(agent.max_iterations)

    # Log conversation turn start for debugging/observability
    _preview_text = _summarize_user_message_for_log(user_message)
    _msg_preview = (_preview_text[:80] + "...") if len(_preview_text) > 80 else _preview_text
    _msg_preview = _msg_preview.replace("\n", " ")
    logger.info(
        "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
        agent.session_id or "none", agent.model, agent.provider or "unknown",
        agent.platform or "unknown", len(conversation_history or []),
        _msg_preview,
    )

    # Initialize conversation (copy to avoid mutating the caller's list)
    messages = list(conversation_history) if conversation_history else []

    # Hydrate todo store from conversation history (gateway creates a fresh
    # AIAgent per message, so the in-memory store is empty -- we need to
    # recover the todo state from the most recent todo tool response in history)
    if conversation_history and not agent._todo_store.has_items():
        agent._hydrate_todo_store(conversation_history)

    # Hydrate per-session nudge counters from persisted history.
    # Gateway creates a fresh AIAgent per inbound message (cache miss /
    # 1h idle eviction / config-signature mismatch / process restart), so
    # _turns_since_memory and _user_turn_count start at 0 every turn and
    # the memory.nudge_interval trigger may never be reached. Reconstruct
    # an effective count from prior user turns in conversation_history.
    # Idempotent: a cached agent that already accumulated counters keeps
    # them; only a freshly-built agent with empty in-memory state hydrates.
    # See issue #22357.
    if conversation_history and agent._user_turn_count == 0:
        prior_user_turns = sum(
            1 for m in conversation_history if m.get("role") == "user"
        )
        if prior_user_turns > 0:
            agent._user_turn_count = prior_user_turns
            if agent._memory_nudge_interval > 0 and agent._turns_since_memory == 0:
                # % preserves original 1-in-N cadence rather than firing a
                # review immediately on resume (which would surprise users
                # whose session happened to land just past a multiple of N).
                agent._turns_since_memory = prior_user_turns % agent._memory_nudge_interval


    # Prefill messages (few-shot priming) are injected at API-call time only,
    # never stored in the messages list. This keeps them ephemeral: they won't
    # be saved to session DB, session logs, or batch trajectories, but they're
    # automatically re-applied on every API call (including session continuations).
    
    # Track user turns for memory flush and periodic nudge logic
    agent._user_turn_count += 1

    # Reset the streaming context scrubber at the top of each turn so a
    # hung span from a prior interrupted stream can't taint this turn's
    # output.
    scrubber = getattr(agent, "_stream_context_scrubber", None)
    if scrubber is not None:
        scrubber.reset()
    # Reset the think scrubber for the same reason — an interrupted
    # prior stream may have left us inside an unterminated block.
    think_scrubber = getattr(agent, "_stream_think_scrubber", None)
    if think_scrubber is not None:
        think_scrubber.reset()

    # Preserve the original user message (no nudge injection).
    original_user_message = persist_user_message if persist_user_message is not None else user_message

    # Track memory nudge trigger (turn-based, checked here).
    # Skill trigger is checked AFTER the agent loop completes, based on
    # how many tool iterations THIS turn used.
    _should_review_memory = False
    if (agent._memory_nudge_interval > 0
            and "memory" in agent.valid_tool_names
            and agent._memory_store):
        agent._turns_since_memory += 1
        if agent._turns_since_memory >= agent._memory_nudge_interval:
            _should_review_memory = True
            agent._turns_since_memory = 0

    # Add user message
    user_msg = {"role": "user", "content": user_message}
    messages.append(user_msg)
    current_turn_user_idx = len(messages) - 1
    agent._persist_user_message_idx = current_turn_user_idx
    
    if not agent.quiet_mode:
        _print_preview = _summarize_user_message_for_log(user_message)
        agent._safe_print(f"💬 Starting conversation: '{_print_preview[:60]}{'...' if len(_print_preview) > 60 else ''}'")
    
    # ── System prompt (cached per session for prefix caching) ──
    # Built once on first call, reused for all subsequent calls.
    # Only rebuilt after context compression events (which invalidate
    # the cache and reload memory from disk).
    #
    # For continuing sessions (gateway creates a fresh AIAgent per
    # message), we load the stored system prompt from the session DB
    # instead of rebuilding.  Rebuilding would pick up memory changes
    # from disk that the model already knows about (it wrote them!),
    # producing a different system prompt and breaking the Anthropic
    # prefix cache.
    if agent._cached_system_prompt is None:
        stored_prompt = None
        if conversation_history and agent._session_db:
            try:
                session_row = agent._session_db.get_session(agent.session_id)
                if session_row:
                    stored_prompt = session_row.get("system_prompt") or None
            except Exception:
                pass  # Fall through to build fresh

        if stored_prompt:
            # Continuing session — reuse the exact system prompt from
            # the previous turn so the Anthropic cache prefix matches.
            agent._cached_system_prompt = stored_prompt
        else:
            # First turn of a new session — build from scratch.
            agent._cached_system_prompt = agent._build_system_prompt(system_message)
            # Plugin hook: on_session_start
            # Fired once when a brand-new session is created (not on
            # continuation).  Plugins can use this to initialise
            # session-scoped state (e.g. warm a memory cache).
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _invoke_hook(
                    "on_session_start",
                    session_id=agent.session_id,
                    model=agent.model,
                    platform=getattr(agent, "platform", None) or "",
                )
            except Exception as exc:
                logger.warning("on_session_start hook failed: %s", exc)

            # Store the system prompt snapshot in SQLite
            if agent._session_db:
                try:
                    agent._session_db.update_system_prompt(agent.session_id, agent._cached_system_prompt)
                except Exception as e:
                    logger.debug("Session DB update_system_prompt failed: %s", e)

    active_system_prompt = agent._cached_system_prompt

    # ── Preflight context compression ──
    # Before entering the main loop, check if the loaded conversation
    # history already exceeds the model's context threshold.  This handles
    # cases where a user switches to a model with a smaller context window
    # while having a large existing session — compress proactively rather
    # than waiting for an API error (which might be caught as a non-retryable
    # 4xx and abort the request entirely).
    if (
        agent.compression_enabled
        and len(messages) > agent.context_compressor.protect_first_n
                            + agent.context_compressor.protect_last_n + 1
    ):
        # Include tool schema tokens — with many tools these can add
        # 20-30K+ tokens that the old sys+msg estimate missed entirely.
        _preflight_tokens = estimate_request_tokens_rough(
            messages,
            system_prompt=active_system_prompt or "",
            tools=agent.tools or None,
        )

        if _preflight_tokens >= agent.context_compressor.threshold_tokens:
            logger.info(
                "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                f"{_preflight_tokens:,}",
                f"{agent.context_compressor.threshold_tokens:,}",
                agent.model,
                f"{agent.context_compressor.context_length:,}",
            )
            agent._emit_status(
                f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                f">= {agent.context_compressor.threshold_tokens:,} threshold. "
                "This may take a moment."
            )
            # May need multiple passes for very large sessions with small
            # context windows (each pass summarises the middle N turns).
            for _pass in range(3):
                _orig_len = len(messages)
                messages, active_system_prompt = agent._compress_context(
                    messages, system_message, approx_tokens=_preflight_tokens,
                    task_id=effective_task_id,
                )
                if len(messages) >= _orig_len:
                    break  # Cannot compress further
                # Compression created a new session — clear the history
                # reference so _flush_messages_to_session_db writes ALL
                # compressed messages to the new session's SQLite, not
                # skipping them because conversation_history is still the
                # pre-compression length.
                conversation_history = None
                # Fix: reset retry counters after compression so the model
                # gets a fresh budget on the compressed context.  Without
                # this, pre-compression retries carry over and the model
                # hits "(empty)" immediately after compression-induced
                # context loss.
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                agent._last_content_with_tools = None
                agent._last_content_tools_all_housekeeping = False
                agent._mute_post_response = False
                # Re-estimate after compression
                _preflight_tokens = estimate_request_tokens_rough(
                    messages,
                    system_prompt=active_system_prompt or "",
                    tools=agent.tools or None,
                )
                if _preflight_tokens < agent.context_compressor.threshold_tokens:
                    break  # Under threshold

    # Plugin hook: pre_llm_call
    # Fired once per turn before the tool-calling loop.  Plugins can
    # return a dict with a ``context`` key (or a plain string) whose
    # value is appended to the current turn's user message.
    #
    # Context is ALWAYS injected into the user message, never the
    # system prompt.  This preserves the prompt cache prefix — the
    # system prompt stays identical across turns so cached tokens
    # are reused.  The system prompt is Hermes's territory; plugins
    # contribute context alongside the user's input.
    #
    # All injected context is ephemeral (not persisted to session DB).
    _plugin_user_context = ""
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _pre_results = _invoke_hook(
            "pre_llm_call",
            session_id=agent.session_id,
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
            _plugin_user_context = "\n\n".join(_ctx_parts)
    except Exception as exc:
        logger.warning("pre_llm_call hook failed: %s", exc)

    # Main conversation loop
    api_call_count = 0
    final_response = None
    interrupted = False
    codex_ack_continuations = 0
    length_continue_retries = 0
    truncated_tool_call_retries = 0
    truncated_response_parts: List[str] = []
    compression_attempts = 0
    _turn_exit_reason = "unknown"  # Diagnostic: why the loop ended

    # Per-turn file-mutation verifier state.  Keyed by resolved path;
    # each failed ``write_file`` / ``patch`` call records the error
    # preview.  Later successful writes to the same path remove the
    # entry (the model recovered).  At end-of-turn, any entries still
    # present are surfaced in an advisory footer so the model cannot
    # over-claim success while the file is actually unchanged on disk.
    agent._turn_failed_file_mutations: Dict[str, Dict[str, Any]] = {}
    
    # Record the execution thread so interrupt()/clear_interrupt() can
    # scope the tool-level interrupt signal to THIS agent's thread only.
    # Must be set before any thread-scoped interrupt syncing.
    agent._execution_thread_id = threading.current_thread().ident

    # Always clear stale per-thread state from a previous turn. If an
    # interrupt arrived before startup finished, preserve it and bind it
    # to this execution thread now instead of dropping it on the floor.
    _ra()._set_interrupt(False, agent._execution_thread_id)
    if agent._interrupt_requested:
        _ra()._set_interrupt(True, agent._execution_thread_id)
        agent._interrupt_thread_signal_pending = False
    else:
        agent._interrupt_message = None
        agent._interrupt_thread_signal_pending = False

    # Notify memory providers of the new turn so cadence tracking works.
    # Must happen BEFORE prefetch_all() so providers know which turn it is
    # and can gate context/dialectic refresh via contextCadence/dialecticCadence.
    if agent._memory_manager:
        try:
            _turn_msg = original_user_message if isinstance(original_user_message, str) else ""
            agent._memory_manager.on_turn_start(agent._user_turn_count, _turn_msg)
        except Exception:
            pass

    # External memory provider: prefetch once before the tool loop.
    # Reuse the cached result on every iteration to avoid re-calling
    # prefetch_all() on each tool call (10 tool calls = 10x latency + cost).
    # Use original_user_message (clean input) — user_message may contain
    # injected skill content that bloats / breaks provider queries.
    _ext_prefetch_cache = ""
    if agent._memory_manager:
        try:
            _query = original_user_message if isinstance(original_user_message, str) else ""
            _ext_prefetch_cache = agent._memory_manager.prefetch_all(_query) or ""
        except Exception:
            pass

    # Optional opt-in runtime: if api_mode == codex_app_server, hand the
    # turn to the codex app-server subprocess (terminal/file ops/patching
    # all run inside Codex). Default Hermes path is bypassed entirely.
    # See agent/transports/codex_app_server_session.py for the adapter
    # and references/codex-app-server-runtime.md for the rationale.
    if agent.api_mode == "codex_app_server":
        return agent._run_codex_app_server_turn(
            user_message=user_message,
            original_user_message=original_user_message,
            messages=messages,
            effective_task_id=effective_task_id,
            should_review_memory=_should_review_memory,
        )

    while (api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0) or agent._budget_grace_call:
        # Reset per-turn checkpoint dedup so each iteration can take one snapshot
        agent._checkpoint_mgr.new_turn()

        # Check for interrupt request (e.g., user sent new message)
        if agent._interrupt_requested:
            interrupted = True
            _turn_exit_reason = "interrupted_by_user"
            if not agent.quiet_mode:
                agent._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
            break
        
        api_call_count += 1
        agent._api_call_count = api_call_count
        agent._touch_activity(f"starting API call #{api_call_count}")

        # Grace call: the budget is exhausted but we gave the model one
        # more chance.  Consume the grace flag so the loop exits after
        # this iteration regardless of outcome.
        if agent._budget_grace_call:
            agent._budget_grace_call = False
        elif not agent.iteration_budget.consume():
            _turn_exit_reason = "budget_exhausted"
            if not agent.quiet_mode:
                agent._safe_print(f"\n⚠️  Iteration budget exhausted ({agent.iteration_budget.used}/{agent.iteration_budget.max_total} iterations used)")
            break

        # Fire step_callback for gateway hooks (agent:step event)
        if agent.step_callback is not None:
            try:
                prev_tools = []
                for _idx, _m in enumerate(reversed(messages)):
                    if _m.get("role") == "assistant" and _m.get("tool_calls"):
                        _fwd_start = len(messages) - _idx
                        _results_by_id = {}
                        for _tm in messages[_fwd_start:]:
                            if _tm.get("role") != "tool":
                                break
                            _tcid = _tm.get("tool_call_id")
                            if _tcid:
                                _results_by_id[_tcid] = _tm.get("content", "")
                        prev_tools = [
                            {
                                "name": tc["function"]["name"],
                                "result": _results_by_id.get(tc.get("id")),
                                "arguments": tc["function"].get("arguments"),
                            }
                            for tc in _m["tool_calls"]
                            if isinstance(tc, dict)
                        ]
                        break
                agent.step_callback(api_call_count, prev_tools)
            except Exception as _step_err:
                logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

        # Track tool-calling iterations for skill nudge.
        # Counter resets whenever skill_manage is actually used.
        if (agent._skill_nudge_interval > 0
                and "skill_manage" in agent.valid_tool_names):
            agent._iters_since_skill += 1
        
        # ── Pre-API-call /steer drain ──────────────────────────────────
        # If a /steer arrived during the previous API call (while the model
        # was thinking), drain it now — before we build api_messages — so
        # the model sees the steer text on THIS iteration.  Without this,
        # steers sent during an API call only land after the NEXT tool batch,
        # which may never come if the model returns a final response.
        #
        # We scan backwards for the last tool-role message in the messages
        # list.  If found, the steer is appended there.  If not (first
        # iteration, no tools yet), the steer stays pending for the next
        # tool batch — injecting into a user message would break role
        # alternation, and there's no tool output to piggyback on.
        _pre_api_steer = agent._drain_pending_steer()
        if _pre_api_steer:
            _injected = False
            for _si in range(len(messages) - 1, -1, -1):
                _sm = messages[_si]
                if isinstance(_sm, dict) and _sm.get("role") == "tool":
                    marker = f"\n\nUser guidance: {_pre_api_steer}"
                    existing = _sm.get("content", "")
                    if isinstance(existing, str):
                        _sm["content"] = existing + marker
                    else:
                        # Multimodal content blocks — append text block
                        try:
                            blocks = list(existing) if existing else []
                            blocks.append({"type": "text", "text": marker})
                            _sm["content"] = blocks
                        except Exception:
                            pass
                    _injected = True
                    logger.debug(
                        "Pre-API-call steer drain: injected into tool msg at index %d",
                        _si,
                    )
                    break
            if not _injected:
                # No tool message to inject into — put it back so
                # the post-tool-execution drain picks it up later.
                _lock = getattr(agent, "_pending_steer_lock", None)
                if _lock is not None:
                    with _lock:
                        if agent._pending_steer:
                            agent._pending_steer = agent._pending_steer + "\n" + _pre_api_steer
                        else:
                            agent._pending_steer = _pre_api_steer
                else:
                    existing = getattr(agent, "_pending_steer", None)
                    agent._pending_steer = (existing + "\n" + _pre_api_steer) if existing else _pre_api_steer

        # Prepare messages for API call
        # If we have an ephemeral system prompt, prepend it to the messages
        # Note: Reasoning is embedded in content via <think> tags for trajectory storage.
        # However, providers like Moonshot AI require a separate 'reasoning_content' field
        # on assistant messages with tool_calls. We handle both cases here.
        request_logger = getattr(agent, "logger", None) or logging.getLogger(__name__)
        repaired_tool_calls = agent._sanitize_tool_call_arguments(
            messages,
            logger=request_logger,
            session_id=agent.session_id,
        )
        if repaired_tool_calls > 0:
            request_logger.info(
                "Sanitized %s corrupted tool_call arguments before request (session=%s)",
                repaired_tool_calls,
                agent.session_id or "-",
            )

        # Defensive: repair malformed role-alternation before API call.
        # Catches cases where the history got wedged into a
        # ``tool → user`` or ``user → user`` tail (e.g. after empty-
        # response scaffolding was stripped and a new user message
        # landed after an orphan tool result). Most providers return
        # empty content on malformed sequences, which would otherwise
        # retrigger the empty-retry loop indefinitely.
        repaired_seq = agent._repair_message_sequence(messages)
        if repaired_seq > 0:
            request_logger.info(
                "Repaired %s message-alternation violations before request (session=%s)",
                repaired_seq,
                agent.session_id or "-",
            )

        api_messages = []
        for idx, msg in enumerate(messages):
            api_msg = msg.copy()

            # Inject ephemeral context into the current turn's user message.
            # Sources: memory manager prefetch + plugin pre_llm_call hooks
            # with target="user_message" (the default).  Both are
            # API-call-time only — the original message in `messages` is
            # never mutated, so nothing leaks into session persistence.
            if idx == current_turn_user_idx and msg.get("role") == "user":
                _injections = []
                if _ext_prefetch_cache:
                    _fenced = build_memory_context_block(_ext_prefetch_cache)
                    if _fenced:
                        _injections.append(_fenced)
                if _plugin_user_context:
                    _injections.append(_plugin_user_context)
                if _injections:
                    _base = api_msg.get("content", "")
                    if isinstance(_base, str):
                        api_msg["content"] = _base + "\n\n" + "\n\n".join(_injections)

            # For ALL assistant messages, pass reasoning back to the API
            # This ensures multi-turn reasoning context is preserved
            agent._copy_reasoning_content_for_api(msg, api_msg)

            # Remove 'reasoning' field - it's for trajectory storage only
            # We've copied it to 'reasoning_content' for the API above
            if "reasoning" in api_msg:
                api_msg.pop("reasoning")
            # Remove finish_reason - not accepted by strict APIs (e.g. Mistral)
            if "finish_reason" in api_msg:
                api_msg.pop("finish_reason")
            # Strip internal thinking-prefill marker
            api_msg.pop("_thinking_prefill", None)
            # Strip Codex Responses API fields (call_id, response_item_id) for
            # strict providers like Mistral, Fireworks, etc. that reject unknown fields.
            # Uses new dicts so the internal messages list retains the fields
            # for Codex Responses compatibility.
            if agent._should_sanitize_tool_calls():
                agent._sanitize_tool_calls_for_strict_api(api_msg)
            # Keep 'reasoning_details' - OpenRouter uses this for multi-turn reasoning context
            # The signature field helps maintain reasoning continuity
            api_messages.append(api_msg)

        # Build the final system message: cached prompt + ephemeral system prompt.
        # Ephemeral additions are API-call-time only (not persisted to session DB).
        # External recall context is injected into the user message, not the system
        # prompt, so the stable cache prefix remains unchanged.
        #
        # NOTE: Plugin context from pre_llm_call hooks is injected into the
        # user message (see injection block above), NOT the system prompt.
        # This is intentional — system prompt modifications break the prompt
        # cache prefix.  The system prompt is reserved for Hermes internals.
        #
        # Hermes invariant: the system prompt is built ONCE per session
        # (cached on ``_cached_system_prompt``) and replayed verbatim on
        # every turn.  We send it as a single content string so the
        # bytes are byte-stable across turns and upstream prompt caches
        # stay warm.
        effective_system = active_system_prompt or ""
        if agent.ephemeral_system_prompt:
            effective_system = (effective_system + "\n\n" + agent.ephemeral_system_prompt).strip()
        if effective_system:
            api_messages = [{"role": "system", "content": effective_system}] + api_messages

        # Inject ephemeral prefill messages right after the system prompt
        # but before conversation history. Same API-call-time-only pattern.
        if agent.prefill_messages:
            sys_offset = 1 if (api_messages and api_messages[0].get("role") == "system") else 0
            for idx, pfm in enumerate(agent.prefill_messages):
                api_messages.insert(sys_offset + idx, pfm.copy())

        # Apply Anthropic prompt caching for Claude models on native
        # Anthropic, OpenRouter, and third-party Anthropic-compatible
        # gateways. Auto-detected: if ``_use_prompt_caching`` is set,
        # inject cache_control breakpoints (system + last 3 messages)
        # to reduce input token costs by ~75% on multi-turn
        # conversations.
        if agent._use_prompt_caching:
            api_messages = apply_anthropic_cache_control(
                api_messages,
                cache_ttl=agent._cache_ttl,
                native_anthropic=agent._use_native_cache_layout,
            )

        # Safety net: strip orphaned tool results / add stubs for missing
        # results before sending to the API.  Runs unconditionally — not
        # gated on context_compressor — so orphans from session loading or
        # manual message manipulation are always caught.
        api_messages = agent._sanitize_api_messages(api_messages)

        # Drop thinking-only assistant turns (reasoning but no visible
        # output and no tool_calls) and merge any adjacent user messages
        # left behind. Prevents Anthropic 400s ("The final block in an
        # assistant message cannot be `thinking`.") and equivalent errors
        # from third-party Anthropic-compatible gateways that can't replay
        # a thinking-only turn. Runs on the per-call copy only — the
        # stored conversation history keeps the reasoning block for the
        # UI transcript and session persistence.
        api_messages = agent._drop_thinking_only_and_merge_users(api_messages)

        # Normalize message whitespace and tool-call JSON for consistent
        # prefix matching.  Ensures bit-perfect prefixes across turns,
        # which enables KV cache reuse on local inference servers
        # (llama.cpp, vLLM, Ollama) and improves cache hit rates for
        # cloud providers.  Operates on api_messages (the API copy) so
        # the original conversation history in `messages` is untouched.
        for am in api_messages:
            if isinstance(am.get("content"), str):
                am["content"] = am["content"].strip()
        for am in api_messages:
            tcs = am.get("tool_calls")
            if not tcs:
                continue
            new_tcs = []
            for tc in tcs:
                if isinstance(tc, dict) and "function" in tc:
                    try:
                        args_obj = json.loads(tc["function"]["arguments"])
                        tc = {**tc, "function": {
                            **tc["function"],
                            "arguments": json.dumps(
                                args_obj, separators=(",", ":"),
                                sort_keys=True,
                            ),
                        }}
                    except Exception:
                        tc["function"]["arguments"] = _repair_tool_call_arguments(
                            tc["function"]["arguments"],
                            tc["function"].get("name", "?"),
                        )
                new_tcs.append(tc)
            am["tool_calls"] = new_tcs

        # Proactively strip any surrogate characters before the API call.
        # Models served via Ollama (Kimi K2.5, GLM-5, Qwen) can return
        # lone surrogates (U+D800-U+DFFF) that crash json.dumps() inside
        # the OpenAI SDK. Sanitizing here prevents the 3-retry cycle.
        _sanitize_messages_surrogates(api_messages)

        # Calculate approximate request size for logging
        total_chars = sum(len(str(msg)) for msg in api_messages)
        approx_tokens = estimate_messages_tokens_rough(api_messages)
        
        # Thinking spinner for quiet mode (animated during API call)
        thinking_spinner = None
        
        if not agent.quiet_mode:
            agent._vprint(f"\n{agent.log_prefix}🔄 Making API call #{api_call_count}/{agent.max_iterations}...")
            agent._vprint(f"{agent.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
            agent._vprint(f"{agent.log_prefix}   🔧 Available tools: {len(agent.tools) if agent.tools else 0}")
        else:
            # Animated thinking spinner in quiet mode
            face = random.choice(KawaiiSpinner.get_thinking_faces())
            verb = random.choice(KawaiiSpinner.get_thinking_verbs())
            if agent.thinking_callback:
                # CLI TUI mode: use prompt_toolkit widget instead of raw spinner
                # (works in both streaming and non-streaming modes)
                agent.thinking_callback(f"{face} {verb}...")
            elif not agent._has_stream_consumers() and agent._should_start_quiet_spinner():
                # Raw KawaiiSpinner only when no streaming consumers and the
                # spinner output has a safe sink.
                spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=agent._print_fn)
                thinking_spinner.start()
        
        # Log request details if verbose
        if agent.verbose_logging:
            logging.debug(f"API Request - Model: {agent.model}, Messages: {len(messages)}, Tools: {len(agent.tools) if agent.tools else 0}")
            logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
            logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
        
        api_start_time = time.time()
        retry_count = 0
        max_retries = agent._api_max_retries
        primary_recovery_attempted = False
        max_compression_attempts = 3
        codex_auth_retry_attempted=False
        anthropic_auth_retry_attempted=False
        nous_auth_retry_attempted=False
        copilot_auth_retry_attempted=False
        thinking_sig_retry_attempted = False
        image_shrink_retry_attempted = False
        oauth_1m_beta_retry_attempted = False
        llama_cpp_grammar_retry_attempted = False
        has_retried_429 = False
        restart_with_compressed_messages = False
        restart_with_length_continuation = False

        finish_reason = "stop"
        response = None  # Guard against UnboundLocalError if all retries fail
        api_kwargs = None  # Guard against UnboundLocalError in except handler

        while retry_count < max_retries:
            # ── Nous Portal rate limit guard ──────────────────────
            # If another session already recorded that Nous is rate-
            # limited, skip the API call entirely.  Each attempt
            # (including SDK-level retries) counts against RPH and
            # deepens the rate limit hole.
            if agent.provider == "nous":
                try:
                    from agent.nous_rate_guard import (
                        nous_rate_limit_remaining,
                        format_remaining as _fmt_nous_remaining,
                    )
                    _nous_remaining = nous_rate_limit_remaining()
                    if _nous_remaining is not None and _nous_remaining > 0:
                        _nous_msg = (
                            f"Nous Portal rate limit active — "
                            f"resets in {_fmt_nous_remaining(_nous_remaining)}."
                        )
                        agent._vprint(
                            f"{agent.log_prefix}⏳ {_nous_msg} Trying fallback...",
                            force=True,
                        )
                        agent._emit_status(f"⏳ {_nous_msg}")
                        if agent._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        # No fallback available — return with clear message
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": (
                                f"⏳ {_nous_msg}\n\n"
                                "No fallback provider available. "
                                "Try again after the reset, or add a "
                                "fallback provider in config.yaml."
                            ),
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": _nous_msg,
                        }
                except ImportError:
                    pass
                except Exception:
                    pass  # Never let rate guard break the agent loop

            try:
                agent._reset_stream_delivery_tracking()
                api_kwargs = agent._build_api_kwargs(api_messages)
                if agent._force_ascii_payload:
                    _sanitize_structure_non_ascii(api_kwargs)
                if agent.api_mode == "codex_responses":
                    api_kwargs = agent._get_transport().preflight_kwargs(api_kwargs, allow_stream=False)

                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    request_messages = api_kwargs.get("messages")
                    if not isinstance(request_messages, list):
                        request_messages = api_kwargs.get("input")
                    if not isinstance(request_messages, list):
                        request_messages = api_messages
                    # Shallow-copy the outer list so plugins that retain the
                    # reference for async snapshotting don't observe later
                    # mutations of api_messages.  The inner dicts are not
                    # mutated by the agent loop, so a shallow copy is
                    # sufficient; a deepcopy would walk every tool result
                    # and base64 image on every API call.
                    _invoke_hook(
                        "pre_api_request",
                        task_id=effective_task_id,
                        session_id=agent.session_id or "",
                        user_message=original_user_message,
                        conversation_history=list(messages),
                        platform=agent.platform or "",
                        model=agent.model,
                        provider=agent.provider,
                        base_url=agent.base_url,
                        api_mode=agent.api_mode,
                        api_call_count=api_call_count,
                        request_messages=list(request_messages) if isinstance(request_messages, list) else [],
                        message_count=len(api_messages),
                        tool_count=len(agent.tools or []),
                        approx_input_tokens=approx_tokens,
                        request_char_count=total_chars,
                        max_tokens=agent.max_tokens,
                    )
                except Exception:
                    pass

                if env_var_enabled("HERMES_DUMP_REQUESTS"):
                    agent._dump_api_request_debug(api_kwargs, reason="preflight")

                # Always prefer the streaming path — even without stream
                # consumers.  Streaming gives us fine-grained health
                # checking (90s stale-stream detection, 60s read timeout)
                # that the non-streaming path lacks.  Without this,
                # subagents and other quiet-mode callers can hang
                # indefinitely when the provider keeps the connection
                # alive with SSE pings but never delivers a response.
                # The streaming path is a no-op for callbacks when no
                # consumers are registered, and falls back to non-
                # streaming automatically if the provider doesn't
                # support it.
                def _stop_spinner():
                    nonlocal thinking_spinner
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if agent.thinking_callback:
                        agent.thinking_callback("")

                _use_streaming = True
                # Provider signaled "stream not supported" on a previous
                # attempt — switch to non-streaming for the rest of this
                # session instead of re-failing every retry.
                if getattr(agent, "_disable_streaming", False):
                    _use_streaming = False
                # CopilotACPClient communicates via subprocess stdio and
                # returns a plain SimpleNamespace — not an iterable
                # stream.  Mirror the ACP exclusion used for Responses
                # API upgrade (lines ~1083-1085).
                elif (
                    agent.provider == "copilot-acp"
                    or str(agent.base_url or "").lower().startswith("acp://copilot")
                    or str(agent.base_url or "").lower().startswith("acp+tcp://")
                ):
                    _use_streaming = False
                elif not agent._has_stream_consumers():
                    # No display/TTS consumer. Still prefer streaming for
                    # health checking, but skip for Mock clients in tests
                    # (mocks return SimpleNamespace, not stream iterators).
                    from unittest.mock import Mock
                    if isinstance(getattr(agent, "client", None), Mock):
                        _use_streaming = False

                if _use_streaming:
                    response = agent._interruptible_streaming_api_call(
                        api_kwargs, on_first_delta=_stop_spinner
                    )
                else:
                    response = agent._interruptible_api_call(api_kwargs)
                
                api_duration = time.time() - api_start_time
                
                # Stop thinking spinner silently -- the response box or tool
                # execution messages that follow are more informative.
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")
                
                if not agent.quiet_mode:
                    agent._vprint(f"{agent.log_prefix}⏱️  API call completed in {api_duration:.2f}s")
                
                if agent.verbose_logging:
                    # Log response with provider info if available
                    resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                    logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                
                # Validate response shape before proceeding
                response_invalid = False
                error_details = []
                if agent.api_mode == "codex_responses":
                    _ct_v = agent._get_transport()
                    if not _ct_v.validate_response(response):
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        else:
                            # Provider returned a terminal failure (e.g. quota exhaustion).
                            # Treat as invalid so the fallback chain is triggered instead of
                            # letting the error bubble up outside the retry/fallback loop.
                            _codex_resp_status = str(getattr(response, "status", "") or "").strip().lower()
                            if _codex_resp_status in {"failed", "cancelled"}:
                                _codex_error_obj = getattr(response, "error", None)
                                _codex_error_msg = (
                                    _codex_error_obj.get("message") if isinstance(_codex_error_obj, dict)
                                    else str(_codex_error_obj) if _codex_error_obj
                                    else f"Responses API returned status '{_codex_resp_status}'"
                                )
                                logging.warning(
                                    "Codex response status='%s' (error=%s). Routing to fallback. %s",
                                    _codex_resp_status, _codex_error_msg,
                                    agent._client_log_context(),
                                )
                                response_invalid = True
                                error_details.append(f"response.status={_codex_resp_status}: {_codex_error_msg}")
                            else:
                                # output_text fallback: stream backfill may have failed
                                # but normalize can still recover from output_text
                                _out_text = getattr(response, "output_text", None)
                                _out_text_stripped = _out_text.strip() if isinstance(_out_text, str) else ""
                                if _out_text_stripped:
                                    logger.debug(
                                        "Codex response.output is empty but output_text is present "
                                        "(%d chars); deferring to normalization.",
                                        len(_out_text_stripped),
                                    )
                                else:
                                    _resp_status = getattr(response, "status", None)
                                    _resp_incomplete = getattr(response, "incomplete_details", None)
                                    logger.warning(
                                        "Codex response.output is empty after stream backfill "
                                        "(status=%s, incomplete_details=%s, model=%s). %s",
                                        _resp_status, _resp_incomplete,
                                        getattr(response, "model", None),
                                        f"api_mode={agent.api_mode} provider={agent.provider}",
                                    )
                                    response_invalid = True
                                    error_details.append("response.output is empty")
                elif agent.api_mode == "anthropic_messages":
                    _tv = agent._get_transport()
                    if not _tv.validate_response(response):
                        response_invalid = True
                        if response is None:
                            error_details.append("response is None")
                        else:
                            error_details.append("response.content invalid (not a non-empty list)")
                elif agent.api_mode == "bedrock_converse":
                    _btv = agent._get_transport()
                    if not _btv.validate_response(response):
                        response_invalid = True
                        if response is None:
                            error_details.append("response is None")
                        else:
                            error_details.append("Bedrock response invalid (no output or choices)")
                else:
                    _ctv = agent._get_transport()
                    if not _ctv.validate_response(response):
                        response_invalid = True
                        if response is None:
                            error_details.append("response is None")
                        elif not hasattr(response, 'choices'):
                            error_details.append("response has no 'choices' attribute")
                        elif response.choices is None:
                            error_details.append("response.choices is None")
                        else:
                            error_details.append("response.choices is empty")

                if response_invalid:
                    # Stop spinner before printing error messages
                    if thinking_spinner:
                        thinking_spinner.stop("(´;ω;`) oops, retrying...")
                        thinking_spinner = None
                    if agent.thinking_callback:
                        agent.thinking_callback("")
                    
                    # Invalid response — could be rate limiting, provider timeout,
                    # upstream server error, or malformed response.
                    retry_count += 1
                    
                    # Eager fallback: empty/malformed responses are a common
                    # rate-limit symptom.  Switch to fallback immediately
                    # rather than retrying with extended backoff.
                    if agent._fallback_index < len(agent._fallback_chain):
                        agent._emit_status("⚠️ Empty/malformed response — switching to fallback...")
                    if agent._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        primary_recovery_attempted = False
                        continue

                    # Check for error field in response (some providers include this)
                    error_msg = "Unknown"
                    provider_name = "Unknown"
                    if response and hasattr(response, 'error') and response.error:
                        error_msg = str(response.error)
                        # Try to extract provider from error metadata
                        if hasattr(response.error, 'metadata') and response.error.metadata:
                            provider_name = response.error.metadata.get('provider_name', 'Unknown')
                    elif response and hasattr(response, 'message') and response.message:
                        error_msg = str(response.message)
                    
                    # Try to get provider from model field (OpenRouter often returns actual model used)
                    if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
                        provider_name = f"model={response.model}"
                    
                    # Check for x-openrouter-provider or similar metadata
                    if provider_name == "Unknown" and response:
                        # Log all response attributes for debugging
                        resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
                        if agent.verbose_logging:
                            logging.debug(f"Response attributes for invalid response: {resp_attrs}")
                    
                    # Extract error code from response for contextual diagnostics
                    _resp_error_code = None
                    if response and hasattr(response, 'error') and response.error:
                        _code_raw = getattr(response.error, 'code', None)
                        if _code_raw is None and isinstance(response.error, dict):
                            _code_raw = response.error.get('code')
                        if _code_raw is not None:
                            try:
                                _resp_error_code = int(_code_raw)
                            except (TypeError, ValueError):
                                pass

                    # Build a human-readable failure hint from the error code
                    # and response time, instead of always assuming rate limiting.
                    if _resp_error_code == 524:
                        _failure_hint = f"upstream provider timed out (Cloudflare 524, {api_duration:.0f}s)"
                    elif _resp_error_code == 504:
                        _failure_hint = f"upstream gateway timeout (504, {api_duration:.0f}s)"
                    elif _resp_error_code == 429:
                        _failure_hint = f"rate limited by upstream provider (429)"
                    elif _resp_error_code in {500, 502}:
                        _failure_hint = f"upstream server error ({_resp_error_code}, {api_duration:.0f}s)"
                    elif _resp_error_code in {503, 529}:
                        _failure_hint = f"upstream provider overloaded ({_resp_error_code})"
                    elif _resp_error_code is not None:
                        _failure_hint = f"upstream error (code {_resp_error_code}, {api_duration:.0f}s)"
                    elif api_duration < 10:
                        _failure_hint = f"fast response ({api_duration:.1f}s) — likely rate limited"
                    elif api_duration > 60:
                        _failure_hint = f"slow response ({api_duration:.0f}s) — likely upstream timeout"
                    else:
                        _failure_hint = f"response time {api_duration:.1f}s"

                    agent._vprint(f"{agent.log_prefix}⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}", force=True)
                    agent._vprint(f"{agent.log_prefix}   🏢 Provider: {provider_name}", force=True)
                    cleaned_provider_error = agent._clean_error_message(error_msg)
                    agent._vprint(f"{agent.log_prefix}   📝 Provider message: {cleaned_provider_error}", force=True)
                    agent._vprint(f"{agent.log_prefix}   ⏱️  {_failure_hint}", force=True)
                    
                    if retry_count >= max_retries:
                        # Try fallback before giving up
                        agent._emit_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
                        if agent._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        agent._emit_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                        logging.error(f"{agent.log_prefix}Invalid API response after {max_retries} retries.")
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Invalid API response after {max_retries} retries: {_failure_hint}",
                            "failed": True  # Mark as failure for filtering
                        }
                    
                    # Backoff before retry — jittered exponential: 5s base, 120s cap
                    wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
                    agent._vprint(f"{agent.log_prefix}⏳ Retrying in {wait_time:.1f}s ({_failure_hint})...", force=True)
                    logging.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")
                    
                    # Sleep in small increments to stay responsive to interrupts
                    sleep_end = time.time() + wait_time
                    _backoff_touch_counter = 0
                    while time.time() < sleep_end:
                        if agent._interrupt_requested:
                            agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                            agent._persist_session(messages, conversation_history)
                            agent.clear_interrupt()
                            return {
                                "final_response": f"Operation interrupted during retry ({_failure_hint}, attempt {retry_count}/{max_retries}).",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)
                        # Touch activity every ~30s so the gateway's inactivity
                        # monitor knows we're alive during backoff waits.
                        _backoff_touch_counter += 1
                        if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                            agent._touch_activity(
                                f"retry backoff ({retry_count}/{max_retries}), "
                                f"{int(sleep_end - time.time())}s remaining"
                            )
                    continue  # Retry the API call

                # Check finish_reason before proceeding
                if agent.api_mode == "codex_responses":
                    status = getattr(response, "status", None)
                    incomplete_details = getattr(response, "incomplete_details", None)
                    incomplete_reason = None
                    if isinstance(incomplete_details, dict):
                        incomplete_reason = incomplete_details.get("reason")
                    else:
                        incomplete_reason = getattr(incomplete_details, "reason", None)
                    if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                        finish_reason = "length"
                    else:
                        finish_reason = "stop"
                elif agent.api_mode == "anthropic_messages":
                    _tfr = agent._get_transport()
                    finish_reason = _tfr.map_finish_reason(response.stop_reason)
                elif agent.api_mode == "bedrock_converse":
                    # Bedrock response already normalized at dispatch — use transport
                    _bt_fr = agent._get_transport()
                    _bedrock_result = _bt_fr.normalize_response(response)
                    finish_reason = _bedrock_result.finish_reason
                else:
                    _cc_fr = agent._get_transport()
                    _finish_result = _cc_fr.normalize_response(response)
                    finish_reason = _finish_result.finish_reason
                    assistant_message = _finish_result
                    if agent._should_treat_stop_as_truncated(
                        finish_reason,
                        assistant_message,
                        messages,
                    ):
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Treating suspicious Ollama/GLM stop response as truncated",
                            force=True,
                        )
                        finish_reason = "length"

                if finish_reason == "length":
                    agent._vprint(f"{agent.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens", force=True)

                    # Normalize the truncated response to a single OpenAI-style
                    # message shape so text-continuation and tool-call retry
                    # work uniformly across chat_completions, bedrock_converse,
                    # and anthropic_messages.  For Anthropic we use the same
                    # adapter the agent loop already relies on so the rebuilt
                    # interim assistant message is byte-identical to what
                    # would have been appended in the non-truncated path.
                    _trunc_msg = None
                    _trunc_transport = agent._get_transport()
                    if agent.api_mode == "anthropic_messages":
                        _trunc_result = _trunc_transport.normalize_response(
                            response, strip_tool_prefix=agent._is_anthropic_oauth
                        )
                    else:
                        _trunc_result = _trunc_transport.normalize_response(response)
                    _trunc_msg = _trunc_result

                    _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
                    _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False

                    # ── Detect thinking-budget exhaustion ──────────────
                    # When the model spends ALL output tokens on reasoning
                    # and has none left for the response, continuation
                    # retries are pointless.  Detect this early and give a
                    # targeted error instead of wasting 3 API calls.
                    # A response is "thinking exhausted" only when the model
                    # actually produced reasoning blocks but no visible text after
                    # them.  Models that do not use <think> tags (e.g. GLM-4.7 on
                    # NVIDIA Build, minimax) may return content=None or an empty
                    # string for unrelated reasons — treat those as normal
                    # truncations that deserve continuation retries, not as
                    # thinking-budget exhaustion.
                    _has_think_tags = bool(
                        _trunc_content and re.search(
                            r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>',
                            _trunc_content,
                            re.IGNORECASE,
                        )
                    )
                    _thinking_exhausted = (
                        not _trunc_has_tool_calls
                        and _has_think_tags
                        and (
                            (_trunc_content is not None and not agent._has_content_after_think_block(_trunc_content))
                            or _trunc_content is None
                        )
                    )

                    if _thinking_exhausted:
                        _exhaust_error = (
                            "Model used all output tokens on reasoning with none left "
                            "for the response. Try lowering reasoning effort or "
                            "increasing max_tokens."
                        )
                        agent._vprint(
                            f"{agent.log_prefix}💭 Reasoning exhausted the output token budget — "
                            f"no visible response was produced.",
                            force=True,
                        )
                        # Return a user-friendly message as the response so
                        # CLI (response box) and gateway (chat message) both
                        # display it naturally instead of a suppressed error.
                        _exhaust_response = (
                            "⚠️ **Thinking Budget Exhausted**\n\n"
                            "The model used all its output tokens on reasoning "
                            "and had none left for the actual response.\n\n"
                            "To fix this:\n"
                            "→ Lower reasoning effort: `/thinkon low` or `/thinkon minimal`\n"
                            "→ Or switch to a larger/non-reasoning model with `/model`"
                        )
                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": _exhaust_response,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": _exhaust_error,
                        }

                    if agent.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
                        assistant_message = _trunc_msg
                        if assistant_message is not None and not _trunc_has_tool_calls:
                            length_continue_retries += 1
                            interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
                            messages.append(interim_msg)
                            if assistant_message.content:
                                truncated_response_parts.append(assistant_message.content)

                            if length_continue_retries < 3:
                                agent._vprint(
                                    f"{agent.log_prefix}↻ Requesting continuation "
                                    f"({length_continue_retries}/3)..."
                                )
                                continue_msg = {
                                    "role": "user",
                                    "content": (
                                        "[System: Your previous response was truncated by the output "
                                        "length limit. Continue exactly where you left off. Do not "
                                        "restart or repeat prior text. Finish the answer directly.]"
                                    ),
                                }
                                messages.append(continue_msg)
                                agent._session_messages = messages
                                agent._save_session_log(messages)
                                restart_with_length_continuation = True
                                break

                            partial_response = agent._strip_think_blocks("".join(truncated_response_parts)).strip()
                            agent._cleanup_task_resources(effective_task_id)
                            agent._persist_session(messages, conversation_history)
                            return {
                                "final_response": partial_response or None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response remained truncated after 3 continuation attempts",
                            }

                    if agent.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
                        assistant_message = _trunc_msg
                        if assistant_message is not None and _trunc_has_tool_calls:
                            if truncated_tool_call_retries < 1:
                                truncated_tool_call_retries += 1
                                agent._vprint(
                                    f"{agent.log_prefix}⚠️  Truncated tool call detected — retrying API call...",
                                    force=True,
                                )
                                # Don't append the broken response to messages;
                                # just re-run the same API call from the current
                                # message state, giving the model another chance.
                                continue
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  Truncated tool call response detected again — refusing to execute incomplete tool arguments.",
                                force=True,
                            )
                            agent._cleanup_task_resources(effective_task_id)
                            agent._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit",
                            }

                    # If we have prior messages, roll back to last complete state
                    if len(messages) > 1:
                        agent._vprint(f"{agent.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                        rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)

                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)

                        return {
                            "final_response": None,
                            "messages": rolled_back_messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Response truncated due to output length limit"
                        }
                    else:
                        # First message was truncated - mark as failed
                        agent._vprint(f"{agent.log_prefix}❌ First response truncated - cannot recover", force=True)
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": "First response truncated due to output length limit"
                        }
                
                # Track actual token usage from response for context management
                if hasattr(response, 'usage') and response.usage:
                    canonical_usage = normalize_usage(
                        response.usage,
                        provider=agent.provider,
                        api_mode=agent.api_mode,
                    )
                    prompt_tokens = canonical_usage.prompt_tokens
                    completion_tokens = canonical_usage.output_tokens
                    total_tokens = canonical_usage.total_tokens
                    usage_dict = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                    agent.context_compressor.update_from_response(usage_dict)

                    # Cache discovered context length after successful call.
                    # Only persist limits confirmed by the provider (parsed
                    # from the error message), not guessed probe tiers.
                    if getattr(agent.context_compressor, "_context_probed", False):
                        ctx = agent.context_compressor.context_length
                        if getattr(agent.context_compressor, "_context_probe_persistable", False):
                            save_context_length(agent.model, agent.base_url, ctx)
                            agent._safe_print(f"{agent.log_prefix}💾 Cached context length: {ctx:,} tokens for {agent.model}")
                        agent.context_compressor._context_probed = False
                        agent.context_compressor._context_probe_persistable = False

                    agent.session_prompt_tokens += prompt_tokens
                    agent.session_completion_tokens += completion_tokens
                    agent.session_total_tokens += total_tokens
                    agent.session_api_calls += 1
                    agent.session_input_tokens += canonical_usage.input_tokens
                    agent.session_output_tokens += canonical_usage.output_tokens
                    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
                    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
                    agent.session_reasoning_tokens += canonical_usage.reasoning_tokens

                    # Log API call details for debugging/observability
                    _cache_pct = ""
                    if canonical_usage.cache_read_tokens and prompt_tokens:
                        _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
                    logger.info(
                        "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s",
                        agent.session_api_calls, agent.model, agent.provider or "unknown",
                        prompt_tokens, completion_tokens, total_tokens,
                        api_duration, _cache_pct,
                    )

                    cost_result = estimate_usage_cost(
                        agent.model,
                        canonical_usage,
                        provider=agent.provider,
                        base_url=agent.base_url,
                        api_key=getattr(agent, "api_key", ""),
                    )
                    if cost_result.amount_usd is not None:
                        agent.session_estimated_cost_usd += float(cost_result.amount_usd)
                    agent.session_cost_status = cost_result.status
                    agent.session_cost_source = cost_result.source

                    # Persist token counts to session DB for /insights.
                    # Do this for every platform with a session_id so non-CLI
                    # sessions (gateway, cron, delegated runs) cannot lose
                    # token/accounting data if a higher-level persistence path
                    # is skipped or fails. Gateway/session-store writes use
                    # absolute totals, so they safely overwrite these per-call
                    # deltas instead of double-counting them.
                    if agent._session_db and agent.session_id:
                        try:
                            # Ensure the session row exists before attempting UPDATE.
                            # Under concurrent load (cron/kanban), the initial
                            # _ensure_db_session() may have failed due to SQLite
                            # locking.  Retry here so per-call token deltas are
                            # not silently lost (UPDATE on a non-existent row
                            # affects 0 rows without error).
                            if not agent._session_db_created:
                                agent._ensure_db_session()
                            agent._session_db.update_token_counts(
                                agent.session_id,
                                input_tokens=canonical_usage.input_tokens,
                                output_tokens=canonical_usage.output_tokens,
                                cache_read_tokens=canonical_usage.cache_read_tokens,
                                cache_write_tokens=canonical_usage.cache_write_tokens,
                                reasoning_tokens=canonical_usage.reasoning_tokens,
                                estimated_cost_usd=float(cost_result.amount_usd)
                                if cost_result.amount_usd is not None else None,
                                cost_status=cost_result.status,
                                cost_source=cost_result.source,
                                billing_provider=agent.provider,
                                billing_base_url=agent.base_url,
                                billing_mode="subscription_included"
                                if cost_result.status == "included" else None,
                                model=agent.model,
                                api_call_count=1,
                            )
                        except Exception as e:
                            # Log token persistence failures so they're
                            # visible in agent.log — silent loss here is
                            # the root cause of undercounted analytics.
                            logger.debug(
                                "Token persistence failed (session=%s, tokens=%d): %s",
                                agent.session_id, total_tokens, e,
                            )
                    
                    if agent.verbose_logging:
                        logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")
                    
                    # Surface cache hit stats for any provider that reports
                    # them — not just those where we inject cache_control
                    # markers.  OpenAI/Kimi/DeepSeek/Qwen all do automatic
                    # server-side prefix caching and return
                    # ``prompt_tokens_details.cached_tokens``; users
                    # previously could not see their cache % because this
                    # line was gated on ``_use_prompt_caching``, which is
                    # only True for Anthropic-style marker injection.
                    # ``canonical_usage`` is already normalised from all
                    # three API shapes (Anthropic / Codex / OpenAI-chat)
                    # so we can rely on its values directly.
                    cached = canonical_usage.cache_read_tokens
                    written = canonical_usage.cache_write_tokens
                    prompt = usage_dict["prompt_tokens"]
                    if (cached or written) and not agent.quiet_mode:
                        hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                        agent._vprint(
                            f"{agent.log_prefix}   💾 Cache: "
                            f"{cached:,}/{prompt:,} tokens "
                            f"({hit_pct:.0f}% hit, {written:,} written)"
                        )
                
                has_retried_429 = False  # Reset on success
                # Clear Nous rate limit state on successful request —
                # proves the limit has reset and other sessions can
                # resume hitting Nous.
                if agent.provider == "nous":
                    try:
                        from agent.nous_rate_guard import clear_nous_rate_limit
                        clear_nous_rate_limit()
                    except Exception:
                        pass
                agent._touch_activity(f"API call #{api_call_count} completed")
                break  # Success, exit retry loop

            except InterruptedError:
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")
                api_elapsed = time.time() - api_start_time
                agent._vprint(f"{agent.log_prefix}⚡ Interrupted during API call.", force=True)
                agent._persist_session(messages, conversation_history)
                interrupted = True
                final_response = f"Operation interrupted: waiting for model response ({api_elapsed:.1f}s elapsed)."
                break

            except Exception as api_error:
                # Stop spinner before printing error messages
                if thinking_spinner:
                    thinking_spinner.stop("(╥_╥) error, retrying...")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")

                # -----------------------------------------------------------
                # UnicodeEncodeError recovery.  Two common causes:
                #   1. Lone surrogates (U+D800..U+DFFF) from clipboard paste
                #      (Google Docs, rich-text editors) — sanitize and retry.
                #   2. ASCII codec on systems with LANG=C or non-UTF-8 locale
                #      (e.g. Chromebooks) — any non-ASCII character fails.
                #      Detect via the error message mentioning 'ascii' codec.
                # We sanitize messages in-place and may retry twice:
                # first to strip surrogates, then once more for pure
                # ASCII-only locale sanitization if needed.
                # -----------------------------------------------------------
                if isinstance(api_error, UnicodeEncodeError) and getattr(agent, '_unicode_sanitization_passes', 0) < 2:
                    _err_str = str(api_error).lower()
                    _is_ascii_codec = "'ascii'" in _err_str or "ascii" in _err_str
                    # Detect surrogate errors — utf-8 codec refusing to
                    # encode U+D800..U+DFFF.  The error text is:
                    #   "'utf-8' codec can't encode characters in position
                    #    N-M: surrogates not allowed"
                    _is_surrogate_error = (
                        "surrogate" in _err_str
                        or ("'utf-8'" in _err_str and not _is_ascii_codec)
                    )
                    # Sanitize surrogates from both the canonical `messages`
                    # list AND `api_messages` (the API-copy, which may carry
                    # `reasoning_content`/`reasoning_details` transformed
                    # from `reasoning` — fields the canonical list doesn't
                    # have directly).  Also clean `api_kwargs` if built and
                    # `prefill_messages` if present.  Mirrors the ASCII
                    # codec recovery below.
                    _surrogates_found = _sanitize_messages_surrogates(messages)
                    if isinstance(api_messages, list):
                        if _sanitize_messages_surrogates(api_messages):
                            _surrogates_found = True
                    if isinstance(api_kwargs, dict):
                        if _sanitize_structure_surrogates(api_kwargs):
                            _surrogates_found = True
                    if isinstance(getattr(agent, "prefill_messages", None), list):
                        if _sanitize_messages_surrogates(agent.prefill_messages):
                            _surrogates_found = True
                    # Gate the retry on the error type, not on whether we
                    # found anything — _force_ascii_payload / the extended
                    # surrogate walker above cover all known paths, but a
                    # new transformed field could still slip through.  If
                    # the error was a surrogate encode failure, always let
                    # the retry run; the proactive sanitizer at line ~8781
                    # runs again on the next iteration.  Bounded by
                    # _unicode_sanitization_passes < 2 (outer guard).
                    if _surrogates_found or _is_surrogate_error:
                        agent._unicode_sanitization_passes += 1
                        if _surrogates_found:
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  Stripped invalid surrogate characters from messages. Retrying...",
                                force=True,
                            )
                        else:
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  Surrogate encoding error — retrying after full-payload sanitization...",
                                force=True,
                            )
                        continue
                    if _is_ascii_codec:
                        agent._force_ascii_payload = True
                        # ASCII codec: the system encoding can't handle
                        # non-ASCII characters at all. Sanitize all
                        # non-ASCII content from messages/tool schemas and retry.
                        # Sanitize both the canonical `messages` list and
                        # `api_messages` (the API-copy built before the retry
                        # loop, which may contain extra fields like
                        # reasoning_content that are not in `messages`).
                        _messages_sanitized = _sanitize_messages_non_ascii(messages)
                        if isinstance(api_messages, list):
                            _sanitize_messages_non_ascii(api_messages)
                        # Also sanitize the last api_kwargs if already built,
                        # so a leftover non-ASCII value in a transformed field
                        # (e.g. extra_body, reasoning_content) doesn't survive
                        # into the next attempt via _build_api_kwargs cache paths.
                        if isinstance(api_kwargs, dict):
                            _sanitize_structure_non_ascii(api_kwargs)
                        _prefill_sanitized = False
                        if isinstance(getattr(agent, "prefill_messages", None), list):
                            _prefill_sanitized = _sanitize_messages_non_ascii(agent.prefill_messages)

                        _tools_sanitized = False
                        if isinstance(getattr(agent, "tools", None), list):
                            _tools_sanitized = _sanitize_tools_non_ascii(agent.tools)

                        _system_sanitized = False
                        if isinstance(active_system_prompt, str):
                            _sanitized_system = _strip_non_ascii(active_system_prompt)
                            if _sanitized_system != active_system_prompt:
                                active_system_prompt = _sanitized_system
                                agent._cached_system_prompt = _sanitized_system
                                _system_sanitized = True
                        if isinstance(getattr(agent, "ephemeral_system_prompt", None), str):
                            _sanitized_ephemeral = _strip_non_ascii(agent.ephemeral_system_prompt)
                            if _sanitized_ephemeral != agent.ephemeral_system_prompt:
                                agent.ephemeral_system_prompt = _sanitized_ephemeral
                                _system_sanitized = True

                        _headers_sanitized = False
                        _default_headers = (
                            agent._client_kwargs.get("default_headers")
                            if isinstance(getattr(agent, "_client_kwargs", None), dict)
                            else None
                        )
                        if isinstance(_default_headers, dict):
                            _headers_sanitized = _sanitize_structure_non_ascii(_default_headers)

                        # Sanitize the API key — non-ASCII characters in
                        # credentials (e.g. ʋ instead of v from a bad
                        # copy-paste) cause httpx to fail when encoding
                        # the Authorization header as ASCII.  This is the
                        # most common cause of persistent UnicodeEncodeError
                        # that survives message/tool sanitization (#6843).
                        _credential_sanitized = False
                        _raw_key = getattr(agent, "api_key", None) or ""
                        if _raw_key:
                            _clean_key = _strip_non_ascii(_raw_key)
                            if _clean_key != _raw_key:
                                agent.api_key = _clean_key
                                if isinstance(getattr(agent, "_client_kwargs", None), dict):
                                    agent._client_kwargs["api_key"] = _clean_key
                                # Also update the live client — it holds its
                                # own copy of api_key which auth_headers reads
                                # dynamically on every request.
                                if getattr(agent, "client", None) is not None and hasattr(agent.client, "api_key"):
                                    agent.client.api_key = _clean_key
                                _credential_sanitized = True
                                agent._vprint(
                                    f"{agent.log_prefix}⚠️  API key contained non-ASCII characters "
                                    f"(bad copy-paste?) — stripped them. If auth fails, "
                                    f"re-copy the key from your provider's dashboard.",
                                    force=True,
                                )

                        # Always retry on ASCII codec detection —
                        # _force_ascii_payload guarantees the full
                        # api_kwargs payload is sanitized on the
                        # next iteration (line ~8475).  Even when
                        # per-component checks above find nothing
                        # (e.g. non-ASCII only in api_messages'
                        # reasoning_content), the flag catches it.
                        # Bounded by _unicode_sanitization_passes < 2.
                        agent._unicode_sanitization_passes += 1
                        _any_sanitized = (
                            _messages_sanitized
                            or _prefill_sanitized
                            or _tools_sanitized
                            or _system_sanitized
                            or _headers_sanitized
                            or _credential_sanitized
                        )
                        if _any_sanitized:
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  System encoding is ASCII — stripped non-ASCII characters from request payload. Retrying...",
                                force=True,
                            )
                        else:
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  System encoding is ASCII — enabling full-payload sanitization for retry...",
                                force=True,
                            )
                        continue

                # ── Image-rejection recovery ──────────────────────────────
                # Some providers (mlx-lm, text-only endpoints, text-only
                # fallbacks on multimodal models) reject any message that
                # contains image_url content with a 4xx error like
                # "Only 'text' content type is supported."  On first hit,
                # strip all images from the message list, mark the session
                # as vision-unsupported, and retry with text only.
                #
                # Detection is best-effort English phrase matching — a
                # locale-translated or heavily-reworded upstream error
                # will bypass this guard and fall through to the normal
                # error handler.  Expand the phrase list when new
                # provider wordings are observed in the wild.
                _err_body = ""
                try:
                    _err_body = str(getattr(api_error, "body", None) or
                                    getattr(api_error, "message", None) or
                                    str(api_error))
                except Exception:
                    pass
                _err_status = getattr(api_error, "status_code", None)
                _IMAGE_REJECTION_PHRASES = (
                    "only 'text' content type is supported",
                    "only text content type is supported",
                    "image_url is not supported",
                    "image content is not supported",
                    "multimodal is not supported",
                    "multimodal content is not supported",
                    "multimodal input is not supported",
                    "vision is not supported",
                    "vision input is not supported",
                    "does not support images",
                    "does not support image input",
                    "does not support multimodal",
                    "does not support vision",
                    "model does not support image",
                    # ChatGPT-account Codex backend
                    # (https://chatgpt.com/backend-api/codex) rejects
                    # data:image/...base64 URLs in input_image fields
                    # with HTTP 400 "Invalid 'input[N].content[K].image_url'.
                    # Expected a valid URL, but got a value with an
                    # invalid format." The OpenAI Responses API on the
                    # public endpoint accepts data URLs, but the
                    # ChatGPT-account variant does not. Without this
                    # phrase the agent cascaded into compression /
                    # context-too-large recovery instead of just
                    # stripping the images. Match is narrow on
                    # purpose — keyed on the field-path apostrophe so
                    # we don't false-trip on other URL validation
                    # errors. (issue #23570)
                    "image_url'. expected",
                    # DeepSeek's OpenAI-compatible API reports text-only
                    # request-body variants as:
                    # "unknown variant `image_url`, expected `text`".
                    "unknown variant `image_url`, expected `text`",
                    "unknown variant image_url, expected text",
                )
                _err_lower = _err_body.lower()
                _looks_like_image_rejection = any(
                    p in _err_lower for p in _IMAGE_REJECTION_PHRASES
                )
                # 4xx-only gate: never interpret 5xx/timeout as "server
                # said no to images" — those are transient and must
                # route to the normal retry path.
                _status_ok = _err_status is None or (400 <= int(_err_status) < 500)
                if (
                    getattr(agent, "_vision_supported", True)
                    and _looks_like_image_rejection
                    and _status_ok
                ):
                    agent._vision_supported = False
                    _imgs_removed = _strip_images_from_messages(messages)
                    if isinstance(api_messages, list):
                        _strip_images_from_messages(api_messages)
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Server rejected image content — "
                        f"switching to text-only mode for this session"
                        + (". Stripped images from history and retrying." if _imgs_removed else "."),
                        force=True,
                    )
                    continue

                status_code = getattr(api_error, "status_code", None)
                error_context = agent._extract_api_error_context(api_error)

                # ── Classify the error for structured recovery decisions ──
                _compressor = getattr(agent, "context_compressor", None)
                _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
                classified = classify_api_error(
                    api_error,
                    provider=getattr(agent, "provider", "") or "",
                    model=getattr(agent, "model", "") or "",
                    approx_tokens=approx_tokens,
                    context_length=_ctx_len,
                    num_messages=len(api_messages) if api_messages else 0,
                )
                logger.debug(
                    "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
                    classified.reason.value, classified.status_code,
                    classified.retryable, classified.should_compress,
                    classified.should_rotate_credential, classified.should_fallback,
                )

                recovered_with_pool, has_retried_429 = agent._recover_with_credential_pool(
                    status_code=status_code,
                    has_retried_429=has_retried_429,
                    classified_reason=classified.reason,
                    error_context=error_context,
                )
                if recovered_with_pool:
                    continue

                # Image-too-large recovery: shrink oversized native image
                # parts in-place and retry once.  Triggered by Anthropic's
                # per-image 5 MB ceiling (400 with "image exceeds 5 MB
                # maximum") or any other provider that complains about
                # image size.  If shrink fails or a second attempt still
                # fails, fall through to normal error handling.
                if (
                    classified.reason == FailoverReason.image_too_large
                    and not image_shrink_retry_attempted
                ):
                    image_shrink_retry_attempted = True
                    if agent._try_shrink_image_parts_in_messages(api_messages):
                        agent._vprint(
                            f"{agent.log_prefix}📐 Image(s) exceeded provider size limit — "
                            f"shrank and retrying...",
                            force=True,
                        )
                        continue
                    else:
                        logger.info(
                            "image-shrink recovery: no data-URL image parts found "
                            "or shrink didn't reduce size; surfacing original error."
                        )

                # Anthropic OAuth subscription rejected the 1M-context beta
                # header ("long context beta is not yet available for this
                # subscription"). Disable the beta for the rest of this
                # session, rebuild the client, and retry once.  1M-capable
                # subscriptions never hit this branch — they accept the
                # beta and keep full 1M context.  See PR #17680 for the
                # original report (we chose reactive recovery over the
                # proposed unconditional omit so capable subscriptions
                # don't silently lose the capability).
                if (
                    classified.reason == FailoverReason.oauth_long_context_beta_forbidden
                    and agent.api_mode == "anthropic_messages"
                    and agent._is_anthropic_oauth
                    and not oauth_1m_beta_retry_attempted
                ):
                    oauth_1m_beta_retry_attempted = True
                    if not getattr(agent, "_oauth_1m_beta_disabled", False):
                        agent._oauth_1m_beta_disabled = True
                        try:
                            agent._anthropic_client.close()
                        except Exception:
                            pass
                        agent._rebuild_anthropic_client()
                        agent._vprint(
                            f"{agent.log_prefix}🔕 OAuth subscription doesn't support "
                            f"the 1M-context beta — disabled for this session and retrying...",
                            force=True,
                        )
                        continue

                if (
                    agent.api_mode == "codex_responses"
                    and agent.provider in {"openai-codex", "xai-oauth"}
                    and status_code == 401
                    and not codex_auth_retry_attempted
                ):
                    codex_auth_retry_attempted = True
                    if agent._try_refresh_codex_client_credentials(force=True):
                        _label = "xAI OAuth" if agent.provider == "xai-oauth" else "Codex"
                        agent._vprint(f"{agent.log_prefix}🔐 {_label} auth refreshed after 401. Retrying request...")
                        continue
                if (
                    agent.api_mode == "chat_completions"
                    and agent.provider == "nous"
                    and status_code == 401
                    and not nous_auth_retry_attempted
                ):
                    nous_auth_retry_attempted = True
                    if agent._try_refresh_nous_client_credentials(force=True):
                        print(f"{agent.log_prefix}🔐 Nous agent key refreshed after 401. Retrying request...")
                        continue
                    # Credential refresh didn't help — show diagnostic info.
                    # Most common causes: Portal OAuth expired/revoked,
                    # account out of credits, or agent key blocked.
                    from hermes_constants import display_hermes_home as _dhh_fn
                    _dhh = _dhh_fn()
                    _body_text = ""
                    try:
                        _body = getattr(api_error, "body", None) or getattr(api_error, "response", None)
                        if _body is not None:
                            _body_text = str(_body)[:200]
                    except Exception:
                        pass
                    print(f"{agent.log_prefix}🔐 Nous 401 — Portal authentication failed.")
                    if _body_text:
                        print(f"{agent.log_prefix}   Response: {_body_text}")
                    print(f"{agent.log_prefix}   Most likely: Portal OAuth expired, account out of credits, or agent key revoked.")
                    print(f"{agent.log_prefix}   Troubleshooting:")
                    print(f"{agent.log_prefix}     • Re-authenticate: hermes login --provider nous")
                    print(f"{agent.log_prefix}     • Check credits / billing: https://portal.nousresearch.com")
                    print(f"{agent.log_prefix}     • Verify stored credentials: {_dhh}/auth.json")
                    print(f"{agent.log_prefix}     • Switch providers temporarily: /model <model> --provider openrouter")
                if (
                    agent.provider == "copilot"
                    and status_code == 401
                    and not copilot_auth_retry_attempted
                ):
                    copilot_auth_retry_attempted = True
                    if agent._try_refresh_copilot_client_credentials():
                        agent._vprint(f"{agent.log_prefix}🔐 Copilot credentials refreshed after 401. Retrying request...")
                        continue
                if (
                    agent.api_mode == "anthropic_messages"
                    and status_code == 401
                    and hasattr(agent, '_anthropic_api_key')
                    and not anthropic_auth_retry_attempted
                ):
                    anthropic_auth_retry_attempted = True
                    from agent.anthropic_adapter import _is_oauth_token
                    if agent._try_refresh_anthropic_client_credentials():
                        print(f"{agent.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request...")
                        continue
                    # Credential refresh didn't help — show diagnostic info
                    key = agent._anthropic_api_key
                    auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
                    print(f"{agent.log_prefix}🔐 Anthropic 401 — authentication failed.")
                    print(f"{agent.log_prefix}   Auth method: {auth_method}")
                    print(f"{agent.log_prefix}   Token prefix: {key[:12]}..." if key and len(key) > 12 else f"{agent.log_prefix}   Token: (empty or short)")
                    print(f"{agent.log_prefix}   Troubleshooting:")
                    from hermes_constants import display_hermes_home as _dhh_fn
                    _dhh = _dhh_fn()
                    print(f"{agent.log_prefix}     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens")
                    print(f"{agent.log_prefix}     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values")
                    print(f"{agent.log_prefix}     • For API keys: verify at https://platform.claude.com/settings/keys")
                    print(f"{agent.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry")
                    print(f"{agent.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_TOKEN \"\"")
                    print(f"{agent.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_API_KEY \"\"")

                # ── Thinking block signature recovery ─────────────────
                # Anthropic signs thinking blocks against the full turn
                # content.  Any upstream mutation (context compression,
                # session truncation, message merging) invalidates the
                # signature → HTTP 400.  Recovery: strip reasoning_details
                # from all messages so the next retry sends no thinking
                # blocks at all.  One-shot — don't retry infinitely.
                if (
                    classified.reason == FailoverReason.thinking_signature
                    and not thinking_sig_retry_attempted
                ):
                    thinking_sig_retry_attempted = True
                    for _m in messages:
                        if isinstance(_m, dict):
                            _m.pop("reasoning_details", None)
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Thinking block signature invalid — "
                        f"stripped all thinking blocks, retrying...",
                        force=True,
                    )
                    logging.warning(
                        "%sThinking block signature recovery: stripped "
                        "reasoning_details from %d messages",
                        agent.log_prefix, len(messages),
                    )
                    continue

                # ── llama.cpp grammar-parse recovery ──────────────────
                # llama.cpp's ``json-schema-to-grammar`` converter rejects
                # regex escape classes (``\d``, ``\w``, ``\s``) and most
                # ``format`` values in tool schemas.  MCP servers emit
                # these routinely for date/phone/email params.  Recovery:
                # strip ``pattern``/``format`` from ``agent.tools`` and
                # retry once.  We keep the keywords by default so cloud
                # providers get the full prompting hints; this branch
                # fires only for users on llama.cpp's OAI server.
                if (
                    classified.reason == FailoverReason.llama_cpp_grammar_pattern
                    and not llama_cpp_grammar_retry_attempted
                ):
                    llama_cpp_grammar_retry_attempted = True
                    try:
                        from tools.schema_sanitizer import strip_pattern_and_format
                        _, _stripped = strip_pattern_and_format(agent.tools)
                    except Exception as _strip_exc:  # pragma: no cover — defensive
                        logging.warning(
                            "%sllama.cpp grammar recovery: strip helper failed: %s",
                            agent.log_prefix, _strip_exc,
                        )
                        _stripped = 0
                    if _stripped:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  llama.cpp rejected tool schema grammar — "
                            f"stripped {_stripped} pattern/format keyword(s), retrying...",
                            force=True,
                        )
                        logging.warning(
                            "%sllama.cpp grammar recovery: stripped %d "
                            "pattern/format keyword(s) from tool schemas",
                            agent.log_prefix, _stripped,
                        )
                        continue
                    # No keywords found to strip — fall through to normal
                    # retry path rather than loop forever on the same error.
                    logging.warning(
                        "%sllama.cpp grammar error but no pattern/format "
                        "keywords to strip — falling through to normal retry",
                        agent.log_prefix,
                    )

                retry_count += 1
                elapsed_time = time.time() - api_start_time
                agent._touch_activity(
                    f"API error recovery (attempt {retry_count}/{max_retries})"
                )
                
                error_type = type(api_error).__name__
                error_msg = str(api_error).lower()
                _error_summary = agent._summarize_api_error(api_error)
                logger.warning(
                    "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
                    retry_count,
                    max_retries,
                    error_type,
                    agent._client_log_context(),
                    _error_summary,
                )

                _provider = getattr(agent, "provider", "unknown")
                _base = getattr(agent, "base_url", "unknown")
                _model = getattr(agent, "model", "unknown")
                _status_code_str = f" [HTTP {status_code}]" if status_code else ""
                agent._vprint(f"{agent.log_prefix}⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}{_status_code_str}", force=True)
                agent._vprint(f"{agent.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                agent._vprint(f"{agent.log_prefix}   🌐 Endpoint: {_base}", force=True)
                agent._vprint(f"{agent.log_prefix}   📝 Error: {_error_summary}", force=True)
                if status_code and status_code < 500:
                    _err_body = getattr(api_error, "body", None)
                    _err_body_str = str(_err_body)[:300] if _err_body else None
                    if _err_body_str:
                        agent._vprint(f"{agent.log_prefix}   📋 Details: {_err_body_str}", force=True)
                agent._vprint(f"{agent.log_prefix}   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")

                # Actionable hint for OpenRouter "no tool endpoints" error.
                # This fires regardless of whether fallback succeeds — the
                # user needs to know WHY their model failed so they can fix
                # their provider routing, not just silently fall back.
                if (
                    agent._is_openrouter_url()
                    and "support tool use" in error_msg
                ):
                    agent._vprint(
                        f"{agent.log_prefix}   💡 No OpenRouter providers for {_model} support tool calling with your current settings.",
                        force=True,
                    )
                    if agent.providers_allowed:
                        agent._vprint(
                            f"{agent.log_prefix}      Your provider_routing.only restriction is filtering out tool-capable providers.",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      Try removing the restriction or adding providers that support tools for this model.",
                            force=True,
                        )
                    agent._vprint(
                        f"{agent.log_prefix}      Check which providers support tools: https://openrouter.ai/models/{_model}",
                        force=True,
                    )

                # Check for interrupt before deciding to retry
                if agent._interrupt_requested:
                    agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                    agent._persist_session(messages, conversation_history)
                    agent.clear_interrupt()
                    return {
                        "final_response": f"Operation interrupted: handling API error ({error_type}: {agent._clean_error_message(str(api_error))}).",
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "interrupted": True,
                    }
                
                # Check for 413 payload-too-large BEFORE generic 4xx handler.
                # A 413 is a payload-size error — the correct response is to
                # compress history and retry, not abort immediately.
                status_code = getattr(api_error, "status_code", None)

                # ── Anthropic Sonnet long-context tier gate ───────────
                # Anthropic returns HTTP 429 "Extra usage is required for
                # long context requests" when a Claude Max (or similar)
                # subscription doesn't include the 1M-context tier.  This
                # is NOT a transient rate limit — retrying or switching
                # credentials won't help.  Reduce context to 200k (the
                # standard tier) and compress.
                if classified.reason == FailoverReason.long_context_tier:
                    _reduced_ctx = 200000
                    compressor = agent.context_compressor
                    old_ctx = compressor.context_length
                    if old_ctx > _reduced_ctx:
                        compressor.update_model(
                            model=agent.model,
                            context_length=_reduced_ctx,
                            base_url=agent.base_url,
                            api_key=getattr(agent, "api_key", ""),
                            provider=agent.provider,
                        )
                        # Context probing flags — only set on built-in
                        # compressor (plugin engines manage their own).
                        if hasattr(compressor, "_context_probed"):
                            compressor._context_probed = True
                            # Don't persist — this is a subscription-tier
                            # limitation, not a model capability.  If the
                            # user later enables extra usage the 1M limit
                            # should come back automatically.
                            compressor._context_probe_persistable = False
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Anthropic long-context tier "
                            f"requires extra usage — reducing context: "
                            f"{old_ctx:,} → {_reduced_ctx:,} tokens",
                            force=True,
                        )

                    compression_attempts += 1
                    if compression_attempts <= max_compression_attempts:
                        original_len = len(messages)
                        messages, active_system_prompt = agent._compress_context(
                            messages, system_message,
                            approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )
                        # Compression created a new session — clear history
                        # so _flush_messages_to_session_db writes compressed
                        # messages to the new session, not skipping them.
                        conversation_history = None
                        if len(messages) < original_len or old_ctx > _reduced_ctx:
                            agent._emit_status(
                                f"🗜️ Context reduced to {_reduced_ctx:,} tokens "
                                f"(was {old_ctx:,}), retrying..."
                            )
                            time.sleep(2)
                            restart_with_compressed_messages = True
                            break
                    # Fall through to normal error handling if compression
                    # is exhausted or didn't help.

                # Eager fallback for rate-limit errors (429 or quota exhaustion).
                # When a fallback model is configured, switch immediately instead
                # of burning through retries with exponential backoff -- the
                # primary provider won't recover within the retry window.
                is_rate_limited = classified.reason in {
                    FailoverReason.rate_limit,
                    FailoverReason.billing,
                }
                if is_rate_limited and agent._fallback_index < len(agent._fallback_chain):
                    # Don't eagerly fallback if credential pool rotation may
                    # still recover.  See _pool_may_recover_from_rate_limit
                    # for the single-credential-pool and CloudCode-quota
                    # exceptions.  Fixes #11314 and #13636.
                    pool_may_recover = _pool_may_recover_from_rate_limit(
                        agent._credential_pool,
                        provider=agent.provider,
                        base_url=getattr(agent, "base_url", None),
                    )
                    if not pool_may_recover:
                        agent._emit_status("⚠️ Rate limited — switching to fallback provider...")
                        if agent._try_activate_fallback(reason=classified.reason):
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue

                # ── Nous Portal: record rate limit & skip retries ─────
                # When Nous returns a 429 that is a genuine account-
                # level rate limit, record the reset time to a shared
                # file so ALL sessions (cron, gateway, auxiliary) know
                # not to pile on, then skip further retries -- each
                # one burns another RPH request and deepens the hole.
                # The retry loop's top-of-iteration guard will catch
                # this on the next pass and try fallback or bail.
                #
                # IMPORTANT: Nous Portal multiplexes multiple upstream
                # providers (DeepSeek, Kimi, MiMo, Hermes).  A 429 can
                # also mean an UPSTREAM provider is out of capacity
                # for one specific model -- transient, clears in
                # seconds, nothing to do with the caller's quota.
                # Tripping the cross-session breaker on that would
                # block every Nous model for minutes.  We use
                # ``is_genuine_nous_rate_limit`` to tell the two
                # apart via the 429's own x-ratelimit-* headers and
                # the last-known-good state captured on the previous
                # successful response.
                if (
                    is_rate_limited
                    and agent.provider == "nous"
                    and classified.reason == FailoverReason.rate_limit
                    and not recovered_with_pool
                ):
                    _genuine_nous_rate_limit = False
                    try:
                        from agent.nous_rate_guard import (
                            is_genuine_nous_rate_limit,
                            record_nous_rate_limit,
                        )
                        _err_resp = getattr(api_error, "response", None)
                        _err_hdrs = (
                            getattr(_err_resp, "headers", None)
                            if _err_resp else None
                        )
                        _genuine_nous_rate_limit = is_genuine_nous_rate_limit(
                            headers=_err_hdrs,
                            last_known_state=agent._rate_limit_state,
                        )
                        if _genuine_nous_rate_limit:
                            record_nous_rate_limit(
                                headers=_err_hdrs,
                                error_context=error_context,
                            )
                        else:
                            logging.info(
                                "Nous 429 looks like upstream capacity "
                                "(no exhausted bucket in headers or "
                                "last-known state) -- not tripping "
                                "cross-session breaker."
                            )
                    except Exception:
                        pass
                    if _genuine_nous_rate_limit:
                        # Skip straight to max_retries -- the
                        # top-of-loop guard will handle fallback or
                        # bail cleanly.
                        retry_count = max_retries
                        continue
                    # Upstream capacity 429: fall through to normal
                    # retry logic.  A different model (or the same
                    # model a moment later) will typically succeed.

                is_payload_too_large = (
                    classified.reason == FailoverReason.payload_too_large
                )

                # Actionable hint for GitHub Models (Azure) 413 errors.
                # The free tier enforces a hard 8K token cap per request,
                # which Hermes' system prompt + tool schemas alone exceed.
                # Compression can't help — the floor is the system prompt
                # itself, not the conversation — so surface a clear "not
                # compatible" message instead of looping into three futile
                # compression attempts.
                if (
                    status_code == 413
                    and isinstance(agent.base_url, str)
                    and "models.inference.ai.azure.com" in agent.base_url
                ):
                    agent._vprint(
                        f"{agent.log_prefix}   💡 GitHub Models free tier (models.inference.ai.azure.com) caps every",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      request at ~8K tokens. Hermes' system prompt + tool schemas baseline",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      exceeds that floor, so this endpoint cannot run an agentic loop.",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      Use the `copilot` provider with a Copilot subscription token (`hermes",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      setup` → GitHub Copilot), or pick any other provider.",
                        force=True,
                    )

                if is_payload_too_large:
                    compression_attempts += 1
                    if compression_attempts > max_compression_attempts:
                        agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logging.error(f"{agent.log_prefix}413 compression failed after {max_compression_attempts} attempts.")
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Request payload too large: max compression attempts ({max_compression_attempts}) reached.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }
                    agent._emit_status(f"⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

                    original_len = len(messages)
                    messages, active_system_prompt = agent._compress_context(
                        messages, system_message, approx_tokens=approx_tokens,
                        task_id=effective_task_id,
                    )
                    # Compression created a new session — clear history
                    # so _flush_messages_to_session_db writes compressed
                    # messages to the new session, not skipping them.
                    conversation_history = None

                    if len(messages) < original_len:
                        agent._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                        time.sleep(2)  # Brief pause between compression retries
                        restart_with_compressed_messages = True
                        break
                    else:
                        agent._vprint(f"{agent.log_prefix}❌ Payload too large and cannot compress further.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logging.error(f"{agent.log_prefix}413 payload too large. Cannot compress further.")
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": "Request payload too large (413). Cannot compress further.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }

                # Check for context-length errors BEFORE generic 4xx handler.
                # The classifier detects context overflow from: explicit error
                # messages, generic 400 + large session heuristic (#1630), and
                # server disconnect + large session pattern (#2153).
                is_context_length_error = (
                    classified.reason == FailoverReason.context_overflow
                )

                if is_context_length_error:
                    compressor = agent.context_compressor
                    old_ctx = compressor.context_length

                    # ── Distinguish two very different errors ───────────
                    # 1. "Prompt too long": the INPUT exceeds the context window.
                    #    Fix: reduce context_length + compress history.
                    # 2. "max_tokens too large": input is fine, but
                    #    input_tokens + requested max_tokens > context_window.
                    #    Fix: reduce max_tokens (the OUTPUT cap) for this call.
                    #    Do NOT shrink context_length — the window is unchanged.
                    #
                    # Note: max_tokens = output token cap (one response).
                    #       context_length = total window (input + output combined).
                    available_out = parse_available_output_tokens_from_error(error_msg)
                    if available_out is not None:
                        # Error is purely about the output cap being too large.
                        # Cap output to the available space and retry without
                        # touching context_length or triggering compression.
                        safe_out = max(1, available_out - 64)  # small safety margin
                        agent._ephemeral_max_output_tokens = safe_out
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Output cap too large for current prompt — "
                            f"retrying with max_tokens={safe_out:,} "
                            f"(available_tokens={available_out:,}; context_length unchanged at {old_ctx:,})",
                            force=True,
                        )
                        # Still count against compression_attempts so we don't
                        # loop forever if the error keeps recurring.
                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                            agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{agent.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                            agent._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }
                        restart_with_compressed_messages = True
                        break

                    # Error is about the INPUT being too large — reduce context_length.
                    # Try to parse the actual limit from the error message
                    parsed_limit = parse_context_limit_from_error(error_msg)
                    _provider_lower = (getattr(agent, "provider", "") or "").lower()
                    _base_lower = (getattr(agent, "base_url", "") or "").rstrip("/").lower()
                    is_minimax_provider = (
                        _provider_lower in {"minimax", "minimax-cn"}
                        or _base_lower.startswith((
                            "https://api.minimax.io/anthropic",
                            "https://api.minimaxi.com/anthropic",
                        ))
                    )
                    minimax_delta_only_overflow = (
                        is_minimax_provider
                        and parsed_limit is None
                        and "context window exceeds limit (" in error_msg
                    )
                    if parsed_limit and parsed_limit < old_ctx:
                        new_ctx = parsed_limit
                        agent._vprint(f"{agent.log_prefix}Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})", force=True)
                    elif minimax_delta_only_overflow:
                        new_ctx = old_ctx
                        agent._vprint(
                            f"{agent.log_prefix}Provider reported overflow amount only; "
                            f"keeping context_length at {old_ctx:,} tokens and compressing.",
                            force=True,
                        )
                    else:
                        # Step down to the next probe tier
                        new_ctx = get_next_probe_tier(old_ctx)

                    if new_ctx and new_ctx < old_ctx:
                        compressor.update_model(
                            model=agent.model,
                            context_length=new_ctx,
                            base_url=agent.base_url,
                            api_key=getattr(agent, "api_key", ""),
                            provider=agent.provider,
                        )
                        # Context probing flags — only set on built-in
                        # compressor (plugin engines manage their own).
                        if hasattr(compressor, "_context_probed"):
                            compressor._context_probed = True
                            # Only persist limits parsed from the provider's
                            # error message (a real number).  Guessed fallback
                            # tiers from get_next_probe_tier() should stay
                            # in-memory only — persisting them pollutes the
                            # cache with wrong values.
                            compressor._context_probe_persistable = bool(
                                parsed_limit and parsed_limit == new_ctx
                            )
                        agent._vprint(f"{agent.log_prefix}⚠️  Context length exceeded — stepping down: {old_ctx:,} → {new_ctx:,} tokens", force=True)
                    else:
                        agent._vprint(f"{agent.log_prefix}⚠️  Context length exceeded at minimum tier — attempting compression...", force=True)

                    compression_attempts += 1
                    if compression_attempts > max_compression_attempts:
                        agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logging.error(f"{agent.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }
                    agent._emit_status(f"🗜️ Context too large (~{approx_tokens:,} tokens) — compressing ({compression_attempts}/{max_compression_attempts})...")

                    original_len = len(messages)
                    messages, active_system_prompt = agent._compress_context(
                        messages, system_message, approx_tokens=approx_tokens,
                        task_id=effective_task_id,
                    )
                    # Compression created a new session — clear history
                    # so _flush_messages_to_session_db writes compressed
                    # messages to the new session, not skipping them.
                    conversation_history = None

                    if len(messages) < original_len or new_ctx and new_ctx < old_ctx:
                        if len(messages) < original_len:
                            agent._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                        time.sleep(2)  # Brief pause between compression retries
                        restart_with_compressed_messages = True
                        break
                    else:
                        # Can't compress further and already at minimum tier
                        agent._vprint(f"{agent.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.", force=True)
                        logging.error(f"{agent.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }

                # Check for non-retryable client errors.  The classifier
                # already accounts for 413, 429, 529 (transient), context
                # overflow, and generic-400 heuristics.  Local validation
                # errors (ValueError, TypeError) are programming bugs.
                # Exclude UnicodeEncodeError — it's a ValueError subclass
                # but is handled separately by the surrogate sanitization
                # path above.  Exclude json.JSONDecodeError — also a
                # ValueError subclass, but it indicates a transient
                # provider/network failure (malformed response body,
                # truncated stream, routing layer corruption), not a
                # local programming bug, and should be retried (#14782).
                is_local_validation_error = (
                    isinstance(api_error, (ValueError, TypeError))
                    and not isinstance(
                        api_error, (UnicodeEncodeError, json.JSONDecodeError)
                    )
                    # ssl.SSLError (and its subclass SSLCertVerificationError)
                    # inherits from OSError *and* ValueError via Python MRO,
                    # so the isinstance(ValueError) check above would
                    # misclassify a TLS transport failure as a local
                    # programming bug and abort without retrying.  Exclude
                    # ssl.SSLError explicitly so the error classifier's
                    # retryable=True mapping takes effect instead.
                    and not isinstance(api_error, ssl.SSLError)
                )
                is_client_error = (
                    is_local_validation_error
                    or (
                        not classified.retryable
                        and not classified.should_compress
                        and classified.reason not in {
                            FailoverReason.rate_limit,
                            FailoverReason.billing,
                            FailoverReason.overloaded,
                            FailoverReason.context_overflow,
                            FailoverReason.payload_too_large,
                            FailoverReason.long_context_tier,
                            FailoverReason.thinking_signature,
                        }
                    )
                ) and not is_context_length_error

                if is_client_error:
                    # Try fallback before aborting — a different provider
                    # may not have the same issue (rate limit, auth, etc.)
                    agent._emit_status(f"⚠️ Non-retryable error (HTTP {status_code}) — trying fallback...")
                    if agent._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        primary_recovery_attempted = False
                        continue
                    if api_kwargs is not None:
                        agent._dump_api_request_debug(
                            api_kwargs, reason="non_retryable_client_error", error=api_error,
                        )
                    agent._emit_status(
                        f"❌ Non-retryable error (HTTP {status_code}): "
                        f"{agent._summarize_api_error(api_error)}"
                    )
                    agent._vprint(f"{agent.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.", force=True)
                    agent._vprint(f"{agent.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                    agent._vprint(f"{agent.log_prefix}   🌐 Endpoint: {_base}", force=True)
                    # Actionable guidance for common auth errors
                    if classified.is_auth or classified.reason == FailoverReason.billing:
                        if _provider in {"openai-codex", "xai-oauth"} and status_code == 401:
                            if _provider == "openai-codex":
                                agent._vprint(f"{agent.log_prefix}   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been", force=True)
                                agent._vprint(f"{agent.log_prefix}      refreshed by another client (Codex CLI, VS Code). To fix:", force=True)
                                agent._vprint(f"{agent.log_prefix}      1. Run `codex` in your terminal to generate fresh tokens.", force=True)
                                agent._vprint(f"{agent.log_prefix}      2. Then run `hermes auth` to re-authenticate.", force=True)
                            else:
                                agent._vprint(f"{agent.log_prefix}   💡 xAI OAuth token was rejected (HTTP 401). To fix:", force=True)
                                agent._vprint(f"{agent.log_prefix}      re-authenticate with xAI Grok OAuth (SuperGrok Subscription) from `hermes model`.", force=True)
                        else:
                            agent._vprint(f"{agent.log_prefix}   💡 Your API key was rejected by the provider. Check:", force=True)
                            agent._vprint(f"{agent.log_prefix}      • Is the key valid? Run: hermes setup", force=True)
                            agent._vprint(f"{agent.log_prefix}      • Does your account have access to {_model}?", force=True)
                            if base_url_host_matches(str(_base), "openrouter.ai"):
                                agent._vprint(f"{agent.log_prefix}      • Check credits: https://openrouter.ai/settings/credits", force=True)
                    else:
                        agent._vprint(f"{agent.log_prefix}   💡 This type of error won't be fixed by retrying.", force=True)
                    logging.error(f"{agent.log_prefix}Non-retryable client error: {api_error}")
                    # Skip session persistence when the error is likely
                    # context-overflow related (status 400 + large session).
                    # Persisting the failed user message would make the
                    # session even larger, causing the same failure on the
                    # next attempt. (#1630)
                    if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Skipping session persistence "
                            f"for large failed session to prevent growth loop.",
                            force=True,
                        )
                    else:
                        agent._persist_session(messages, conversation_history)
                    return {
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "failed": True,
                        "error": str(api_error),
                    }

                if retry_count >= max_retries:
                    # Before falling back, try rebuilding the primary
                    # client once for transient transport errors (stale
                    # connection pool, TCP reset).  Only attempted once
                    # per API call block.
                    if not primary_recovery_attempted and agent._try_recover_primary_transport(
                        api_error, retry_count=retry_count, max_retries=max_retries,
                    ):
                        primary_recovery_attempted = True
                        retry_count = 0
                        continue
                    # Try fallback before giving up entirely
                    agent._emit_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
                    if agent._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        primary_recovery_attempted = False
                        continue
                    _final_summary = agent._summarize_api_error(api_error)
                    if is_rate_limited:
                        agent._emit_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
                    else:
                        agent._emit_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
                    agent._vprint(f"{agent.log_prefix}   💀 Final error: {_final_summary}", force=True)

                    # Detect SSE stream-drop pattern (e.g. "Network
                    # connection lost") and surface actionable guidance.
                    # This typically happens when the model generates a
                    # very large tool call (write_file with huge content)
                    # and the proxy/CDN drops the stream mid-response.
                    _is_stream_drop = (
                        not getattr(api_error, "status_code", None)
                        and any(p in error_msg for p in (
                            "connection lost", "connection reset",
                            "connection closed", "network connection",
                            "network error", "terminated",
                        ))
                    )
                    if _is_stream_drop:
                        agent._vprint(
                            f"{agent.log_prefix}   💡 The provider's stream "
                            f"connection keeps dropping. This often happens "
                            f"when the model tries to write a very large "
                            f"file in a single tool call.",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      Try asking the model "
                            f"to use execute_code with Python's open() for "
                            f"large files, or to write the file in smaller "
                            f"sections.",
                            force=True,
                        )

                    logging.error(
                        "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
                        agent.log_prefix, max_retries, _final_summary,
                        _provider, _model, len(api_messages), f"{approx_tokens:,}",
                    )
                    if api_kwargs is not None:
                        agent._dump_api_request_debug(
                            api_kwargs, reason="max_retries_exhausted", error=api_error,
                        )
                    agent._persist_session(messages, conversation_history)
                    _final_response = f"API call failed after {max_retries} retries: {_final_summary}"
                    if _is_stream_drop:
                        _final_response += (
                            "\n\nThe provider's stream connection keeps "
                            "dropping — this often happens when generating "
                            "very large tool call responses (e.g. write_file "
                            "with long content). Try asking me to use "
                            "execute_code with Python's open() for large "
                            "files, or to write in smaller sections."
                        )
                    return {
                        "final_response": _final_response,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "failed": True,
                        "error": _final_summary,
                    }

                # For rate limits, respect the Retry-After header if present
                _retry_after = None
                if is_rate_limited:
                    _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
                    if _resp_headers and hasattr(_resp_headers, "get"):
                        _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
                        if _ra_raw:
                            try:
                                _retry_after = min(float(_ra_raw), 120)  # Cap at 2 minutes
                            except (TypeError, ValueError):
                                pass
                wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
                if is_rate_limited:
                    agent._emit_status(f"⏱️ Rate limited. Waiting {wait_time:.1f}s (attempt {retry_count + 1}/{max_retries})...")
                else:
                    agent._emit_status(f"⏳ Retrying in {wait_time:.1f}s (attempt {retry_count}/{max_retries})...")
                logger.warning(
                    "Retrying API call in %ss (attempt %s/%s) %s error=%s",
                    wait_time,
                    retry_count,
                    max_retries,
                    agent._client_log_context(),
                    api_error,
                )
                # Sleep in small increments so we can respond to interrupts quickly
                # instead of blocking the entire wait_time in one sleep() call
                sleep_end = time.time() + wait_time
                _backoff_touch_counter = 0
                while time.time() < sleep_end:
                    if agent._interrupt_requested:
                        agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                        agent._persist_session(messages, conversation_history)
                        agent.clear_interrupt()
                        return {
                            "final_response": f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }
                    time.sleep(0.2)  # Check interrupt every 200ms
                    # Touch activity every ~30s so the gateway's inactivity
                    # monitor knows we're alive during backoff waits.
                    _backoff_touch_counter += 1
                    if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                        agent._touch_activity(
                            f"error retry backoff ({retry_count}/{max_retries}), "
                            f"{int(sleep_end - time.time())}s remaining"
                        )
        
        # If the API call was interrupted, skip response processing
        if interrupted:
            _turn_exit_reason = "interrupted_during_api_call"
            break

        if restart_with_compressed_messages:
            api_call_count -= 1
            agent.iteration_budget.refund()
            # Count compression restarts toward the retry limit to prevent
            # infinite loops when compression reduces messages but not enough
            # to fit the context window.
            retry_count += 1
            restart_with_compressed_messages = False
            continue

        if restart_with_length_continuation:
            # Progressively boost the output token budget on each retry.
            # Retry 1 → 2× base, retry 2 → 3× base, capped at 32 768.
            # Applies to all providers via _ephemeral_max_output_tokens.
            _boost_base = agent.max_tokens if agent.max_tokens else 4096
            _boost = _boost_base * (length_continue_retries + 1)
            agent._ephemeral_max_output_tokens = min(_boost, 32768)
            continue

        # Guard: if all retries exhausted without a successful response
        # (e.g. repeated context-length errors that exhausted retry_count),
        # the `response` variable is still None. Break out cleanly.
        if response is None:
            _turn_exit_reason = "all_retries_exhausted_no_response"
            print(f"{agent.log_prefix}❌ All API retries exhausted with no successful response.")
            agent._persist_session(messages, conversation_history)
            break

        try:
            _transport = agent._get_transport()
            _normalize_kwargs = {}
            if agent.api_mode == "anthropic_messages":
                _normalize_kwargs["strip_tool_prefix"] = agent._is_anthropic_oauth
            normalized = _transport.normalize_response(response, **_normalize_kwargs)
            assistant_message = normalized
            finish_reason = normalized.finish_reason
            
            # Normalize content to string — some OpenAI-compatible servers
            # (llama-server, etc.) return content as a dict or list instead
            # of a plain string, which crashes downstream .strip() calls.
            if assistant_message.content is not None and not isinstance(assistant_message.content, str):
                raw = assistant_message.content
                if isinstance(raw, dict):
                    assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
                elif isinstance(raw, list):
                    # Multimodal content list — extract text parts
                    parts = []
                    for part in raw:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif isinstance(part, dict) and "text" in part:
                            parts.append(str(part["text"]))
                    assistant_message.content = "\n".join(parts)
                else:
                    assistant_message.content = str(raw)

            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _assistant_tool_calls = getattr(assistant_message, "tool_calls", None) or []
                _assistant_text = assistant_message.content or ""
                _invoke_hook(
                    "post_api_request",
                    task_id=effective_task_id,
                    session_id=agent.session_id or "",
                    platform=agent.platform or "",
                    model=agent.model,
                    provider=agent.provider,
                    base_url=agent.base_url,
                    api_mode=agent.api_mode,
                    api_call_count=api_call_count,
                    api_duration=api_duration,
                    finish_reason=finish_reason,
                    message_count=len(api_messages),
                    response_model=getattr(response, "model", None),
                    response=response,
                    usage=agent._usage_summary_for_api_request_hook(response),
                    assistant_message=assistant_message,
                    assistant_content_chars=len(_assistant_text),
                    assistant_tool_call_count=len(_assistant_tool_calls),
                )
            except Exception:
                pass

            # Handle assistant response
            if assistant_message.content and not agent.quiet_mode:
                if agent.verbose_logging:
                    agent._vprint(f"{agent.log_prefix}🤖 Assistant: {assistant_message.content}")
                else:
                    agent._vprint(f"{agent.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

            # Notify progress callback of model's thinking (used by subagent
            # delegation to relay the child's reasoning to the parent display).
            if (assistant_message.content and agent.tool_progress_callback):
                _think_text = assistant_message.content.strip()
                # Strip reasoning XML tags that shouldn't leak to parent display
                _think_text = re.sub(
                    r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
                ).strip()
                # For subagents: relay first line to parent display (existing behaviour).
                # For all agents with a structured callback: emit reasoning.available event.
                first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                if first_line and getattr(agent, '_delegate_depth', 0) > 0:
                    try:
                        agent.tool_progress_callback("_thinking", first_line)
                    except Exception:
                        pass
                elif _think_text:
                    try:
                        agent.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
                    except Exception:
                        pass
            
            # Check for incomplete <REASONING_SCRATCHPAD> (opened but never closed)
            # This means the model ran out of output tokens mid-reasoning — retry up to 2 times
            if has_incomplete_scratchpad(assistant_message.content or ""):
                agent._incomplete_scratchpad_retries += 1
                
                agent._vprint(f"{agent.log_prefix}⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
                
                if agent._incomplete_scratchpad_retries <= 2:
                    agent._vprint(f"{agent.log_prefix}🔄 Retrying API call ({agent._incomplete_scratchpad_retries}/2)...")
                    # Don't add the broken message, just retry
                    continue
                else:
                    # Max retries - discard this turn and save as partial
                    agent._vprint(f"{agent.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                    agent._incomplete_scratchpad_retries = 0
                    
                    rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)
                    agent._cleanup_task_resources(effective_task_id)
                    agent._persist_session(messages, conversation_history)
                    
                    return {
                        "final_response": None,
                        "messages": rolled_back_messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                    }
            
            # Reset incomplete scratchpad counter on clean response
            agent._incomplete_scratchpad_retries = 0

            if agent.api_mode == "codex_responses" and finish_reason == "incomplete":
                agent._codex_incomplete_retries += 1

                interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
                interim_has_content = bool((interim_msg.get("content") or "").strip())
                interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False
                interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))
                interim_has_codex_message_items = bool(interim_msg.get("codex_message_items"))

                if (
                    interim_has_content
                    or interim_has_reasoning
                    or interim_has_codex_reasoning
                    or interim_has_codex_message_items
                ):
                    last_msg = messages[-1] if messages else None
                    # Duplicate detection: two consecutive incomplete assistant
                    # messages with identical content AND reasoning are collapsed.
                    # For provider-state-only changes (encrypted reasoning
                    # items or replayable message ids/phases/statuses differ
                    # while visible content/reasoning are unchanged), compare
                    # those opaque payloads too so we don't silently drop the
                    # newer continuation state.
                    last_codex_items = last_msg.get("codex_reasoning_items") if isinstance(last_msg, dict) else None
                    interim_codex_items = interim_msg.get("codex_reasoning_items")
                    last_codex_message_items = last_msg.get("codex_message_items") if isinstance(last_msg, dict) else None
                    interim_codex_message_items = interim_msg.get("codex_message_items")
                    duplicate_interim = (
                        isinstance(last_msg, dict)
                        and last_msg.get("role") == "assistant"
                        and last_msg.get("finish_reason") == "incomplete"
                        and (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                        and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
                        and last_codex_items == interim_codex_items
                        and last_codex_message_items == interim_codex_message_items
                    )
                    if not duplicate_interim:
                        messages.append(interim_msg)
                        agent._emit_interim_assistant_message(interim_msg)

                if agent._codex_incomplete_retries < 3:
                    if not agent.quiet_mode:
                        agent._vprint(f"{agent.log_prefix}↻ Codex response incomplete; continuing turn ({agent._codex_incomplete_retries}/3)")
                    agent._session_messages = messages
                    agent._save_session_log(messages)
                    continue

                agent._codex_incomplete_retries = 0
                agent._persist_session(messages, conversation_history)
                return {
                    "final_response": None,
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "partial": True,
                    "error": "Codex response remained incomplete after 3 continuation attempts",
                }
            elif hasattr(agent, "_codex_incomplete_retries"):
                agent._codex_incomplete_retries = 0
            
            # Check for tool calls
            if assistant_message.tool_calls:
                if not agent.quiet_mode:
                    agent._vprint(f"{agent.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                
                if agent.verbose_logging:
                    for tc in assistant_message.tool_calls:
                        logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")
                
                # Validate tool call names - detect model hallucinations
                # Repair mismatched tool names before validating
                for tc in assistant_message.tool_calls:
                    if tc.function.name not in agent.valid_tool_names:
                        repaired = agent._repair_tool_call(tc.function.name)
                        if repaired:
                            print(f"{agent.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                            tc.function.name = repaired
                invalid_tool_calls = [
                    tc.function.name for tc in assistant_message.tool_calls
                    if tc.function.name not in agent.valid_tool_names
                ]
                if invalid_tool_calls:
                    # Track retries for invalid tool calls
                    agent._invalid_tool_retries += 1

                    # Return helpful error to model — model can agent-correct next turn
                    available = ", ".join(sorted(agent.valid_tool_names))
                    invalid_name = invalid_tool_calls[0]
                    invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
                    agent._vprint(f"{agent.log_prefix}⚠️  Unknown tool '{invalid_preview}' — sending error to model for agent-correction ({agent._invalid_tool_retries}/3)")

                    if agent._invalid_tool_retries >= 3:
                        agent._vprint(f"{agent.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
                        agent._invalid_tool_retries = 0
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": f"Model generated invalid tool call: {invalid_preview}"
                        }

                    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
                    messages.append(assistant_msg)
                    for tc in assistant_message.tool_calls:
                        if tc.function.name not in agent.valid_tool_names:
                            content = f"Tool '{tc.function.name}' does not exist. Available tools: {available}"
                        else:
                            content = "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                        messages.append({
                            "role": "tool",
                            "name": tc.function.name,
                            "tool_call_id": tc.id,
                            "content": content,
                        })
                    continue
                # Reset retry counter on successful tool call validation
                agent._invalid_tool_retries = 0
                
                # Validate tool call arguments are valid JSON
                # Handle empty strings as empty objects (common model quirk)
                invalid_json_args = []
                for tc in assistant_message.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, (dict, list)):
                        tc.function.arguments = json.dumps(args)
                        continue
                    if args is not None and not isinstance(args, str):
                        tc.function.arguments = str(args)
                        args = tc.function.arguments
                    # Treat empty/whitespace strings as empty object
                    if not args or not args.strip():
                        tc.function.arguments = "{}"
                        continue
                    try:
                        json.loads(args)
                    except json.JSONDecodeError as e:
                        invalid_json_args.append((tc.function.name, str(e)))
                
                if invalid_json_args:
                    # Check if the invalid JSON is due to truncation rather
                    # than a model formatting mistake.  Routers sometimes
                    # rewrite finish_reason from "length" to "tool_calls",
                    # hiding the truncation from the length handler above.
                    # Detect truncation: args that don't end with } or ]
                    # (after stripping whitespace) are cut off mid-stream.
                    _truncated = any(
                        not (tc.function.arguments or "").rstrip().endswith(("}", "]"))
                        for tc in assistant_message.tool_calls
                        if tc.function.name in {n for n, _ in invalid_json_args}
                    )
                    if _truncated:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Truncated tool call arguments detected "
                            f"(finish_reason={finish_reason!r}) — refusing to execute.",
                            force=True,
                        )
                        agent._invalid_json_retries = 0
                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Response truncated due to output length limit",
                        }

                    # Track retries for invalid JSON arguments
                    agent._invalid_json_retries += 1

                    tool_name, error_msg = invalid_json_args[0]
                    agent._vprint(f"{agent.log_prefix}⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

                    if agent._invalid_json_retries < 3:
                        agent._vprint(f"{agent.log_prefix}🔄 Retrying API call ({agent._invalid_json_retries}/3)...")
                        # Don't add anything to messages, just retry the API call
                        continue
                    else:
                        # Instead of returning partial, inject tool error results so the model can recover.
                        # Using tool results (not user messages) preserves role alternation.
                        agent._vprint(f"{agent.log_prefix}⚠️  Injecting recovery tool results for invalid JSON...")
                        agent._invalid_json_retries = 0  # Reset for next attempt
                        
                        # Append the assistant message with its (broken) tool_calls
                        recovery_assistant = agent._build_assistant_message(assistant_message, finish_reason)
                        messages.append(recovery_assistant)
                        
                        # Respond with tool error results for each tool call
                        invalid_names = {name for name, _ in invalid_json_args}
                        for tc in assistant_message.tool_calls:
                            if tc.function.name in invalid_names:
                                err = next(e for n, e in invalid_json_args if n == tc.function.name)
                                tool_result = (
                                    f"Error: Invalid JSON arguments. {err}. "
                                    f"For tools with no required parameters, use an empty object: {{}}. "
                                    f"Please retry with valid JSON."
                                )
                            else:
                                tool_result = "Skipped: other tool call in this response had invalid JSON."
                            messages.append({
                                "role": "tool",
                                "name": tc.function.name,
                                "tool_call_id": tc.id,
                                "content": tool_result,
                            })
                        continue
                
                # Reset retry counter on successful JSON validation
                agent._invalid_json_retries = 0

                # ── Post-call guardrails ──────────────────────────
                assistant_message.tool_calls = agent._cap_delegate_task_calls(
                    assistant_message.tool_calls
                )
                assistant_message.tool_calls = agent._deduplicate_tool_calls(
                    assistant_message.tool_calls
                )

                assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
                
                # If this turn has both content AND tool_calls, capture the content
                # as a fallback final response. Common pattern: model delivers its
                # answer and calls memory/skill tools as a side-effect in the same
                # turn. If the follow-up turn after tools is empty, we use this.
                turn_content = assistant_message.content or ""
                if turn_content and agent._has_content_after_think_block(turn_content):
                    agent._last_content_with_tools = turn_content
                    # Only mute subsequent output when EVERY tool call in
                    # this turn is post-response housekeeping (memory, todo,
                    # skill_manage, etc.).  If any substantive tool is present
                    # (search_files, read_file, write_file, terminal, ...),
                    # keep output visible so the user sees progress.
                    _HOUSEKEEPING_TOOLS = frozenset({
                        "memory", "todo", "skill_manage", "session_search",
                    })
                    _all_housekeeping = all(
                        tc.function.name in _HOUSEKEEPING_TOOLS
                        for tc in assistant_message.tool_calls
                    )
                    agent._last_content_tools_all_housekeeping = _all_housekeeping
                    if _all_housekeeping and agent._has_stream_consumers():
                        agent._mute_post_response = True
                    elif agent._should_emit_quiet_tool_messages():
                        clean = agent._strip_think_blocks(turn_content).strip()
                        if clean:
                            agent._vprint(f"  ┊ 💬 {clean}")
                
                # Pop thinking-only prefill message(s) before appending
                # (tool-call path — same rationale as the final-response path).
                _had_prefill = False
                while (
                    messages
                    and isinstance(messages[-1], dict)
                    and messages[-1].get("_thinking_prefill")
                ):
                    messages.pop()
                    _had_prefill = True

                # Reset prefill counter when tool calls follow a prefill
                # recovery.  Without this, the counter accumulates across
                # the whole conversation — a model that intermittently
                # empties (empty → prefill → tools → empty → prefill →
                # tools) burns both prefill attempts and the third empty
                # gets zero recovery.  Resetting here treats each tool-
                # call success as a fresh start.
                if _had_prefill:
                    agent._thinking_prefill_retries = 0
                    agent._empty_content_retries = 0
                # Successful tool execution — reset the post-tool nudge
                # flag so it can fire again if the model goes empty on
                # a LATER tool round.
                agent._post_tool_empty_retried = False

                messages.append(assistant_msg)
                agent._emit_interim_assistant_message(assistant_msg)

                # Close any open streaming display (response box, reasoning
                # box) before tool execution begins.  Intermediate turns may
                # have streamed early content that opened the response box;
                # flushing here prevents it from wrapping tool feed lines.
                # Only signal the display callback — TTS (_stream_callback)
                # should NOT receive None (it uses None as end-of-stream).
                if agent.stream_delta_callback:
                    try:
                        agent.stream_delta_callback(None)
                    except Exception:
                        pass

                agent._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

                if agent._tool_guardrail_halt_decision is not None:
                    decision = agent._tool_guardrail_halt_decision
                    _turn_exit_reason = "guardrail_halt"
                    final_response = agent._toolguard_controlled_halt_response(decision)
                    agent._emit_status(
                        f"⚠️ Tool guardrail halted {decision.tool_name}: {decision.code}"
                    )
                    messages.append({"role": "assistant", "content": final_response})
                    break

                # Reset per-turn retry counters after successful tool
                # execution so a single truncation doesn't poison the
                # entire conversation.
                truncated_tool_call_retries = 0

                # Signal that a paragraph break is needed before the next
                # streamed text.  We don't emit it immediately because
                # multiple consecutive tool iterations would stack up
                # redundant blank lines.  Instead, _fire_stream_delta()
                # will prepend a single "\n\n" the next time real text
                # arrives.
                agent._stream_needs_break = True

                # Refund the iteration if the ONLY tool(s) called were
                # execute_code (programmatic tool calling).  These are
                # cheap RPC-style calls that shouldn't eat the budget.
                _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
                if _tc_names == {"execute_code"}:
                    agent.iteration_budget.refund()
                
                # Use real token counts from the API response to decide
                # compression.  prompt_tokens + completion_tokens is the
                # actual context size the provider reported plus the
                # assistant turn — a tight lower bound for the next prompt.
                # Tool results appended above aren't counted yet, but the
                # threshold (default 50%) leaves ample headroom; if tool
                # results push past it, the next API call will report the
                # real total and trigger compression then.
                #
                # If last_prompt_tokens is 0 (stale after API disconnect
                # or provider returned no usage data), fall back to rough
                # estimate to avoid missing compression.  Without this,
                # a session can grow unbounded after disconnects because
                # should_compress(0) never fires.  (#2153)
                _compressor = agent.context_compressor
                if _compressor.last_prompt_tokens > 0:
                    # Only use prompt_tokens — completion/reasoning
                    # tokens don't consume context window space.
                    # Thinking models (GLM-5.1, QwQ, DeepSeek R1)
                    # inflate completion_tokens with reasoning,
                    # causing premature compression.  (#12026)
                    _real_tokens = _compressor.last_prompt_tokens
                else:
                    # Include tool schemas — with 50+ tools enabled
                    # these add 20-30K tokens the messages-only
                    # estimate misses, which can skip compression
                    # past the configured threshold (#14695).
                    _real_tokens = estimate_request_tokens_rough(
                        messages, tools=agent.tools or None
                    )

                if agent.compression_enabled and _compressor.should_compress(_real_tokens):
                    agent._safe_print("  ⟳ compacting context…")
                    messages, active_system_prompt = agent._compress_context(
                        messages, system_message,
                        approx_tokens=agent.context_compressor.last_prompt_tokens,
                        task_id=effective_task_id,
                    )
                    # Compression created a new session — clear history so
                    # _flush_messages_to_session_db writes compressed messages
                    # to the new session (see preflight compression comment).
                    conversation_history = None
                
                # Save session log incrementally (so progress is visible even if interrupted)
                agent._session_messages = messages
                agent._save_session_log(messages)
                
                # Continue loop for next response
                continue
            
            else:
                # No tool calls - this is the final response
                final_response = assistant_message.content or ""
                
                # Fix: unmute output when entering the no-tool-call branch
                # so the user can see empty-response warnings and recovery
                # status messages.  _mute_post_response was set during a
                # prior housekeeping tool turn and should not silence the
                # final response path.
                agent._mute_post_response = False
                
                # Check if response only has think block with no actual content after it
                if not agent._has_content_after_think_block(final_response):
                    # ── Partial stream recovery ─────────────────────
                    # If content was already streamed to the user before
                    # the connection died, use it as the final response
                    # instead of falling through to prior-turn fallback
                    # or wasting API calls on retries.
                    _partial_streamed = (
                        getattr(agent, "_current_streamed_assistant_text", "") or ""
                    )
                    if agent._has_content_after_think_block(_partial_streamed):
                        _turn_exit_reason = "partial_stream_recovery"
                        _recovered = agent._strip_think_blocks(_partial_streamed).strip()
                        logger.info(
                            "Partial stream content delivered (%d chars) "
                            "— using as final response",
                            len(_recovered),
                        )
                        agent._emit_status(
                            "↻ Stream interrupted — using delivered content "
                            "as final response"
                        )
                        final_response = _recovered
                        agent._response_was_previewed = True
                        break

                    # If the previous turn already delivered real content alongside
                    # HOUSEKEEPING tool calls (e.g. "You're welcome!" + memory save),
                    # the model has nothing more to say. Use the earlier content
                    # immediately instead of wasting API calls on retries.
                    # NOTE: Only use this shortcut when ALL tools in that turn were
                    # housekeeping (memory, todo, etc.).  When substantive tools
                    # were called (terminal, search_files, etc.), the content was
                    # likely mid-task narration ("I'll scan the directory...") and
                    # the empty follow-up means the model choked — let the
                    # post-tool nudge below handle that instead of exiting early.
                    fallback = getattr(agent, '_last_content_with_tools', None)
                    if fallback and getattr(agent, '_last_content_tools_all_housekeeping', False):
                        _turn_exit_reason = "fallback_prior_turn_content"
                        logger.info("Empty follow-up after tool calls — using prior turn content as final response")
                        agent._emit_status("↻ Empty response after tool calls — using earlier content as final answer")
                        agent._last_content_with_tools = None
                        agent._last_content_tools_all_housekeeping = False
                        agent._empty_content_retries = 0
                        # Do NOT modify the assistant message content — the
                        # old code injected "Calling the X tools..." which
                        # poisoned the conversation history.  Just use the
                        # fallback text as the final response and break.
                        final_response = agent._strip_think_blocks(fallback).strip()
                        agent._response_was_previewed = True
                        break

                    # ── Post-tool-call empty response nudge ───────────
                    # The model returned empty after executing tool calls.
                    # This covers two cases:
                    #  (a) No prior-turn content at all — model went silent
                    #  (b) Prior turn had content + SUBSTANTIVE tools (the
                    #      fallback above was skipped because the content
                    #      was mid-task narration, not a final answer)
                    # Instead of giving up, nudge the model to continue by
                    # appending a user-level hint.  This is the #9400 case:
                    # weaker models (mimo-v2-pro, GLM-5, etc.) sometimes
                    # return empty after tool results instead of continuing
                    # to the next step.  One retry with a nudge usually
                    # fixes it.
                    _prior_was_tool = any(
                        m.get("role") == "tool"
                        for m in messages[-5:]  # check recent messages
                    )
                    # Detect Qwen3/Ollama-style in-content thinking blocks.
                    # Ollama puts <think> in the content field (not in
                    # reasoning_content), so _has_structured below would
                    # miss it.  We check here so thinking-only responses
                    # after tool calls route to prefill instead of nudge.
                    _has_inline_thinking = bool(
                        re.search(
                            r'<think>|<thinking>|<reasoning>',
                            final_response or "",
                            re.IGNORECASE,
                        )
                    )
                    if (
                        _prior_was_tool
                        and not getattr(agent, "_post_tool_empty_retried", False)
                        and not _has_inline_thinking  # thinking model still working — let prefill handle
                    ):
                        agent._post_tool_empty_retried = True
                        # Clear stale narration so it doesn't resurface
                        # on a later empty response after the nudge.
                        agent._last_content_with_tools = None
                        agent._last_content_tools_all_housekeeping = False
                        logger.info(
                            "Empty response after tool calls — nudging model "
                            "to continue processing"
                        )
                        agent._emit_status(
                            "⚠️ Model returned empty after tool calls — "
                            "nudging to continue"
                        )
                        # Append the empty assistant message first so the
                        # message sequence stays valid:
                        #   tool(result) → assistant("(empty)") → user(nudge)
                        # Without this, we'd have tool → user which most
                        # APIs reject as an invalid sequence.
                        _nudge_msg = agent._build_assistant_message(assistant_message, finish_reason)
                        _nudge_msg["content"] = "(empty)"
                        _nudge_msg["_empty_recovery_synthetic"] = True
                        messages.append(_nudge_msg)
                        messages.append({
                            "role": "user",
                            "content": (
                                "You just executed tool calls but returned an "
                                "empty response. Please process the tool "
                                "results above and continue with the task."
                            ),
                            "_empty_recovery_synthetic": True,
                        })
                        continue

                    # ── Thinking-only prefill continuation ──────────
                    # The model produced structured reasoning (via API
                    # fields) but no visible text content.  Rather than
                    # giving up, append the assistant message as-is and
                    # continue — the model will see its own reasoning
                    # on the next turn and produce the text portion.
                    # Inspired by clawdbot's "incomplete-text" recovery.
                    # Also covers Qwen3/Ollama in-content <think> blocks
                    # (detected above as _has_inline_thinking).
                    _has_structured = bool(
                        getattr(assistant_message, "reasoning", None)
                        or getattr(assistant_message, "reasoning_content", None)
                        or getattr(assistant_message, "reasoning_details", None)
                        or _has_inline_thinking
                    )
                    if _has_structured and agent._thinking_prefill_retries < 2:
                        agent._thinking_prefill_retries += 1
                        logger.info(
                            "Thinking-only response (no visible content) — "
                            "prefilling to continue (%d/2)",
                            agent._thinking_prefill_retries,
                        )
                        agent._emit_status(
                            f"↻ Thinking-only response — prefilling to continue "
                            f"({agent._thinking_prefill_retries}/2)"
                        )
                        interim_msg = agent._build_assistant_message(
                            assistant_message, "incomplete"
                        )
                        interim_msg["_thinking_prefill"] = True
                        messages.append(interim_msg)
                        agent._session_messages = messages
                        agent._save_session_log(messages)
                        continue

                    # ── Empty response retry ──────────────────────
                    # Model returned nothing usable.  Retry up to 3
                    # times before attempting fallback.  This covers
                    # both truly empty responses (no content, no
                    # reasoning) AND reasoning-only responses after
                    # prefill exhaustion — models like mimo-v2-pro
                    # always populate reasoning fields via OpenRouter,
                    # so the old `not _has_structured` guard blocked
                    # retries for every reasoning model after prefill.
                    _truly_empty = not agent._strip_think_blocks(
                        final_response
                    ).strip()
                    _prefill_exhausted = (
                        _has_structured
                        and agent._thinking_prefill_retries >= 2
                    )
                    if _truly_empty and (not _has_structured or _prefill_exhausted) and agent._empty_content_retries < 3:
                        agent._empty_content_retries += 1
                        logger.warning(
                            "Empty response (no content or reasoning) — "
                            "retry %d/3 (model=%s)",
                            agent._empty_content_retries, agent.model,
                        )
                        agent._emit_status(
                            f"⚠️ Empty response from model — retrying "
                            f"({agent._empty_content_retries}/3)"
                        )
                        continue

                    # ── Exhausted retries — try fallback provider ──
                    # Before giving up with "(empty)", attempt to
                    # switch to the next provider in the fallback
                    # chain.  This covers the case where a model
                    # (e.g. GLM-4.5-Air) consistently returns empty
                    # due to context degradation or provider issues.
                    if _truly_empty and agent._fallback_chain:
                        logger.warning(
                            "Empty response after %d retries — "
                            "attempting fallback (model=%s, provider=%s)",
                            agent._empty_content_retries, agent.model,
                            agent.provider,
                        )
                        agent._emit_status(
                            "⚠️ Model returning empty responses — "
                            "switching to fallback provider..."
                        )
                        if agent._try_activate_fallback():
                            agent._empty_content_retries = 0
                            agent._emit_status(
                                f"↻ Switched to fallback: {agent.model} "
                                f"({agent.provider})"
                            )
                            logger.info(
                                "Fallback activated after empty responses: "
                                "now using %s on %s",
                                agent.model, agent.provider,
                            )
                            continue

                    # Exhausted retries and fallback chain (or no
                    # fallback configured).  Fall through to the
                    # "(empty)" terminal.
                    _turn_exit_reason = "empty_response_exhausted"
                    reasoning_text = agent._extract_reasoning(assistant_message)
                    agent._drop_trailing_empty_response_scaffolding(messages)
                    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
                    assistant_msg["content"] = "(empty)"
                    # This is a user-facing failure sentinel for the gateway,
                    # not real assistant content. Persisting it makes later
                    # "continue" turns replay assistant("(empty)") as if it
                    # were a meaningful model response, which can keep long
                    # tool-heavy sessions stuck in empty-response loops.
                    assistant_msg["_empty_terminal_sentinel"] = True
                    messages.append(assistant_msg)

                    if reasoning_text:
                        reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                        logger.warning(
                            "Reasoning-only response (no visible content) "
                            "after exhausting retries and fallback. "
                            "Reasoning: %s", reasoning_preview,
                        )
                        agent._emit_status(
                            "⚠️ Model produced reasoning but no visible "
                            "response after all retries. Returning empty."
                        )
                    else:
                        logger.warning(
                            "Empty response (no content or reasoning) "
                            "after %d retries. No fallback available. "
                            "model=%s provider=%s",
                            agent._empty_content_retries, agent.model,
                            agent.provider,
                        )
                        agent._emit_status(
                            "❌ Model returned no content after all retries"
                            + (" and fallback attempts." if agent._fallback_chain else
                               ". No fallback providers configured.")
                        )

                    final_response = "(empty)"
                    break
                
                # Reset retry counter/signature on successful content
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0

                if (
                    agent.api_mode == "codex_responses"
                    and agent.valid_tool_names
                    and codex_ack_continuations < 2
                    and agent._looks_like_codex_intermediate_ack(
                        user_message=user_message,
                        assistant_content=final_response,
                        messages=messages,
                    )
                ):
                    codex_ack_continuations += 1
                    interim_msg = agent._build_assistant_message(assistant_message, "incomplete")
                    messages.append(interim_msg)
                    agent._emit_interim_assistant_message(interim_msg)

                    continue_msg = {
                        "role": "user",
                        "content": (
                            "[System: Continue now. Execute the required tool calls and only "
                            "send your final answer after completing the task.]"
                        ),
                    }
                    messages.append(continue_msg)
                    agent._session_messages = messages
                    agent._save_session_log(messages)
                    continue

                codex_ack_continuations = 0

                if truncated_response_parts:
                    final_response = "".join(truncated_response_parts) + final_response
                    truncated_response_parts = []
                    length_continue_retries = 0
                
                final_response = agent._strip_think_blocks(final_response).strip()
                
                final_msg = agent._build_assistant_message(assistant_message, finish_reason)

                # Pop thinking-only prefill and empty-response retry
                # scaffolding before appending the final response.  These
                # internal turns are only for the next API retry and should
                # not become durable transcript context.
                while (
                    messages
                    and isinstance(messages[-1], dict)
                    and (
                        messages[-1].get("_thinking_prefill")
                        or messages[-1].get("_empty_recovery_synthetic")
                        or messages[-1].get("_empty_terminal_sentinel")
                    )
                ):
                    messages.pop()

                messages.append(final_msg)
                
                _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
                if not agent.quiet_mode:
                    agent._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                break
            
        except Exception as e:
            error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
            try:
                print(f"❌ {error_msg}")
            except (OSError, ValueError):
                logger.error(error_msg)
            
            logger.debug("Outer loop error in API call #%d", api_call_count, exc_info=True)
            
            # If an assistant message with tool_calls was already appended,
            # the API expects a role="tool" result for every tool_call_id.
            # Fill in error results for any that weren't answered yet.
            for idx in range(len(messages) - 1, -1, -1):
                msg = messages[idx]
                if not isinstance(msg, dict):
                    break
                if msg.get("role") == "tool":
                    continue
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    answered_ids = {
                        m["tool_call_id"]
                        for m in messages[idx + 1:]
                        if isinstance(m, dict) and m.get("role") == "tool"
                    }
                    for tc in msg["tool_calls"]:
                        if not tc or not isinstance(tc, dict): continue
                        if tc["id"] not in answered_ids:
                            err_msg = {
                                "role": "tool",
                                "name": _ra().AIAgent._get_tool_call_name_static(tc),
                                "tool_call_id": tc["id"],
                                "content": f"Error executing tool: {error_msg}",
                            }
                            messages.append(err_msg)
                break
            
            # Non-tool errors don't need a synthetic message injected.
            # The error is already printed to the user (line above), and
            # the retry loop continues.  Injecting a fake user/assistant
            # message pollutes history, burns tokens, and risks violating
            # role-alternation invariants.

            # If we're near the limit, break to avoid infinite loops
            if api_call_count >= agent.max_iterations - 1:
                _turn_exit_reason = f"error_near_max_iterations({error_msg[:80]})"
                final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                # Append as assistant so the history stays valid for
                # session resume (avoids consecutive user messages).
                messages.append({"role": "assistant", "content": final_response})
                break
    
    if final_response is None and (
        api_call_count >= agent.max_iterations
        or agent.iteration_budget.remaining <= 0
    ):
        # Budget exhausted — ask the model for a summary via one extra
        # API call with tools stripped.  _handle_max_iterations injects a
        # user message and makes a single toolless request.
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{agent.max_iterations})"
        agent._emit_status(
            f"⚠️ Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
            "— asking model to summarise"
        )
        if not agent.quiet_mode:
            agent._safe_print(
                f"\n⚠️  Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
                "— requesting summary..."
            )
        final_response = agent._handle_max_iterations(messages, api_call_count)

        # If running as a kanban worker, block the task so the dispatcher
        # knows the worker could not complete (rather than treating it as a
        # protocol violation).  The agent loop strips tools before calling
        # _handle_max_iterations, so the model cannot call kanban_block
        # itself — we must do it on its behalf.
        _kanban_task = os.environ.get("HERMES_KANBAN_TASK")
        if _kanban_task:
            try:
                _ra().handle_function_call(
                    "kanban_block",
                    {
                        "task_id": _kanban_task,
                        "reason": (
                            f"Iteration budget exhausted "
                            f"({api_call_count}/{agent.max_iterations}) — "
                            "task could not complete within the allowed "
                            "iterations"
                        ),
                    },
                    task_id=effective_task_id,
                )
                logger.info(
                    "kanban_block called for task %s after iteration "
                    "exhaustion (%d/%d)",
                    _kanban_task, api_call_count, agent.max_iterations,
                )
            except Exception:
                logger.warning(
                    "Failed to call kanban_block after iteration "
                    "exhaustion for task %s",
                    _kanban_task,
                    exc_info=True,
                )

    # Determine if conversation completed successfully
    completed = final_response is not None and api_call_count < agent.max_iterations

    # Save trajectory if enabled.  ``user_message`` may be a multimodal
    # list of parts; the trajectory format wants a plain string.
    agent._save_trajectory(messages, _summarize_user_message_for_log(user_message), completed)

    # Clean up VM and browser for this task after conversation completes
    agent._cleanup_task_resources(effective_task_id)

    # Persist session to both JSON log and SQLite only after private retry
    # scaffolding has been removed. Otherwise a later user "continue" turn
    # can replay assistant("(empty)") / recovery nudges and fall into the
    # same empty-response loop again.
    agent._drop_trailing_empty_response_scaffolding(messages)
    agent._persist_session(messages, conversation_history)

    # ── Turn-exit diagnostic log ─────────────────────────────────────
    # Always logged at INFO so agent.log captures WHY every turn ended.
    # When the last message is a tool result (agent was mid-work), log
    # at WARNING — this is the "just stops" scenario users report.
    _last_msg_role = messages[-1].get("role") if messages else None
    _last_tool_name = None
    if _last_msg_role == "tool":
        # Walk back to find the assistant message with the tool call
        for _m in reversed(messages):
            if _m.get("role") == "assistant" and _m.get("tool_calls"):
                _tcs = _m["tool_calls"]
                if _tcs and isinstance(_tcs[0], dict):
                    _last_tool_name = _tcs[-1].get("function", {}).get("name")
                break

    _turn_tool_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
    )
    _resp_len = len(final_response) if final_response else 0
    _budget_used = agent.iteration_budget.used if agent.iteration_budget else 0
    _budget_max = agent.iteration_budget.max_total if agent.iteration_budget else 0

    _diag_msg = (
        "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
        "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
    )
    _diag_args = (
        _turn_exit_reason, agent.model, api_call_count, agent.max_iterations,
        _budget_used, _budget_max,
        _turn_tool_count, _last_msg_role, _resp_len,
        agent.session_id or "none",
    )

    if _last_msg_role == "tool" and not interrupted:
        # Agent was mid-work — this is the "just stops" case.
        logger.warning(
            "Turn ended with pending tool result (agent may appear stuck). "
            + _diag_msg + " last_tool=%s",
            *_diag_args, _last_tool_name,
        )
    else:
        logger.info(_diag_msg, *_diag_args)

    # File-mutation verifier footer.
    # If one or more ``write_file`` / ``patch`` calls failed during this
    # turn and were never superseded by a successful write to the same
    # path, append an advisory footer to the assistant response.  This
    # catches the specific case — reported by Ben Eng (#15524-adjacent)
    # — where a model issues a batch of parallel patches, half of them
    # fail with "Could not find old_string", and the model summarises
    # the turn claiming every file was edited.  The user then has to
    # manually run ``git status`` to catch the lie.  With this footer
    # the truth is surfaced on every turn, so over-claiming is
    # structurally impossible past the model.
    #
    # Gate: only applied when a real text response exists for this
    # turn and the user didn't interrupt.  Empty/interrupted turns
    # already have other surface text that shouldn't be augmented.
    if final_response and not interrupted:
        try:
            _failed = getattr(agent, "_turn_failed_file_mutations", None) or {}
            if _failed and agent._file_mutation_verifier_enabled():
                footer = agent._format_file_mutation_failure_footer(_failed)
                if footer:
                    final_response = final_response.rstrip() + "\n\n" + footer
        except Exception as _ver_err:
            logger.debug("file-mutation verifier footer failed: %s", _ver_err)

    # Plugin hook: transform_llm_output
    # Fired once per turn after the tool-calling loop completes.
    # Plugins can transform the LLM's output text before it's returned.
    # First hook to return a string wins; None/empty return leaves text unchanged.
    if final_response and not interrupted:
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _transform_results = _invoke_hook(
                "transform_llm_output",
                response_text=final_response,
                session_id=agent.session_id or "",
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
            for _hook_result in _transform_results:
                if isinstance(_hook_result, str) and _hook_result:
                    final_response = _hook_result
                    break  # First non-empty string wins
        except Exception as exc:
            logger.warning("transform_llm_output hook failed: %s", exc)

    # Plugin hook: post_llm_call
    # Fired once per turn after the tool-calling loop completes.
    # Plugins can use this to persist conversation data (e.g. sync
    # to an external memory system).
    if final_response and not interrupted:
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "post_llm_call",
                session_id=agent.session_id,
                user_message=original_user_message,
                assistant_response=final_response,
                conversation_history=list(messages),
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("post_llm_call hook failed: %s", exc)

    # Extract reasoning from the CURRENT turn only.  Walk backwards
    # but stop at the user message that started this turn — anything
    # earlier is from a prior turn and must not leak into the reasoning
    # box (confusing stale display; #17055).  Within the current turn
    # we still want the *most recent* non-empty reasoning: many
    # providers (Claude thinking, DeepSeek v4, Codex Responses) emit
    # reasoning on the tool-call step and leave the final-answer step
    # with reasoning=None, so picking only the last assistant would
    # silently drop legitimate same-turn reasoning.
    last_reasoning = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            break  # turn boundary — don't cross into prior turns
        if msg.get("role") == "assistant" and msg.get("reasoning"):
            last_reasoning = msg["reasoning"]
            break

    # Build result with interrupt info if applicable
    result = {
        "final_response": final_response,
        "last_reasoning": last_reasoning,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": completed,
        "turn_exit_reason": _turn_exit_reason,
        "partial": False,  # True only when stopped due to invalid tool calls
        "interrupted": interrupted,
        "response_previewed": getattr(agent, "_response_was_previewed", False),
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "input_tokens": agent.session_input_tokens,
        "output_tokens": agent.session_output_tokens,
        "cache_read_tokens": agent.session_cache_read_tokens,
        "cache_write_tokens": agent.session_cache_write_tokens,
        "reasoning_tokens": agent.session_reasoning_tokens,
        "prompt_tokens": agent.session_prompt_tokens,
        "completion_tokens": agent.session_completion_tokens,
        "total_tokens": agent.session_total_tokens,
        "last_prompt_tokens": getattr(agent.context_compressor, "last_prompt_tokens", 0) or 0,
        "estimated_cost_usd": agent.session_estimated_cost_usd,
        "cost_status": agent.session_cost_status,
        "cost_source": agent.session_cost_source,
    }
    if agent._tool_guardrail_halt_decision is not None:
        result["guardrail"] = agent._tool_guardrail_halt_decision.to_metadata()
    # If a /steer landed after the final assistant turn (no more tool
    # batches to drain into), hand it back to the caller so it can be
    # delivered as the next user turn instead of being silently lost.
    _leftover_steer = agent._drain_pending_steer()
    if _leftover_steer:
        result["pending_steer"] = _leftover_steer
    agent._response_was_previewed = False
    
    # Include interrupt message if one triggered the interrupt
    if interrupted and agent._interrupt_message:
        result["interrupt_message"] = agent._interrupt_message
    
    # Clear interrupt state after handling
    agent.clear_interrupt()

    # Clear stream callback so it doesn't leak into future calls
    agent._stream_callback = None

    # Check skill trigger NOW — based on how many tool iterations THIS turn used.
    _should_review_skills = False
    if (agent._skill_nudge_interval > 0
            and agent._iters_since_skill >= agent._skill_nudge_interval
            and "skill_manage" in agent.valid_tool_names):
        _should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider: sync the completed turn + queue next prefetch.
    agent._sync_external_memory_for_turn(
        original_user_message=original_user_message,
        final_response=final_response,
        interrupted=interrupted,
    )

    # Background memory/skill review — runs AFTER the response is delivered
    # so it never competes with the user's task for model attention.
    if final_response and not interrupted and (_should_review_memory or _should_review_skills):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )
        except Exception:
            pass  # Background review is best-effort

    # Note: Memory provider on_session_end() + shutdown_all() are NOT
    # called here — run_conversation() is called once per user message in
    # multi-turn sessions. Shutting down after every turn would kill the
    # provider before the second message. Actual session-end cleanup is
    # handled by the CLI (atexit / /reset) and gateway (session expiry /
    # _reset_session).

    # Plugin hook: on_session_end
    # Fired at the very end of every run_conversation call.
    # Plugins can use this for cleanup, flushing buffers, etc.
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_end",
            session_id=agent.session_id,
            completed=completed,
            interrupted=interrupted,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_end hook failed: %s", exc)

    return result



__all__ = ["run_conversation"]
