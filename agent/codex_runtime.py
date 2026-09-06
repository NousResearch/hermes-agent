"""Codex API runtime — App Server and Responses-API streaming paths. Every entry point takes the parent
AIAgent first: ``run_codex_app_server_turn`` drives one ``codex app-server`` subprocess turn;
``run_codex_stream`` runs one streaming Codex Responses call."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

from agent.stream_single_writer import claim_stream_writer, stream_writer_is_current

logger = logging.getLogger(__name__)


def _call_guarded(fn: Callable | None, fail_msg: str, *fail_args: Any, args: tuple = (), kwargs: dict | None = None):
    """Invoke an optional display/debug callback; a buggy hook must never tear down the turn."""
    if fn is None:
        return
    try:
        fn(*args, **(kwargs or {}))
    except Exception:
        logger.debug(fail_msg, *fail_args, exc_info=True)


def _codex_request_failure_details(error: BaseException) -> tuple[int | None, str]:
    """(serialized request bytes, exception class chain); the buffered ``httpx.Request`` content
    on OpenAI connection errors gives the exact byte count without logging payloads or URLs."""
    request_body_bytes: int | None = None
    exception_classes: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        exception_classes.append(type(current).__name__)
        if request_body_bytes is None:
            content = None
            with suppress(Exception):
                content = getattr(getattr(current, "request", None), "content", None)
            if isinstance(content, str):
                request_body_bytes = len(content.encode("utf-8"))
            elif isinstance(content, (bytes, bytearray, memoryview)):
                request_body_bytes = len(content)
        implicit_chain = current.__cause__ is None and not current.__suppress_context__
        current = current.__context__ if implicit_chain else current.__cause__
    return request_body_bytes, " <- ".join(exception_classes)


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        # Only the str->int parse is guarded; a float NaN still raises like it always has.
        with suppress(ValueError):
            return max(int(value), 0)
    return 0


def _queue_token_counts(agent, fail_msg: str, *fail_extra: Any, counts: Callable[[], dict]) -> None:
    """Enqueue per-call accounting for the SessionDB background writer. ``counts`` is built
    lazily inside the guarded try so a stub agent without a session DB is never touched."""
    if not (agent._session_db and agent.session_id):
        return
    try:
        if not agent._session_db_created:
            agent._ensure_db_session()
        agent._session_db.queue_token_counts(agent.session_id, **counts())
    except Exception as exc:
        logger.debug(fail_msg, agent.session_id, *fail_extra, exc)


def _record_codex_app_server_usage(agent, turn) -> dict[str, Any]:
    """Translate Codex app-server token usage into Hermes accounting. Prompt bucket = uncached + cached
    input (the protocol exposes no cache-write tokens); a turn with no usage still counts as one API call."""
    agent.session_api_calls += 1
    usage = getattr(turn, "token_usage_last", None)
    compressor = getattr(agent, "context_compressor", None)

    def billing(**extra):
        return dict(model=agent.model, billing_provider=agent.provider, billing_base_url=agent.base_url, api_call_count=1, **extra)
    if not isinstance(usage, dict) or not usage:
        if compressor is not None and getattr(compressor, "awaiting_real_usage_after_compression", False):
            # No usage cannot adjudicate the pending compaction; unlatch preflight deferral.
            compressor.update_from_response({})
        _queue_token_counts(agent, "Codex app-server api-call persistence failed (session=%s): %s",
                            counts=lambda: billing(billing_mode="subscription_included"))
        return {}
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
    canonical_usage = CanonicalUsage(
        input_tokens=_coerce_usage_int(usage.get("inputTokens")), output_tokens=_coerce_usage_int(usage.get("outputTokens")),
        cache_read_tokens=_coerce_usage_int(usage.get("cachedInputTokens")), cache_write_tokens=0,
        reasoning_tokens=_coerce_usage_int(usage.get("reasoningOutputTokens")), raw_usage=usage,
    )
    prompt_tokens = canonical_usage.prompt_tokens
    total_tokens = _coerce_usage_int(usage.get("totalTokens")) or canonical_usage.total_tokens
    token_counts = {f: getattr(canonical_usage, f) for f in
                    ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")}
    usage_dict = {"prompt_tokens": prompt_tokens, "completion_tokens": canonical_usage.output_tokens,
                  "total_tokens": total_tokens, **token_counts}
    if compressor is not None:
        try:
            compressor.update_from_response(usage_dict)
            context_window = getattr(turn, "model_context_window", None)
            if isinstance(context_window, int) and context_window > 0:
                compressor.context_length = context_window
        except Exception:
            logger.debug("codex app-server usage update failed", exc_info=True)
    for key, value in usage_dict.items():
        setattr(agent, f"session_{key}", getattr(agent, f"session_{key}") + value)
    cost_result = estimate_usage_cost(
        agent.model, canonical_usage, provider=agent.provider, base_url=agent.base_url, api_key=getattr(agent, "api_key", ""),
    )
    cost_usd = float(cost_result.amount_usd) if cost_result.amount_usd is not None else None
    if cost_usd is not None:
        agent.session_estimated_cost_usd += cost_usd
    agent.session_cost_status, agent.session_cost_source = cost_result.status, cost_result.source
    cost_fields = {"estimated_cost_usd": cost_usd, "cost_status": cost_result.status, "cost_source": cost_result.source}
    _queue_token_counts(
        agent, "Codex app-server token persistence failed (session=%s, tokens=%d): %s", total_tokens,
        counts=lambda: billing(**token_counts, **cost_fields,
                               billing_mode="subscription_included" if cost_result.status == "included" else None),
    )
    return {**usage_dict, "last_prompt_tokens": prompt_tokens, **cost_fields}


def _record_codex_app_server_compaction(agent, turn, *, approx_tokens: int | None = None, force: bool = False) -> bool:
    """Record a Codex-native compaction boundary: the app-server owns the compacted thread,
    so local transcript rows are NOT rewritten — only session event/usage counters."""
    if not force and not getattr(turn, "compacted", False):
        return False
    thread_id, turn_id = getattr(turn, "thread_id", None) or "", getattr(turn, "turn_id", None) or ""
    logger.info("codex app-server compaction observed: session=%s thread=%s turn=%s force=%s",
                getattr(agent, "session_id", None) or "none", thread_id, turn_id, force)
    if not force:
        with suppress(Exception):
            from agent.conversation_compression import COMPACTION_STATUS
            agent._emit_status(COMPACTION_STATUS)
    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        compressor.compression_count = getattr(compressor, "compression_count", 0) + 1
        compressor.last_compression_rough_tokens = approx_tokens or 0
        # Codex owns this summary: a prior Hermes deterministic-fallback flag must not leak into it.
        record_boundary = getattr(type(compressor), "record_completed_compaction", None)
        if callable(record_boundary):
            record_boundary(compressor, used_fallback=False)
        elif hasattr(compressor, "_verify_compaction_cleared_threshold"):
            compressor._verify_compaction_cleared_threshold = True
        if not getattr(turn, "token_usage_last", None):
            compressor.last_prompt_tokens, compressor.last_completion_tokens = -1, 0
            compressor.awaiting_real_usage_after_compression = True
    # Provider-side context was rewritten; the usage anchor's transcript snapshot no longer matches.
    agent._usage_anchor = None
    agent._turn_base_usage_anchor = None
    agent._last_compaction_in_place = False
    _call_guarded(getattr(agent, "event_callback", None) or None, "event_callback error on codex session:compress",
                  args=("session:compress", {
                      "platform": getattr(agent, "platform", None) or "", "session_id": getattr(agent, "session_id", None) or "",
                      "old_session_id": "", "in_place": False,
                      "compression_count": getattr(compressor, "compression_count", 0) if compressor is not None else 0,
                      "runtime": "codex_app_server", "thread_id": thread_id, "turn_id": turn_id,
                  }))
    return True


# --- Codex app-server → Hermes UI bridge -------------------------------------
# The app-server bypasses the Hermes tool loop, so the bridge translates JSON-RPC notifications
# into the callbacks the standard runtime fires (tool_progress_callback, _fire_stream_delta, ...).

# Item types that project to a Hermes tool_call (keep in sync with agent/transports/codex_event_projector.py
# so UI names match recorded names). webSearch is codex's built-in tool: no projector entry, still gets a bubble.
_CODEX_TOOL_ITEM_TYPES = frozenset({"commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall", "webSearch"})
# Internal MCP server wrapping Hermes' native tools: its inner dispatch has no tool_progress_callback, so the
# codex-level mcpToolCall IS the display event and the mcp.hermes-tools.* prefix is stripped (users see Hermes tools).
_INTERNAL_MCP_SERVER = "hermes-tools"
_STATIC_TOOL_NAMES = {"commandExecution": "exec_command", "fileChange": "apply_patch", "webSearch": "web_search"}
_STABLE_ID_PREFIXES = {"commandExecution": "exec", "fileChange": "apply_patch"}
_MCP_LIKE_ITEM_TYPES = {"mcpToolCall", "dynamicToolCall"}
# Item types whose preview is the first 120 chars of one string field.
_PREVIEW_FIELDS = {"commandExecution": "command", "webSearch": "query"}


def _item_changes(item: dict) -> list[dict]:
    return [c for c in (item.get("changes") or []) if isinstance(c, dict)]


def _codex_item_to_tool_name(item: dict) -> str:
    """Synthetic Hermes tool name for a codex item (mirrors CodexEventProjector)."""
    item_type = item.get("type") or ""
    if item_type == "mcpToolCall":
        server, tool = item.get("server") or "mcp", item.get("tool") or "unknown"
        return tool if server == _INTERNAL_MCP_SERVER else f"mcp.{server}.{tool}"
    if item_type == "dynamicToolCall":
        return item.get("tool") or "dynamic"
    return _STATIC_TOOL_NAMES.get(item_type) or item_type or "unknown"


def _codex_item_to_args(item: dict) -> dict:
    """Args dict for tool_progress_callback("tool.started"); mirrors the projector shapes."""
    item_type = item.get("type") or ""
    if item_type == "commandExecution":
        return {"command": item.get("command") or "", "cwd": item.get("cwd") or ""}
    if item_type == "fileChange":
        return {"changes": [{"kind": (c.get("kind") or {}).get("type") or "update", "path": c.get("path") or ""}
                            for c in _item_changes(item)]}
    if item_type in _MCP_LIKE_ITEM_TYPES:
        args = item.get("arguments") or {}
        return args if isinstance(args, dict) else {"arguments": args}
    return {"query": item.get("query") or ""} if item_type == "webSearch" else {}


def _codex_item_to_preview(item: dict) -> Any:
    """Short preview for the tool.started bubble; None when nothing useful (UI tolerates None)."""
    item_type = item.get("type") or ""
    if item_type in _PREVIEW_FIELDS:
        return (item.get(_PREVIEW_FIELDS[item_type]) or "")[:120] or None
    if item_type == "fileChange":
        paths = [c.get("path") for c in _item_changes(item) if c.get("path")]
        return (", ".join(paths[:3]) + (f", +{len(paths) - 3} more" if len(paths) > 3 else "")) if paths else None
    if item_type in _MCP_LIKE_ITEM_TYPES:
        args = item.get("arguments") or {}
        if isinstance(args, dict) and args:
            with suppress(TypeError, ValueError):
                return json.dumps(args, ensure_ascii=False)[:120]
    return None


def _codex_item_completion_payload(item: dict) -> tuple[str, bool]:
    """(result_text, is_error) for a completed tool item; mirrors the projector's tool-result content."""
    item_type = item.get("type") or ""
    if item_type == "commandExecution":
        out, exit_code = item.get("aggregatedOutput") or "", item.get("exitCode")
        is_error = bool(exit_code is not None and exit_code != 0)
        return (f"[exit {exit_code}]\n{out}" if is_error else out), is_error
    if item_type == "fileChange":
        status = item.get("status") or "unknown"
        n = len(item.get("changes") or [])
        return f"apply_patch status={status}, {n} change(s)", status not in {"completed", "applied", "success"}
    if item_type == "mcpToolCall":
        if error := item.get("error"):
            return f"[error] {json.dumps(error, ensure_ascii=False)[:1000]}", True
        result = item.get("result")
        return (json.dumps(result, ensure_ascii=False)[:4000] if result is not None else ""), False
    if item_type == "dynamicToolCall":
        content_items, success = item.get("contentItems") or [], item.get("success", True)
        has_items = isinstance(content_items, list) and content_items
        return (json.dumps(content_items, ensure_ascii=False)[:4000] if has_items else f"success={success}"), not bool(success)
    return "", False


def _stable_call_id(item: dict, name: str) -> str:
    """Deterministic tool_call id mirroring CodexEventProjector (live TUI card correlates with projected history)."""
    from agent.transports.codex_event_projector import _deterministic_call_id
    item_type = item.get("type") or ""
    tool = item.get("tool") or "unknown"
    prefix = {"mcpToolCall": f"mcp__{item.get('server') or 'mcp'}__{tool}", "dynamicToolCall": f"dyn_{tool}"}.get(item_type)
    return _deterministic_call_id(prefix or _STABLE_ID_PREFIXES.get(item_type, name), item.get("id") or "")


def make_codex_app_server_event_bridge(agent) -> Callable[[dict], None]:
    """Build the ``on_event`` callback for ``CodexAppServerSession(on_event=...)``.

    Tool items fire ``tool_progress_callback`` plus the stable-ID ``tool_start_callback`` /
    ``tool_complete_callback`` card hooks; deltas go to ``_fire_stream_delta`` / ``_fire_reasoning_delta``;
    a completed agentMessage goes to ``_emit_interim_assistant_message`` (the gateway's ``already_streamed``
    check dedupes against streamed deltas). Every callback is guarded so a buggy display hook cannot
    tear down the turn loop."""
    # item_id -> (tool_name, args, started_monotonic); duration even when codex omits durationMs.
    started: dict[str, tuple[str, dict, float]] = {}

    def agent_cb(attr: str, fail_msg: str, *fail_args: Any, args: tuple = (), kwargs: dict | None = None) -> None:
        _call_guarded(getattr(agent, attr, None), fail_msg, *fail_args, args=args, kwargs=kwargs)

    def _fire_tool_started(item: dict) -> None:
        item_id, name = item.get("id") or "", _codex_item_to_tool_name(item)
        args = _codex_item_to_args(item)
        if item_id:
            started[item_id] = (name, args, time.monotonic())
        agent_cb("tool_progress_callback", "tool_progress_callback raised on tool.started for %s", name,
                 args=("tool.started", name, _codex_item_to_preview(item), args))
        # Stable-ID tool card (TUI/desktop) fires alongside the progress bubble.
        agent_cb("tool_start_callback", "tool_start_callback raised for %s", name,
                 args=(_stable_call_id(item, name), name, args))

    def _fire_tool_completed(item: dict) -> None:
        name = _codex_item_to_tool_name(item)
        prior = started.pop(item_id, None) if (item_id := item.get("id") or "") else None
        # Prefer codex's durationMs; else our started timestamp; else None (some codex
        # versions only emit completed for fast items).
        codex_ms = item.get("durationMs")
        has_codex_ms = isinstance(codex_ms, (int, float)) and codex_ms >= 0
        duration: Any = codex_ms / 1000.0 if has_codex_ms else (time.monotonic() - prior[2] if prior else None)
        result, is_error = _codex_item_completion_payload(item)
        agent_cb("tool_progress_callback", "tool_progress_callback raised on tool.completed for %s", name,
                 args=("tool.completed", name, None, None),
                 kwargs={"duration": duration, "is_error": is_error, "result": result})
        args = prior[1] if prior is not None else _codex_item_to_args(item)
        agent_cb("tool_complete_callback", "tool_complete_callback raised for %s", name,
                 args=(_stable_call_id(item, name), name, args, result))

    def _fire_delta(params: dict, attr: str) -> None:
        text = params.get("delta") or params.get("text") or ""
        # Single-writer guard (#65991): a superseded stream must not pollute the turn's accumulated text
        # (which also feeds the interim-visible-text de-dup comparison), even when a caller reaches this
        # directly (the tool-suppressed content path) rather than through _fire_stream_delta.
        if isinstance(text, str) and text:
            agent_cb(attr, f"{attr} raised", args=(text,))

    def _fire_agent_message_completed(item: dict) -> None:
        text = item.get("text") or ""
        # display.show_commentary=false keeps mid-turn narration off the interim path too (codex_responses contract).
        if isinstance(text, str) and text.strip() and getattr(agent, "show_commentary", True):
            agent_cb("_emit_interim_assistant_message", "_emit_interim_assistant_message raised",
                     args=({"role": "assistant", "content": text},))

    def _on_item(params: dict, completed: bool) -> None:
        item = params.get("item")
        if not isinstance(item, dict):
            return
        item_type = item.get("type") or ""
        if item_type in _CODEX_TOOL_ITEM_TYPES:
            (_fire_tool_completed if completed else _fire_tool_started)(item)
        elif completed and item_type == "agentMessage":
            _fire_agent_message_completed(item)
    handlers: dict[str, Callable[[dict], None]] = {
        "item/agentMessage/delta": lambda p: _fire_delta(p, "_fire_stream_delta"),
        "item/reasoning/delta": lambda p: _fire_delta(p, "_fire_reasoning_delta"),
        "item/reasoning/summaryDelta": lambda p: _fire_delta(p, "_fire_reasoning_delta"),
        "item/started": lambda p: _on_item(p, completed=False), "item/completed": lambda p: _on_item(p, completed=True),
    }

    def on_event(note: dict) -> None:
        handler = handlers.get(note.get("method") or "") if isinstance(note, dict) else None
        if handler is not None:
            params = note.get("params")
            handler(params if isinstance(params, dict) else {})
    return on_event


# --- Codex app-server turn ----------------------------------------------------


def _close_codex_session(agent) -> None:
    """Drop the session so the next turn respawns codex instead of reusing a dead client."""
    with suppress(Exception):
        agent._codex_session.close()
    agent._codex_session = None


def _consume_user_interrupt(agent, active: bool = True) -> tuple[bool, Any]:
    """(user_interrupted, interrupt_message); clears the agent-level interrupt so a hard
    stop cannot poison the next turn (mirrors the conversation-loop finalizer)."""
    interrupted = bool(active and getattr(agent, "_interrupt_requested", False))
    message = getattr(agent, "_interrupt_message", None) if interrupted else None
    if interrupted:
        agent.clear_interrupt()
    return interrupted, message


def _ensure_codex_session(agent) -> None:
    """Lazily spawn one CodexAppServerSession per AIAgent (reused across turns, closed by the _cleanup hook)."""
    if getattr(agent, "_codex_session", None) is not None:
        return
    from agent.runtime_cwd import resolve_agent_cwd
    from agent.transports.codex_app_server_session import CodexAppServerSession, _ServerRequestRouting
    # Approval callback: Hermes' standard prompt flow when a CLI thread installed one.
    approval_callback = None
    with suppress(Exception):
        from tools.terminal_tool import _get_approval_callback
        approval_callback = _get_approval_callback()
    # Gateway/cron have no UI for codex approval requests, so exec/apply_patch fail closed by default. Only an
    # explicit approval bypass (approvals.mode: off, /yolo, --yolo, HERMES_YOLO_MODE) hands policy to codex's sandbox.
    auto_approve_requests = False
    try:
        from tools.approval import is_approval_bypass_active
        auto_approve_requests = is_approval_bypass_active()
    except Exception:
        logger.debug("codex app-server: approval-bypass lookup failed; keeping fail-closed default", exc_info=True)
    # Bridge codex JSON-RPC notifications (item/started, item/completed, item/agentMessage/delta, ...) into
    # Hermes' gateway UI callbacks (tool_progress_callback, _fire_stream_delta,
    # _emit_interim_assistant_message). Without this, Discord/Telegram users see no live tool-progress or
    # interim commentary while codex_app_server is running — only the final answer (#33200). Supersedes the
    # narrower item/started-only bridge from #38835.
    agent._codex_session = CodexAppServerSession(
        cwd=getattr(agent, "session_cwd", None) or str(resolve_agent_cwd()), approval_callback=approval_callback,
        request_routing=_ServerRequestRouting(auto_approve_exec=auto_approve_requests, auto_approve_apply_patch=auto_approve_requests),
        on_event=make_codex_app_server_event_bridge(agent),
    )


def _persist_projected_messages(agent, turn, messages: List[Dict[str, Any]]) -> None:
    """Splice the projected messages into ``messages`` and flush them to the session DB.

    Bypasses conversation_loop's per-step _persist_session(); the flush dedups via _DB_PERSISTED_MARKER so
    only the new codex rows are written. The agent stays the sole persister (agent_persisted=True): a
    gateway re-write would re-INSERT the user turn."""
    if not turn.projected_messages:
        return
    from agent.message_metadata import append_message
    for projected_message in turn.projected_messages:
        append_message(messages, projected_message)
    if getattr(agent, "_session_db", None) is None:
        return
    flush_ok = False
    try:
        flush_ok = agent._flush_messages_to_session_db(messages)
    except Exception:
        logger.warning("codex app-server projected-message flush failed", exc_info=True)
    if flush_ok is False:
        # Output already streamed and agent_persisted cannot flip to False: surface the gap loudly.
        logger.warning("codex app-server turn was delivered but could NOT be persisted to the session DB "
                       "(session=%s) — this turn will be missing after restart/resume", getattr(agent, "session_id", None))


def _finish_codex_turn(agent, turn, messages: List[Dict[str, Any]], *, original_user_message: Any,
                       should_review_memory: bool) -> dict[str, Any]:
    """Post-turn bookkeeping mirroring the chat_completions loop; returns usage fields."""
    # run_conversation() already bumped _turns_since_memory / _user_turn_count; only _iters_since_skill is ours.
    agent._iters_since_skill = getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    _record_codex_app_server_compaction(agent, turn)
    usage_result = _record_codex_app_server_usage(agent, turn)
    # Skill nudge check AFTER iters were incremented (same as chat_completions).
    should_review_skills = (0 < agent._skill_nudge_interval <= agent._iters_since_skill
                            and "skill_manage" in agent.valid_tool_names)
    if should_review_skills:
        agent._iters_since_skill = 0
    # External memory sync skipped on interrupt/error (no partial transcripts).
    if not turn.interrupted and turn.error is None:
        _call_guarded(getattr(agent, "_sync_external_memory_for_turn", None), "external memory sync raised", kwargs=dict(
            original_user_message=original_user_message, final_response=turn.final_text, interrupted=False, messages=messages,
        ))
    # Background review fork: only when a trigger tripped AND a real final response exists.
    if turn.final_text and not turn.interrupted and (should_review_memory or should_review_skills):
        _call_guarded(getattr(agent, "_spawn_background_review", None), "background review spawn raised", kwargs=dict(
            messages_snapshot=list(messages), review_memory=should_review_memory, review_skills=should_review_skills,
        ))
    return usage_result


def run_codex_app_server_turn(agent, *, user_message: str, original_user_message: Any, messages: List[Dict[str, Any]],
                              effective_task_id: str, should_review_memory: bool = False) -> Dict[str, Any]:
    """Hand the turn to a ``codex app-server`` subprocess and project its events into ``messages``.
    Returns the chat_completions result shape. The user message is ALREADY in ``messages`` — never append it again."""
    # Defense in depth for compression.checkpoint_required: agent init refuses the combination, but
    # api_mode is mutable. Explicit-True check matches compress_context().
    if getattr(agent, "compression_checkpoint_required", False) is True:
        from agent.conversation_compression import _checkpoint_blocked
        raise _checkpoint_blocked("codex_app_server owns the authoritative thread and compacts it "
                                  "without a truthful pre-compaction transcript boundary")
    _ensure_codex_session(agent)
    try:
        turn = agent._codex_session.run_turn(user_input=user_message)
    except Exception as exc:
        logger.exception("codex app-server turn failed")
        _close_codex_session(agent)
        return _turn_result(
            _consume_user_interrupt(agent), messages, api_calls=0, completed=False, error=str(exc),
            final_response=f"Codex app-server turn failed: {exc}. Fall back to default runtime with `/codex-runtime auto`.",
        )
    interrupt = _consume_user_interrupt(agent, turn.interrupted)
    # Wedged client (deadline blown, watchdog tripped, OAuth refresh died, subprocess exited): retire it.
    if getattr(turn, "should_retire", False):
        logger.warning("codex app-server session retired (turn error: %s)", turn.error)
        _close_codex_session(agent)
    _persist_projected_messages(agent, turn, messages)
    usage_result = _finish_codex_turn(
        agent, turn, messages, original_user_message=original_user_message, should_review_memory=should_review_memory,
    )
    return _turn_result(
        interrupt, messages, api_calls=1, completed=not turn.interrupted and turn.error is None, error=turn.error,
        # We flushed the projected rows ourselves (agent_persisted); the gateway must skip its own DB write.
        final_response=turn.final_text, agent_persisted=True, codex_thread_id=turn.thread_id, codex_turn_id=turn.turn_id,
        **usage_result,
    )


def _turn_result(interrupt: tuple[bool, Any], messages: List[Dict[str, Any]], *, api_calls: int, completed: bool,
                 error: Any, final_response: Any, **extra: Any) -> Dict[str, Any]:
    """Result shape shared with the chat_completions path (``partial`` == ``not completed``)."""
    user_interrupted, interrupt_message = interrupt
    return {
        "final_response": final_response, "messages": messages, "api_calls": api_calls,
        "completed": completed, "partial": not completed, "interrupted": user_interrupted,
        **({"interrupt_message": interrupt_message} if interrupt_message else {}),
        "error": error, **extra,
    }


# --- Event-driven Responses streaming -----------------------------------------
# The SDK's ``responses.stream(...)`` helper rebuilds a typed Response from ``response.completed.response.output``
# and crashes when it is null. We consume raw ``responses.create(stream=True)`` SSE events and assemble the final
# response from ``output_item.done``, so the terminal ``output`` may be null / [] / a string / absent.


def _event_field(event: Any, name: str, default: Any = None) -> Any:
    """Field access for attr-style (SDK objects) and dict (raw JSON) events/items."""
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        value = event.get(name, default)
    return value if value is not None else default


def _raise_stream_error(event: Any) -> None:
    """Raise ``_StreamErrorEvent`` from a ``type=error`` SSE frame. The spec puts code/message/param at the
    top level, but the SDK and several proxies nest them under ``error``; read top-level first, then the envelope."""
    from run_agent import _StreamErrorEvent
    nested = _event_field(event, "error")

    def _error_field(name: str) -> Any:
        value = _event_field(event, name)
        return _event_field(nested, name) if value is None and nested is not None else value
    raw_message = _error_field("message")
    message = (str(raw_message) if raw_message is not None else "stream emitted error event").strip() or "stream emitted error event"
    raise _StreamErrorEvent(message, code=_error_field("code"), param=_error_field("param"))


def _message_phase(item: Any) -> str | None:
    phase = _event_field(item, "phase", None)
    return phase.strip().lower() if isinstance(phase, str) else None


def _output_text_of(item: Any) -> str:
    """Concatenated ``output_text`` parts of a message item ("" if content is not a list)."""
    content_parts = _event_field(item, "content", [])
    parts = content_parts if isinstance(content_parts, list) else []
    return "".join(
        str(_event_field(part, "text", "") or "") for part in parts if _event_field(part, "type", "") == "output_text"
    ).strip()


class _CodexResponseAssembler:
    """Assemble a Response-shaped ``SimpleNamespace`` from raw Responses SSE events.

    Only ``usage`` / ``status`` / ``id`` are read from the terminal frame — never ``response.output``. Output
    items come from ``output_item.done``, or are synthesized from text deltas, or settled from function calls
    announced via ``output_item.added`` but never confirmed (some backends omit per-item done events on success)."""

    has_tool_calls = first_delta_fired = saw_terminal = False
    next_output_sequence = 0
    active_message_phase: str | None = None
    # Reasoning summary parts carry no separator; a summary_index change is where the blank line belongs.
    active_summary_index: Any = None
    terminal_status: str = "completed"
    terminal_usage = terminal_response_id = terminal_incomplete_details = terminal_error = None
    # terminal_status defaults to "completed", so settlement needs an explicitly observed response.completed frame.
    saw_response_completed = False

    def __init__(self, *, model, on_text_delta, on_reasoning_delta, on_commentary_message, on_first_delta):
        self.model, self.on_text_delta, self.on_reasoning_delta = model, on_text_delta, on_reasoning_delta
        self.on_commentary_message, self.on_first_delta = on_commentary_message, on_first_delta
        self.output_items: List[Any] = []
        # output_index / first-observed sequence per output item, in lockstep, so settled pending calls merge
        # back in stream order.
        self.output_indexes, self.output_sequences = [], []
        self.text_deltas, self.commentary_text_deltas = [], []
        # pending_function_calls: announced-but-unconfirmed function calls keyed by item id. announced_output_order:
        # first-observed (sequence, output_index) per announced item id so a later .done keeps its announced position.
        self.pending_function_calls: Dict[str, Dict[str, Any]] = {}
        self.announced_output_order: Dict[str, tuple] = {}

    def _safe(self, cb: Callable | None, label: str, *args: Any) -> None:
        _call_guarded(cb, f"Codex stream {label} raised", args=args)

    def _on_item_added(self, event: Any, event_type: str) -> None:
        item = _event_field(event, "item")
        item_type = _event_field(item, "type", "")
        self.active_message_phase = _message_phase(item) if item_type == "message" else None
        if self.active_message_phase == "commentary":
            self.commentary_text_deltas = []
        # Record first-observed ordering for EVERY announced item; .done must reuse it or a mixed
        # announced/pending stream without output_index values reorders the calls.
        item_id = str(_event_field(item, "id", ""))
        if item_id and item_id not in self.announced_output_order:
            self.announced_output_order[item_id] = (self.next_output_sequence, _event_field(event, "output_index"))
            self.next_output_sequence += 1
        if "function_call" in str(item_type):
            self.has_tool_calls = True
            if item_id:
                announced_sequence, announced_index = self.announced_output_order[item_id]
                self.pending_function_calls[item_id] = {
                    "item": item, "arguments": str(_event_field(item, "arguments", "") or ""),
                    "output_index": announced_index, "sequence": announced_sequence,
                }

    def _on_text_delta(self, event: Any, event_type: str) -> None:
        delta_text = _event_field(event, "delta", "")
        if not delta_text:
            return
        # Harmony commentary/analysis text is mid-turn narration, never the final answer: route to the
        # reasoning callback, keep only the item for replay.
        if self.active_message_phase == "commentary":
            self.commentary_text_deltas.append(delta_text)
            # Legacy fallback when no first-class commentary consumer is installed.
            if self.on_commentary_message is None:
                self._safe(self.on_reasoning_delta, "on_reasoning_delta", delta_text)
        elif self.active_message_phase == "analysis":
            self._safe(self.on_reasoning_delta, "on_reasoning_delta", delta_text)
        else:
            self.text_deltas.append(delta_text)
            if self.has_tool_calls:
                return
            if not self.first_delta_fired:
                self.first_delta_fired = True
                self._safe(self.on_first_delta, "on_first_delta")
            self._safe(self.on_text_delta, "on_text_delta", delta_text)

    def _on_function_call(self, event: Any, event_type: str) -> None:
        self.has_tool_calls = True
        pending = self.pending_function_calls.get(str(_event_field(event, "item_id", "")))
        if pending is None:
            return  # the item itself lands on output_item.done
        if "delta" in event_type:
            pending["arguments"] += _event_field(event, "delta", "") or ""
        elif event_type.endswith("function_call_arguments.done"):
            # Authoritative for the accumulated string; an explicit "" (zero-arg call) counts, only a
            # missing field keeps the streamed deltas.
            if (done_args := _event_field(event, "arguments", None)) is not None:
                pending["arguments"] = str(done_args)

    def _on_reasoning_delta(self, event: Any, event_type: str) -> None:
        reasoning_text = _event_field(event, "delta", "")
        if not reasoning_text or self.on_reasoning_delta is None:
            return
        summary_index = _event_field(event, "summary_index")
        if summary_index is not None:
            if self.active_summary_index is not None and summary_index != self.active_summary_index:
                reasoning_text = f"\n\n{reasoning_text}"
            self.active_summary_index = summary_index
        self._safe(self.on_reasoning_delta, "on_reasoning_delta", reasoning_text)

    def _on_item_done(self, event: Any, event_type: str) -> None:
        done_item = _event_field(event, "item")
        if done_item is None:
            return
        self.output_items.append(done_item)
        # Reuse the announced position when known (fresh tail sequence for unannounced items); the .done
        # event's own output_index wins over the announced one.
        done_id = str(_event_field(done_item, "id", ""))
        announced_sequence, announced_index = self.announced_output_order.get(done_id, (None, None))
        if announced_sequence is None:
            announced_sequence, self.next_output_sequence = self.next_output_sequence, self.next_output_sequence + 1
        self.output_indexes.append(_event_field(event, "output_index", announced_index))
        self.output_sequences.append(announced_sequence)
        # Confirmed by the authoritative done event; never settle it twice.
        self.pending_function_calls.pop(done_id, None)
        if _message_phase(done_item) == "commentary" and self.on_commentary_message is not None:
            commentary_text = "".join(self.commentary_text_deltas).strip() or _output_text_of(done_item)
            if commentary_text:
                self._safe(self.on_commentary_message, "on_commentary_message", commentary_text)
            self.commentary_text_deltas = []

    def _on_terminal(self, event: Any, event_type: str) -> bool:
        self.saw_terminal = True
        resp_obj = _event_field(event, "response")
        if resp_obj is not None:
            self.terminal_usage, self.terminal_response_id = _event_field(resp_obj, "usage"), _event_field(resp_obj, "id")
            rstatus = _event_field(resp_obj, "status")
            if isinstance(rstatus, str):
                self.terminal_status = rstatus
            if event_type == "response.incomplete":
                self.terminal_incomplete_details = _event_field(resp_obj, "incomplete_details")
            elif event_type == "response.failed":
                self.terminal_error = _event_field(resp_obj, "error")
        self.saw_response_completed = self.saw_response_completed or event_type == "response.completed"
        self.terminal_status = self.terminal_status or event_type.removeprefix("response.")
        return True

    # Exact-type handlers first, then substring-matched ones in priority order. ``error`` frames
    # carry the provider's real failure reason; raise so the credential pool + classifier see the body.
    _EXACT_HANDLERS = {
        "error": lambda self, event, event_type: _raise_stream_error(event),
        "response.output_item.added": _on_item_added, "response.output_item.done": _on_item_done,
        "response.completed": _on_terminal, "response.incomplete": _on_terminal, "response.failed": _on_terminal,
    }
    _FUZZY_HANDLERS = (
        (lambda t: "output_text.delta" in t, _on_text_delta), (lambda t: "function_call" in t, _on_function_call),
        (lambda t: "reasoning" in t and "delta" in t, _on_reasoning_delta),
    )

    def feed(self, event: Any) -> bool:
        """Process one event; True when the stream hit a terminal frame."""
        event_type = _event_field(event, "type", "")
        event_type = event_type if isinstance(event_type, str) else ""
        handler = self._EXACT_HANDLERS.get(event_type) or next((h for m, h in self._FUZZY_HANDLERS if m(event_type)), None)
        return bool(handler(self, event, event_type)) if handler is not None else False

    def _settled_output(self) -> List[Any]:
        """Merge .done items with settled pending calls, keeping stream order."""
        indexed = list(zip(self.output_indexes, self.output_sequences, self.output_items))
        for pending in self.pending_function_calls.values():
            item = pending["item"]
            indexed.append((pending.get("output_index"), pending["sequence"], SimpleNamespace(
                type="function_call", id=_event_field(item, "id", None), call_id=_event_field(item, "call_id", None),
                name=_event_field(item, "name", None), status="completed",
                # Empty/whitespace arguments become "{}" so zero-delta calls stay executable; malformed
                # non-empty JSON passes through untouched.
                arguments=(pending["arguments"] or "").strip() or "{}",
            )))
        # output_index is optional: protocol order only when every entry has one, else wire order.
        if all(entry[0] is not None for entry in indexed):
            with suppress(TypeError):  # non-comparable index values: keep wire order
                indexed.sort(key=lambda entry: entry[0])
        else:
            indexed.sort(key=lambda entry: entry[1])
        return [entry[2] for entry in indexed]

    def result(self) -> SimpleNamespace:
        # With only plain text deltas (no tool calls), synthesize one message item.
        output: List[Any] = list(self.output_items)
        if not output and self.text_deltas and not self.has_tool_calls:
            content = [SimpleNamespace(type="output_text", text="".join(self.text_deltas))]
            output = [SimpleNamespace(type="message", role="assistant", status="completed", content=content)]
        # Done items stay authoritative; settlement only fills the gap left by backends that omit
        # per-item done events on a successful completion.
        if self.pending_function_calls and self.saw_response_completed:
            output = self._settled_output()
        # No terminal frame AND no usable content = truncated / rejected stream.
        if not self.saw_terminal and not output:
            raise RuntimeError("Codex Responses stream did not emit a terminal response")
        return SimpleNamespace(
            output=output, output_text="".join(self.text_deltas), usage=self.terminal_usage, status=self.terminal_status,
            id=self.terminal_response_id, model=self.model, incomplete_details=self.terminal_incomplete_details,
            error=self.terminal_error)


def _consume_codex_event_stream(
    event_iter: Any, *, model: str, on_text_delta=None, on_reasoning_delta=None, on_commentary_message=None,
    on_first_delta=None, on_event=None, interrupt_check=None,
) -> SimpleNamespace:
    """Consume a Codex Responses SSE stream into a Response-shaped ``SimpleNamespace`` (see
    :class:`_CodexResponseAssembler`; ``status`` is ``completed`` when the stream ended with content but no
    terminal frame; ``model`` comes from kwargs).

    Callbacks: ``on_text_delta`` per output_text delta, suppressed once a function_call is seen;
    ``on_reasoning_delta`` for reasoning and ``phase=analysis`` deltas (also commentary without a commentary
    callback); ``on_commentary_message`` once per completed ``phase=commentary`` message, before any following
    tool item; ``on_first_delta`` one-shot; ``on_event`` every event before any processing; ``interrupt_check()``
    True breaks the loop and may raise ``TimeoutError`` / ``InterruptedError`` for request retirement that
    must not become a partial final response."""
    assembler = _CodexResponseAssembler(model=model, on_text_delta=on_text_delta, on_reasoning_delta=on_reasoning_delta,
                                        on_commentary_message=on_commentary_message, on_first_delta=on_first_delta)
    for event in event_iter:
        if on_event is not None:
            try:
                on_event(event)
            except (TimeoutError, InterruptedError):
                raise  # watchdog / cancellation control flow must propagate
            except Exception:
                logger.debug("Codex stream on_event hook raised", exc_info=True)
        if (interrupt_check is not None and interrupt_check()) or assembler.feed(event):
            break
    return assembler.result()


def _sanitize_consumer_codex_request(agent: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Drop fields the ChatGPT OAuth Codex endpoint rejects, at the final wire boundary (after Relay /
    middleware / ``request_overrides``): a late ``prompt_cache_retention``, top-level or nested in
    ``extra_body``, would otherwise HTTP 400 a valid follow-up."""
    sanitized = dict(request)
    # getattr: run_codex_stream is also driven with stand-in agents carrying only the attrs a path needs.
    backend_predicate = getattr(agent, "_is_codex_backend", None)
    if not (callable(backend_predicate) and bool(backend_predicate())):
        return sanitized
    dropped_from = ["top-level"] if "prompt_cache_retention" in sanitized else []
    sanitized.pop("prompt_cache_retention", None)
    # Copy before editing (caller's mapping must not mutate); drop when emptied.
    extra_body = sanitized.get("extra_body")
    if isinstance(extra_body, dict) and "prompt_cache_retention" in extra_body:
        sanitized["extra_body"] = {k: v for k, v in extra_body.items() if k != "prompt_cache_retention"}
        if not sanitized["extra_body"]:
            sanitized.pop("extra_body")
        dropped_from.append("extra_body")
    if dropped_from:
        logger.warning("Dropped unsupported prompt_cache_retention at consumer Codex wire boundary (model=%s, via %s).",
                       sanitized.get("model", getattr(agent, "model", "unknown")), ", ".join(dropped_from))
    return sanitized


# Bulk request fields carrying the conversation payload; the rest is scalar config the SDK transform handles fast.
_SDK_TRANSFORM_BYPASS_FIELDS = ("input", "tools")


def _is_plain_json_data(value: Any) -> bool:
    """True when ``value`` is purely JSON wire types; pydantic models / generators must keep the typed SDK path."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_plain_json_data(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_is_plain_json_data(item) for item in value)
    return False


def _bypass_sdk_request_transform(stream_kwargs: dict) -> dict:
    """Route bulk payload fields around the SDK's ``maybe_transform``.

    ``responses.create`` re-walks the whole body against the ResponseCreateParams union with the GIL held —
    multi-MB conversations can wedge for hours, pre-network, where no watchdog socket kill helps. The SDK
    merges ``extra_body`` AFTER the transform, so moving wire-format bulk fields there yields a byte-identical
    request without the walk. HERMES_CODEX_SDK_TRANSFORM=1 disables."""
    if os.environ.get("HERMES_CODEX_SDK_TRANSFORM", "").strip().lower() in {"1", "true", "yes", "on"}:
        return stream_kwargs
    moved = {f: stream_kwargs[f] for f in _SDK_TRANSFORM_BYPASS_FIELDS
             if isinstance(stream_kwargs.get(f), (dict, list)) and _is_plain_json_data(stream_kwargs[f])}
    if not moved:
        return stream_kwargs
    bypassed = {key: value for key, value in stream_kwargs.items() if key not in moved}
    extra_body = bypassed.get("extra_body")
    merged = dict(extra_body) if isinstance(extra_body, dict) else {}
    # An explicit caller-provided extra_body entry keeps precedence (SDK post-transform merge).
    bypassed["extra_body"] = {**merged, **{f: v for f, v in moved.items() if f not in merged}}
    return bypassed


def run_codex_stream(agent, api_kwargs: dict, client: Any = None, on_first_delta=None):
    """One streaming Responses API request over raw ``responses.create(stream=True)`` events."""
    import httpx as _httpx
    from openai import APIConnectionError as _APIConnectionError
    from agent import relay_llm
    transport_errors = (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError)
    active_client = client or agent._ensure_primary_openai_client(reason="codex_stream_direct")
    max_stream_retries, model = 1, api_kwargs.get("model")
    # Accumulate streamed text so callers / compat shims can read it.
    agent._codex_streamed_text_parts: list = []
    # Retirement token for THIS request (installed by ``interruptible_api_call``). A watchdog that kills the
    # connection clears the agent-level token, so a worker still draining frames can tell it was retired.
    # ``None`` = no watchdog; every check passes.
    request_token = getattr(agent, "_active_codex_stream_request_token", None)
    # Delta-sink claim for the CURRENT physical attempt (None until the stream opens).
    writer_token = {"value": None}

    def _request_is_current() -> bool:
        return request_token is None or getattr(agent, "_active_codex_stream_request_token", None) is request_token

    def _fenced(fn: Callable[[Any], None]) -> Callable[[Any], None]:
        """Wrap a callback so a retired request's late frames never reach the agent."""
        return lambda value: fn(value) if _request_is_current() else None

    def _on_text_delta(text: str) -> None:
        agent._codex_streamed_text_parts.append(text)
        agent._fire_stream_delta(text)

    def _on_event(event: Any) -> None:  # TTFB watchdog and activity touch — once per SSE event.
        agent._codex_stream_last_event_ts = time.time()
        agent._touch_activity("receiving stream response")

    def _interrupt_or_superseded() -> bool:
        # A retired request must NOT break out of the consume loop (that returns a partial ``final`` with
        # status "completed"); raise so the watchdog's TimeoutError is seen.
        if not _request_is_current():
            raise TimeoutError("Codex Responses stream request retired before terminal response")
        return bool(agent._interrupt_requested)

    def _open_codex_stream(next_api_kwargs: dict[str, Any]):
        stream_kwargs = _sanitize_consumer_codex_request(agent, next_api_kwargs)
        stream_kwargs["stream"] = True
        return active_client.responses.create(**_bypass_sdk_request_transform(stream_kwargs))

    def _log_failure(exc: BaseException) -> None:
        request_body_bytes, exception_chain = _codex_request_failure_details(exc)
        logger.warning("Codex Responses request failed: serialized_request_body_bytes=%s stream_opened=%s "
                       "exception_chain=%s model=%s", "unknown" if request_body_bytes is None else request_body_bytes,
                       str(writer_token["value"] is not None).lower(), exception_chain, getattr(agent, "model", "unknown"))

    def _codex_stream_created(_raw_stream: Any) -> None:
        # Claim the delta sink for THIS attempt; a newer attempt supersedes this token.
        writer_token["value"] = claim_stream_writer(agent)

    def _accept_codex_chunk(_chunk: Any) -> bool:
        token = writer_token["value"]
        if token is None or stream_writer_is_current(agent, token):
            return True
        logger.warning("Codex streaming attempt superseded by a newer stream; stopping consumption to preserve "
                       "the single-writer invariant (model=%s).", api_kwargs.get("model", "unknown"))
        return False

    def _drain_for_finalizer(event_stream: Any) -> None:
        # ``final`` is already assembled; draining only lets Relay run its finalizer. A transport error
        # here must NOT discard the completed, already-billed response.
        try:
            for _ignored in event_stream:
                pass
        except (*transport_errors, _APIConnectionError) as exc:
            if not isinstance(exc, transport_errors):
                _log_failure(exc)
            logger.warning("Codex Responses stream transport finalization failed after a terminal response was already "
                           "received; returning the completed response instead of retrying. %s error=%s",
                           agent._client_log_context(), exc)

    def _close_event_stream(event_stream: Any) -> None:
        close_fn = getattr(event_stream, "close", None)  # None while connect never succeeded
        try:
            if callable(close_fn):
                close_fn()
        except Exception:
            # A failed close can leave this connection checked out of the httpx pool while the caller
            # reuse-caches the client; poison the slot so close really closes the pool. ``client is None``
            # is the shared primary client — never force-shut.
            if client is not None:
                agent._abort_request_openai_client(active_client, reason="codex_stream_close_failed")
    show_commentary = getattr(agent, "show_commentary", True)
    wants_commentary = getattr(agent, "interim_assistant_callback", None) is not None and show_commentary
    on_commentary_message = _fenced(lambda text: agent._fire_streamed_codex_commentary(text)) if wants_commentary else None
    call_role = ("delegated" if getattr(agent, "is_subagent", False)
                 else "fallback" if int(getattr(agent, "_fallback_index", 0) or 0) > 0 else "primary")
    for attempt in range(max_stream_retries + 1):
        if agent._interrupt_requested:
            raise InterruptedError("Agent interrupted before Codex stream retry")
        intercepted_events: list = []
        writer_token["value"] = event_stream = None
        try:
            try:
                event_stream = relay_llm.stream(
                    dict(api_kwargs), _open_codex_stream,
                    session_id=str(getattr(agent, "session_id", "") or ""),
                    name=str(getattr(agent, "provider", "") or "codex"), model_name=str(model or ""),
                    finalizer=lambda: _consume_codex_event_stream(list(intercepted_events), model=model),
                    on_stream_created=_codex_stream_created, on_chunk=intercepted_events.append,
                    chunk_adapter=lambda chunk: chunk, accept_chunk=_accept_codex_chunk,
                    completed_response_predicate=lambda r: bool(hasattr(r, "output") and not hasattr(r, "__iter__")),
                    metadata={"api_mode": "codex_responses", "call_role": call_role, "retry_count": attempt,
                              "api_request_id": getattr(agent, "_current_api_request_id", None)},
                    defer_logical_completion=True,
                )
                final = _consume_codex_event_stream(
                    event_stream, model=model, on_text_delta=_fenced(_on_text_delta),
                    on_reasoning_delta=_fenced(lambda text: agent._fire_reasoning_delta(text)),
                    on_commentary_message=on_commentary_message, on_first_delta=on_first_delta,
                    on_event=_fenced(_on_event), interrupt_check=_interrupt_or_superseded,
                )
            except transport_errors as exc:
                if attempt >= max_stream_retries:
                    _log_failure(exc)
                    raise
                logger.debug(
                    "Codex Responses stream connect failed (attempt %s/%s); retrying. %s error=%s" if event_stream is None
                    else "Codex Responses stream transport failed mid-iteration (attempt %s/%s); retrying. %s error=%s",
                    attempt + 1, max_stream_retries + 1, agent._client_log_context(), exc,
                )
                continue
            except RuntimeError:
                # "No terminal response"; Relay may still hold a finalizer-assembled response.
                if event_stream is not None and event_stream.final_response is not None:
                    return event_stream.final_response
                raise
            except _APIConnectionError as exc:
                _log_failure(exc)
                raise
            if not agent._interrupt_requested:
                _drain_for_finalizer(event_stream)
            if final.status in {"incomplete", "failed"}:
                logger.warning("Codex Responses stream terminal status=%s "
                               "(incomplete_details=%s, error=%s, streamed_chars=%d). %s",
                               final.status, final.incomplete_details, final.error,
                               sum(len(p) for p in agent._codex_streamed_text_parts), agent._client_log_context())
            return final
        finally:
            _close_event_stream(event_stream)


__all__ = [
    "run_codex_app_server_turn", "run_codex_stream",
    "_consume_codex_event_stream", "make_codex_app_server_event_bridge",
]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def run_codex_create_stream_fallback(agent, api_kwargs: dict, client: Any = None):
    """Backward-compatible alias for the unified event-driven path.

    Historically this was the fallback when the SDK's high-level
    ``responses.stream(...)`` helper raised on shape drift.  The primary
    path now does exactly what the fallback did, so this just forwards.
    Kept as a public symbol because tests and a small number of call sites
    still reference it by name.
    """
    return run_codex_stream(agent, api_kwargs, client=client)
# ---- END PLUGIN-COMPAT ----
