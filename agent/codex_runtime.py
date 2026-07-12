"""Codex API runtime — App Server and Responses-API streaming paths.

Extracted from :class:`AIAgent` to keep the agent loop file focused.
Each function takes the parent ``AIAgent`` as its first argument
(``agent``).  AIAgent keeps thin forwarder methods for backward
compatibility.

* ``run_codex_app_server_turn`` — drives one turn through the
  ``codex_app_server`` subprocess client (used when a Codex CLI install
  is the active provider).
* ``run_codex_stream`` — streams a Codex Responses API call (the
  ``codex_responses`` api_mode).
* ``run_codex_create_stream_fallback`` — recovery path when the
  Responses ``stream=True`` initial create fails.
"""

from __future__ import annotations

import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

from agent.stream_single_writer import claim_stream_writer, stream_writer_is_current

logger = logging.getLogger(__name__)


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


def _record_codex_app_server_usage(agent, turn) -> dict[str, Any]:
    """Translate Codex app-server token usage into Hermes accounting.

    Codex app-server reports usage via thread/tokenUsage/updated as:
    inputTokens, cachedInputTokens, outputTokens, reasoningOutputTokens,
    totalTokens.

    Hermes' canonical prompt bucket includes uncached input + cached input.
    The Codex app-server protocol does not currently expose cache-write tokens,
    so that bucket remains zero on this runtime.

    Even when Codex omits usage for a turn, Hermes should still count that turn
    as one API call for session/status accounting.
    """
    agent.session_api_calls += 1

    usage = getattr(turn, "token_usage_last", None)
    if not isinstance(usage, dict) or not usage:
        compressor = getattr(agent, "context_compressor", None)
        if (
            compressor is not None
            and getattr(compressor, "awaiting_real_usage_after_compression", False)
        ):
            # No usage means this turn cannot adjudicate the pending compaction.
            # Consume the marker so a later unrelated reading is not charged to
            # it and preflight deferral cannot stay latched indefinitely.
            compressor.update_from_response({})
        if agent._session_db and agent.session_id:
            try:
                if not agent._session_db_created:
                    agent._ensure_db_session()
                agent._session_db.update_token_counts(
                    agent.session_id,
                    model=agent.model,
                    billing_provider=agent.provider,
                    billing_base_url=agent.base_url,
                    billing_mode="subscription_included",
                    api_call_count=1,
                )
            except Exception as exc:
                logger.debug(
                    "Codex app-server api-call persistence failed (session=%s): %s",
                    agent.session_id, exc,
                )
        return {}

    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    input_tokens = _coerce_usage_int(usage.get("inputTokens"))
    cache_read_tokens = _coerce_usage_int(usage.get("cachedInputTokens"))
    output_tokens = _coerce_usage_int(usage.get("outputTokens"))
    reasoning_tokens = _coerce_usage_int(usage.get("reasoningOutputTokens"))
    reported_total = _coerce_usage_int(usage.get("totalTokens"))

    canonical_usage = CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=0,
        reasoning_tokens=reasoning_tokens,
        raw_usage=usage,
    )
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = reported_total or canonical_usage.total_tokens
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": canonical_usage.reasoning_tokens,
    }

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        try:
            compressor.update_from_response(usage_dict)
            context_window = getattr(turn, "model_context_window", None)
            if isinstance(context_window, int) and context_window > 0:
                compressor.context_length = context_window
        except Exception:
            logger.debug("codex app-server usage update failed", exc_info=True)

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
    agent.session_reasoning_tokens += canonical_usage.reasoning_tokens

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
        except Exception as exc:
            logger.debug(
                "Codex app-server token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, exc,
            )

    return {
        **usage_dict,
        "last_prompt_tokens": prompt_tokens,
        "estimated_cost_usd": float(cost_result.amount_usd)
        if cost_result.amount_usd is not None else None,
        "cost_status": cost_result.status,
        "cost_source": cost_result.source,
    }


def _record_codex_app_server_compaction(
    agent,
    turn,
    *,
    approx_tokens: int | None = None,
    force: bool = False,
) -> bool:
    """Record a Codex-native context compaction boundary in Hermes state.

    The app-server owns the compacted thread context, so Hermes should not
    rewrite local transcript rows here; state.db records the boundary via the
    session event/usage counters while preserving the visible transcript.
    """
    if not force and not getattr(turn, "compacted", False):
        return False

    thread_id = getattr(turn, "thread_id", None) or ""
    turn_id = getattr(turn, "turn_id", None) or ""
    logger.info(
        "codex app-server compaction observed: session=%s thread=%s turn=%s force=%s",
        getattr(agent, "session_id", None) or "none",
        thread_id,
        turn_id,
        force,
    )
    if not force:
        try:
            from agent.conversation_compression import COMPACTION_STATUS

            agent._emit_status(COMPACTION_STATUS)
        except Exception:
            pass

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        compressor.compression_count = getattr(
            compressor, "compression_count", 0
        ) + 1
        compressor.last_compression_rough_tokens = approx_tokens or 0
        # The app server has already completed a real compaction boundary. Its
        # usage update (when supplied) is therefore the same real-vs-real
        # effectiveness verdict used by the normal compression path.
        record_boundary = getattr(
            type(compressor), "record_completed_compaction", None
        )
        if callable(record_boundary):
            # Codex owns this summary. A prior Hermes deterministic-fallback
            # flag must not leak into the native boundary's quality verdict.
            record_boundary(compressor, used_fallback=False)
        elif hasattr(compressor, "_verify_compaction_cleared_threshold"):
            compressor._verify_compaction_cleared_threshold = True
        if not getattr(turn, "token_usage_last", None):
            compressor.last_prompt_tokens = -1
            compressor.last_completion_tokens = 0
            compressor.awaiting_real_usage_after_compression = True

    agent._last_compaction_in_place = False
    try:
        if getattr(agent, "event_callback", None):
            agent.event_callback(
                "session:compress",
                {
                    "platform": getattr(agent, "platform", None) or "",
                    "session_id": getattr(agent, "session_id", None) or "",
                    "old_session_id": "",
                    "in_place": False,
                    "compression_count": getattr(
                        compressor, "compression_count", 0
                    )
                    if compressor is not None
                    else 0,
                    "runtime": "codex_app_server",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                },
            )
    except Exception:
        logger.debug("event_callback error on codex session:compress", exc_info=True)

    return True


# ---------------------------------------------------------------------------
# Codex app-server → Hermes UI bridge (#33200)
#
# The codex_app_server runtime hands the entire turn to a subprocess and
# bypasses the normal Hermes tool loop. Without this bridge gateway
# adapters (Discord, Telegram, TUI) never see live tool-progress bubbles
# or interim assistant commentary while codex is working — the user just
# stares at a quiet channel until the final answer lands. The bridge
# translates raw codex JSON-RPC notifications into the same three agent
# callbacks the standard runtime fires:
#   - tool_progress_callback("tool.started"|"tool.completed", name, ...)
#   - _fire_stream_delta(text) for streaming agentMessage chunks
#   - _emit_interim_assistant_message({...}) for completed agentMessages
# ---------------------------------------------------------------------------

# Codex item types that map to a Hermes tool_call in the projector (and
# therefore deserve a tool_progress bubble pair). The projector lives in
# agent/transports/codex_event_projector.py — keep these in sync so the
# tool name shown in the UI matches the name recorded in messages.
# webSearch is codex's built-in web search tool — it has no projector
# entry (codex handles it internally) but still deserves a bubble.
_CODEX_TOOL_ITEM_TYPES = frozenset(
    {"commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall", "webSearch"}
)

# Internal MCP server that wraps Hermes' native tools for codex. When
# codex calls back through it, the inner dispatch runs in a SEPARATE
# hermes-tools-mcp-server subprocess that has no access to the parent
# agent's tool_progress_callback — so the inner call can never surface
# its own native progress event. The codex-level mcpToolCall event IS
# the display event for those calls; we strip the mcp.hermes-tools.*
# namespacing and emit the bare tool name (web_search, browser_navigate,
# vision_analyze, ...) since the user thinks of these as Hermes tools,
# not as MCP calls.
_INTERNAL_MCP_SERVER = "hermes-tools"


def _codex_item_to_tool_name(item: dict) -> str:
    """Synthetic Hermes tool name for a codex item. Mirrors
    CodexEventProjector so the progress bubble and the projected
    tool_calls entry use the same identifier."""
    item_type = item.get("type") or ""
    if item_type == "commandExecution":
        return "exec_command"
    if item_type == "fileChange":
        return "apply_patch"
    if item_type == "mcpToolCall":
        server = item.get("server") or "mcp"
        tool = item.get("tool") or "unknown"
        if server == _INTERNAL_MCP_SERVER:
            return tool
        return f"mcp.{server}.{tool}"
    if item_type == "dynamicToolCall":
        return item.get("tool") or "dynamic"
    if item_type == "webSearch":
        return "web_search"
    return item_type or "unknown"


def _codex_item_to_args(item: dict) -> dict:
    """Args dict surfaced to tool_progress_callback("tool.started", ...).
    Mirrors the projector's _project_command / _project_file_change /
    _project_mcp_tool_call / _project_dynamic_tool_call shapes."""
    item_type = item.get("type") or ""
    if item_type == "commandExecution":
        return {"command": item.get("command") or "",
                "cwd": item.get("cwd") or ""}
    if item_type == "fileChange":
        return {"changes": [
            {"kind": (c.get("kind") or {}).get("type") or "update",
             "path": c.get("path") or ""}
            for c in (item.get("changes") or []) if isinstance(c, dict)
        ]}
    if item_type in {"mcpToolCall", "dynamicToolCall"}:
        args = item.get("arguments") or {}
        return args if isinstance(args, dict) else {"arguments": args}
    if item_type == "webSearch":
        return {"query": item.get("query") or ""}
    return {}


def _codex_item_to_preview(item: dict) -> Any:
    """Short human-readable preview for the tool.started bubble. Returns
    None when no useful preview is available (Hermes' UI tolerates None)."""
    item_type = item.get("type") or ""
    if item_type == "commandExecution":
        cmd = item.get("command") or ""
        return cmd[:120] if cmd else None
    if item_type == "fileChange":
        paths = [c.get("path") for c in (item.get("changes") or [])
                 if isinstance(c, dict) and c.get("path")]
        if not paths:
            return None
        preview = ", ".join(paths[:3])
        if len(paths) > 3:
            preview += f", +{len(paths) - 3} more"
        return preview
    if item_type in {"mcpToolCall", "dynamicToolCall"}:
        args = item.get("arguments") or {}
        if not isinstance(args, dict) or not args:
            return None
        try:
            return json.dumps(args, ensure_ascii=False)[:120]
        except (TypeError, ValueError):
            return None
    if item_type == "webSearch":
        query = item.get("query") or ""
        return query[:120] if query else None
    return None


def _codex_item_completion_payload(item: dict) -> tuple[str, bool]:
    """Return (result_text, is_error) for a completed codex tool item.
    Mirrors the projector's tool-result content so the bubble shows the
    same outcome string that ends up in the messages list."""
    item_type = item.get("type") or ""
    if item_type == "commandExecution":
        out = item.get("aggregatedOutput") or ""
        exit_code = item.get("exitCode")
        is_error = bool(exit_code is not None and exit_code != 0)
        if is_error:
            out = f"[exit {exit_code}]\n{out}"
        return out, is_error
    if item_type == "fileChange":
        status = item.get("status") or "unknown"
        n = len(item.get("changes") or [])
        return (
            f"apply_patch status={status}, {n} change(s)",
            status not in {"completed", "applied", "success"},
        )
    if item_type == "mcpToolCall":
        error = item.get("error")
        if error:
            return (
                f"[error] {json.dumps(error, ensure_ascii=False)[:1000]}",
                True,
            )
        result = item.get("result")
        return (
            json.dumps(result, ensure_ascii=False)[:4000]
            if result is not None else "",
            False,
        )
    if item_type == "dynamicToolCall":
        content_items = item.get("contentItems") or []
        if isinstance(content_items, list) and content_items:
            return (
                json.dumps(content_items, ensure_ascii=False)[:4000],
                not bool(item.get("success", True)),
            )
        success = item.get("success", True)
        return f"success={success}", not bool(success)
    return "", False


def make_codex_app_server_event_bridge(agent) -> Callable[[dict], None]:
    """Build an ``on_event`` callback that wires codex app-server JSON-RPC
    notifications into Hermes' gateway UI callbacks.

    Returns a single-argument callable suitable for
    ``CodexAppServerSession(on_event=...)``.

    Translation map:
      * ``item/started`` for tool-shaped items → ``tool_progress_callback(
        "tool.started", name, preview, args)``
      * ``item/completed`` for tool-shaped items → ``tool_progress_callback(
        "tool.completed", name, None, None, duration=..., is_error=...,
        result=...)``
      * ``item/agentMessage/delta`` → ``_fire_stream_delta(text)`` so chat
        adapters can render the assistant's reply as it streams.
      * ``item/reasoning/delta`` → ``_fire_reasoning_delta(text)``
      * ``item/completed`` for ``agentMessage`` →
        ``_emit_interim_assistant_message({"role": "assistant",
        "content": text})``. The gateway's ``already_streamed`` check
        dedupes against any text the stream-delta callback already
        rendered for the same message.

    All callback invocations are guarded — a buggy display callback must
    not tear down the codex turn loop. Errors are logged at DEBUG so the
    notification stream keeps flowing regardless.
    """
    # item_id -> (tool_name, args, started_wall_time). Populated on
    # item/started and consumed on item/completed so duration is correct
    # even when codex doesn't report durationMs.
    started: dict[str, tuple[str, dict, float]] = {}

    def _stable_call_id(item: dict, name: str) -> str:
        """Deterministic tool_call id mirroring CodexEventProjector, so a
        live TUI tool card correlates with the same tool call after the
        session is resumed and history is projected."""
        from agent.transports.codex_event_projector import _deterministic_call_id

        item_id = item.get("id") or ""
        item_type = item.get("type") or ""
        if item_type == "commandExecution":
            return _deterministic_call_id("exec", item_id)
        if item_type == "fileChange":
            return _deterministic_call_id("apply_patch", item_id)
        if item_type == "mcpToolCall":
            server = item.get("server") or "mcp"
            tool = item.get("tool") or "unknown"
            return _deterministic_call_id(f"mcp__{server}__{tool}", item_id)
        if item_type == "dynamicToolCall":
            tool = item.get("tool") or "unknown"
            return _deterministic_call_id(f"dyn_{tool}", item_id)
        return _deterministic_call_id(name, item_id)

    def _fire_tool_started(item: dict) -> None:
        item_id = item.get("id") or ""
        name = _codex_item_to_tool_name(item)
        args = _codex_item_to_args(item)
        if item_id:
            started[item_id] = (name, args, time.monotonic())
        cb = getattr(agent, "tool_progress_callback", None)
        if cb is not None:
            try:
                cb("tool.started", name, _codex_item_to_preview(item), args)
            except Exception:
                logger.debug(
                    "tool_progress_callback raised on tool.started for %s",
                    name, exc_info=True,
                )
        # Authoritative stable-ID tool card (TUI / desktop). Fires
        # alongside tool_progress so surfaces that render structured tool
        # cards (not just progress bubbles) stay correlated with the
        # projected history entry after a resume.
        start_cb = getattr(agent, "tool_start_callback", None)
        if start_cb is not None:
            try:
                start_cb(_stable_call_id(item, name), name, args)
            except Exception:
                logger.debug(
                    "tool_start_callback raised for %s", name, exc_info=True,
                )

    def _fire_tool_completed(item: dict) -> None:
        item_id = item.get("id") or ""
        name = _codex_item_to_tool_name(item)
        prior = started.pop(item_id, None)
        # Prefer codex's own durationMs when present so the bubble shows
        # exact tool wall-time; fall back to our started timestamp; fall
        # back to None if we never saw an item/started (some codex
        # versions only emit completed for fast items).
        duration: Any = None
        codex_ms = item.get("durationMs")
        if isinstance(codex_ms, (int, float)) and codex_ms >= 0:
            duration = codex_ms / 1000.0
        elif prior is not None:
            duration = time.monotonic() - prior[2]
        result, is_error = _codex_item_completion_payload(item)
        cb = getattr(agent, "tool_progress_callback", None)
        if cb is not None:
            try:
                cb("tool.completed", name, None, None,
                   duration=duration, is_error=is_error, result=result)
            except Exception:
                logger.debug(
                    "tool_progress_callback raised on tool.completed for %s",
                    name, exc_info=True,
                )
        complete_cb = getattr(agent, "tool_complete_callback", None)
        if complete_cb is not None:
            args = prior[1] if prior is not None else _codex_item_to_args(item)
            try:
                complete_cb(_stable_call_id(item, name), name, args, result)
            except Exception:
                logger.debug(
                    "tool_complete_callback raised for %s", name, exc_info=True,
                )

    def _fire_text_delta(params: dict) -> None:
        text = params.get("delta") or params.get("text") or ""
        if not isinstance(text, str) or not text:
            return
        fn = getattr(agent, "_fire_stream_delta", None)
        if fn is None:
            return
        try:
            fn(text)
        except Exception:
            logger.debug("_fire_stream_delta raised", exc_info=True)

    def _fire_reasoning_delta(params: dict) -> None:
        text = params.get("delta") or params.get("text") or ""
        if not isinstance(text, str) or not text:
            return
        fn = getattr(agent, "_fire_reasoning_delta", None)
        if fn is None:
            return
        try:
            fn(text)
        except Exception:
            logger.debug("_fire_reasoning_delta raised", exc_info=True)

    def _fire_agent_message_completed(item: dict) -> None:
        text = item.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            return
        # display.show_commentary=false — mid-turn narration stays off the
        # visible interim path on this runtime too (same contract as the
        # codex_responses commentary channel).
        if not getattr(agent, "show_commentary", True):
            return
        emit = getattr(agent, "_emit_interim_assistant_message", None)
        if emit is None:
            return
        try:
            emit({"role": "assistant", "content": text})
        except Exception:
            logger.debug(
                "_emit_interim_assistant_message raised", exc_info=True,
            )

    def on_event(note: dict) -> None:
        if not isinstance(note, dict):
            return
        method = note.get("method") or ""
        params = note.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        if method == "item/agentMessage/delta":
            _fire_text_delta(params)
            return
        if method in {"item/reasoning/delta", "item/reasoning/summaryDelta"}:
            _fire_reasoning_delta(params)
            return
        item = params.get("item")
        if not isinstance(item, dict):
            return
        item_type = item.get("type") or ""
        if method == "item/started" and item_type in _CODEX_TOOL_ITEM_TYPES:
            _fire_tool_started(item)
            return
        if method == "item/completed":
            if item_type in _CODEX_TOOL_ITEM_TYPES:
                _fire_tool_completed(item)
            elif item_type == "agentMessage":
                _fire_agent_message_completed(item)

    return on_event


def run_codex_app_server_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Codex app-server runtime path. Hands the entire turn to a `codex
    app-server` subprocess and projects its events back into Hermes'
    messages list so memory/skill review keep working.

    Called from run_conversation() when agent.api_mode == "codex_app_server".
    Returns the same dict shape as the chat_completions path.
    """
    from agent.transports.codex_app_server_session import (
        CodexAppServerSession,
        _ServerRequestRouting,
    )

    # Lazy session: one CodexAppServerSession per AIAgent instance.
    # Spawned on first turn, reused across turns, closed at AIAgent
    # shutdown (see _cleanup hook).
    if not hasattr(agent, "_codex_session") or agent._codex_session is None:
        from agent.runtime_cwd import resolve_agent_cwd

        cwd = getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())
        # Approval callback: defer to Hermes' standard prompt flow if a
        # CLI thread has installed one. Gateway / cron contexts get the
        # codex-side fail-closed default.
        try:
            from tools.terminal_tool import _get_approval_callback
            approval_callback = _get_approval_callback()
        except Exception:
            approval_callback = None

        # Gateway / cron contexts have no UI to surface codex's approval
        # requests through, so codex app-server exec / apply_patch requests
        # fail closed (silently decline) by default. When the user has
        # explicitly opted out of Hermes approvals — via `approvals.mode: off`
        # in config, the /yolo session toggle, or --yolo / HERMES_YOLO_MODE —
        # honor that and let codex's own sandbox permission profile
        # (~/.codex/config.toml) be the policy gate instead of double-gating
        # with a missing Hermes UI. Defaults (manual/smart/unset) preserve the
        # current fail-closed behavior — this is a no-op for those users.
        auto_approve_requests = False
        try:
            from tools.approval import is_approval_bypass_active

            auto_approve_requests = is_approval_bypass_active()
        except Exception:
            logger.debug(
                "codex app-server: approval-bypass lookup failed; "
                "keeping fail-closed default",
                exc_info=True,
            )

        # Bridge codex JSON-RPC notifications (item/started, item/completed,
        # item/agentMessage/delta, ...) into Hermes' gateway UI callbacks
        # (tool_progress_callback, _fire_stream_delta,
        # _emit_interim_assistant_message). Without this, Discord/Telegram
        # users see no live tool-progress or interim commentary while
        # codex_app_server is running — only the final answer (#33200).
        # Supersedes the narrower item/started-only bridge from #38835.
        agent._codex_session = CodexAppServerSession(
            cwd=cwd,
            approval_callback=approval_callback,
            request_routing=_ServerRequestRouting(
                auto_approve_exec=auto_approve_requests,
                auto_approve_apply_patch=auto_approve_requests,
            ),
            on_event=make_codex_app_server_event_bridge(agent),
        )

    # NOTE: the user message is ALREADY appended to messages by the
    # standard run_conversation() flow (line ~11823) before the early
    # return reaches us. Do NOT append again — that would duplicate.

    try:
        turn = agent._codex_session.run_turn(user_input=user_message)
    except Exception as exc:
        logger.exception("codex app-server turn failed")
        # Crash → unconditionally drop the session so the next turn
        # respawns from scratch instead of reusing a dead client.
        try:
            agent._codex_session.close()
        except Exception:
            pass
        agent._codex_session = None
        return {
            "final_response": (
                f"Codex app-server turn failed: {exc}. "
                f"Fall back to default runtime with `/codex-runtime auto`."
            ),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
        }

    # If the turn signalled the underlying client is wedged (deadline
    # blown, post-tool watchdog tripped, OAuth refresh died, subprocess
    # exited), retire the session so the next turn respawns codex
    # rather than riding the broken process. Mirrors openclaw beta.8's
    # "retire timed-out app-server clients" fix.
    if getattr(turn, "should_retire", False):
        logger.warning(
            "codex app-server session retired (turn error: %s)",
            turn.error,
        )
        try:
            agent._codex_session.close()
        except Exception:
            pass
        agent._codex_session = None

    # Splice projected messages into the conversation. The projector emits
    # standard {role, content, tool_calls, tool_call_id} entries, which
    # is exactly what curator.py / sessions DB expect.
    if turn.projected_messages:
        messages.extend(turn.projected_messages)

        # Persist the newly-projected assistant/tool messages ourselves.
        # This path is an early return that bypasses conversation_loop, whose
        # normal per-step _persist_session() calls would otherwise flush them.
        # The inbound user turn was already flushed at turn start
        # (turn_context.py _persist_session), and _flush_messages_to_session_db
        # is idempotent via the intrinsic _DB_PERSISTED_MARKER — so this writes
        # ONLY the new codex projected rows and does NOT re-write the user turn.
        # Keeping the agent as the sole persister lets us return
        # agent_persisted=True below, so the gateway skips its own DB write and
        # we avoid the #860/#42039 duplicate user-message write (append_message
        # is a raw INSERT with no dedup, so a gateway re-write would duplicate
        # the already-flushed user turn). See gateway/run.py agent_persisted.
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug(
                    "codex app-server projected-message flush failed",
                    exc_info=True,
                )


    # Counter ticks for the agent-improvement loop.
    # _turns_since_memory and _user_turn_count are ALREADY incremented
    # in the run_conversation() pre-loop block (lines ~11793-11817) so we
    # do NOT touch them here — that would double-count.
    # Only _iters_since_skill needs explicit increment, since the
    # chat_completions loop bumps it per tool iteration (line ~12110)
    # and that loop is bypassed on this path.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    _record_codex_app_server_compaction(agent, turn)
    usage_result = _record_codex_app_server_usage(agent, turn)
    api_calls = 1

    # Now check the skill nudge AFTER iters were incremented — same
    # pattern the chat_completions path uses (line ~15432).
    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider sync (mirrors line ~15439). Skipped on
    # interrupt/error to avoid feeding partial transcripts to memory.
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

    # Background review fork — same cadence + signature as the default
    # path (line ~15449). Only fires when a trigger actually tripped AND
    # we have a real final response.
    if (
        turn.final_text
        and not turn.interrupted
        and (should_review_memory or should_review_skills)
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )
        except Exception:
            logger.debug("background review spawn raised", exc_info=True)

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": api_calls,
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "error": turn.error,
        # The codex app-server runtime IS an early-return path that bypasses
        # conversation_loop, but we flush the projected assistant/tool messages
        # ourselves above (see the _flush_messages_to_session_db call after
        # messages.extend). The inbound user turn was already flushed at turn
        # start (turn_context._persist_session) and the flush dedups via
        # _DB_PERSISTED_MARKER, so state.db ends up with each real message
        # exactly once and session_search / conversation-distill see the full
        # gateway conversation. Report agent_persisted=True so the gateway
        # skips its own append_to_transcript DB write — writing again there
        # would re-INSERT the already-flushed user turn (append_message has no
        # dedup), reintroducing the #860 / #42039 duplicate-write bug.
        "agent_persisted": True,
        "codex_thread_id": turn.thread_id,
        "codex_turn_id": turn.turn_id,
        **usage_result,
    }


# ---------------------------------------------------------------------------
# Event-driven Responses streaming
#
# OpenAI ships its consumer Codex backend (chatgpt.com/backend-api/codex) on
# a different schedule from the openai Python SDK.  The high-level
# ``client.responses.stream(...)`` helper reconstructs a typed Response from
# the terminal ``response.completed`` event's ``response.output`` field, and
# when that field drifts to ``null`` (gpt-5.5, May 2026) the SDK raises
# ``TypeError: 'NoneType' object is not iterable`` mid-iteration.
#
# We sidestep the whole class of failure by going one level lower:
# ``client.responses.create(stream=True)`` returns the raw AsyncIterable of
# SSE events, and we assemble the final response object purely from
# ``response.output_item.done`` events as they arrive.  We never read
# ``response.completed.response.output`` for content reconstruction, so the
# backend can return ``null``, ``[]``, a string, or omit the field entirely
# and we don't care.
#
# This mirrors what the OpenClaw TS implementation does for the same backend
# and is structurally immune to the bug class rather than patched.
# ---------------------------------------------------------------------------


_TERMINAL_EVENT_TYPES = frozenset({
    "response.completed",
    "response.incomplete",
    "response.failed",
})


def _event_field(event: Any, name: str, default: Any = None) -> Any:
    """Field access that handles both attr-style (SDK objects) and dict (raw JSON) events."""
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        value = event.get(name, default)
    return value if value is not None else default


def _item_field(item: Any, name: str, default: Any = None) -> Any:
    """Field access for nested Response items (attr-style SDK object or dict)."""
    value = getattr(item, name, None)
    if value is None and isinstance(item, dict):
        value = item.get(name, default)
    return value if value is not None else default


def _raise_stream_error(event: Any) -> None:
    """Raise a ``_StreamErrorEvent`` from a ``type=error`` SSE frame.

    The Responses spec puts the failure details at the top level of the
    frame (``{"type": "error", "code": ..., "message": ..., "param": ...}``),
    but the official OpenAI SDK and several OpenAI-compatible proxies wrap
    them in an HTTP-style nested envelope instead
    (``{"type": "error", "error": {"code": ..., "message": ..., "param": ...}}``).
    Read the top-level fields first, then fall back to the nested envelope so
    the error classifier sees the provider's real code/message (rate-limit vs
    context-overflow vs entitlement) rather than the generic placeholder.
    Port of anomalyco/opencode#36130.

    Imported lazily so this module stays importable from places that don't
    pull in ``run_agent`` (e.g. plugin code, doc tools).
    """
    from run_agent import _StreamErrorEvent

    nested = _event_field(event, "error")

    def _error_field(name: str) -> Any:
        value = _event_field(event, name)
        if value is None and nested is not None:
            value = _item_field(nested, name)
        return value

    raw_message = _error_field("message")
    if raw_message is not None and not isinstance(raw_message, str):
        raw_message = str(raw_message)
    message = (raw_message or "stream emitted error event").strip() or "stream emitted error event"
    raise _StreamErrorEvent(
        message,
        code=_error_field("code"),
        param=_error_field("param"),
    )


def _consume_codex_event_stream(
    event_iter: Any,
    *,
    model: str,
    on_text_delta=None,
    on_reasoning_delta=None,
    on_commentary_message=None,
    on_first_delta=None,
    on_event=None,
    interrupt_check=None,
) -> SimpleNamespace:
    """Consume a Codex Responses SSE event stream and return a final response.

    The returned object is a ``SimpleNamespace`` shaped like the SDK's typed
    ``Response`` for the fields downstream code actually reads:

    * ``output``: list of output items, assembled from ``response.output_item.done``.
      For tool-call turns this contains the function_call items; for plain-text
      turns it contains a synthesized ``message`` item built from streamed deltas
      if no message item was emitted directly.
    * ``output_text``: assembled text from ``response.output_text.delta`` deltas.
    * ``usage``: copied from the terminal event's ``response.usage`` (when present).
    * ``status``: ``completed`` / ``incomplete`` / ``failed`` (or ``completed`` if
      the stream ended without a terminal frame but produced content).
    * ``id``: ``response.id`` when present.
    * ``incomplete_details``: passed through for ``response.incomplete`` frames.
    * ``error``: passed through for ``response.failed`` frames.
    * ``model``: from kwargs (the wire model name is not authoritative).

    Critically, we never read ``response.output`` from the terminal event for
    content reconstruction — only ``usage``, ``status``, ``id``.  That field
    being ``null`` / ``[]`` / missing is fine.

    Callbacks:

    * ``on_text_delta(str)`` — fires per ``response.output_text.delta``, suppressed
      once a function_call event is seen (so tool-call turns don't bleed text
      into the chat).
    * ``on_reasoning_delta(str)`` — fires per ``response.reasoning.*.delta`` and
      ``phase=analysis`` message deltas. When no dedicated commentary callback
      is supplied, commentary also uses this legacy fallback.
    * ``on_commentary_message(str)`` — fires once per completed
      ``phase=commentary`` message, before any following tool item executes.
    * ``on_first_delta()`` — one-shot, fires on the first text delta only.
    * ``on_event(event)`` — fires for every event before any other processing.
      Used for watchdog activity, debug logging, anything wire-shape-agnostic.
    * ``interrupt_check()`` — returns True to break the loop early.
    """
    collected_output_items: List[Any] = []
    collected_text_deltas: List[str] = []
    has_tool_calls = False
    first_delta_fired = False
    active_message_phase: str | None = None
    commentary_text_deltas: List[str] = []
    terminal_status: str = "completed"
    terminal_usage: Any = None
    terminal_response_id: str = None
    terminal_incomplete_details: Any = None
    terminal_error: Any = None
    saw_terminal = False

    for event in event_iter:
        if on_event is not None:
            try:
                on_event(event)
            except (TimeoutError, InterruptedError):
                # Control-flow signals from watchdog/cancellation hooks must
                # propagate, not get swallowed as "debug noise".
                raise
            except Exception:
                # Genuine bugs in third-party debug/log hooks shouldn't break
                # stream consumption.
                logger.debug("Codex stream on_event hook raised", exc_info=True)
        if interrupt_check is not None and interrupt_check():
            break

        event_type = _event_field(event, "type", "")
        if not isinstance(event_type, str):
            event_type = ""

        # ``error`` SSE frames carry the provider's real failure reason
        # (subscription / quota / model-not-available / rejected-reasoning-replay)
        # but never appear in the terminal set.  Surface them as a structured
        # exception so the credential pool + error classifier see the body.
        if event_type == "error":
            _raise_stream_error(event)

        # Track the phase of the active streamed message item.  Codex/Harmony
        # ``commentary``/``analysis`` text is mid-turn preamble/progress
        # narration, never the final answer.  We still collect completed output
        # items for replay, but route those deltas to the reasoning callback so
        # they display like thinking text instead of assistant content.
        if event_type == "response.output_item.added":
            item = _event_field(event, "item")
            item_type = _item_field(item, "type", "")
            if item_type == "message":
                phase = _item_field(item, "phase", None)
                active_message_phase = phase.strip().lower() if isinstance(phase, str) else None
                if active_message_phase == "commentary":
                    commentary_text_deltas = []
            else:
                active_message_phase = None
            if "function_call" in str(item_type):
                has_tool_calls = True
            continue

        if "output_text.delta" in event_type or event_type == "response.output_text.delta":
            delta_text = _event_field(event, "delta", "")
            if delta_text and active_message_phase == "commentary":
                commentary_text_deltas.append(delta_text)
                # Preserve CLI/backward compatibility when no first-class
                # commentary consumer is installed.
                if on_commentary_message is None and on_reasoning_delta is not None:
                    try:
                        on_reasoning_delta(delta_text)
                    except Exception:
                        logger.debug("Codex stream on_reasoning_delta raised", exc_info=True)
            elif delta_text and active_message_phase == "analysis":
                if on_reasoning_delta is not None:
                    try:
                        on_reasoning_delta(delta_text)
                    except Exception:
                        logger.debug("Codex stream on_reasoning_delta raised", exc_info=True)
            elif delta_text:
                collected_text_deltas.append(delta_text)
                if not has_tool_calls:
                    if not first_delta_fired:
                        first_delta_fired = True
                        if on_first_delta is not None:
                            try:
                                on_first_delta()
                            except Exception:
                                logger.debug("Codex stream on_first_delta raised", exc_info=True)
                    if on_text_delta is not None:
                        try:
                            on_text_delta(delta_text)
                        except Exception:
                            logger.debug("Codex stream on_text_delta raised", exc_info=True)
            continue

        if "function_call" in event_type:
            has_tool_calls = True
            # fall through — function_call items still get added on output_item.done

        if "reasoning" in event_type and "delta" in event_type:
            reasoning_text = _event_field(event, "delta", "")
            if reasoning_text and on_reasoning_delta is not None:
                try:
                    on_reasoning_delta(reasoning_text)
                except Exception:
                    logger.debug("Codex stream on_reasoning_delta raised", exc_info=True)
            continue

        if event_type == "response.output_item.done":
            done_item = _event_field(event, "item")
            if done_item is not None:
                collected_output_items.append(done_item)
                done_phase = _item_field(done_item, "phase", None)
                done_phase = done_phase.strip().lower() if isinstance(done_phase, str) else None
                if done_phase == "commentary" and on_commentary_message is not None:
                    commentary_text = "".join(commentary_text_deltas).strip()
                    if not commentary_text:
                        content_parts = _item_field(done_item, "content", [])
                        if isinstance(content_parts, list):
                            commentary_text = "".join(
                                str(_item_field(part, "text", "") or "")
                                for part in content_parts
                                if _item_field(part, "type", "") == "output_text"
                            ).strip()
                    if commentary_text:
                        try:
                            on_commentary_message(commentary_text)
                        except Exception:
                            logger.debug(
                                "Codex stream on_commentary_message raised",
                                exc_info=True,
                            )
                    commentary_text_deltas = []
            continue

        if event_type in _TERMINAL_EVENT_TYPES:
            saw_terminal = True
            resp_obj = _event_field(event, "response")
            if resp_obj is not None:
                terminal_usage = getattr(resp_obj, "usage", None)
                if terminal_usage is None and isinstance(resp_obj, dict):
                    terminal_usage = resp_obj.get("usage")
                rid = getattr(resp_obj, "id", None)
                if rid is None and isinstance(resp_obj, dict):
                    rid = resp_obj.get("id")
                terminal_response_id = rid
                rstatus = getattr(resp_obj, "status", None)
                if rstatus is None and isinstance(resp_obj, dict):
                    rstatus = resp_obj.get("status")
                if isinstance(rstatus, str):
                    terminal_status = rstatus
                if event_type == "response.incomplete":
                    terminal_incomplete_details = getattr(resp_obj, "incomplete_details", None)
                    if terminal_incomplete_details is None and isinstance(resp_obj, dict):
                        terminal_incomplete_details = resp_obj.get("incomplete_details")
                if event_type == "response.failed":
                    terminal_error = getattr(resp_obj, "error", None)
                    if terminal_error is None and isinstance(resp_obj, dict):
                        terminal_error = resp_obj.get("error")
            if event_type == "response.completed":
                terminal_status = terminal_status or "completed"
            elif event_type == "response.incomplete":
                terminal_status = terminal_status or "incomplete"
            elif event_type == "response.failed":
                terminal_status = terminal_status or "failed"
            # Stop on terminal event.
            break

    # Build the final output list.  Prefer items observed via output_item.done;
    # if none arrived but we streamed plain text deltas (no tool calls), synthesize
    # a single message item so downstream normalization has something to work with.
    if collected_output_items:
        output = list(collected_output_items)
    elif collected_text_deltas and not has_tool_calls:
        assembled = "".join(collected_text_deltas)
        output = [SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text=assembled)],
        )]
    else:
        output = []

    # If the stream ended without any terminal event AND produced no usable
    # content (no items, no text deltas), surface that as a RuntimeError so
    # callers can distinguish "stream truncated mid-flight / provider rejected
    # the call" from "stream completed with empty body".  This preserves the
    # signal the SDK's high-level helper used to raise as
    # ``RuntimeError("Didn't receive a `response.completed` event.")``.
    if not saw_terminal and not output:
        raise RuntimeError(
            "Codex Responses stream did not emit a terminal response"
        )

    assembled_text = "".join(collected_text_deltas)

    final = SimpleNamespace(
        output=output,
        output_text=assembled_text,
        usage=terminal_usage,
        status=terminal_status,
        id=terminal_response_id,
        model=model,
        incomplete_details=terminal_incomplete_details,
        error=terminal_error,
    )
    return final


def run_codex_stream(agent, api_kwargs: dict, client: Any = None, on_first_delta=None):
    """Execute one streaming Responses API request and return the final response.

    Uses ``responses.create(stream=True)`` (low-level raw event iteration)
    rather than the high-level ``responses.stream(...)`` helper.  This makes
    us structurally immune to backend drift in the ``response.completed``
    payload shape — we never let the SDK reconstruct a typed object from
    the terminal event's ``output`` field.
    """
    import httpx as _httpx

    active_client = client or agent._ensure_primary_openai_client(reason="codex_stream_direct")
    max_stream_retries = 1
    # Accumulate streamed text so callers / compat shims can read it.
    agent._codex_streamed_text_parts: list = []

    def _on_text_delta(text: str) -> None:
        agent._codex_streamed_text_parts.append(text)
        agent._fire_stream_delta(text)

    def _on_reasoning_delta(text: str) -> None:
        agent._fire_reasoning_delta(text)

    def _on_commentary_message(text: str) -> None:
        agent._fire_streamed_codex_commentary(text)

    def _on_event(event: Any) -> None:
        # TTFB watchdog and activity touch — runs once per SSE event.
        agent._codex_stream_last_event_ts = time.time()
        agent._touch_activity("receiving stream response")

    def _interrupt_check() -> bool:
        return bool(agent._interrupt_requested)

    transport = getattr(agent, "responses_transport", "sse") or "sse"
    if transport != "sse":
        from agent.codex_websocket_transport import (
            WebSocketNotStartedError,
            run_codex_websocket_stream,
        )

        timeout = api_kwargs.get("timeout")
        timeout = float(timeout) if isinstance(timeout, (int, float)) and timeout > 0 else None

        def _consume_websocket_events(events, _get_final_response=None):
            return _consume_codex_event_stream(
                events,
                model=api_kwargs.get("model"),
                on_text_delta=_on_text_delta,
                on_reasoning_delta=_on_reasoning_delta,
                on_commentary_message=(
                    _on_commentary_message
                    if (
                        getattr(agent, "interim_assistant_callback", None) is not None
                        and getattr(agent, "show_commentary", True)
                    )
                    else None
                ),
                on_first_delta=on_first_delta,
                on_event=_on_event,
                interrupt_check=_interrupt_check,
            )

        try:
            return run_codex_websocket_stream(
                api_kwargs=api_kwargs,
                client=active_client,
                provider=agent.provider,
                base_url=agent.base_url,
                session_id=agent.session_id,
                transport=transport,
                collect_events=_consume_websocket_events,
                interrupted=_interrupt_check,
                timeout=timeout,
            )
        except WebSocketNotStartedError as exc:
            if transport != "auto":
                raise
            logger.debug(
                "Codex WebSocket auto mode fell back to SSE before request start: %s",
                exc,
            )

    for attempt in range(max_stream_retries + 1):
        if agent._interrupt_requested:
            raise InterruptedError("Agent interrupted before Codex stream retry")

        stream_kwargs = dict(api_kwargs)
        stream_kwargs["stream"] = True

        try:
            event_stream = active_client.responses.create(**stream_kwargs)
        except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
            if attempt < max_stream_retries:
                logger.debug(
                    "Codex Responses stream connect failed (attempt %s/%s); retrying. %s error=%s",
                    attempt + 1, max_stream_retries + 1,
                    agent._client_log_context(), exc,
                )
                continue
            raise

        # Claim the delta sink for THIS attempt (#65991) — parity with the
        # chat_completions/anthropic/bedrock paths. If a prior attempt's
        # stream is somehow still alive, this claim supersedes it so its
        # late deltas are fenced out of the turn; conversely, a newer
        # attempt supersedes us and the interrupt_check below stops our
        # consumption immediately.
        _writer_token = claim_stream_writer(agent)

        def _interrupt_or_superseded(_tok=_writer_token) -> bool:
            if agent._interrupt_requested:
                return True
            if not stream_writer_is_current(agent, _tok):
                logger.warning(
                    "Codex streaming attempt superseded by a newer stream; "
                    "stopping consumption to preserve the single-writer "
                    "invariant (model=%s).",
                    api_kwargs.get("model", "unknown"),
                )
                return True
            return False

        try:
            # Compatibility: some mocks/providers return a concrete response
            # instead of an iterable.  Pass it straight through.
            if hasattr(event_stream, "output") and not hasattr(event_stream, "__iter__"):
                return event_stream

            try:
                final = _consume_codex_event_stream(
                    event_stream,
                    model=api_kwargs.get("model"),
                    on_text_delta=_on_text_delta,
                    on_reasoning_delta=_on_reasoning_delta,
                    on_commentary_message=(
                        _on_commentary_message
                        if (
                            getattr(agent, "interim_assistant_callback", None) is not None
                            and getattr(agent, "show_commentary", True)
                        )
                        else None
                    ),
                    on_first_delta=on_first_delta,
                    on_event=_on_event,
                    interrupt_check=_interrupt_or_superseded,
                )
            except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
                if attempt < max_stream_retries:
                    logger.debug(
                        "Codex Responses stream transport failed mid-iteration "
                        "(attempt %s/%s); retrying. %s error=%s",
                        attempt + 1, max_stream_retries + 1,
                        agent._client_log_context(), exc,
                    )
                    continue
                raise

            if final.status in {"incomplete", "failed"}:
                logger.warning(
                    "Codex Responses stream terminal status=%s "
                    "(incomplete_details=%s, error=%s, streamed_chars=%d). %s",
                    final.status, final.incomplete_details, final.error,
                    sum(len(p) for p in agent._codex_streamed_text_parts),
                    agent._client_log_context(),
                )

            return final
        finally:
            close_fn = getattr(event_stream, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass


def run_codex_create_stream_fallback(agent, api_kwargs: dict, client: Any = None):
    """Backward-compatible alias for the unified event-driven path.

    Historically this was the fallback when the SDK's high-level
    ``responses.stream(...)`` helper raised on shape drift.  The primary
    path now does exactly what the fallback did, so this just forwards.
    Kept as a public symbol because tests and a small number of call sites
    still reference it by name.
    """
    return run_codex_stream(agent, api_kwargs, client=client)


__all__ = [
    "run_codex_app_server_turn",
    "run_codex_stream",
    "run_codex_create_stream_fallback",
    "_consume_codex_event_stream",
    "make_codex_app_server_event_bridge",
]
