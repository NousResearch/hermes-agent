"""OpenAI-compatible routes for the API server adapter.

``OpenAICompatRoutesMixin`` (inherited by ``APIServerAdapter``) carries ``/v1/chat/completions``,
``/v1/responses`` (+ GET/DELETE), their SSE writers and the Responses-transcript helpers.
api_server-internal helpers are imported lazily inside each method: the origin imports this
module (top-level import = cycle), and lazy lookup keeps ``patch("...api_server.X")`` effective.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import suppress
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - mirrors api_server's optional import
    web = None  # type: ignore[assignment]

# Logger parity with the origin module (moved log records keep their name).
logger = logging.getLogger("gateway.platforms.api_server")

async def _iter_stream_items(stream_q, agent_task, response):
    """Yield agent stream items until EOS, writing SSE keepalives while idle.

    Yields the ``None`` sentinel once so callers can run EOS-only work; when ``agent_task``
    is already done the remaining queue is drained and the sentinel swallowed.
    """
    from gateway.platforms.api_server import CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS

    last_activity = time.monotonic()
    while True:
        try:
            item = await asyncio.wait_for(stream_q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if agent_task.done():
                while True:
                    try:
                        item = stream_q.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    if item is None:
                        return
                    yield item
                    last_activity = time.monotonic()
            if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                await response.write(b": keepalive\n\n")
                last_activity = time.monotonic()
            continue
        if item is None:
            yield None
            return
        yield item
        last_activity = time.monotonic()


def _result_flags(result: Any) -> tuple:
    """``(completed, partial, failed, error)`` from an agent result dict (defaults if not a dict)."""
    if not isinstance(result, dict):
        return True, False, False, None
    return (bool(result.get("completed", True)), bool(result.get("partial")),
            bool(result.get("failed")), result.get("error"))


def _finish_reason(completed, is_partial, is_failed, err_msg, agent_error=None) -> str:
    """OpenAI ``finish_reason``: "length" for truncation, "error" for failure, else "stop"."""
    # OpenAI uses "length" for truncation, "stop" for normal completion, and downstream SDKs accept "error"
    # / custom codes. See issue #22496.
    if is_partial and err_msg and "truncat" in err_msg.lower():
        return "length"
    if agent_error is not None or is_failed or (not completed and err_msg):
        return "error"
    return "stop"


def _hermes_extras(completed, is_partial, is_failed, err_msg, finish_reason: str) -> Dict[str, Any]:
    return {
        "completed": completed, "partial": is_partial, "failed": is_failed, "error": err_msg,
        "error_code": "output_truncated" if finish_reason == "length" else "agent_error"}


def _message_item(text: Any) -> Dict[str, Any]:
    """Responses ``message`` output item carrying one ``output_text`` part."""
    return {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": text}]}


def _trim_tool_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Trim large tool payloads in place so response.completed stays under ~100KB (clients
    already received the full details via the incremental events)."""
    for item in items:
        if item.get("type") == "function_call":
            try:
                raw = item.get("arguments")
                args = json.loads(raw) if isinstance(raw, str) else item.get("arguments", {})
                if isinstance(args, dict):
                    for k in ("content", "query", "pattern", "old_string", "new_string"):
                        if isinstance(args.get(k), str) and len(args[k]) > 500:
                            args[k] = f"[{len(args[k])} chars — truncated for response.completed]"
                    item["arguments"] = json.dumps(args)
            except Exception:
                pass
        elif item.get("type") == "function_call_output":
            output = item.get("output", [])
            if isinstance(output, list) and output:
                first = output[0]
                if isinstance(first, dict) and first.get("type") == "input_text":
                    text = first.get("text", "")
                    if len(text) > 1000:
                        first["text"] = text[:500] + "...[" + str(len(text) - 500) + " more chars]"
                        item["output"] = [first]
    return items


class _ResponsesStream:
    """Per-request state and event emitters for the POST /v1/responses SSE writer.

    Every event carries a monotonic ``sequence_number`` (canonical Responses SSE schema). Text
    deltas are batched (50ms) against Open WebUI re-render storms; tool events flush first.
    """

    def __init__(self, adapter, response, *, response_id: str, model: str, created_at: int,
                 conversation_history: List[Dict[str, str]], user_message: str,
                 instructions: Optional[str], conversation: Optional[str], store: bool, session_id: str):
        from gateway.platforms import api_server as api
        self._api = api
        self.adapter, self.response, self.response_id = adapter, response, response_id
        self.model, self.created_at, self.conversation_history = model, created_at, conversation_history
        self.user_message, self.instructions = user_message, instructions
        self.conversation, self.store, self.session_id = conversation, store, session_id
        self.final_text_parts: List[str] = []
        self.pending_tool_calls: List[Dict[str, Any]] = []  # open function_call items, in order
        self.emitted_items: List[Dict[str, Any]] = []  # output items so far (terminal payload)
        self.output_index = 0
        self.call_counter = 0  # call_id fallback when the agent supplies no tool_call_id
        self.sequence_number = 0
        self.message_item_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.message_output_index: Optional[int] = None
        self.message_opened = False
        self.final_response_text = ""
        self.agent_error: Optional[str] = None
        self.usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.terminal_snapshot_persisted = False
        self.result: Any = None
        self._batch_buf: List[str] = []
        self._batch_timer: Optional[asyncio.Task] = None
        self._batch_lock = asyncio.Lock()

    async def write_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if "sequence_number" not in data:
            data["sequence_number"] = self.sequence_number
        self.sequence_number += 1
        await self.response.write(self._api._sse_frame(data, event=event_type))

    def envelope(self, status: str) -> Dict[str, Any]:
        return {"id": self.response_id, "object": "response", "status": status,
                "created_at": self.created_at, "model": self.model}

    def terminal_envelope(self, status: str, output: List[Dict[str, Any]], *, error=None) -> dict:
        """``envelope`` + ``output`` (+ ``error`` when given) + ``usage``, in wire key order."""
        env = self.envelope(status)
        env["output"] = output
        if error is not None:
            env["error"] = {"message": error, "type": "server_error"}
        env["usage"] = self._api._responses_usage_payload(self.usage)
        return env

    def _history_with_user(self) -> List[Dict[str, Any]]:
        return list(self.conversation_history) + [{"role": "user", "content": self.user_message}]

    def persist_snapshot(self, response_env: Dict[str, Any], *, history=None, session_id=None):
        if not self.store:
            return
        self.adapter._response_store.put(self.response_id, {
            "response": response_env,
            "conversation_history": self._history_with_user() if history is None else history,
            "instructions": self.instructions,
            "session_id": session_id or self.session_id})
        if self.conversation:
            self.adapter._response_store.set_conversation(self.conversation, self.response_id)

    def persist_incomplete_if_needed(self) -> None:
        """Persist an ``incomplete`` snapshot when no terminal one was written (disconnect /
        cancel paths), so GET /v1/responses/{id} and ``previous_response_id`` chaining survive."""
        if not self.store or self.terminal_snapshot_persisted:
            return
        text = "".join(self.final_text_parts) or self.final_response_text
        items = list(self.emitted_items)
        history = self._history_with_user()
        if text:
            items.append(_message_item(text))
            history.append({"role": "assistant", "content": text})
        self.persist_snapshot(self.terminal_envelope("incomplete", items), history=history)

    async def emit_created(self) -> None:
        env = self.envelope("in_progress")
        env["output"] = []
        await self.write_event("response.created", {"type": "response.created", "response": env})
        self.persist_snapshot(env)

    async def _open_message_item(self) -> None:
        """Emit output_item.added for the assistant message on the first text delta."""
        if self.message_opened:
            return
        self.message_opened = True
        self.message_output_index = self.output_index
        self.output_index += 1
        await self.write_event("response.output_item.added", {
            "type": "response.output_item.added", "output_index": self.message_output_index,
            "item": {"id": self.message_item_id, "type": "message", "status": "in_progress",
                     "role": "assistant", "content": []}})

    async def emit_text_delta(self, delta_text: str) -> None:
        await self._open_message_item()
        self.final_text_parts.append(delta_text)
        await self.write_event("response.output_text.delta", {
            "type": "response.output_text.delta", "item_id": self.message_item_id,
            "output_index": self.message_output_index, "content_index": 0, "delta": delta_text,
            "logprobs": []})

    async def emit_tool_started(self, payload: Dict[str, Any]) -> None:
        """function_call ``output_item.added``; the agent's tool_call_id beats a generated call id."""
        self.call_counter += 1
        call_id = payload.get("tool_call_id") or f"call_{self.response_id[5:]}_{self.call_counter}"
        args = payload.get("arguments", {})
        arguments_str = json.dumps(args) if isinstance(args, dict) else str(args)
        name = payload.get("name", "")
        item = {"id": f"fc_{uuid.uuid4().hex[:24]}", "type": "function_call",
                "status": "in_progress", "name": name, "call_id": call_id, "arguments": arguments_str}
        idx = self.output_index
        self.output_index += 1
        self.pending_tool_calls.append({
            "call_id": call_id, "name": name, "arguments": arguments_str, "item_id": item["id"],
            "output_index": idx})
        self.emitted_items.append(
            {"type": "function_call", "name": name, "arguments": arguments_str, "call_id": call_id})
        await self.write_event("response.output_item.added", {
            "type": "response.output_item.added", "output_index": idx, "item": item})

    async def emit_tool_completed(self, payload: Dict[str, Any]) -> None:
        """function_call ``output_item.done`` + function_call_output added/done; orphans skipped."""
        call_id = payload.get("tool_call_id")
        pending = next((p for p in self.pending_tool_calls if p["call_id"] == call_id), None)
        if not call_id or pending is None:
            return
        self.pending_tool_calls.remove(pending)
        done_item = {"id": pending["item_id"], "type": "function_call", "status": "completed",
                     "name": pending["name"], "call_id": pending["call_id"],
                     "arguments": pending["arguments"]}
        await self.write_event("response.output_item.done", {
            "type": "response.output_item.done", "output_index": pending["output_index"],
            "item": done_item})
        result = payload.get("result", "")
        result_str = result if isinstance(result, str) else json.dumps(result)
        output_parts = [{"type": "input_text", "text": result_str}]
        output_item = {"id": f"fco_{uuid.uuid4().hex[:24]}", "type": "function_call_output",
                       "call_id": pending["call_id"], "output": output_parts, "status": "completed"}
        idx = self.output_index
        self.output_index += 1
        self.emitted_items.append(
            {"type": "function_call_output", "call_id": pending["call_id"], "output": output_parts})
        for event in ("response.output_item.added", "response.output_item.done"):
            await self.write_event(event, {"type": event, "output_index": idx, "item": output_item})

    async def dispatch(self, item: Any) -> None:
        """Route one queue item: tool tuples emit immediately, strings are batched, others dropped."""
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            tag, payload = item
            await self.flush_batch()
            if tag == "__tool_started__":
                await self.emit_tool_started(payload)
            elif tag == "__tool_completed__":
                await self.emit_tool_completed(payload)
        elif isinstance(item, str):
            self._batch_buf.append(item)
            if self._batch_timer is None:
                self._batch_timer = asyncio.create_task(self._batch_flush_after(0.05))

    async def _batch_flush_after(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        # Clear the timer BEFORE flushing so new deltas can start a fresh timer while we emit.
        self._batch_timer = None
        await self.flush_batch()

    def cancel_batch_timer(self) -> None:
        if self._batch_timer and not self._batch_timer.done():
            self._batch_timer.cancel()
            self._batch_timer = None

    async def flush_batch(self) -> None:
        """Emit a single delta for all buffered text."""
        if not self._batch_buf:
            return
        async with self._batch_lock:
            if self._batch_buf:
                combined = "".join(self._batch_buf)
                self._batch_buf = []
                await self.emit_text_delta(combined)

    async def collect_result(self, agent_task) -> None:
        """Await the agent; when it produced a final_response but streamed no deltas
        (some providers only emit the full text at the end), emit one fallback delta."""
        try:
            result, agent_usage = await agent_task
            self.result = result
            self.usage = agent_usage or self.usage
            agent_final = result.get("final_response", "") if isinstance(result, dict) else ""
            if agent_final and not self.final_text_parts:
                await self.emit_text_delta(agent_final)
            if agent_final and not self.final_response_text:
                self.final_response_text = agent_final
            if isinstance(result, dict) and result.get("error") and not self.final_response_text:
                self.agent_error = self._api._redact_api_error_text(result["error"])
        except Exception as e:  # noqa: BLE001
            logger.error("Error running agent for streaming responses: %s", e, exc_info=True)
            self.agent_error = self._api._redact_api_error_text(e)

    async def close_message_item(self) -> None:
        self.final_response_text = "".join(self.final_text_parts) or self.final_response_text
        if not self.message_opened:
            return
        await self.write_event("response.output_text.done", {
            "type": "response.output_text.done", "item_id": self.message_item_id,
            "output_index": self.message_output_index, "content_index": 0,
            "text": self.final_response_text, "logprobs": []})
        await self.write_event("response.output_item.done", {
            "type": "response.output_item.done", "output_index": self.message_output_index,
            "item": {"id": self.message_item_id, "type": "message", "status": "completed",
                     "role": "assistant",
                     "content": [{"type": "output_text", "text": self.final_response_text}]}})

    def _final_items(self) -> List[Dict[str, Any]]:
        """Emitted items (trimmed) plus a final message item, so clients that only parse
        the terminal payload still see the assistant text (mirrors _extract_output_items)."""
        items = _trim_tool_items(list(self.emitted_items))
        redact = self._api._redact_api_error_text
        text = self.final_response_text or (redact(self.agent_error) if self.agent_error else "")
        items.append(_message_item(text))
        return items

    async def emit_failed(self) -> None:
        redact = self._api._redact_api_error_text
        env = self.terminal_envelope("failed", self._final_items(), error=redact(self.agent_error))
        history = self._history_with_user()
        history.append(
            {"role": "assistant", "content": self.final_response_text or redact(self.agent_error)})
        self.persist_snapshot(env, history=history)
        self.terminal_snapshot_persisted = True
        await self.write_event("response.failed", {"type": "response.failed", "response": env})

    async def emit_completed(self) -> None:
        env = self.terminal_envelope("completed", self._final_items())
        result = self.result
        full_history = self.adapter._build_response_conversation_history(
            self.conversation_history, self.user_message, result, self.final_response_text)
        # Transcript substitution for result["_compressed"] happens in the history builder; only
        # a compression-rotated session_id is propagated so chaining resumes the child session.
        sid = result.get("session_id") if isinstance(result, dict) else None
        self.persist_snapshot(
            env, history=full_history, session_id=sid if isinstance(sid, str) and sid else None)
        self.terminal_snapshot_persisted = True
        await self.write_event(
            "response.completed", {"type": "response.completed", "response": env})

    async def emit_crash(self, exc: BaseException) -> None:
        error = self._api._redact_api_error_text(exc, limit=500)
        env = self.terminal_envelope("failed", list(self.emitted_items), error=error)
        await self.write_event("response.failed", {"type": "response.failed", "response": env})


class OpenAICompatRoutesMixin:
    """/v1/chat/completions and /v1/responses handlers + SSE writers."""

    def _select_request_route(
        self, body: Dict[str, Any], *, session_id, gateway_session_key, model_alias) -> tuple:
        """Resolve the model_routes alias + per-request overrides ->
        ``(route, agent_overrides, error_response_or_None)``."""
        from gateway.platforms.api_server import _error_response, _request_agent_overrides
        route = self._resolve_route(model_alias)
        overrides = _request_agent_overrides(
            body, virtual_model=self._model_name, allow_bare_model=self._direct_model_requests)
        err = self._request_route_conflict_error(
            session_id=session_id, gateway_session_key=gateway_session_key,
            requested_model=overrides.get("requested_model"),
            requested_provider=overrides.get("requested_provider"), route=route)
        return route, overrides, (_error_response(err, 400) if err else None)

    def _spawn_stream_agent(self, stream_q, **run_kwargs) -> tuple:
        """Start ``_run_agent`` for an SSE writer -> ``(agent_task, agent_ref)``. ``agent_ref[0]``
        lets the writer interrupt on disconnect; the EOS sentinel is enqueued from the task's done
        callback so drain loops never race a polled ``agent_task.done()``."""
        def _on_delta(delta):
            # None from the agent is a CLI box-close signal, not EOS — forwarding it would end
            # the stream early. Called from the run_conversation worker thread: put_threadsafe.
            if delta is not None:
                stream_q.put_threadsafe(delta)
        agent_ref = [None]
        agent_task = asyncio.ensure_future(self._run_agent(
            stream_delta_callback=_on_delta, agent_ref=agent_ref, **run_kwargs))
        agent_task.add_done_callback(lambda _fut: stream_q.put_nowait(None))
        return agent_task, agent_ref

    async def _handle_chat_completions(self, request: "web.Request") -> "web.Response":
        """POST /v1/chat/completions — OpenAI Chat Completions format."""
        from gateway.platforms.api_server import (
            ThreadSafeAsyncQueue, _chat_usage_payload, _coerce_request_bool,
            _content_has_visible_payload, _derive_chat_session_id, _error_response, _invalid_request,
            _multimodal_validation_error, _normalize_chat_content, _normalize_multimodal_content,
            _openai_error, _redact_api_error_text, _resolve_media_to_data_urls)
        # Bound total in-flight agent runs (configurable; #7483).
        limited = self._concurrency_limited_response()
        if limited is not None:
            return limited
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON in request body", 400)
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return _invalid_request("Missing or invalid 'messages' field")
        stream = _coerce_request_bool(body.get("stream"), default=False)

        # System messages -> ephemeral system prompt layered ON TOP of core, flattened to text
        # (Anthropic rejects images there, OpenAI text models ignore them).
        system_prompt = None
        conversation_messages: List[Dict[str, str]] = []
        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            raw_content = msg.get("content", "")
            if role == "system":
                content = _normalize_chat_content(raw_content)
                system_prompt = content if system_prompt is None else system_prompt + "\n" + content
            elif role in {"user", "assistant"}:
                try:
                    content = _normalize_multimodal_content(raw_content)
                except ValueError as exc:
                    return _multimodal_validation_error(exc, param=f"messages[{idx}].content")
                conversation_messages.append({"role": role, "content": content})
        user_message: Any = (conversation_messages[-1].get("content", "") if conversation_messages else "")
        history = conversation_messages[:-1]
        if not _content_has_visible_payload(user_message):
            return _invalid_request("No user message found in messages")

        # X-Hermes-Session-Key scopes long-term memory per channel; independent of
        # X-Hermes-Session-Id (the key persists across transcripts, the id rotates on /new).
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err
        # X-Hermes-Session-Id continues an existing session (history from state.db, not the body);
        # requires a configured API key or any client could read history by guessing ids.
        provided_session_id = request.headers.get("X-Hermes-Session-Id", "").strip()
        if provided_session_id:
            if not self._api_key:
                logger.warning(
                    "Session continuation via X-Hermes-Session-Id rejected: "
                    "no API key configured.  Set API_SERVER_KEY to enable "
                    "session continuity.")
                return _error_response("Session continuation requires API key authentication. "
                        "Configure API_SERVER_KEY to enable this feature.", 403)
            # Same guard as the native gateway: ids are interpolated into on-disk filenames.
            from gateway.session import _is_path_unsafe
            if re.search(r'[\r\n\x00]', provided_session_id) or _is_path_unsafe(provided_session_id):
                return _invalid_request("Invalid session ID")
            if len(provided_session_id) > self._MAX_SESSION_HEADER_LEN:
                return _invalid_request("Session ID too long")
            session_id = provided_session_id
            try:
                db = await self._ensure_session_db_async()
                if db is not None:
                    history = await asyncio.to_thread(db.get_messages_as_conversation, session_id)
            except Exception as e:
                logger.warning("Failed to load session history for %s: %s", session_id, e)
                history = []
        else:
            # Stable id from the conversation fingerprint so Open WebUI-style clients map onto
            # one Hermes session.
            first_user = next(
                (cm.get("content", "") for cm in conversation_messages if cm.get("role") == "user"), "")
            session_id = _derive_chat_session_id(system_prompt, first_user)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        model_name = body.get("model", self._model_name)
        created = int(time.time())
        route, agent_overrides, selection_error = self._select_request_route(
            body, session_id=session_id, gateway_session_key=gateway_session_key,
            model_alias=model_name)
        if selection_error is not None:
            return selection_error
        run_kwargs = dict(
            user_message=user_message, conversation_history=history,
            ephemeral_system_prompt=system_prompt, session_id=session_id,
            gateway_session_key=gateway_session_key, **agent_overrides, route=route)
        if stream:
            _stream_q = ThreadSafeAsyncQueue()
            # tool_call_ids with an emitted "running": a "completed" without one (internal/
            # filtered tools) is dropped rather than orphaned on the wire.
            _started_tool_call_ids: set[str] = set()

            def _on_tool_start(tool_call_id, function_name, function_args):
                """``hermes.tool.progress`` status=running; ``_``-prefixed tools stay off the wire."""
                if not tool_call_id or function_name.startswith("_"):
                    return
                _started_tool_call_ids.add(tool_call_id)
                from agent.display import build_tool_preview, get_tool_emoji
                label = build_tool_preview(function_name, function_args) or function_name
                _stream_q.put_threadsafe(("__tool_progress__", {
                    "tool": function_name, "emoji": get_tool_emoji(function_name), "label": label,
                    "toolCallId": tool_call_id, "status": "running"}))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                if not tool_call_id or tool_call_id not in _started_tool_call_ids:
                    return
                _started_tool_call_ids.discard(tool_call_id)
                _stream_q.put_threadsafe(("__tool_progress__", {
                    "tool": function_name, "toolCallId": tool_call_id, "status": "completed"}))

            # tool_progress_callback deliberately NOT wired: it would duplicate the structured
            # start/complete callbacks (which carry the tool_call id).
            agent_task, agent_ref = self._spawn_stream_agent(
                _stream_q, tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete, **run_kwargs)
            return await self._write_sse_chat_completion(
                request, completion_id, model_name, created, _stream_q,
                agent_task, agent_ref, session_id=session_id,
                gateway_session_key=gateway_session_key)

        async def _compute_completion():
            return await self._run_agent(**run_kwargs)
        outcome, err = await self._run_idempotent(
            request, body, _compute_completion, log_label="chat completions",
            fingerprint_keys=["model", "provider", "model_options", "messages", "tools", "tool_choice", "stream"],
        )
        if err is not None:
            return err
        result, usage = outcome
        final_response = _resolve_media_to_data_urls(result.get("final_response") or "")
        completed, is_partial, is_failed, err_msg = _result_flags(result)
        if err_msg:
            err_msg = _redact_api_error_text(err_msg)
        finish_reason = _finish_reason(completed, is_partial, is_failed, err_msg)
        response_headers = {"X-Hermes-Session-Id": result.get("session_id", session_id)}
        if gateway_session_key:
            response_headers["X-Hermes-Session-Key"] = gateway_session_key
        # Hard fail (no usable text AND a real failure) -> 502 OpenAI error envelope so SDK
        # clients raise instead of rendering the failure string as message.content.
        if not final_response and (is_failed or is_partial):
            err_body = _openai_error(
                err_msg or "Agent run did not produce a response.", err_type="server_error",
                code="agent_incomplete")
            err_body["error"]["hermes"] = {
                "completed": completed, "partial": is_partial, "failed": is_failed}
            response_headers["X-Hermes-Completed"] = "false"
            response_headers["X-Hermes-Partial"] = "true" if is_partial else "false"
            return web.json_response(err_body, status=502, headers=response_headers)
        # Soft partial (some text, run incomplete): 200 + finish_reason="length"/Hermes extras.
        response_data = {
            "id": completion_id, "object": "chat.completion", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": final_response},
                         "finish_reason": finish_reason}],
            "usage": _chat_usage_payload(usage)}
        if is_partial or is_failed or not completed:
            response_data["hermes"] = _hermes_extras(
                completed, is_partial, is_failed, err_msg, finish_reason)
            response_headers["X-Hermes-Completed"] = "false"
            response_headers["X-Hermes-Partial"] = "true" if is_partial else "false"
            if err_msg:
                response_headers["X-Hermes-Error"] = _redact_api_error_text(err_msg, limit=200)
        return web.json_response(response_data, headers=response_headers)

    async def _run_idempotent(
        self, request: "web.Request", body: Dict[str, Any], compute, *,
        log_label: str, fingerprint_keys: List[str]) -> tuple:
        """Run ``compute()`` once per Idempotency-Key + body fingerprint ->
        ``((result, usage), None)`` or ``(None, 500 response)``."""
        from gateway.platforms.api_server import (
            _error_response, _idem_cache, _make_request_fingerprint)
        idempotency_key = request.headers.get("Idempotency-Key")
        try:
            if idempotency_key:
                fp = _make_request_fingerprint(body, keys=fingerprint_keys)
                result, usage = await _idem_cache.get_or_set(idempotency_key, fp, compute)
            else:
                result, usage = await compute()
            return (result, usage), None
        except Exception as e:
            logger.error("Error running agent for %s: %s", log_label, e, exc_info=True)
            return None, _error_response(f"Internal server error: {e}", 500, err_type="server_error")

    async def _prepare_sse_response(
        self, request: "web.Request", session_id: Optional[str], gateway_session_key: Optional[str],
    ) -> "web.StreamResponse":
        """Open a prepared SSE StreamResponse with CORS + session headers (the CORS middleware
        can't inject headers after ``prepare()`` flushes them, so they are resolved here)."""
        sse_headers = {
            "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        origin = request.headers.get("Origin", "")
        if origin:
            sse_headers.update(self._cors_headers_for_origin(origin) or {})
        if session_id:
            sse_headers["X-Hermes-Session-Id"] = session_id
        if gateway_session_key:
            sse_headers["X-Hermes-Session-Key"] = gateway_session_key
        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)
        return response

    async def _write_sse_chat_completion(
        self, request: "web.Request", completion_id: str, model: str,
        created: int, stream_q, agent_task, agent_ref=None, session_id: str = None,
        gateway_session_key: str = None) -> "web.StreamResponse":
        """Stream ``chat.completion.chunk`` frames from the agent's delta queue. On client
        disconnect the agent is interrupted (stops LLM calls), then its task wrapper cancelled."""
        from gateway.platforms.api_server import (
            _abandon_agent_task, _chat_usage_payload, _sse_frame)
        response = await self._prepare_sse_response(request, session_id, gateway_session_key)

        def _chunk(delta: Dict[str, Any], finish_reason=None, **extra) -> Dict[str, Any]:
            return {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}], **extra}
        try:
            await response.write(_sse_frame(_chunk({"role": "assistant"})))
            async for delta in _iter_stream_items(stream_q, agent_task, response):
                if delta is None:
                    break
                if isinstance(delta, tuple) and len(delta) == 2 and delta[0] == "__tool_progress__":
                    # Custom event: tool lifecycle for frontends without markers in history.
                    await response.write(_sse_frame(delta[1], event="hermes.tool.progress"))
                else:
                    await response.write(_sse_frame(_chunk({"content": delta})))
            # The agent can fail after the queue drains (task raises / result flagged failed or
            # partial): surface a non-"stop" finish_reason like the non-streaming path.
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            result = agent_error = None
            try:
                result, agent_usage = await agent_task
                usage = agent_usage or usage
            except Exception as exc:
                agent_error = exc
                logger.error("Agent task %s failed during SSE streaming: %s", completion_id, exc)
            completed, is_partial, is_failed, err_msg = _result_flags(result)
            if agent_error is not None:
                is_failed = True
                err_msg = err_msg or str(agent_error)
            finish_reason = _finish_reason(completed, is_partial, is_failed, err_msg, agent_error)
            finish_chunk = _chunk({}, finish_reason, usage=_chat_usage_payload(usage))
            if finish_reason != "stop":
                if err_msg:
                    finish_chunk["error"] = {
                        "message": err_msg,
                        "type": type(agent_error).__name__ if agent_error else "agent_error"}
                finish_chunk["hermes"] = _hermes_extras(
                    completed, is_partial, is_failed, err_msg, finish_reason)
            await response.write(_sse_frame(finish_chunk))
            await response.write(b"data: [DONE]\n\n")
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            await _abandon_agent_task(agent_ref, agent_task, "SSE client disconnected")
            logger.info("SSE client disconnected; interrupted agent task %s", completion_id)
        except Exception:
            # Agent crashed mid-stream: an error chunk beats a TransferEncodingError.
            import traceback as _tb
            logger.error("Agent crashed mid-stream for %s: %s", completion_id, _tb.format_exc()[:300])
            with suppress(Exception):
                await response.write(_sse_frame(_chunk({}, "error")))
                await response.write(b"data: [DONE]\n\n")
        return response

    async def _write_sse_responses(
        self, request: "web.Request", response_id: str, model: str, created_at: int, stream_q,
        agent_task, agent_ref, conversation_history: List[Dict[str, str]], user_message: str,
        instructions: Optional[str], conversation: Optional[str], store: bool, session_id: str,
        gateway_session_key: Optional[str] = None) -> "web.StreamResponse":
        """Write the SSE stream for POST /v1/responses.

        Events: ``response.created`` -> ``output_text.delta/done`` + ``output_item.added/done``
        (function_call / function_call_output) -> ``response.completed`` (non-streaming envelope)
        or ``response.failed``. On disconnect the agent is interrupted and, with ``store=True``,
        an ``incomplete`` snapshot replaces ``in_progress`` so GET / chaining still work.
        """
        from gateway.platforms.api_server import _abandon_agent_task, _redact_api_error_text
        response = await self._prepare_sse_response(request, session_id, gateway_session_key)
        st = _ResponsesStream(
            self, response, response_id=response_id, model=model, created_at=created_at,
            conversation_history=conversation_history, user_message=user_message,
            instructions=instructions, conversation=conversation, store=store, session_id=session_id)
        try:
            await st.emit_created()
            async for item in _iter_stream_items(stream_q, agent_task, response):
                if item is None:  # EOS sentinel
                    st.cancel_batch_timer()
                    await st.flush_batch()
                    break
                await st.dispatch(item)
            await st.flush_batch()
            await st.collect_result(agent_task)
            await st.close_message_item()
            if st.agent_error:
                await st.emit_failed()
            else:
                await st.emit_completed()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            st.persist_incomplete_if_needed()
            await _abandon_agent_task(agent_ref, agent_task, "SSE client disconnected")
            logger.info("SSE client disconnected; interrupted agent task %s", response_id)
        except asyncio.CancelledError:
            # Server-side cancellation (shutdown, timeout): persist incomplete, then re-raise.
            st.persist_incomplete_if_needed()
            await _abandon_agent_task(
                agent_ref, agent_task, "SSE task cancelled",
                reap_source="api_server_sse_cancelled", await_cancel=False)
            logger.info("SSE task cancelled; persisted incomplete snapshot for %s", response_id)
            raise
        except Exception as exc:
            # Unhandled agent error (BadRequestError, AuthenticationError, ...): emit
            # response.failed and end the stream cleanly (no TransferEncodingError).
            import traceback as _tb
            st.persist_incomplete_if_needed()
            st.agent_error = _redact_api_error_text(_tb.format_exc())
            with suppress(Exception):
                await st.emit_crash(exc)
            logger.error("Agent crashed mid-stream for %s: %s", response_id, str(st.agent_error)[:300])
        return response

    async def _handle_responses(self, request: "web.Request") -> "web.Response":
        """POST /v1/responses — OpenAI Responses API format."""
        from gateway.platforms.api_server import (
            ThreadSafeAsyncQueue, _auto_truncate_response_history, _coerce_request_bool,
            _content_has_visible_payload, _error_response, _invalid_request,
            _multimodal_validation_error, _normalize_multimodal_content, _redact_api_error_text,
            _resolve_media_to_data_urls, _responses_usage_payload)
        # Bound total in-flight agent runs (configurable; #7483).
        limited = self._concurrency_limited_response()
        if limited is not None:
            return limited
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err
        try:
            body = await request.json()
        except Exception:
            return _invalid_request("Invalid JSON in request body")
        raw_input = body.get("input")
        if raw_input is None:
            return _error_response("Missing 'input' field", 400)
        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        conversation = body.get("conversation")
        store = _coerce_request_bool(body.get("store"), default=True)
        if conversation and previous_response_id:
            return _error_response("Cannot use both 'conversation' and 'previous_response_id'", 400)
        if conversation:
            # A conversation name resolves to its latest response_id (unknown = new conversation).
            previous_response_id = self._response_store.get_conversation(conversation)

        input_messages: List[Dict[str, Any]] = []
        if isinstance(raw_input, str):
            input_messages = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list):
            for idx, item in enumerate(raw_input):
                if isinstance(item, str):
                    input_messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    try:
                        content = _normalize_multimodal_content(item.get("content", ""))
                    except ValueError as exc:
                        return _multimodal_validation_error(exc, param=f"input[{idx}].content")
                    input_messages.append({"role": item.get("role", "user"), "content": content})
        else:
            return _error_response("'input' must be a string or array", 400)

        # Explicit conversation_history (stateless clients) beats previous_response_id chaining.
        conversation_history: List[Dict[str, Any]] = []
        raw_history = body.get("conversation_history")
        if raw_history:
            if not isinstance(raw_history, list):
                return _error_response("'conversation_history' must be an array of message objects", 400)
            for i, entry in enumerate(raw_history):
                if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                    return _error_response(f"conversation_history[{i}] must have 'role' and 'content' fields", 400)
                try:
                    entry_content = _normalize_multimodal_content(entry["content"])
                except ValueError as exc:
                    return _multimodal_validation_error(exc, param=f"conversation_history[{i}].content")
                conversation_history.append({"role": str(entry["role"]), "content": entry_content})
            if previous_response_id:
                logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")
        stored_session_id = None
        if not conversation_history and previous_response_id:
            stored = self._response_store.get(previous_response_id)
            if stored is None:
                return _error_response(f"Previous response not found: {previous_response_id}", 404)
            conversation_history = list(stored.get("conversation_history", []))
            stored_session_id = stored.get("session_id")
            if instructions is None:
                instructions = stored.get("instructions")
        # All input messages but the last become history; the last is the user message.
        conversation_history.extend(input_messages[:-1])
        user_message: Any = input_messages[-1].get("content", "") if input_messages else ""
        if not _content_has_visible_payload(user_message):
            return _error_response("No user message found in input", 400)
        if body.get("truncation") == "auto":
            conversation_history = _auto_truncate_response_history(conversation_history)

        # Session precedence: previous_response_id chain > declared X-Hermes-Session-Key > fresh
        # id. Binding the declared key follows the same precedence: a chain-selected session must
        # not have its routing key rewritten to this header.
        _declared_selected = not stored_session_id and bool(gateway_session_key)
        session_id = (
            stored_session_id
            or self._declared_conversation_session(gateway_session_key)
            or str(uuid.uuid4()))
        stream = _coerce_request_bool(body.get("stream"), default=False)
        route, agent_overrides, selection_error = self._select_request_route(
            body, session_id=session_id, gateway_session_key=gateway_session_key,
            model_alias=body.get("model"))
        if selection_error is not None:
            return selection_error
        run_kwargs = dict(
            user_message=user_message, conversation_history=conversation_history,
            ephemeral_system_prompt=instructions, session_id=session_id,
            gateway_session_key=gateway_session_key, bind_declared_conversation=_declared_selected,
            **agent_overrides, route=route)
        if stream:
            _stream_q = ThreadSafeAsyncQueue()

            def _on_tool_progress(event_type, name, preview, args, **kwargs):
                return  # structured start/complete callbacks carry the call id; progress ignored

            def _on_tool_start(tool_call_id, function_name, function_args):
                _stream_q.put_threadsafe(("__tool_started__", {
                    "tool_call_id": tool_call_id, "name": function_name,
                    "arguments": function_args or {}}))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                _stream_q.put_threadsafe(("__tool_completed__", {
                    "tool_call_id": tool_call_id, "name": function_name,
                    "arguments": function_args or {}, "result": function_result}))
            agent_task, agent_ref = self._spawn_stream_agent(
                _stream_q, tool_progress_callback=_on_tool_progress,
                tool_start_callback=_on_tool_start, tool_complete_callback=_on_tool_complete,
                **run_kwargs)
            return await self._write_sse_responses(
                request=request, response_id=f"resp_{uuid.uuid4().hex[:28]}",
                model=body.get("model", self._model_name), created_at=int(time.time()),
                stream_q=_stream_q, agent_task=agent_task, agent_ref=agent_ref,
                conversation_history=conversation_history, user_message=user_message,
                instructions=instructions, conversation=conversation, store=store,
                session_id=session_id, gateway_session_key=gateway_session_key)

        async def _compute_response():
            return await self._run_agent(**run_kwargs)
        outcome, err = await self._run_idempotent(
            request, body, _compute_response, log_label="responses",
            fingerprint_keys=["input", "instructions", "previous_response_id", "conversation", "model", "provider", "model_options", "tools"],
        )
        if err is not None:
            return err
        result, usage = outcome
        final_response = _resolve_media_to_data_urls(result.get("final_response", ""))
        if not final_response:
            final_response = _redact_api_error_text(result.get("error", "(No response generated)"))
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        created_at = int(time.time())
        full_history = self._build_response_conversation_history(
            conversation_history, user_message, result, final_response)
        # _run_agent's effective session id carries compression rotations; storing it keeps
        # previous_response_id chaining off the pre-rotation session (else compression re-fires).
        _result_sid = result.get("session_id") if isinstance(result, dict) else None
        _effective_session_id = (
            _result_sid if isinstance(_result_sid, str) and _result_sid else session_id)
        # Output items = current turn only (AIAgent returns a full transcript; mocked paths
        # only the current-turn suffix).
        output_start_index = self._response_messages_turn_start_index(
            conversation_history, user_message, result)
        response_data = {
            "id": response_id, "object": "response", "status": "completed",
            "created_at": created_at, "model": body.get("model", self._model_name),
            "output": self._extract_output_items(result, start_index=output_start_index),
            "usage": _responses_usage_payload(usage)}
        if store:
            self._response_store.put(response_id, {
                "response": response_data, "conversation_history": full_history,
                "instructions": instructions, "session_id": _effective_session_id})
            if conversation:
                self._response_store.set_conversation(conversation, response_id)
        response_headers = {"X-Hermes-Session-Id": _effective_session_id}
        if gateway_session_key:
            response_headers["X-Hermes-Session-Key"] = gateway_session_key
        return web.json_response(response_data, headers=response_headers)

    async def _handle_get_response(self, request: "web.Request") -> "web.Response":
        """GET /v1/responses/{response_id} — retrieve a stored response."""
        from gateway.platforms.api_server import _error_response
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        response_id = request.match_info["response_id"]
        stored = self._response_store.get(response_id)
        if stored is None:
            return _error_response(f"Response not found: {response_id}", 404)
        return web.json_response(stored["response"])

    async def _handle_delete_response(self, request: "web.Request") -> "web.Response":
        """DELETE /v1/responses/{response_id} — delete a stored response."""
        from gateway.platforms.api_server import _error_response
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        response_id = request.match_info["response_id"]
        if not self._response_store.delete(response_id):
            return _error_response(f"Response not found: {response_id}", 404)
        return web.json_response({"id": response_id, "object": "response", "deleted": True})

    @staticmethod
    def _build_response_conversation_history(
        conversation_history: List[Dict[str, Any]], user_message: Any, result: Dict[str, Any],
        final_response: Any) -> List[Dict[str, Any]]:
        """Build the stored Responses transcript without duplicating history.

        A compressed transcript (``result["_compressed"]``) shares no input-history prefix, so
        turn-start detection fails; prepending the uncompressed history would bloat the stored
        context and re-trigger compression every request — it is stored as-is instead.
        """
        from gateway.platforms.api_server import APIServerAdapter
        prior = list(conversation_history)
        current_user = {"role": "user", "content": user_message}
        agent_messages = result.get("messages") if isinstance(result, dict) else None
        if isinstance(agent_messages, list) and agent_messages:
            turn_start = APIServerAdapter._response_messages_turn_start_index(
                conversation_history, user_message, result)
            # turn_start == 0: compression rewrote the transcript or agent_messages is turn-only.
            if turn_start or result.get("_compressed"):
                return list(agent_messages)
            return prior + [current_user] + agent_messages
        return prior + [current_user, {"role": "assistant", "content": final_response}]

    @staticmethod
    def _response_messages_turn_start_index(
        conversation_history: List[Dict[str, Any]], user_message: Any, result: Dict[str, Any],
    ) -> int:
        """Detect transcript-shaped result["messages"] and return turn start."""
        agent_messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(agent_messages, list) or not agent_messages:
            return 0
        prior = list(conversation_history)
        expected_prefix = prior + [{"role": "user", "content": user_message}]
        if agent_messages[:len(expected_prefix)] == expected_prefix:
            return len(expected_prefix)
        if prior and agent_messages[:len(prior)] == prior:
            return len(prior)
        return 0

    @classmethod
    def _turn_transcript_messages(
        cls, conversation_history: List[Dict[str, Any]], user_message: Any, result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """This turn's assistant/tool messages in client-safe shape: clients accumulating
        ``assistant.delta`` into one buffer cannot reconstruct assistant segments that preceded
        tool calls, so ``run.completed`` carries the authoritative per-turn transcript.

        Emitting the authoritative per-turn transcript on ``run.completed`` lets any SSE consumer reconcile
        its live view against ground truth without a separate ``GET /messages`` round-trip. Purely additive:
        clients that ignore the field are unaffected. Refs #34703.
        """
        agent_messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(agent_messages, list) or not agent_messages:
            return []
        start = cls._response_messages_turn_start_index(conversation_history, user_message, result)
        out: List[Dict[str, Any]] = []
        for msg in agent_messages[start:]:
            if not isinstance(msg, dict) or msg.get("role") not in {"assistant", "tool"}:
                continue
            # _message_response projects compaction scaffolding; pure handoffs are "hidden".
            projected = cls._message_response(msg)
            if projected.get("display_kind") != "hidden":
                out.append(projected)
        return out

    @staticmethod
    def _extract_output_items(result: Dict[str, Any], start_index: int = 0) -> List[Dict[str, Any]]:
        """Output items from ``result["messages"][start_index:]``: ``function_call`` per assistant
        tool_call, ``function_call_output`` per tool message, then the final ``message``."""
        from gateway.platforms.api_server import _redact_api_error_text
        items: List[Dict[str, Any]] = []
        messages = result.get("messages", [])
        if start_index > 0:
            messages = messages[start_index:]
        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    # Already executed server-side; replayed for structured tool UI only, so
                    # marked completed (matching the SSE path) — never pending client calls.
                    items.append({
                        "id": f"fc_{uuid.uuid4().hex[:24]}", "type": "function_call",
                        "status": "completed", "name": func.get("name", ""),
                        "arguments": func.get("arguments", ""),
                        "call_id": tc.get("id", "")})
            elif role == "tool":
                items.append({
                    "id": f"fco_{uuid.uuid4().hex[:24]}", "type": "function_call_output",
                    "status": "completed", "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", "")})
        final = result.get("final_response", "") or _redact_api_error_text(
            result.get("error", "(No response generated)"))
        items.append(_message_item(final))
        return items
