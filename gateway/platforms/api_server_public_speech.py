"""Request-owned public speech projection for opt-in Responses SSE consumers.

Protocol callbacks stay private before enqueue. Raw text deltas are never public in this
mode: the producer classifies interim messages as commentary, while its successful result
supplies the authoritative final answer. Final-token delivery is intentionally delayed.
"""
import asyncio
import logging
from contextlib import suppress
from typing import Any

from gateway.platforms.api_server_openai_routes import _ResponsesStream, _iter_stream_items, _message_item

logger = logging.getLogger("gateway.platforms.api_server")


def spawn_public_speech(adapter, run_kwargs, *, run_agent=None):
    queue = PublicSpeechQueue()
    agent_task, agent_ref = adapter._spawn_stream_agent(
        queue, run_agent=run_agent, interim_assistant_callback=public_callbacks(queue),
        tool_start_callback=lambda *_: None, tool_complete_callback=lambda *_: None,
        tool_progress_callback=lambda *_args, **_kwargs: None, **run_kwargs)
    return queue, agent_task, agent_ref


async def start_public_speech(adapter, request, *, body, run_kwargs, conversation_history,
                              user_message, instructions, conversation, store, session_id,
                              gateway_session_key, run_agent, response_id, created_at,
                              response_model, durable_key=None, terminal_replay=False):
    queue, task, agent_ref = spawn_public_speech(adapter, run_kwargs, run_agent=run_agent)
    return await write_public_speech(
        adapter, request, response_id=response_id,
        model=response_model, created_at=created_at,
        stream_q=queue, agent_task=task, agent_ref=agent_ref,
        conversation_history=conversation_history, user_message=user_message,
        instructions=instructions, conversation=conversation, store=store,
        session_id=session_id, gateway_session_key=gateway_session_key,
        durable_key=durable_key, terminal_replay=terminal_replay)


class PublicSpeechQueue:
    """Only speech, classification markers and EOS can cross this queue boundary."""

    def __init__(self):
        from gateway.platforms.api_server import ThreadSafeAsyncQueue
        self._queue = ThreadSafeAsyncQueue()

    def put_threadsafe(self, item):
        if isinstance(item, str):
            self._queue.put_threadsafe(item)
        elif isinstance(item, tuple) and len(item) == 2 and item[0] == "__commentary__":
            text = item[1]
            if isinstance(text, dict) and isinstance(text.get("text"), str):
                self._queue.put_threadsafe(item)

    def put_nowait(self, item):
        if item is None:
            self._queue.put_nowait(None)

    async def get(self):
        return await self._queue.get()

    def get_nowait(self):
        return self._queue.get_nowait()


def public_callbacks(queue):
    """Bind agent callback semantics to one request, without changing generic producer callbacks."""
    # A callback runs on the worker thread, after ordinary-provider raw deltas.
    def on_commentary(text, *, already_streamed=False):
        if isinstance(text, str) and text.strip():
            queue.put_threadsafe(("__commentary__", {"text": text, "already_streamed": already_streamed}))
    return on_commentary


def valid_effective_session_id(value, adapter):
    """Do not reflect an untrusted result identity or persist it for the next turn."""
    from gateway.session import _is_path_unsafe
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if any(ord(char) < 32 or ord(char) == 127 for char in value) or _is_path_unsafe(value):
        return False
    try:
        return len(value.encode("utf-8")) <= adapter._MAX_SESSION_HEADER_LEN
    except UnicodeError:
        return False


class PublicSpeechStream(_ResponsesStream):
    """The opt-in writer owns its own result classification and stored public envelope."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.public_text: list[str] = []
        self._commentary: list[dict[str, Any]] = []
        self._terminal_message: dict[str, Any] | None = None
        self._partial_commentary: dict[str, Any] | None = None
        self._delivered_final = ""
        self.terminal_written = False

    async def write_event(self, event_type, data):
        # Public ordering is projection-local, never the private tool/reasoning ledger.
        # Record only accepted writes; failed writes are not replay/delivery evidence.
        import copy
        data.setdefault('sequence_number', self.sequence_number)
        self.sequence_number += 1
        await self.response.write(self._api._sse_frame(data, event=event_type))
        if self.durable_key:
            self.recorded_events.append((event_type, copy.deepcopy(data)))

    def _output_index(self, item_id):
        index = self.output_index
        self.output_index += 1
        return index

    def persist_snapshot(self, response_env, *, history=None, session_id=None):
        if self.terminal_replay:
            saved = self.response_store.get(self.response_id)
            if response_env['status'] != 'completed' or (
                    saved and saved['response']['status'] != 'incomplete'):
                # Snapshot ownership does not depend on the derivative replay cache.
                # Only a disconnected publisher's incomplete snapshot may be reconstructed.
                return
        return super().persist_snapshot(response_env, history=history, session_id=session_id)

    async def dispatch(self, item: Any) -> None:
        if self.terminal_replay:
            return  # late observers cannot assign identities to a missing commentary prefix
        if isinstance(item, str):
            return  # unclassified model/provider deltas never cross the public wire
        elif isinstance(item, tuple) and len(item) == 2 and item[0] == "__commentary__":
            payload = item[1]
            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                return
            text = payload["text"]
            # The producer classifies interim speech, including preambles previously
            # sent as raw internal deltas. Only its completed result defines final text.
            await self.emit_commentary(text)

    async def emit_commentary(self, text: str) -> None:
        item = {"id": self._item_id('message', commentary=True), "type": "message",
                "role": "assistant", "status": "completed", "phase": "commentary",
                "content": [{"type": "output_text", "text": text}]}
        idx = self.output_index
        self.output_index += 1
        await self.write_event("response.output_item.added", {
            "type": "response.output_item.added", "output_index": idx, "item": item})
        # An added item has visible text but no done frame yet.
        self._partial_commentary = {**item, "status": "in_progress"}
        await self.write_event("response.output_item.done", {
            "type": "response.output_item.done", "output_index": idx, "item": item})
        self._commentary.append(item)
        self._partial_commentary = None

    def _output(self):
        return [*self._commentary, *([self._partial_commentary] if self._partial_commentary else []),
                *([self._terminal_message] if self._terminal_message else [])]

    def _history(self):
        # Public delivery is not the private execution transcript. Retain observed
        # tool calls/results on failure and writer recovery without exposing them.
        if isinstance(self.result, dict) and isinstance(self.result.get('messages'), list) and self.result['messages']:
            return self.adapter._build_response_conversation_history(
                self.conversation_history, self.user_message, self.result, self._delivered_final,
                tool_output_max_chars=self.adapter._history_tool_output_max_chars)
        history = self._history_with_user()
        if self._delivered_final:
            history.append({"role": "assistant", "content": self._delivered_final})
        return history

    def _unsuccessful_envelope(self, status, *, interrupted=False):
        items = self._output()
        if self._delivered_final and self._terminal_message is None:
            items.append({"id": self.message_item_id, "status": "incomplete",
                          **_message_item(self._delivered_final)})
        env = self.terminal_envelope(status, items, error="Agent did not complete.")
        env["hermes"] = {"completed": False, "failed": status == "failed",
                         "partial": bool(items), "interrupted": interrupted}
        return env

    def persist_incomplete_if_needed(self):
        if self.store and not self.terminal_snapshot_persisted:
            self.persist_snapshot(self._unsuccessful_envelope("incomplete"), history=self._history(),
                                  session_id=self.session_id)

    async def _close_public_final(self):
        text = "".join(self.public_text)
        if not text:
            return
        await self._open_message_item()
        await self.write_event("response.output_text.delta", {
            "type": "response.output_text.delta", "item_id": self.message_item_id,
            "output_index": self.message_output_index, "content_index": 0,
            "delta": text, "logprobs": []})
        self._delivered_final = text
        await self.write_event("response.output_text.done", {
            "type": "response.output_text.done", "item_id": self.message_item_id,
            "output_index": self.message_output_index, "content_index": 0,
            "text": text, "logprobs": []})
        done_item = {"id": self.message_item_id, "status": "completed", **_message_item(text)}
        await self.write_event("response.output_item.done", {
            "type": "response.output_item.done", "output_index": self.message_output_index,
            "item": done_item})
        self._terminal_message = done_item

    async def finish(self, agent_task):
        try:
            result, usage = await agent_task
            self.usage = usage or self.usage
        except Exception:
            logger.exception("Agent failed during public speech Responses stream")
            result = {"failed": True, "completed": False}
        self.result = result
        failed = not isinstance(result, dict) or not result or bool(result.get("failed") or result.get("error"))
        # _ProviderAuthResolutionError is caught by _run_agent and returned as a
        # guidance-only result: no model call and no failure flags. Its final text
        # is diagnostic, not model speech.
        if isinstance(result, dict) and result.get("api_calls") == 0 and not result.get("messages"):
            failed = True
        incomplete = isinstance(result, dict) and bool(
            result.get("interrupted") or result.get("partial") or result.get("completed") is False)
        status = "failed" if failed else "incomplete" if incomplete else "completed"
        history = None
        sid = self.session_id
        if status == "completed":
            # Held deltas have never been delivered publicly. Only the successful
            # agent's final is authoritative (including empty transforms).
            final = result.get("final_response")
            if result.get("_notification_presentation_suppressed") is True:
                self.public_text = []
            elif isinstance(final, str):
                self.public_text = [final]
            else:
                self.public_text = []
            self.final_response_text = "".join(self.public_text)
            # Preflight the private continuation before publishing the final item.
            history = self.adapter._build_response_conversation_history(
                self.conversation_history, self.user_message, result, self.final_response_text,
                tool_output_max_chars=self.adapter._history_tool_output_max_chars)
            sid = result.get("session_id")
            sid = sid if valid_effective_session_id(sid, self.adapter) else self.session_id
            # Exercise fallible storage before final delivery, but expose only an
            # in-progress snapshot with the original session and no future speech.
            # GET and previous_response_id may use that safe state while SSE is open.
            self.persist_snapshot(self.envelope("in_progress") | {"output": self._output()},
                                  history=self._history_with_user(), session_id=self.session_id)
            await self._close_public_final()
            env = self.terminal_envelope(status, self._output())
            env["hermes"] = {"session_id": sid}
        else:
            # Failed raw deltas and result.final_response may be provider diagnostics.
            self.public_text = []
            env = self._unsuccessful_envelope(status, interrupted=bool(
                isinstance(result, dict) and result.get("interrupted")))
            self.persist_snapshot(env, history=self._history(), session_id=self.session_id)
            self.terminal_snapshot_persisted = True
        await self.write_event("response." + status, {"type": "response." + status, "response": env})
        self.terminal_written = True
        if status == "completed":
            # The write completed: publish the prepared private continuation and
            # rotated identity synchronously, with no await between wire and store.
            self.persist_snapshot(env, history=history, session_id=sid)
        self.terminal_snapshot_persisted = True
        if self.durable_key and not self.terminal_replay:
            # Keep only the public projection in the derivative retry cache. The
            # authority's private execution receipt remains untouched.
            headers = {key: value for key, value in self.response.headers.items()
                       if key.startswith('X-Hermes-')}
            self.response_store.put(self.durable_key[0], {
                'fingerprint': self.durable_key[1], 'response': env, 'headers': headers,
                'events': self.recorded_events})


async def write_public_speech(adapter, request, *, response_id, model, created_at, stream_q,
                              agent_task, agent_ref, conversation_history, user_message,
                              instructions, conversation, store, session_id, gateway_session_key=None,
                              durable_key=None, terminal_replay=False):
    """Opt-in writer, separate from the generic Responses writer and batch projection."""
    from gateway.platforms.api_server import _abandon_agent_task
    response = None
    st = None
    try:
        response = await adapter._prepare_sse_response(request, session_id, gateway_session_key)
        st = PublicSpeechStream(
            adapter, response, response_id=response_id, model=model, created_at=created_at,
            conversation_history=conversation_history, user_message=user_message,
            instructions=instructions, conversation=conversation, store=store, session_id=session_id,
            durable_key=durable_key, terminal_replay=terminal_replay)
        await st.emit_created()
        async for item in _iter_stream_items(stream_q, agent_task, response):
            if item is None:
                break
            await st.dispatch(item)
        await st.finish(agent_task)
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
        if st is not None and not st.terminal_written:
            with suppress(Exception):
                st.persist_incomplete_if_needed()
        await _abandon_agent_task(agent_ref, agent_task, "SSE client disconnected")
    except asyncio.CancelledError:
        if st is not None and not st.terminal_written:
            with suppress(Exception):
                st.persist_incomplete_if_needed()
        await _abandon_agent_task(agent_ref, agent_task, "SSE task cancelled",
                                  reap_source="api_server_sse_cancelled", await_cancel=False)
        raise
    except Exception:
        logger.exception("Public speech Responses writer failed")
        if st is not None and st.terminal_written:
            # A completed frame was already accepted by the transport; a late
            # store failure cannot retract it. Never send a contradictory terminal.
            logger.error("Public speech terminal was written but snapshot persistence failed")
        elif st is not None:
            with suppress(Exception):
                env = st._unsuccessful_envelope("failed")
                st.persist_snapshot(env, history=st._history(), session_id=st.session_id)
                st.terminal_snapshot_persisted = True
                await st.write_event("response.failed", {"type": "response.failed", "response": env})
        await _abandon_agent_task(agent_ref, agent_task, "Public speech writer failed")
        if response is None:
            raise
    return response
