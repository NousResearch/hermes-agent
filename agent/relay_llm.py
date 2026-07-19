"""Core NeMo Relay adapters for physical Hermes provider attempts."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

from agent import relay_runtime


_PROVIDER_MESSAGE_EXTENSION_KEYS = frozenset(
    {"reasoning_content", "reasoning_details"}
)


def execute(
    request: dict[str, Any],
    callback: Callable[[dict[str, Any]], Any],
    *,
    session_id: str,
    name: str,
    model_name: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Run one non-streaming physical provider attempt through Relay."""
    runtime, session, parent = _execution_context(session_id)
    if runtime is None or session is None:
        return callback(request)
    logical = _logical_parent(runtime, session, parent, metadata)
    parent = logical[1] if logical is not None else parent

    relay_request_body = _relay_request_body(request, metadata)
    relay_request = runtime.relay.LLMRequest({}, relay_request_body)
    raw_response: dict[str, Any] = {}
    callback_error: BaseException | None = None

    def invoke(next_request: Any) -> Any:
        nonlocal callback_error
        try:
            final_request = _provider_request(
                request,
                next_request,
                relay_request_body=relay_request_body,
                metadata=metadata,
            )
            raw = callback(final_request)
        except BaseException as exc:
            callback_error = exc
            raise
        raw_response["value"] = raw
        raw_response["json"] = _jsonable(raw)
        return raw_response["json"]

    try:
        managed = _run_awaitable(
            runtime.run_in_session_async(
                session,
                runtime.relay.llm.execute,
                name,
                relay_request,
                invoke,
                handle=parent,
                metadata=_jsonable(metadata or {}),
                model_name=model_name,
                codec=_codec(runtime.relay, metadata),
                response_codec=_codec(runtime.relay, metadata),
            )
        )
    except BaseException as exc:
        if (
            callback_error is not None
            and relay_runtime._is_relay_wrapped_callback_error(exc, callback_error)
        ):
            raise callback_error
        raise

    if "value" in raw_response and _json_equal(managed, raw_response["json"]):
        _complete_logical(logical, outcome="success")
        return raw_response["value"]
    _complete_logical(logical, outcome="success")
    return managed


async def execute_async(
    request: dict[str, Any],
    callback: Callable[[dict[str, Any]], Any],
    *,
    session_id: str,
    name: str,
    model_name: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Run one asynchronous physical provider attempt through Relay."""
    runtime, session, parent = _execution_context(session_id)
    if runtime is None or session is None:
        return await callback(request)
    logical = _logical_parent(runtime, session, parent, metadata)
    parent = logical[1] if logical is not None else parent

    relay_request_body = _relay_request_body(request, metadata)
    relay_request = runtime.relay.LLMRequest({}, relay_request_body)
    raw_response: dict[str, Any] = {}
    callback_error: BaseException | None = None

    async def invoke(next_request: Any) -> Any:
        nonlocal callback_error
        try:
            final_request = _provider_request(
                request,
                next_request,
                relay_request_body=relay_request_body,
                metadata=metadata,
            )
            raw = await callback(final_request)
        except BaseException as exc:
            callback_error = exc
            raise
        raw_response["value"] = raw
        raw_response["json"] = _jsonable(raw)
        return raw_response["json"]

    try:
        managed = await runtime.run_in_session_async(
            session,
            runtime.relay.llm.execute,
            name,
            relay_request,
            invoke,
            handle=parent,
            metadata=_jsonable(metadata or {}),
            model_name=model_name,
            codec=_codec(runtime.relay, metadata),
            response_codec=_codec(runtime.relay, metadata),
        )
    except BaseException as exc:
        if (
            callback_error is not None
            and relay_runtime._is_relay_wrapped_callback_error(exc, callback_error)
        ):
            raise callback_error
        raise

    _complete_logical(logical, outcome="success")
    if "value" in raw_response and _json_equal(managed, raw_response["json"]):
        return raw_response["value"]
    return managed


def execute_current(
    request: dict[str, Any],
    callback: Callable[[dict[str, Any]], Any],
    *,
    name: str,
    model_name: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Run a provider attempt under the inherited Hermes turn when present."""
    turn = relay_runtime.current_turn()
    if turn is None:
        return callback(request)
    return execute(
        request,
        callback,
        session_id=turn.lease.session_id,
        name=name,
        model_name=model_name,
        metadata=metadata,
    )


async def execute_current_async(
    request: dict[str, Any],
    callback: Callable[[dict[str, Any]], Any],
    *,
    name: str,
    model_name: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Run an async provider attempt under the inherited turn when present."""
    turn = relay_runtime.current_turn()
    if turn is None:
        return await callback(request)
    return await execute_async(
        request,
        callback,
        session_id=turn.lease.session_id,
        name=name,
        model_name=model_name,
        metadata=metadata,
    )


def stream_current(
    request: dict[str, Any],
    stream_factory: Callable[[dict[str, Any]], Any],
    *,
    name: str,
    model_name: str,
    finalizer: Callable[[], Any],
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Run a provider stream under the inherited Hermes turn when present."""
    turn = relay_runtime.current_turn()
    if turn is None:
        return stream_factory(request)
    return stream(
        request,
        stream_factory,
        session_id=turn.lease.session_id,
        name=name,
        model_name=model_name,
        finalizer=finalizer,
        metadata=metadata,
    )


def stream(
    request: dict[str, Any],
    stream_factory: Callable[[dict[str, Any]], Any],
    *,
    session_id: str,
    name: str,
    model_name: str,
    finalizer: Callable[[], Any],
    on_stream_created: Callable[[Any], None] | None = None,
    on_chunk: Callable[[Any], None] | None = None,
    chunk_adapter: Callable[[Any], Any] | None = None,
    accept_chunk: Callable[[Any], bool] | None = None,
    completed_response_predicate: Callable[[Any], bool] | None = None,
    metadata: dict[str, Any] | None = None,
) -> "ManagedLlmStream":
    """Return a synchronous view of one Relay-managed provider stream."""
    return ManagedLlmStream(
        request,
        stream_factory,
        session_id=session_id,
        name=name,
        model_name=model_name,
        finalizer=finalizer,
        on_stream_created=on_stream_created,
        on_chunk=on_chunk,
        chunk_adapter=chunk_adapter,
        accept_chunk=accept_chunk,
        completed_response_predicate=completed_response_predicate,
        metadata=metadata,
    )


class ManagedLlmStream(Iterator[Any]):
    """Drive Relay's async stream from Hermes's provider worker thread."""

    def __init__(
        self,
        request: dict[str, Any],
        stream_factory: Callable[[dict[str, Any]], Any],
        *,
        session_id: str,
        name: str,
        model_name: str,
        finalizer: Callable[[], Any],
        on_stream_created: Callable[[Any], None] | None,
        on_chunk: Callable[[Any], None] | None,
        chunk_adapter: Callable[[Any], Any] | None,
        accept_chunk: Callable[[Any], bool] | None,
        completed_response_predicate: Callable[[Any], bool] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        self.final_response: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: Any = None
        self._closed = False
        self._callback_error: BaseException | None = None
        self._logical: tuple[relay_runtime.RelayTurnContext, Any, str] | None = None
        self._on_chunk = on_chunk
        self._chunk_adapter = chunk_adapter or _namespace
        self._accept_chunk = accept_chunk
        self._relay_observes_chunks = False
        self._raw_chunks: list[tuple[Any, Any]] = []

        runtime, session, parent = _execution_context(session_id)
        if runtime is None or session is None:
            raw_stream = stream_factory(request)
            if completed_response_predicate is not None and completed_response_predicate(
                raw_stream
            ):
                self.final_response = raw_stream
                self._stream = iter(())
            else:
                if on_stream_created is not None:
                    on_stream_created(raw_stream)
                self._stream = iter(raw_stream)
            return

        self._logical = _logical_parent(runtime, session, parent, metadata)
        if self._logical is not None:
            parent = self._logical[1]
        relay_request_body = _relay_request_body(request, metadata)
        relay_request = runtime.relay.LLMRequest({}, relay_request_body)

        async def provider_stream(next_request: Any):
            raw_stream = None
            try:
                raw_stream = stream_factory(
                    _provider_request(
                        request,
                        next_request,
                        relay_request_body=relay_request_body,
                        metadata=metadata,
                    )
                )
                if (
                    completed_response_predicate is not None
                    and completed_response_predicate(raw_stream)
                ):
                    self.final_response = raw_stream
                    return
                if on_stream_created is not None:
                    on_stream_created(raw_stream)
                for chunk in raw_stream:
                    if self._accept_chunk is not None and not self._accept_chunk(
                        chunk
                    ):
                        break
                    encoded_chunk = _jsonable(chunk)
                    self._raw_chunks.append((encoded_chunk, chunk))
                    yield encoded_chunk
            except BaseException as exc:
                self._callback_error = exc
                raise
            finally:
                close = getattr(raw_stream, "close", None)
                if callable(close):
                    close()

        def observe_chunk(chunk: Any) -> None:
            if self._on_chunk is not None:
                self._on_chunk(_jsonable(chunk))

        def relay_finalizer() -> Any:
            if self.final_response is not None:
                return _jsonable(self.final_response)
            return _jsonable(finalizer())

        loop = asyncio.new_event_loop()
        self._loop = loop
        self._relay_observes_chunks = True
        try:
            self._stream = loop.run_until_complete(
                runtime.run_in_session_async(
                    session,
                    runtime.relay.llm.stream_execute,
                    name,
                    relay_request,
                    provider_stream,
                    observe_chunk,
                    relay_finalizer,
                    handle=parent,
                    metadata=_jsonable(metadata or {}),
                    model_name=model_name,
                    codec=_codec(runtime.relay, metadata),
                    response_codec=_codec(runtime.relay, metadata),
                )
            )
        except BaseException:
            loop.close()
            self._loop = None
            raise

    def __iter__(self) -> "ManagedLlmStream":
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        if self._loop is None:
            try:
                return next(self._stream)
            except StopIteration:
                self.close()
                raise

        async def next_chunk() -> Any:
            return await anext(self._stream)

        try:
            chunk = self._loop.run_until_complete(next_chunk())
        except StopAsyncIteration:
            _complete_logical(self._logical, outcome="success")
            self._logical = None
            self.close()
            raise StopIteration from None
        except BaseException as exc:
            callback_error = self._callback_error
            self.close()
            if (
                callback_error is not None
                and relay_runtime._is_relay_wrapped_callback_error(exc, callback_error)
            ):
                raise callback_error
            raise
        if not self._relay_observes_chunks and self._on_chunk is not None:
            self._on_chunk(chunk)
        for index, (encoded, raw) in enumerate(self._raw_chunks):
            if _json_equal(chunk, encoded):
                del self._raw_chunks[: index + 1]
                return raw
        return self._chunk_adapter(chunk)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        self._loop = None
        if loop is None:
            close = getattr(self._stream, "close", None)
            if callable(close):
                close()
            return
        close = getattr(self._stream, "aclose", None)
        if callable(close):

            async def close_stream() -> None:
                await close()

            try:
                loop.run_until_complete(close_stream())
            except Exception:
                pass
        loop.close()

    def __del__(self) -> None:
        self.close()


class AnthropicStreamAccumulator:
    """Rebuild an Anthropic Message from post-intercept SSE events."""

    def __init__(self) -> None:
        self._message: dict[str, Any] = {}
        self._blocks: dict[int, dict[str, Any]] = {}

    def observe(self, event: Any) -> None:
        payload = _jsonable(event)
        if not isinstance(payload, dict):
            return
        event_type = payload.get("type")
        if event_type == "message_start":
            message = payload.get("message")
            if isinstance(message, dict):
                for key in ("id", "type", "role", "model", "usage"):
                    if key in message:
                        self._message[key] = message[key]
            return
        if event_type == "content_block_start":
            index = payload.get("index")
            block = payload.get("content_block")
            if isinstance(index, int) and isinstance(block, dict):
                self._blocks[index] = dict(block)
            return
        if event_type == "content_block_delta":
            index = payload.get("index")
            delta = payload.get("delta")
            if not isinstance(index, int) or not isinstance(delta, dict):
                return
            block = self._blocks.setdefault(index, {})
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                block["text"] = str(block.get("text") or "") + str(
                    delta.get("text") or ""
                )
            elif delta_type == "thinking_delta":
                block["thinking"] = str(block.get("thinking") or "") + str(
                    delta.get("thinking") or ""
                )
            elif delta_type == "signature_delta":
                block["signature"] = str(block.get("signature") or "") + str(
                    delta.get("signature") or ""
                )
            elif delta_type == "input_json_delta":
                partial = str(block.pop("_partial_json", "")) + str(
                    delta.get("partial_json") or ""
                )
                block["_partial_json"] = partial
            elif delta_type == "citations_delta" and "citation" in delta:
                block.setdefault("citations", []).append(delta["citation"])
            return
        if event_type == "message_delta":
            delta = payload.get("delta")
            if isinstance(delta, dict):
                for key in ("stop_reason", "stop_sequence"):
                    if key in delta:
                        self._message[key] = delta[key]
            if "usage" in payload:
                self._message["usage"] = payload["usage"]

    def finalize(self) -> dict[str, Any]:
        blocks = []
        for index in sorted(self._blocks):
            block = dict(self._blocks[index])
            partial = block.pop("_partial_json", None)
            if partial is not None:
                try:
                    block["input"] = json.loads(partial)
                except (TypeError, ValueError):
                    block["input"] = partial
            blocks.append(block)
        return {**self._message, "content": blocks}

    def response(self, base: Any = None) -> Any:
        """Return the attribute-shaped response consumed by Hermes."""
        assembled = self.finalize()
        if base is not None and base.__class__.__module__ == "unittest.mock":
            base_payload = {}
            for key in (
                "id",
                "type",
                "role",
                "model",
                "content",
                "stop_reason",
                "stop_sequence",
                "usage",
            ):
                value = getattr(base, key, None)
                if value is not None and value.__class__.__module__ != "unittest.mock":
                    base_payload[key] = _jsonable(value)
        else:
            base_payload = _jsonable(base)
        if not isinstance(base_payload, dict):
            base_payload = {}
        content = assembled.pop("content", [])
        merged = {**base_payload, **assembled}
        if content or "content" not in merged:
            merged["content"] = content
        return _namespace(merged)


def _execution_context(
    session_id: str,
) -> tuple[relay_runtime.RelayRuntime | None, Any, Any]:
    turn = relay_runtime.current_turn()
    if turn is not None and isinstance(turn.lease.host, relay_runtime.RelayRuntime):
        return turn.lease.host, turn.lease.session, turn.handle
    runtime = relay_runtime.get_runtime()
    if runtime is None:
        return None, None, None
    session = runtime.get_session(session_id)
    if session is None:
        session = runtime.ensure_session({"session_id": session_id})
    return runtime, session, None if session is None else session.handle


def _logical_parent(
    runtime: relay_runtime.RelayRuntime,
    session: Any,
    parent: Any,
    metadata: dict[str, Any] | None,
) -> tuple[relay_runtime.RelayTurnContext, Any, str] | None:
    turn = relay_runtime.current_turn()
    request_id = str((metadata or {}).get("api_request_id") or "")
    if turn is None or not request_id or turn.lease.host is not runtime:
        return None
    with turn.logical_llm_lock:
        handle = turn.logical_llm_calls.get(request_id)
        if handle is None:
            handle = runtime.run_in_session(
                session,
                runtime.relay.scope.push,
                relay_runtime.LOGICAL_LLM_SCOPE,
                runtime.relay.ScopeType.Function,
                handle=parent,
                input={},
                metadata={
                    relay_runtime.RUNTIME_SCHEMA_KEY: relay_runtime.RUNTIME_SCHEMA_VERSION,
                    relay_runtime.RUNTIME_INSTANCE_KEY: runtime.runtime_id,
                    "hermes.call_role": str(
                        (metadata or {}).get("call_role") or "primary"
                    ),
                },
            )
            turn.logical_llm_calls[request_id] = handle
    return turn, handle, request_id


def _complete_logical(
    logical: tuple[relay_runtime.RelayTurnContext, Any, str] | None,
    *,
    outcome: str,
) -> None:
    if logical is None:
        return
    turn, handle, request_id = logical
    lease = turn.lease
    if not isinstance(lease.host, relay_runtime.RelayRuntime):
        return
    with turn.logical_llm_lock:
        if turn.logical_llm_calls.get(request_id) is not handle:
            return
        turn.logical_llm_calls.pop(request_id, None)
    if lease.session is None:
        return
    lease.host.run_in_session(
        lease.session,
        lease.host.relay.scope.pop,
        handle,
        output={"outcome": outcome},
        metadata={
            relay_runtime.RUNTIME_SCHEMA_KEY: relay_runtime.RUNTIME_SCHEMA_VERSION,
            relay_runtime.RUNTIME_INSTANCE_KEY: lease.host.runtime_id,
        },
    )


def _provider_request(
    original: dict[str, Any],
    request: Any,
    *,
    relay_request_body: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    content = getattr(request, "content", request)
    if not isinstance(content, dict):
        content = relay_request_body
    if _json_equal(content, relay_request_body):
        final = dict(original)
    else:
        final = _provider_request_body(content, metadata)
        # Codec-only normalization must not silently change the provider wire
        # request when an unrelated interceptor edits another field.
        for key, value in original.items():
            if key not in relay_request_body and value is None:
                final.setdefault(key, value)
            elif (
                key in relay_request_body
                and key in final
                and _json_equal(final[key], relay_request_body[key])
            ):
                final[key] = value
        _restore_provider_message_extensions(original, final)
    headers = getattr(request, "headers", None)
    if isinstance(headers, dict) and headers:
        final["extra_headers"] = {
            **dict(final.get("extra_headers") or {}),
            **headers,
        }
    return final


def _relay_request_body(
    request: dict[str, Any], metadata: dict[str, Any] | None
) -> dict[str, Any]:
    body = _jsonable(request)
    if not isinstance(body, dict):
        return {}
    # The Responses SDK accepts ``tools=None`` as "no tools", while Relay's
    # typed Responses codec correctly expects either an array or an absent
    # field. Normalize only the codec-facing copy; the original provider
    # request is restored when no interceptor changes it.
    if str((metadata or {}).get("api_mode") or "") == "codex_responses":
        body = dict(body)
        if body.get("tools") is None:
            body.pop("tools", None)
        elif isinstance(body.get("tools"), list):
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        key: value
                        for key, value in tool.items()
                        if key != "type"
                    },
                }
                if isinstance(tool, dict)
                and tool.get("type") == "function"
                and "function" not in tool
                else tool
                for tool in body["tools"]
            ]
    elif str((metadata or {}).get("api_mode") or "") == "chat_completions":
        tools = body.get("tools")
        if isinstance(tools, list):
            body = dict(body)
            body["tools"] = [
                {"type": "function", **tool}
                if isinstance(tool, dict)
                and "function" in tool
                and "type" not in tool
                else tool
                for tool in tools
            ]
    return body


def _restore_provider_message_extensions(
    original: dict[str, Any], final: dict[str, Any]
) -> None:
    """Restore provider wire fields that Relay's typed codec cannot represent."""
    original_messages = original.get("messages")
    final_messages = final.get("messages")
    if not isinstance(original_messages, list) or not isinstance(final_messages, list):
        return
    if len(original_messages) != len(final_messages):
        return
    for original_message, final_message in zip(
        original_messages, final_messages, strict=True
    ):
        if not isinstance(original_message, dict) or not isinstance(final_message, dict):
            continue
        for key in _PROVIDER_MESSAGE_EXTENSION_KEYS:
            if key in original_message and key not in final_message:
                final_message[key] = original_message[key]


def _provider_request_body(
    content: dict[str, Any], metadata: dict[str, Any] | None
) -> dict[str, Any]:
    body = dict(content)
    if str((metadata or {}).get("api_mode") or "") != "codex_responses":
        return body
    tools = body.get("tools")
    if not isinstance(tools, list):
        return body
    body["tools"] = [
        {
            "type": "function",
            **dict(tool["function"]),
        }
        if isinstance(tool, dict)
        and tool.get("type") == "function"
        and isinstance(tool.get("function"), dict)
        else tool
        for tool in tools
    ]
    return body


def _codec(relay: Any, metadata: dict[str, Any] | None) -> Any:
    api_mode = str((metadata or {}).get("api_mode") or "")
    codecs = getattr(relay, "codecs", None)
    if codecs is None:
        return None
    if api_mode == "chat_completions":
        codec = getattr(codecs, "OpenAIChatCodec", None)
    elif api_mode == "anthropic_messages":
        codec = getattr(codecs, "AnthropicMessagesCodec", None)
    elif api_mode == "codex_responses":
        codec = getattr(codecs, "OpenAIResponsesCodec", None)
    else:
        codec = None
    return codec() if callable(codec) else None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Test doubles synthesize arbitrary callable attributes such as
    # ``model_dump``. Treat them as opaque instead of recursively invoking an
    # endless chain of child mocks.
    if value.__class__.__module__ == "unittest.mock":
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"))
        except Exception:
            pass
    try:
        return _jsonable(vars(value))
    except (TypeError, AttributeError):
        return str(value)


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{
            str(key): _namespace(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _json_equal(left: Any, right: Any) -> bool:
    try:
        return json.dumps(
            _jsonable(left), sort_keys=True, separators=(",", ":")
        ) == json.dumps(_jsonable(right), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return False


def _run_awaitable(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError(
        "Synchronous Relay LLM execution cannot run on an event-loop thread"
    )
