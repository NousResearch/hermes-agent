"""Scripted, recording loopback server for OpenAI **chat-completions** dialect variants.

The vendor boundary behind Hermes' ``chat_completions`` transport as spoken by the
routes that bend the base dialect:

* OpenRouter / Nous Portal: a unified ``reasoning_details`` array on the assistant
  message (streamed as ``delta.reasoning_details``) that the client must replay, and
  provider errors delivered as an ``{"error": {...}}`` chunk INSIDE a 200 SSE stream
  (an upstream ban or moderation block after the stream opened);
* DeepSeek-style ``reasoning_content`` (and the ``reasoning`` alias) deltas;
* Ollama / llama.cpp / custom servers: ``strict_user_turn`` reproduces Ollama's
  renderer refusing a payload with no ``user`` message (HTTP 500
  ``no user query found in messages``); ``reasoning_budget_chars`` reproduces a route
  that 400s once the cumulative replayed reasoning text passes a budget.

Faults: HTTP errors with ``Retry-After``, a stream that drops after emitting part of a
tool call, and an in-stream error object. Every request body is recorded and validated
against the installed ``openai`` SDK's ``CompletionCreateParams`` (pydantic
``TypeAdapter``); every chunk is built from ``openai.types.chat.ChatCompletionChunk``.

Vendor-host impersonation: routes Hermes gates by hostname (``openrouter.ai``,
``nousresearch.com``) are reached by configuring ``base_url: http://<vendor host>/...``
and pointing the child's ``HTTP_PROXY`` at this server — the client sends absolute-form
requests here, no DNS or real network involved (``https://`` URLs get a refused
``CONNECT``, so no request can escape to the real vendor).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Union

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming
from pydantic import TypeAdapter

MODEL_ID = "fake-model"


@dataclass
class CText:
    """Assistant text; optional reasoning in any of the dialect's three fields."""

    text: str
    reasoning_content: str | None = None
    reasoning: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    prompt_tokens: int = 100
    completion_tokens: int = 20


@dataclass
class CTools:
    """One assistant message with tool calls. ``str`` args go out verbatim (malformed JSON)."""

    calls: list[tuple[str, dict[str, Any] | str]]
    text: str | None = None
    reasoning_content: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    finish_reason: str = "tool_calls"


@dataclass
class CError:
    status: int = 500
    message: str = "scripted failure"
    code: Any = None
    retry_after: float | None = None


@dataclass
class CStreamError:
    """HTTP 200, stream opens, optional partial text, then an in-stream ``error`` object."""

    error: dict[str, Any] = field(default_factory=lambda: {"code": 403, "message": "scripted in-stream error"})
    partial_text: str = ""


@dataclass
class CDropToolCall:
    """Stream the tool call's id/name and ``partial_args``, then close the socket."""

    name: str
    partial_args: str = '{"path": "x'


Step = Union[CText, CTools, CError, CStreamError, CDropToolCall]
Responder = Callable[[dict[str, Any]], Step]

_STREAM_ADAPTER = TypeAdapter(CompletionCreateParamsStreaming)


def validate_request(body: dict[str, Any]) -> str | None:
    """SDK-oracle validation; ``None`` when the body is a well-formed streaming request.
    Dialect extensions (``reasoning_details`` on messages, ``reasoning``/``provider``
    top-level keys) are extra keys the TypedDicts ignore, never type errors."""
    if not body.get("stream"):
        return None  # non-streaming aux calls: not part of the conformance claim
    try:
        _STREAM_ADAPTER.dump_json(_STREAM_ADAPTER.validate_python(body), warnings=False)
    except Exception as exc:  # noqa: BLE001
        return str(exc)[:4000]
    return None


def _chunk(delta: dict[str, Any], finish: str | None = None, usage: dict | None = None) -> dict[str, Any]:
    payload = {"id": "chatcmpl-variant", "object": "chat.completion.chunk", "created": int(time.time()),
               "model": MODEL_ID, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        payload["usage"] = usage
    # Build through the SDK model (dialect extras survive as extra fields), then dump.
    return ChatCompletionChunk.model_validate(payload).model_dump(mode="json", exclude_none=True)


def replayed_reasoning_chars(body: dict[str, Any]) -> int:
    """Cumulative ``reasoning_details`` text replayed on assistant messages."""
    total = 0
    for m in body.get("messages") or []:
        for d in (m.get("reasoning_details") or []) if isinstance(m, dict) else []:
            if isinstance(d, dict):
                total += len(str(d.get("text") or d.get("summary") or d.get("data") or ""))
    return total


class FakeChatVariantServer:
    def __init__(self, script: list[Step] | Responder | None = None, *, default_text: str = "ok",
                 strict_user_turn: bool = False, reasoning_budget_chars: int | None = None) -> None:
        self._script: list[Step] = list(script) if isinstance(script, list) else []
        self._responder = script if callable(script) else None
        self.default_text = default_text
        self.strict_user_turn = strict_user_turn
        self.reasoning_budget_chars = reasoning_budget_chars
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._server: ThreadingHTTPServer | None = None

    def __enter__(self) -> "FakeChatVariantServer":
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self))
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="fake-chat-variant", daemon=True).start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    @property
    def port(self) -> int:
        assert self._server is not None
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def proxy_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def push(self, *steps: Step) -> None:
        with self._lock:
            self._script.extend(steps)

    def _next(self, record: dict[str, Any]) -> Step:
        if self._responder is not None:
            return self._responder(record)
        with self._lock:
            if self._script:
                return self._script.pop(0)
        return CText(self.default_text)

    def next_call_id(self) -> str:
        with self._lock:
            self._seq += 1
            return f"call_variant_{self._seq}"

    def main_records(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["kind"] == "main"]

    def main_requests(self) -> list[dict[str, Any]]:
        return [r["body"] for r in self.main_records()]

    def invalid_requests(self) -> list[tuple[int, str]]:
        return [(i, r["invalid"]) for i, r in enumerate(self.main_records()) if r.get("invalid")]


def _handler_for(server: FakeChatVariantServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a: object) -> None:
            pass

        def _json(self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_CONNECT(self) -> None:  # noqa: N802 - https through the proxy: refused, never tunnelled
            self._json(403, {"error": {"message": "tunnel refused by test proxy"}})
            self.close_connection = True

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/").endswith("/models"):
                self._json(200, {"object": "list", "data": [
                    {"id": MODEL_ID, "object": "model", "context_length": 128000}]})
                return
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": {"message": "invalid json"}})
                return
            kind = "main" if body.get("tools") else "aux"
            record = {"path": self.path, "kind": kind, "body": body, "t": time.time(),
                      "host": self.headers.get("Host", ""), "invalid": validate_request(body)}
            with server._lock:
                server.requests.append(record)
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._json(404, {"error": {"message": f"unsupported path {self.path}"}})
                return
            if kind == "aux":
                self._render(CText("Fake title"), bool(body.get("stream")))
                return
            rejection = self._route_rejection(body)
            if rejection is not None:
                record["response"] = "route_rejection"
                self._json(*rejection)
                return
            step = server._next(record)
            record["response"] = type(step).__name__
            self._render(step, bool(body.get("stream")))

        def _route_rejection(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
            msgs = [m for m in body.get("messages") or [] if isinstance(m, dict)]
            if server.strict_user_turn and not any(m.get("role") == "user" for m in msgs):
                return 500, {"error": {"message": "no user query found in messages", "type": "api_error"}}
            budget = server.reasoning_budget_chars
            if budget is not None and replayed_reasoning_chars(body) > budget:
                return 400, {"error": {"message": "Provider returned error", "code": 400,
                                       "metadata": {"raw": "This request is not valid."}}}
            return None

        def _start_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

        def _sse(self, payload: dict[str, Any]) -> None:
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        def _render(self, step: Step, stream: bool) -> None:
            if isinstance(step, CError):
                headers = {"Retry-After": str(step.retry_after)} if step.retry_after is not None else {}
                self._json(step.status, {"error": {"message": step.message, "code": step.code,
                                                   "type": "server_error"}}, headers)
                return
            if not stream:
                self._render_json(step)
                return
            self._start_sse()
            self._sse(_chunk({"role": "assistant", "content": ""}))
            if isinstance(step, CStreamError):
                if step.partial_text:
                    self._sse(_chunk({"content": step.partial_text}))
                self.wfile.write(f"data: {json.dumps({'error': step.error})}\n\n".encode())
                self.wfile.flush()
                self.close_connection = True
                return
            if isinstance(step, CDropToolCall):
                self._sse(_chunk({"tool_calls": [{"index": 0, "id": server.next_call_id(), "type": "function",
                                                  "function": {"name": step.name, "arguments": ""}}]}))
                self._sse(_chunk({"tool_calls": [{"index": 0, "function": {"arguments": step.partial_args}}]}))
                self.close_connection = True
                return
            self._stream_reasoning(step)
            if isinstance(step, CText):
                for i in range(0, len(step.text), 8):
                    self._sse(_chunk({"content": step.text[i:i + 8]}))
                finish, usage = "stop", _usage(step.prompt_tokens, step.completion_tokens)
            else:
                if step.text:
                    self._sse(_chunk({"content": step.text}))
                for i, (name, args) in enumerate(step.calls):
                    arg_s = args if isinstance(args, str) else json.dumps(args)
                    self._sse(_chunk({"tool_calls": [{"index": i, "id": server.next_call_id(), "type": "function",
                                                      "function": {"name": name, "arguments": ""}}]}))
                    self._sse(_chunk({"tool_calls": [{"index": i, "function": {"arguments": arg_s}}]}))
                finish, usage = step.finish_reason, _usage(100, 10)
            self._sse(_chunk({}, finish, usage))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

        def _stream_reasoning(self, step: CText | CTools) -> None:
            if step.reasoning_content:
                self._sse(_chunk({"reasoning_content": step.reasoning_content}))
            if isinstance(step, CText) and step.reasoning:
                self._sse(_chunk({"reasoning": step.reasoning}))
            if step.reasoning_details:
                self._sse(_chunk({"reasoning_details": step.reasoning_details}))

        def _render_json(self, step: Step) -> None:
            if isinstance(step, CText):
                msg = {"role": "assistant", "content": step.text}
                if step.reasoning_details:
                    msg["reasoning_details"] = step.reasoning_details
                finish = "stop"
            elif isinstance(step, CTools):
                msg = {"role": "assistant", "content": step.text, "tool_calls": [
                    {"id": server.next_call_id(), "type": "function",
                     "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a)}}
                    for n, a in step.calls]}
                finish = step.finish_reason
            else:
                self._json(502, {"error": {"message": "stream-only fault on a non-streaming request"}})
                return
            self._json(200, {"id": "chatcmpl-variant", "object": "chat.completion", "created": int(time.time()),
                             "model": MODEL_ID, "usage": _usage(100, 20),
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish}]})

    return Handler


def _usage(prompt: int, completion: int) -> dict[str, Any]:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
