"""Scripted, recording loopback LLM provider for end-to-end tests.

One real HTTP server on 127.0.0.1 that speaks the OpenAI Chat Completions
wire format (JSON and SSE streaming). Every request body is recorded so a test
can assert on exactly what Hermes sent (history integrity, prompt-cache prefix
stability, routing/credential isolation), and every response is scripted so a
test can drive tool calls, reasoning, long streams and provider faults through
the real client stack instead of mocking the agent loop.

Main-turn requests (those carrying ``tools``) consume the script in order;
requests without ``tools`` are auxiliary calls (title generation, compression
summaries, judges) and are answered by ``aux`` so they never eat a scripted
turn. When the script is exhausted, main turns answer ``default_text``.

Usage::

    with FakeLLMServer([ToolCall("terminal", {"command": "echo hi"}), Text("done")]) as srv:
        write_hermes_home(home, srv.base_url)
        ...run hermes...
        assert srv.main_requests()[1]["messages"][-1]["role"] == "tool"

Run standalone for manual probes: ``python -m tests.fakes.fake_llm_provider 8765``.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Union

MODEL_ID = "fake-model"


# Scripted responses ---------------------------------------------------------


@dataclass
class Text:
    """A plain assistant answer, optionally with reasoning, streamed in chunks."""

    text: str
    reasoning: str | None = None
    chunk_chars: int = 8
    delay_per_chunk: float = 0.0
    prompt_tokens: int = 100
    completion_tokens: int = 20
    cached_tokens: int = 0
    finish_reason: str = "stop"


@dataclass
class ToolCall:
    """One assistant turn issuing one or more tool calls.

    ``calls`` may be a single ``(name, args)`` or pass ``name``/``args`` directly;
    ``parallel`` adds more calls to the same assistant message. A ``str`` args is
    sent verbatim as the ``arguments`` string (e.g. malformed/truncated JSON).
    """

    name: str
    args: dict[str, Any] | str = field(default_factory=dict)
    parallel: list[tuple[str, dict[str, Any] | str]] = field(default_factory=list)
    text: str | None = None


@dataclass
class Error:
    """An HTTP error response (429/500/400 ...)."""

    status: int = 500
    message: str = "scripted failure"
    retry_after: float | None = None


@dataclass
class Hang:
    """Accept the request and never answer; the connection is dropped after ``seconds``."""

    seconds: float = 3600.0


@dataclass
class DropMidStream:
    """Stream ``text[:after_chars]`` then close the socket without a finish chunk."""

    text: str = "partial answer that never finishes"
    after_chars: int = 12


@dataclass
class StallMidStream:
    """Open the SSE stream, send ``text[:after_chars]``, then go silent for ``seconds``
    without closing (a wedged upstream that keeps the socket open)."""

    text: str = "partial answer that stalls"
    after_chars: int = 8
    seconds: float = 3600.0


@dataclass
class Raw:
    """Send an arbitrary body verbatim (malformed JSON, HTML error pages, ...)."""

    body: str = "this is not json"
    status: int = 200
    content_type: str = "application/json"


Response = Union[Text, ToolCall, Error, Hang, DropMidStream, StallMidStream, Raw]
Responder = Callable[[dict[str, Any]], Response]


# Server ---------------------------------------------------------------------


class FakeLLMServer:
    """Threaded loopback provider. Use as a context manager."""

    def __init__(
        self,
        script: list[Response] | Responder | None = None,
        *,
        default_text: str = "ok",
        aux: Responder | None = None,
        api_key: str | None = None,
        prompt_tokens_fn: Callable[[dict[str, Any]], int] | None = None,
    ) -> None:
        self._script: list[Response] = list(script) if isinstance(script, list) else []
        self._responder: Responder | None = script if callable(script) else None
        self.default_text = default_text
        self._aux = aux or (lambda _req: Text("Fake summary of the earlier conversation."))
        self.expected_api_key = api_key
        # Optional: derive reported ``usage.prompt_tokens`` from each request body (so token-driven
        # logic such as compaction triggers sees a realistic, growing count instead of a constant).
        self.prompt_tokens_fn = prompt_tokens_fn
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._tool_seq = 0

    # lifecycle
    def __enter__(self) -> "FakeLLMServer":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self))
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="fake-llm", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    @property
    def port(self) -> int:
        assert self._server is not None, "server not started"
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    # scripting
    def push(self, *responses: Response) -> None:
        with self._lock:
            self._script.extend(responses)

    def _next_main(self, record: dict[str, Any]) -> Response:
        if self._responder is not None:
            return self._responder(record)
        with self._lock:
            if self._script:
                return self._script.pop(0)
        return Text(self.default_text)

    # inspection
    def main_requests(self) -> list[dict[str, Any]]:
        return [r["body"] for r in self.requests if r["kind"] == "main"]

    def aux_requests(self) -> list[dict[str, Any]]:
        return [r["body"] for r in self.requests if r["kind"] == "aux"]

    def wait_for_requests(self, n: int, timeout: float = 30.0, kind: str = "main") -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if sum(1 for r in self.requests if r["kind"] == kind) >= n:
                return
            time.sleep(0.02)
        raise AssertionError(f"expected {n} {kind} requests, saw {len(self.requests)} total")

    def next_tool_call_id(self) -> str:
        with self._lock:
            self._tool_seq += 1
            return f"call_fake_{self._tool_seq}"


def _handler_for(server: FakeLLMServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a: object) -> None:
            pass

        def _send_json(self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/").endswith("/models"):
                self._send_json(200, {"object": "list", "data": [
                    {"id": MODEL_ID, "object": "model", "context_length": 128000},
                ]})
                return
            self._send_json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": {"message": "invalid json"}})
                return
            auth = self.headers.get("Authorization", "")
            kind = "main" if body.get("tools") else "aux"
            record = {
                "path": self.path,
                "kind": kind,
                "auth": auth,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": body,
                "t": time.time(),
            }
            with server._lock:
                server.requests.append(record)
            if server.expected_api_key is not None and auth != f"Bearer {server.expected_api_key}":
                self._send_json(401, {"error": {"message": "invalid api key", "type": "authentication_error"}})
                return
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send_json(404, {"error": {"message": f"unsupported path {self.path}"}})
                return
            resp = server._next_main(record) if kind == "main" else server._aux(record)
            record["response"] = type(resp).__name__
            prompt_tokens = server.prompt_tokens_fn(body) if server.prompt_tokens_fn else None
            self._respond(resp, bool(body.get("stream")), prompt_tokens, record)

        # response rendering
        def _respond(self, resp: Response, stream: bool, prompt_tokens: int | None = None,
                     record: dict[str, Any] | None = None) -> None:
            if isinstance(resp, Error):
                headers = {"Retry-After": str(resp.retry_after)} if resp.retry_after is not None else {}
                self._send_json(resp.status, {"error": {"message": resp.message, "type": "server_error"}}, headers)
                return
            if isinstance(resp, Hang):
                server._stop.wait(resp.seconds)
                # Drop the socket at the deadline: on a kept-alive HTTP/1.1 connection the client
                # would otherwise wait for a response that never comes, far past ``seconds``.
                self.close_connection = True
                return
            if isinstance(resp, Raw):
                body = resp.body.encode()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if isinstance(resp, StallMidStream):
                self._start_sse()
                self._sse(_chunk({"role": "assistant", "content": ""}))
                self._sse(_chunk({"content": resp.text[: resp.after_chars]}))
                server._stop.wait(resp.seconds)
                self.close_connection = True
                return
            if isinstance(resp, DropMidStream):
                self._start_sse()
                self._sse(_chunk({"role": "assistant", "content": ""}))
                self._sse(_chunk({"content": resp.text[: resp.after_chars]}))
                self.wfile.flush()
                self.close_connection = True
                return
            message, finish, usage = _message_for(resp, server, prompt_tokens)
            # What the provider billed for this request, so usage/cost accounting can be checked
            # against state.db (faulted requests never get a ``usage`` key).
            if record is not None:
                record["usage"] = usage
            if not stream:
                self._send_json(200, {
                    "id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": usage,
                })
                return
            self._start_sse()
            self._sse(_chunk({"role": "assistant", "content": ""}))
            if isinstance(resp, Text):
                if resp.reasoning:
                    for piece in _pieces(resp.reasoning, resp.chunk_chars):
                        self._sse(_chunk({"reasoning_content": piece}))
                for piece in _pieces(resp.text, resp.chunk_chars):
                    if resp.delay_per_chunk:
                        time.sleep(resp.delay_per_chunk)
                    self._sse(_chunk({"content": piece}))
            else:
                if message.get("content"):
                    self._sse(_chunk({"content": message["content"]}))
                for i, tc in enumerate(message["tool_calls"]):
                    self._sse(_chunk({"tool_calls": [{
                        "index": i, "id": tc["id"], "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": ""},
                    }]}))
                    self._sse(_chunk({"tool_calls": [{
                        "index": i, "function": {"arguments": tc["function"]["arguments"]},
                    }]}))
            last = _chunk({}, finish)
            last["usage"] = usage
            self._sse(last)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

        def _start_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

        def _sse(self, payload: dict[str, Any]) -> None:
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

    return Handler


def _pieces(text: str, size: int) -> list[str]:
    size = max(1, size)
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": int(time.time()),
        "model": MODEL_ID, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _message_for(
    resp: Text | ToolCall, server: FakeLLMServer, prompt_tokens: int | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if isinstance(resp, Text):
        message: dict[str, Any] = {"role": "assistant", "content": resp.text}
        if resp.reasoning:
            message["reasoning_content"] = resp.reasoning
        pt = resp.prompt_tokens if prompt_tokens is None else prompt_tokens
        usage = {
            "prompt_tokens": pt,
            "completion_tokens": resp.completion_tokens,
            "total_tokens": pt + resp.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": resp.cached_tokens},
        }
        return message, resp.finish_reason, usage
    calls = [(resp.name, resp.args), *resp.parallel]
    tool_calls = [
        {"id": server.next_tool_call_id(), "type": "function",
         "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}
        for name, args in calls
    ]
    message = {"role": "assistant", "content": resp.text, "tool_calls": tool_calls}
    pt = 100 if prompt_tokens is None else prompt_tokens
    usage = {"prompt_tokens": pt, "completion_tokens": 10, "total_tokens": pt + 10}
    return message, "tool_calls", usage


# HERMES_HOME wiring ---------------------------------------------------------


def write_hermes_home(
    home: Path,
    base_url: str,
    *,
    api_key: str = "sk-fake-e2e",
    extra_config: str = "",
) -> Path:
    """Write a minimal config.yaml + .env routing the main model to ``base_url``.

    Auxiliary tasks use the same endpoint (``auto`` resolves to the main
    provider), retries are capped so fault tests finish quickly, and no real
    provider credential is ever present.
    """
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: custom\n"
        f"  base_url: {base_url}\n"
        f"  default: {MODEL_ID}\n"
        "  context_length: 128000\n"
        "agent:\n"
        "  api_max_retries: 1\n"
        + extra_config,
        encoding="utf-8",
    )
    (home / ".env").write_text(f"OPENAI_API_KEY={api_key}\n", encoding="utf-8")
    return home


if __name__ == "__main__":  # pragma: no cover - manual probe entry point
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    srv = FakeLLMServer()
    srv._server = ThreadingHTTPServer(("127.0.0.1", port), _handler_for(srv))
    print(f"fake provider on {srv.base_url}", flush=True)
    srv._server.serve_forever()
