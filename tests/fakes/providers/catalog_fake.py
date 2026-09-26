"""Multi-dialect recording loopback provider for the provider-catalog E2E matrix.

One real HTTP server on 127.0.0.1 that answers every wire dialect the bundled
model-provider plugins speak, at EXACT paths only (``routes``: path -> dialect, built by the
caller from the base URL it configured; anything else is a 404, so a request that lands under
the right prefix but at the wrong path is visible, never silently answered):

* ``chat``      — OpenAI Chat Completions (JSON or SSE)
* ``anthropic`` — Anthropic Messages (JSON or SSE)
* ``responses`` — OpenAI Responses (SSE event stream)
* ``listing``   — ``GET`` model listing (OpenAI ``data`` shape), or a scripted
  status / hang so picker fallbacks can be exercised

Without ``routes`` the server answers nothing (egress sentinel only).

Every request (method, path, headers, body) is recorded, so a test can assert on
exactly which credential reached which host in which header.

The model script is stateless and dialect-neutral: a main-turn request (one that
offers tools) whose history carries no tool result gets ONE tool call
(``tool_name``/``tool_args``); once a tool result is present the model answers
``final_text``. Requests without tools are auxiliary (titles, summaries) and get
a short plain answer. ``fail_status`` turns every inference request into that
HTTP error (a dead provider for fallback tests).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

USAGE_IN = 1234
USAGE_OUT = 56


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: Any
    # Dialect of the exact route the request hit, or "unknown" (answered 404).
    dialect: str = "unknown"
    t: float = field(default_factory=time.time)


def bare_path(path: str) -> str:
    return path.split("?", 1)[0]


class CatalogFake:
    """Threaded loopback server. Use as a context manager."""

    def __init__(
        self,
        *,
        tool_name: str = "read_file",
        tool_args: dict[str, Any] | None = None,
        final_text: str = "CATALOG-TURN-COMPLETE",
        models: list[str] | None = None,
        models_status: int = 200,
        models_hang_s: float = 0.0,
        fail_status: int | None = None,
        routes: dict[str, str] | None = None,
    ) -> None:
        self.routes = dict(routes or {})
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.final_text = final_text
        self.models = list(models or ["catalog-model-a", "catalog-model-b"])
        self.models_status = models_status
        self.models_hang_s = models_hang_s
        self.fail_status = fail_status
        self.requests: list[Recorded] = []
        # Egress sentinel: when a child process gets ``HTTPS_PROXY``/``HTTP_PROXY`` = ``origin``
        # (see ``proxy_env``), every request aimed at a NON-loopback host lands here instead of
        # the real vendor and is refused with 403 — ``egress`` names each host it tried to reach.
        self.egress: list[Recorded] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._server: ThreadingHTTPServer | None = None

    def __enter__(self) -> "CatalogFake":
        server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self))
        server.daemon_threads = True
        self._server = server
        threading.Thread(target=server.serve_forever, name="catalog-fake", daemon=True).start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    @property
    def origin(self) -> str:
        assert self._server is not None, "server not started"
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def proxy_env(self) -> dict[str, str]:
        """Env that routes every non-loopback request of a child through the egress sentinel."""
        return {"HTTPS_PROXY": self.origin, "HTTP_PROXY": self.origin, "https_proxy": self.origin,
                "http_proxy": self.origin, "ALL_PROXY": "", "all_proxy": "",
                "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}

    def egress_hosts(self) -> list[str]:
        return sorted({r.path for r in self.egress})

    def inference(self) -> list[Recorded]:
        return [r for r in self.requests if r.method == "POST"]

    def listings(self) -> list[Recorded]:
        return [r for r in self.requests if r.method == "GET"]

    def _record(self, rec: Recorded) -> None:
        with self._lock:
            self.requests.append(rec)


# --- dialect-neutral script ---------------------------------------------------


def _has_tool_result(dialect: str, body: dict[str, Any]) -> bool:
    if dialect == "chat":
        return any(m.get("role") == "tool" for m in body.get("messages") or [])
    if dialect == "anthropic":
        return any(
            isinstance(m.get("content"), list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
            for m in body.get("messages") or [])
    items = body.get("input") if isinstance(body.get("input"), list) else []
    return any(isinstance(i, dict) and i.get("type") == "function_call_output" for i in items)


def _plan(fake: CatalogFake, dialect: str, body: dict[str, Any]) -> tuple[str, str | None]:
    """(``text``, ``tool_call_id`` or None) for the next model turn."""
    if not body.get("tools"):
        return "Catalog aux answer", None
    if _has_tool_result(dialect, body):
        return fake.final_text, None
    return "", f"call_{uuid.uuid4().hex[:10]}"


# --- wire renderers -------------------------------------------------------------


def _chat_payloads(fake: CatalogFake, text: str, call_id: str | None, stream: bool) -> list[dict[str, Any]] | dict:
    usage = {"prompt_tokens": USAGE_IN, "completion_tokens": USAGE_OUT, "total_tokens": USAGE_IN + USAGE_OUT}
    base = {"id": "chatcmpl-catalog", "created": int(time.time()), "model": "catalog-model-a"}
    args = json.dumps(fake.tool_args)
    tool_calls = [{"id": call_id, "type": "function", "function": {"name": fake.tool_name, "arguments": args}}]
    finish = "tool_calls" if call_id else "stop"
    if not stream:
        msg: dict[str, Any] = {"role": "assistant", "content": text or None}
        if call_id:
            msg["tool_calls"] = tool_calls
        return {**base, "object": "chat.completion", "usage": usage,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}]}

    def chunk(delta: dict[str, Any], fin: str | None = None) -> dict[str, Any]:
        return {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}

    out = [chunk({"role": "assistant", "content": ""})]
    if call_id:
        out.append(chunk({"tool_calls": [{"index": 0, "id": call_id, "type": "function",
                                          "function": {"name": fake.tool_name, "arguments": ""}}]}))
        out.append(chunk({"tool_calls": [{"index": 0, "function": {"arguments": args}}]}))
    else:
        out.append(chunk({"content": text}))
    last = chunk({}, finish)
    last["usage"] = usage
    out.append(last)
    return out


def _anthropic_events(fake: CatalogFake, text: str, call_id: str | None) -> list[tuple[str, dict[str, Any]]]:
    msg = {"id": "msg_catalog", "type": "message", "role": "assistant", "model": "catalog-model-a",
           "content": [], "stop_reason": None, "stop_sequence": None,
           "usage": {"input_tokens": USAGE_IN, "output_tokens": 1}}
    ev: list[tuple[str, dict[str, Any]]] = [("message_start", {"type": "message_start", "message": msg})]
    if call_id:
        ev += [
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {
                "type": "tool_use", "id": call_id.replace("call_", "toolu_"), "name": fake.tool_name, "input": {}}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(fake.tool_args)}}),
        ]
    else:
        ev += [
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": text}}),
        ]
    ev += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {
            "stop_reason": "tool_use" if call_id else "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": USAGE_OUT}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return ev


def _anthropic_json(fake: CatalogFake, text: str, call_id: str | None) -> dict[str, Any]:
    content: list[dict[str, Any]] = (
        [{"type": "tool_use", "id": call_id.replace("call_", "toolu_"), "name": fake.tool_name, "input": fake.tool_args}]
        if call_id else [{"type": "text", "text": text}])
    return {"id": "msg_catalog", "type": "message", "role": "assistant", "model": "catalog-model-a",
            "content": content, "stop_reason": "tool_use" if call_id else "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": USAGE_IN, "output_tokens": USAGE_OUT}}


def _responses_events(fake: CatalogFake, text: str, call_id: str | None) -> list[dict[str, Any]]:
    rid = f"resp_{uuid.uuid4().hex[:10]}"
    usage = {"input_tokens": USAGE_IN, "output_tokens": USAGE_OUT, "total_tokens": USAGE_IN + USAGE_OUT,
             "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}
    shell = {"id": rid, "object": "response", "created_at": int(time.time()), "model": "catalog-model-a",
             "status": "in_progress", "output": []}
    if call_id:
        item = {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "name": fake.tool_name,
                "arguments": json.dumps(fake.tool_args), "status": "completed"}
        middle = [
            {"type": "response.output_item.added", "output_index": 0, "item": {**item, "arguments": "", "status": "in_progress"}},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": item["id"],
             "delta": item["arguments"]},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
        ]
    else:
        item = {"type": "message", "id": f"msg_{rid}", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}
        middle = [
            {"type": "response.output_item.added", "output_index": 0,
             "item": {**item, "content": [], "status": "in_progress"}},
            {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "item_id": item["id"],
             "delta": text},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
        ]
    done = {**shell, "status": "completed", "output": [item], "usage": usage}
    events = [{"type": "response.created", "response": shell}, *middle,
              {"type": "response.completed", "response": done}]
    for seq, e in enumerate(events):
        e["sequence_number"] = seq
    return events


def _handler_for(fake: CatalogFake) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a: object) -> None:
            pass

        def _headers(self) -> dict[str, str]:
            return {k.lower(): v for k, v in self.headers.items()}

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _sse_open(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def _sse(self, data: Any, event: str | None = None) -> None:
            prefix = f"event: {event}\n" if event else ""
            self.wfile.write(f"{prefix}data: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        def _refuse_egress(self, method: str) -> bool:
            """Proxy-form requests (CONNECT host:port, or an absolute-URI GET/POST) are egress."""
            if method != "CONNECT" and not self.path.startswith(("http://", "https://")):
                return False
            target = self.path.split("://", 1)[-1].split("/", 1)[0]
            with fake._lock:
                fake.egress.append(Recorded(method, target, self._headers(), None))
            self._json(403, {"error": {"message": f"catalog egress sentinel refused {target}"}})
            self.close_connection = True
            return True

        def do_CONNECT(self) -> None:  # noqa: N802
            self._refuse_egress("CONNECT")

        def do_GET(self) -> None:  # noqa: N802
            if self._refuse_egress("GET"):
                return
            dialect = fake.routes.get(bare_path(self.path), "unknown")
            dialect = dialect if dialect == "listing" else "unknown"
            fake._record(Recorded("GET", self.path, self._headers(), None, dialect))
            if dialect != "listing":
                self._json(404, {"error": {"message": f"not found: {self.path}"}})
                return
            if fake.models_hang_s:
                fake._stop.wait(fake.models_hang_s)
                self.close_connection = True
                return
            if fake.models_status != 200:
                self._json(fake.models_status, {"error": {"message": "no listing here"}})
                return
            self._json(200, {"object": "list", "data": [
                {"id": m, "object": "model", "type": "model", "display_name": m, "created": 1,
                 "created_at": "2026-01-01T00:00:00Z", "owned_by": "catalog", "context_length": 131072}
                for m in fake.models]})

        def do_POST(self) -> None:  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            if self._refuse_egress("POST"):
                return
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                body = {"_raw": raw.decode("utf-8", "replace")}
            dialect = fake.routes.get(bare_path(self.path), "unknown")
            dialect = dialect if dialect in ("chat", "anthropic", "responses") else "unknown"
            fake._record(Recorded("POST", self.path, self._headers(), body, dialect))
            if dialect == "unknown":
                self._json(404, {"error": {"message": f"unsupported path {self.path}"}})
                return
            if fake.fail_status is not None:
                self._json(fake.fail_status, {"error": {
                    "message": "catalog fake: provider down", "type": "server_error"}})
                return
            text, call_id = _plan(fake, dialect, body)
            stream = bool(body.get("stream"))
            if dialect == "chat":
                payload = _chat_payloads(fake, text, call_id, stream)
                if not stream:
                    self._json(200, payload)
                    return
                self._sse_open()
                for c in payload:
                    self._sse(c)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            if dialect == "anthropic":
                if not stream:
                    self._json(200, _anthropic_json(fake, text, call_id))
                    return
                self._sse_open()
                for name, data in _anthropic_events(fake, text, call_id):
                    self._sse(data, event=name)
                return
            events = _responses_events(fake, text, call_id)
            if not stream:
                self._json(200, events[-1]["response"])
                return
            self._sse_open()
            for e in events:
                self._sse(e, event=e["type"])

    return Handler
