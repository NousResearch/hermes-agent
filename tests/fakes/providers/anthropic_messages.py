"""Scripted, recording loopback server speaking the Anthropic Messages API.

The installed ``anthropic`` SDK is the ORACLE for both directions, so the fake
cannot invent wire shapes:

* every request body is validated against the SDK's request TypedDicts
  (``MessageCreateParams``; the beta variant when an ``anthropic-beta`` header is
  sent) with a pydantic ``TypeAdapter``; validation errors AND keys the schema
  does not know are recorded on the request (``record["schema_errors"]``);
* every response, SSE event and error envelope is built from ``anthropic.types``
  models and serialised with ``model_dump`` (the one exception is ``ping``, a
  documented stream event the SDK exposes no model for).

Scripted responses (``Reply`` of ``Thinking``/``Text``/``ToolUse`` blocks,
``ApiError``, ``DropStream``) are consumed by main-turn requests (those carrying
``tools``); tool-less requests (title generation, compression summaries) are
auxiliary and answered by ``aux`` so they never eat a scripted turn.

Record mode (documented, never run in CI or against a paid key by this suite):
``AnthropicMessagesServer(record_upstream="https://api.anthropic.com",
cassette="<name>")`` proxies each POST to the upstream base URL and appends a
sanitised exchange (no auth headers, no API keys) to
``tests/fakes/providers/cassettes/anthropic/<name>.jsonl`` (``cassette_dir``
overrides the directory).

A ``Responder`` runs on the request's handler thread, so it may block (hold a
reply until an event fires) and decide at send time; each request record
carries ``bearer`` / ``x_api_key`` so a responder can branch on the credential.
"""

from __future__ import annotations

import json
import threading
import time
import typing
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Union

import anthropic.types as at
import typing_extensions
from anthropic.types import message_create_params as ga_params
from anthropic.types.beta import BetaMessageParam
from anthropic.types.beta import message_create_params as beta_params
from anthropic.types.raw_message_delta_event import Delta as MessageDelta
from pydantic import TypeAdapter, ValidationError

MODEL_ID = "claude-sonnet-4-5-e2e"
CASSETTE_DIR = Path(__file__).with_name("cassettes") / "anthropic"
_GA_ADAPTER = TypeAdapter(ga_params.MessageCreateParams)
_BETA_ADAPTER = TypeAdapter(beta_params.MessageCreateParams)
_SANITISED_HEADERS = frozenset({"authorization", "x-api-key", "cookie", "set-cookie"})


# Scripted responses ---------------------------------------------------------


@dataclass
class Thinking:
    thinking: str
    signature: str


@dataclass
class Text:
    text: str


@dataclass
class ToolUse:
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


Block = Union[Thinking, Text, ToolUse]


@dataclass
class Reply:
    """One assistant message. ``stop_reason`` defaults to tool_use/end_turn by content."""

    blocks: list[Block]
    stop_reason: str | None = None
    input_tokens: int = 100
    output_tokens: int = 20
    chunk_chars: int = 7


@dataclass
class ApiError:
    """A documented error envelope (400 invalid_request_error, 429 rate_limit_error, 529 overloaded_error ...)."""

    status: int
    type: str
    message: str = "scripted failure"
    retry_after: float | None = None


@dataclass
class DropStream:
    """Stream ``reply`` and close the socket after ``after_deltas`` content deltas (no message_stop)."""

    reply: Reply
    after_deltas: int = 2


Response = Union[Reply, ApiError, DropStream]
Responder = Callable[[dict[str, Any]], Response]

_ERROR_MODELS: dict[str, type] = {
    "invalid_request_error": at.InvalidRequestError,
    "authentication_error": at.AuthenticationError,
    "permission_error": at.PermissionError,
    "not_found_error": at.NotFoundError,
    "rate_limit_error": at.RateLimitError,
    "api_error": at.APIErrorObject,
    "overloaded_error": at.OverloadedError,
}


# SDK-oracle request validation ------------------------------------------------


def _materialise(value: Any) -> Any:
    """Drain pydantic's lazy ``Iterable`` validators so nested blocks are validated too."""
    if isinstance(value, dict):
        return {k: _materialise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_materialise(v) for v in value]
    if type(value).__name__ == "ValidatorIterator":
        return [_materialise(v) for v in value]
    return value


def _param_adapters(message_param: Any) -> dict[str, TypeAdapter]:
    """``type`` literal -> TypeAdapter of each request-side ``*Param`` TypedDict in a message's content union."""
    content = typing.get_type_hints(message_param)["content"]
    iterable = next(arg for arg in typing.get_args(content) if arg is not str)
    out: dict[str, TypeAdapter] = {}
    for member in typing.get_args(typing.get_args(iterable)[0]):
        if typing_extensions.is_typeddict(member):  # the SDK builds its Params with typing_extensions
            for literal in typing.get_args(typing.get_type_hints(member)["type"]):
                out[literal] = TypeAdapter(member)
    return out


# A message's content union also admits the RESPONSE BaseModels (ThinkingBlock, ToolUseBlock ...),
# which accept extra keys, so a replayed assistant block would validate with any bogus key. Content
# blocks are re-validated against the request-side *Param TypedDicts only: the shapes the real API
# holds a body to (it 400s "Extra inputs are not permitted").
_GA_BLOCKS = _param_adapters(at.MessageParam)
_BETA_BLOCKS = _param_adapters(BetaMessageParam)


def _unknown_keys(sent: Any, known: Any, path: str = "$") -> list[str]:
    # A block the union resolved to a (permissive) response BaseModel stops this walk; content
    # blocks are checked key-for-key against their *Param TypedDict by _block_problems instead.
    if isinstance(sent, dict) and isinstance(known, dict):
        out = [f"{path}.{k}" for k in sent if k not in known]
        for k in sent:
            if k in known:
                out += _unknown_keys(sent[k], known[k], f"{path}.{k}")
        return out
    if isinstance(sent, list) and isinstance(known, list):
        return [p for i, (s, k) in enumerate(zip(sent, known)) for p in _unknown_keys(s, k, f"{path}[{i}]")]
    return []


def validate_request(body: dict[str, Any], *, beta: bool) -> list[str]:
    """Schema problems of one request body per the SDK's request TypedDicts ([] = conformant)."""
    adapter = _BETA_ADAPTER if beta else _GA_ADAPTER
    try:
        known = _materialise(adapter.validate_python(body))
        problems = [f"unknown key {p}" for p in _unknown_keys(body, known)]
    except ValidationError as exc:
        problems = [_loc(e, "$") for e in exc.errors()]
    for path, block in _content_blocks(body):
        problems += [p for p in _block_problems(block, _BETA_BLOCKS if beta else _GA_BLOCKS, path)
                     if p not in problems]
    return problems


def _loc(err: Any, prefix: str) -> str:
    return f"{prefix}.{'.'.join(map(str, err['loc']))}: {err['msg']}"


def _content_blocks(body: dict[str, Any]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for i, msg in enumerate(body.get("messages") or []):
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            out += [(f"$.messages[{i}].content[{j}]", b) for j, b in enumerate(content)]
    if isinstance(body.get("system"), list):
        out += [(f"$.system[{j}]", b) for j, b in enumerate(body["system"])]
    return out


def _block_problems(block: Any, adapters: dict[str, TypeAdapter], path: str) -> list[str]:
    if not isinstance(block, dict):
        return [f"{path}: content block is {type(block).__name__}, not an object"]
    adapter = adapters.get(block.get("type"))
    if adapter is None:
        return [f"{path}.type: {block.get('type')!r} is not a request content block type"]
    try:
        known = _materialise(adapter.validate_python(block))
    except ValidationError as exc:
        return [_loc(e, path) for e in exc.errors()]
    return [f"unknown key {p}" for p in _unknown_keys(block, known, path)]


# SDK-built responses --------------------------------------------------------------


def _sdk_block(block: Block, tool_id: str) -> Any:
    if isinstance(block, Thinking):
        return at.ThinkingBlock(type="thinking", thinking=block.thinking, signature=block.signature)
    if isinstance(block, Text):
        return at.TextBlock(type="text", text=block.text)
    return at.ToolUseBlock(type="tool_use", id=block.id or tool_id, name=block.name, input=block.input)


def _stop_reason(reply: Reply) -> str:
    if reply.stop_reason:
        return reply.stop_reason
    return "tool_use" if any(isinstance(b, ToolUse) for b in reply.blocks) else "end_turn"


def build_message(reply: Reply, tool_ids: list[str], content: bool = True) -> at.Message:
    return at.Message(
        id=f"msg_{uuid.uuid4().hex[:20]}", type="message", role="assistant", model=MODEL_ID,
        content=[_sdk_block(b, tid) for b, tid in zip(reply.blocks, tool_ids)] if content else [],
        stop_reason=_stop_reason(reply) if content else None, stop_sequence=None,
        usage=at.Usage(input_tokens=reply.input_tokens, output_tokens=reply.output_tokens if content else 1),
    )


def _pieces(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), max(1, size))] or [""]


def _block_events(index: int, block: Block, tool_id: str, chunk: int) -> list[Any]:
    """content_block_start / *_delta... / content_block_stop for one block (SDK event models)."""
    if isinstance(block, Thinking):
        start = at.ThinkingBlock(type="thinking", thinking="", signature="")
        deltas: list[Any] = [at.ThinkingDelta(type="thinking_delta", thinking=p) for p in _pieces(block.thinking, chunk)]
        deltas.append(at.SignatureDelta(type="signature_delta", signature=block.signature))
    elif isinstance(block, Text):
        start = at.TextBlock(type="text", text="")
        deltas = [at.TextDelta(type="text_delta", text=p) for p in _pieces(block.text, chunk)]
    else:
        start = at.ToolUseBlock(type="tool_use", id=block.id or tool_id, name=block.name, input={})
        deltas = [at.InputJSONDelta(type="input_json_delta", partial_json=p)
                  for p in _pieces(json.dumps(block.input), chunk)]
    return [
        at.RawContentBlockStartEvent(type="content_block_start", index=index, content_block=start),
        *[at.RawContentBlockDeltaEvent(type="content_block_delta", index=index, delta=d) for d in deltas],
        at.RawContentBlockStopEvent(type="content_block_stop", index=index),
    ]


def stream_events(reply: Reply, tool_ids: list[str]) -> list[Any]:
    """The full documented event sequence for ``reply`` (``"ping"`` marks the ping event)."""
    events: list[Any] = [at.RawMessageStartEvent(type="message_start", message=build_message(reply, tool_ids, False)),
                         "ping"]
    for i, (block, tid) in enumerate(zip(reply.blocks, tool_ids)):
        events += _block_events(i, block, tid, reply.chunk_chars)
    events.append(at.RawMessageDeltaEvent(
        type="message_delta", delta=MessageDelta(stop_reason=_stop_reason(reply), stop_sequence=None),
        usage=at.MessageDeltaUsage(output_tokens=reply.output_tokens)))
    events.append(at.RawMessageStopEvent(type="message_stop"))
    return events


def _event_bytes(event: Any) -> bytes:
    payload = {"type": "ping"} if event == "ping" else event.model_dump(mode="json")
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n".encode()


def error_envelope(err: ApiError) -> dict[str, Any]:
    model = _ERROR_MODELS[err.type]
    return at.ErrorResponse(type="error", error=model(type=err.type, message=err.message),
                            request_id=f"req_{uuid.uuid4().hex[:16]}").model_dump(mode="json")


# Server ---------------------------------------------------------------------------


class AnthropicMessagesServer:
    """Threaded loopback Messages endpoint. ``base_url`` ends in ``/anthropic`` so Hermes'
    native ``anthropic`` provider accepts it as an Anthropic-protocol override."""

    def __init__(self, script: list[Response] | Responder | None = None, *, default_text: str = "ok",
                 aux: Responder | None = None, models: list[str] | None = None,
                 record_upstream: str | None = None, cassette: str = "session",
                 cassette_dir: Path = CASSETTE_DIR) -> None:
        self._script: list[Response] = list(script) if isinstance(script, list) else []
        self._responder: Responder | None = script if callable(script) else None
        self.default_text = default_text
        self._aux = aux or (lambda _r: Reply([Text("Fake summary of the earlier conversation.")]))
        self.models = models or [MODEL_ID]
        self.record_upstream = record_upstream
        self.cassette = Path(cassette_dir) / f"{cassette}.jsonl"
        self.requests: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._httpd: ThreadingHTTPServer | None = None
        self._tool_seq = 0

    def __enter__(self) -> "AnthropicMessagesServer":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> "AnthropicMessagesServer":
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self))
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05},
                         name="fake-anthropic", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    @property
    def base_url(self) -> str:
        assert self._httpd is not None, "server not started"
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/anthropic"

    def handler_class(self) -> type[BaseHTTPRequestHandler]:
        """Request handler, for serving TLS-intercepted connections (``TLSInterceptProxy``)."""
        return _handler_for(self)

    def push(self, *responses: Response) -> None:
        with self._lock:
            self._script.extend(responses)

    def main_requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return [r for r in self.requests if r["kind"] == "main"]

    def schema_errors(self) -> list[tuple[int, list[str]]]:
        with self._lock:
            return [(i, r["schema_errors"]) for i, r in enumerate(self.requests) if r["schema_errors"]]

    def _next(self, record: dict[str, Any]) -> Response:
        if record["kind"] == "aux":
            return self._aux(record)
        if self._responder is not None:
            return self._responder(record)
        with self._lock:
            if self._script:
                return self._script.pop(0)
        return Reply([Text(self.default_text)])

    def _tool_ids(self, reply: Reply) -> list[str]:
        with self._lock:
            ids = []
            for _ in reply.blocks:
                self._tool_seq += 1
                ids.append(f"toolu_e2e_{self._tool_seq:04d}")
            return ids

    def _record_exchange(self, record: dict[str, Any], status: int, body: bytes) -> None:
        self.cassette.parent.mkdir(parents=True, exist_ok=True)
        headers = {k: v for k, v in record["headers"].items() if k not in _SANITISED_HEADERS}
        with self.cassette.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"request": {"path": record["path"], "headers": headers, "body": record["body"]},
                                 "response": {"status": status, "body": body.decode("utf-8", "replace")}}) + "\n")


def _handler_for(server: AnthropicMessagesServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a: object) -> None:
            pass

        def _json(self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("request-id", f"req_{uuid.uuid4().hex[:16]}")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - model discovery (/v1/models)
            with server._lock:
                server.gets.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}})
            if self.path.split("?", 1)[0].rstrip("/").endswith("/v1/models"):
                page = at.ModelInfo  # the SDK's /v1/models item model
                data = [page(id=m, type="model", display_name=m, created_at="2026-01-01T00:00:00Z")
                        .model_dump(mode="json") for m in server.models]
                self._json(200, {"data": data, "has_more": False,
                                 "first_id": server.models[0], "last_id": server.models[-1]})
                return
            self._json(404, error_envelope(ApiError(404, "not_found_error", f"no route {self.path}")))

        def do_POST(self) -> None:  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if not self.path.split("?", 1)[0].endswith("/v1/messages"):
                self._json(404, error_envelope(ApiError(404, "not_found_error", f"no route {self.path}")))
                return
            body = json.loads(raw or b"{}")
            headers = {k.lower(): v for k, v in self.headers.items()}
            record = {
                "path": self.path, "headers": headers, "body": body, "t": time.monotonic(),
                "bearer": headers.get("authorization", "").removeprefix("Bearer ").strip(),
                "x_api_key": headers.get("x-api-key", ""),
                "kind": "main" if body.get("tools") else "aux",
                "schema_errors": validate_request(body, beta=bool(headers.get("anthropic-beta"))),
            }
            with server._lock:
                server.requests.append(record)
            if server.record_upstream:
                self._proxy(record, raw)
                return
            resp = server._next(record)
            record["response"] = type(resp).__name__
            self._respond(resp, bool(body.get("stream")))

        def _proxy(self, record: dict[str, Any], raw: bytes) -> None:
            fwd = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
            req = urllib.request.Request(str(server.record_upstream).rstrip("/") + "/v1/messages", data=raw,
                                         headers=fwd, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=300) as up:  # noqa: S310 - opt-in record mode
                    status, data, ctype = up.status, up.read(), up.headers.get("Content-Type", "")
            except urllib.error.HTTPError as exc:
                status, data, ctype = exc.code, exc.read(), exc.headers.get("Content-Type", "")
            server._record_exchange(record, status, data)
            self.send_response(status)
            self.send_header("Content-Type", ctype or "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _respond(self, resp: Response, stream: bool) -> None:
            if isinstance(resp, ApiError):
                headers = {"retry-after": str(resp.retry_after)} if resp.retry_after is not None else {}
                self._json(resp.status, error_envelope(resp), headers)
                return
            reply = resp.reply if isinstance(resp, DropStream) else resp
            tool_ids = server._tool_ids(reply)
            if not stream and not isinstance(resp, DropStream):
                self._json(200, build_message(reply, tool_ids).model_dump(mode="json"))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            deltas = 0
            for event in stream_events(reply, tool_ids):
                if isinstance(resp, DropStream) and deltas >= resp.after_deltas:
                    break  # socket closes with no content_block_stop / message_stop
                self.wfile.write(_event_bytes(event))
                self.wfile.flush()
                deltas += isinstance(event, at.RawContentBlockDeltaEvent)
            self.close_connection = True

    return Handler
