"""Scripted, recording loopback server for the OpenAI **Responses** wire dialect.

The external boundary Hermes' ``codex_responses`` transport talks to (ChatGPT Codex,
api.openai.com ``/v1/responses``, Responses-speaking relays). One real HTTP server on
127.0.0.1 that:

* answers ``POST .../responses`` with a streamed SSE event sequence
  (``response.created`` → ``response.output_item.added``/``.done`` per item, text and
  function-call argument deltas → ``response.completed`` with ``usage``) or a scripted
  fault: an HTTP error with ``Retry-After``, the HTTP-200 soft failure
  (``response.failed`` carrying ``invalid_encrypted_content``), or a socket drop after
  N events;
* records every request body and validates it against the installed ``openai`` SDK's
  own request schema (``ResponseCreateParamsStreaming`` through a pydantic
  ``TypeAdapter``) — the SDK is the oracle for what a well-formed request is, so a test
  can assert ``srv.invalid_requests() == []`` instead of hand-pinning field names;
* builds every emitted event from ``openai.types.responses`` models, so the stream is
  exactly what the SDK itself would parse from the real vendor.

Main-turn requests (those carrying ``tools``) consume the script in order; tool-less
requests (title generation and other auxiliary calls, on ``/responses`` or
``/chat/completions``) get a canned answer and never eat a scripted turn.

Record mode (manual, never run by the suite — no paid call is ever made by tests)::

    python -m tests.fakes.providers.openai_responses record \\
        --upstream https://api.openai.com/v1 --out tests/fakes/providers/cassettes/openai/<name>.json

starts a pass-through proxy on 127.0.0.1; point a Hermes home's ``base_url`` at it,
run one turn, and the proxy writes a *sanitized* cassette (auth headers dropped,
``encrypted_content`` and ids replaced with stable placeholders, bodies kept). A
cassette's recorded event list can be replayed with ``Cassette(path)`` in a script.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Union

from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseError,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import TypeAdapter

MODEL_ID = "fake-responses-model"

# Script items ------------------------------------------------------------------


@dataclass
class Reasoning:
    """A reasoning output item carrying an opaque ``encrypted_content`` blob."""

    encrypted: str
    summary: str = ""


@dataclass
class Message:
    """An assistant ``message`` output item (streamed as ``output_text`` deltas)."""

    text: str


@dataclass
class FunctionCall:
    """A ``function_call`` output item. ``args`` as ``str`` is sent verbatim."""

    name: str
    args: dict[str, Any] | str = field(default_factory=dict)


Item = Union[Reasoning, Message, FunctionCall]


@dataclass
class Turn:
    """One successful streamed response made of ``items`` (in output order)."""

    items: list[Item]
    input_tokens: int = 100
    output_tokens: int = 20
    cached_tokens: int = 0
    drop_after_events: int | None = None  # close the socket after N events (no terminal event)


@dataclass
class SoftFail:
    """HTTP 200 whose stream ends in ``response.failed`` (the error lives in the body)."""

    code: str = "invalid_encrypted_content"
    message: str = "Encrypted content could not be decrypted or parsed."


@dataclass
class HttpError:
    """A non-2xx JSON error (``{"error": {...}}``), optionally with ``Retry-After``."""

    status: int = 500
    message: str = "scripted failure"
    code: str | None = None
    type: str = "server_error"
    retry_after: float | None = None


@dataclass
class Cassette:
    """Replay the event list of a sanitized cassette written by record mode."""

    path: Path


Step = Union[Turn, SoftFail, HttpError, Cassette]
Responder = Callable[[dict[str, Any]], Step]

_REQUEST_ADAPTER = TypeAdapter(ResponseCreateParamsStreaming)


# Replayed output items the vendor accepts WITHOUT ``id`` under ``store: false``. The SDK
# TypedDicts mark ``id`` required, but with nothing persisted server-side a replayed id has
# nothing to resolve against (a foreign/unpersisted id 404s), so Hermes replays these items
# by content alone (``encrypted_content`` for reasoning, the text for assistant messages).
# This is the ONLY relaxation of the oracle; every other field is still validated.
_ID_OPTIONAL_REPLAY_TYPES = frozenset({"reasoning", "message"})


def _apply_vendor_deviations(body: dict[str, Any]) -> dict[str, Any]:
    items = body.get("input")
    if not isinstance(items, list) or body.get("store") is not False:
        return body
    return {**body, "input": [
        {**i, "id": "replayed_without_id"}
        if isinstance(i, dict) and i.get("type") in _ID_OPTIONAL_REPLAY_TYPES and "id" not in i
        and i.get("role", "assistant") == "assistant" else i
        for i in items]}


def _item_members() -> dict[str, list[Any]]:
    """``type`` literal -> the SDK input-item TypedDicts declaring it (for readable errors)."""
    from typing import get_args, get_type_hints

    from openai.types.responses.response_input_item_param import ResponseInputItemParam

    out: dict[str, list[Any]] = {}
    for member in get_args(ResponseInputItemParam):
        for lit in get_args(get_type_hints(member).get("type")):
            out.setdefault(lit, []).append(member)
    return out


def _explain_items(items: list[Any]) -> str:
    """Per-item errors against the members sharing the item's ``type`` (the union error
    pydantic reports for the whole request is thousands of lines of non-matching members)."""
    members, lines = _item_members(), []
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or "type" not in item:
            continue
        errors = []
        for member in members.get(item["type"], []):
            ta = TypeAdapter(member)
            try:
                ta.dump_json(ta.validate_python(item), warnings=False)
                errors = []
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{member.__name__}: {exc}")
        if errors:
            lines.append(f"input[{idx}] type={item['type']}: " + " | ".join(errors))
    return "\n".join(lines)


def validate_request(body: dict[str, Any]) -> str | None:
    """SDK-oracle validation of one ``/responses`` request body; ``None`` when well-formed.

    ``dump_json`` forces pydantic's lazy ``Iterable[...]`` validators (``input``,
    ``tools``) to walk every element, so a malformed nested item is reported too."""
    body = _apply_vendor_deviations(body)
    try:
        _REQUEST_ADAPTER.dump_json(_REQUEST_ADAPTER.validate_python(body), warnings=False)
    except Exception as exc:  # noqa: BLE001 - reported to the test verbatim
        detail = _explain_items(body["input"]) if isinstance(body.get("input"), list) else ""
        return (detail or str(exc))[:4000]
    return None


# Event rendering ---------------------------------------------------------------


class _Seq:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self) -> int:
        self.n += 1
        return self.n


def _response(resp_id: str, status: str, output: list[Any], usage: ResponseUsage | None = None) -> Response:
    return Response(
        id=resp_id, created_at=time.time(), model=MODEL_ID, object="response", output=output,
        parallel_tool_calls=True, tool_choice="auto", tools=[], status=status, usage=usage,
    )


def _built_item(item: Item, idx: int) -> Any:
    if isinstance(item, Reasoning):
        summary = [Summary(text=item.summary, type="summary_text")] if item.summary else []
        return ResponseReasoningItem(id=f"rs_{idx}_{uuid.uuid4().hex[:8]}", summary=summary, type="reasoning",
                                     encrypted_content=item.encrypted)
    if isinstance(item, Message):
        return ResponseOutputMessage(
            id=f"msg_{idx}_{uuid.uuid4().hex[:8]}", role="assistant", status="completed", type="message",
            content=[ResponseOutputText(annotations=[], text=item.text, type="output_text")])
    args = item.args if isinstance(item.args, str) else json.dumps(item.args)
    return ResponseFunctionToolCall(id=f"fc_{idx}_{uuid.uuid4().hex[:8]}", call_id=f"call_{uuid.uuid4().hex[:10]}",
                                    name=item.name, arguments=args, type="function_call", status="completed")


def _item_events(built: Any, idx: int, seq: _Seq) -> list[Any]:
    events: list[Any] = []
    if isinstance(built, ResponseOutputMessage):
        opening = built.model_copy(update={"content": [], "status": "in_progress"})
    elif isinstance(built, ResponseFunctionToolCall):
        opening = built.model_copy(update={"arguments": "", "status": "in_progress"})
    else:
        opening = built
    events.append(ResponseOutputItemAddedEvent(item=opening, output_index=idx, sequence_number=seq(),
                                               type="response.output_item.added"))
    if isinstance(built, ResponseOutputMessage):
        text = built.content[0].text
        for i in range(0, len(text), 16):
            events.append(ResponseTextDeltaEvent(content_index=0, delta=text[i:i + 16], item_id=built.id, logprobs=[],
                                                 output_index=idx, sequence_number=seq(),
                                                 type="response.output_text.delta"))
    if isinstance(built, ResponseFunctionToolCall):
        events.append(ResponseFunctionCallArgumentsDeltaEvent(
            delta=built.arguments, item_id=built.id or "", output_index=idx, sequence_number=seq(),
            type="response.function_call_arguments.delta"))
        events.append(ResponseFunctionCallArgumentsDoneEvent(
            arguments=built.arguments, item_id=built.id or "", name=built.name, output_index=idx,
            sequence_number=seq(), type="response.function_call_arguments.done"))
    events.append(ResponseOutputItemDoneEvent(item=built, output_index=idx, sequence_number=seq(),
                                              type="response.output_item.done"))
    return events


def turn_events(turn: Turn) -> list[dict[str, Any]]:
    """The full SSE event sequence for ``turn``, each event a JSON-mode dict."""
    seq, resp_id = _Seq(), f"resp_{uuid.uuid4().hex[:12]}"
    events: list[Any] = [ResponseCreatedEvent(response=_response(resp_id, "in_progress", []),
                                              sequence_number=seq(), type="response.created")]
    built = [_built_item(item, i) for i, item in enumerate(turn.items)]
    for i, b in enumerate(built):
        events.extend(_item_events(b, i, seq))
    usage = ResponseUsage(
        input_tokens=turn.input_tokens, output_tokens=turn.output_tokens,
        total_tokens=turn.input_tokens + turn.output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=turn.cached_tokens),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0))
    events.append(ResponseCompletedEvent(response=_response(resp_id, "completed", built, usage),
                                         sequence_number=seq(), type="response.completed"))
    return [e.model_dump(mode="json", exclude_none=True) for e in events]


def soft_fail_events(fail: SoftFail) -> list[dict[str, Any]]:
    """``response.created`` then ``response.failed``. ``model_construct`` for the error: the
    SDK's ``ResponseError.code`` literal lags the codes the vendor actually sends in this slot
    (``invalid_encrypted_content`` is observed in production and absent from the literal)."""
    seq, resp_id = _Seq(), f"resp_{uuid.uuid4().hex[:12]}"
    created = ResponseCreatedEvent(response=_response(resp_id, "in_progress", []), sequence_number=seq(),
                                   type="response.created")
    failed_resp = _response(resp_id, "failed", []).model_copy(
        update={"error": ResponseError.model_construct(code=fail.code, message=fail.message)})
    failed = ResponseFailedEvent.model_construct(response=failed_resp, sequence_number=seq(), type="response.failed")
    out = [created.model_dump(mode="json", exclude_none=True)]
    dumped = failed.model_dump(mode="json", exclude_none=True, warnings=False)
    dumped["response"]["error"] = {"code": fail.code, "message": fail.message}
    return out + [dumped]


# Server ------------------------------------------------------------------------


class FakeResponsesServer:
    """Threaded loopback Responses provider. Use as a context manager."""

    def __init__(self, script: list[Step] | Responder | None = None, *, default_text: str = "ok",
                 api_key: str | None = None) -> None:
        self._script: list[Step] = list(script) if isinstance(script, list) else []
        self._responder: Responder | None = script if callable(script) else None
        self.default_text = default_text
        self.api_key = api_key
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None

    def __enter__(self) -> "FakeResponsesServer":
        server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self))
        server.daemon_threads = True
        self._server = server
        threading.Thread(target=server.serve_forever, name="fake-responses", daemon=True).start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    @property
    def base_url(self) -> str:
        assert self._server is not None, "server not started"
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def push(self, *steps: Step) -> None:
        with self._lock:
            self._script.extend(steps)

    def next_step(self, record: dict[str, Any]) -> Step:
        if self._responder is not None:
            return self._responder(record)
        with self._lock:
            if self._script:
                return self._script.pop(0)
        return Turn([Message(self.default_text)])

    def main_requests(self) -> list[dict[str, Any]]:
        return [r["body"] for r in self.requests if r["kind"] == "main"]

    def invalid_requests(self) -> list[tuple[int, str]]:
        """``(index, error)`` for every main request the SDK schema rejects."""
        mains = [r for r in self.requests if r["kind"] == "main"]
        return [(i, r["invalid"]) for i, r in enumerate(mains) if r.get("invalid")]


def _handler_for(server: FakeResponsesServer) -> type[BaseHTTPRequestHandler]:
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
            path = self.path.rstrip("/")
            is_responses = path.endswith("/responses")
            kind = "main" if is_responses and body.get("tools") else "aux"
            record = {"path": self.path, "kind": kind, "body": body, "t": time.time(),
                      "headers": {k.lower(): v for k, v in self.headers.items()}}
            if is_responses:
                record["invalid"] = validate_request(body)
            with server._lock:
                server.requests.append(record)
            if server.api_key and self.headers.get("Authorization", "") != f"Bearer {server.api_key}":
                self._json(401, {"error": {"message": "invalid api key", "type": "authentication_error"}})
                return
            if path.endswith("/chat/completions"):
                self._aux_chat(bool(body.get("stream")))
                return
            if not is_responses:
                self._json(404, {"error": {"message": f"unsupported path {self.path}"}})
                return
            step = server.next_step(record) if kind == "main" else Turn([Message("Fake title")])
            record["response"] = type(step).__name__
            self._render(step)

        def _render(self, step: Step) -> None:
            if isinstance(step, HttpError):
                headers = {"Retry-After": str(step.retry_after)} if step.retry_after is not None else {}
                self._json(step.status, {"error": {"message": step.message, "type": step.type, "code": step.code}},
                           headers)
                return
            if isinstance(step, SoftFail):
                events, limit = soft_fail_events(step), None
            elif isinstance(step, Cassette):
                events, limit = json.loads(Path(step.path).read_text(encoding="utf-8"))["events"], None
            else:
                events, limit = turn_events(step), step.drop_after_events
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for i, event in enumerate(events):
                if limit is not None and i >= limit:
                    break
                self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
            self.close_connection = True

        def _aux_chat(self, stream: bool) -> None:
            msg = {"role": "assistant", "content": "Fake title"}
            usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
            if not stream:
                self._json(200, {"id": "chatcmpl-aux", "object": "chat.completion", "created": int(time.time()),
                                 "model": MODEL_ID, "usage": usage,
                                 "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for delta, fin in ((msg, None), ({}, "stop")):
                chunk = {"id": "chatcmpl-aux", "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": MODEL_ID, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return Handler


# Record mode (manual) ------------------------------------------------------------

_SECRETISH = re.compile(r"(sk-[A-Za-z0-9_-]{8,}|eyJ[A-Za-z0-9_.-]{20,})")


def sanitize_event(event: dict[str, Any], counter: list[int]) -> dict[str, Any]:
    """Replace opaque/secret-bearing values with stable placeholders (recursive)."""
    def walk(v: Any, key: str = "") -> Any:
        if isinstance(v, dict):
            return {k: walk(x, k) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, str):
            if key == "encrypted_content":
                counter[0] += 1
                return f"ENC-PLACEHOLDER-{counter[0]}"
            return _SECRETISH.sub("REDACTED", v)
        return v
    return walk(event)


def record_proxy(upstream: str, out: Path, port: int = 0) -> None:  # pragma: no cover - manual tool
    """Pass-through proxy that writes one sanitized cassette per ``/responses`` call."""
    import httpx

    def handler_cls() -> type[BaseHTTPRequestHandler]:
        class H(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                headers = {k: v for k, v in self.headers.items() if k.lower() in {"authorization", "content-type"}}
                url = upstream.rstrip("/") + self.path.removeprefix("/v1")
                events: list[dict[str, Any]] = []
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                with httpx.stream("POST", url, content=raw, headers=headers, timeout=300) as resp:
                    for line in resp.iter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            events.append(json.loads(line[6:]))
                        self.wfile.write((line + "\n").encode())
                        self.wfile.flush()
                counter = [0]
                out.write_text(json.dumps({
                    "request": sanitize_event(json.loads(raw or b"{}"), counter),
                    "events": [sanitize_event(e, counter) for e in events]}, indent=1), encoding="utf-8")
        return H

    srv = ThreadingHTTPServer(("127.0.0.1", port), handler_cls())
    print(f"recording proxy on http://127.0.0.1:{srv.server_address[1]}/v1 -> {upstream}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record"])
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--port", type=int, default=0)
    ns = ap.parse_args()
    record_proxy(ns.upstream, ns.out, ns.port)
