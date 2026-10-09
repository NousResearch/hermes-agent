"""Loopback fake of the AWS Bedrock Runtime Converse / ConverseStream API.

The real ``boto3`` ``bedrock-runtime`` client inside Hermes is pointed here through botocore's
documented endpoint override (``AWS_ENDPOINT_URL_BEDROCK_RUNTIME``), so the SDK serializes, signs
(SigV4) and parses exactly as against AWS; only the service is fake.

What the fake enforces, the way Bedrock does:

* SigV4: every request must carry ``Authorization: AWS4-HMAC-SHA256 Credential=<key>/<date>/<region>/
  bedrock/aws4_request, SignedHeaders=..., Signature=...``; the signature is RE-COMPUTED with the known
  fake secret and must match (a tampered body or wrong key is a 403, like AWS).
* Request shape: the JSON body plus the ``modelId`` from the URI is validated against the botocore
  service model's ``Converse`` / ``ConverseStream`` input shape (``ParamValidator``: types, required
  members, tagged unions). Violations are a 400 ``ValidationException``.
* Conversation semantics the service model cannot express but Converse rejects (messages the real
  endpoint answers with a ValidationException): first message must be ``user``; roles alternate;
  text blocks are non-blank; every ``toolResult`` answers a ``toolUse`` of the immediately
  preceding assistant turn and every such ``toolUse`` is answered; a replayed ``reasoningText`` must
  carry exactly the signature this fake issued for that text (unsigned or altered thinking is
  rejected, as signed-thinking models do).

Errors use the AWS JSON error shape (``x-amzn-ErrorType`` header + ``{"message": ...}``);
ConverseStream answers real ``application/vnd.amazon.eventstream`` binary frames (prelude, typed
headers, CRC32s), including ``:message-type exception`` frames and mid-stream connection drops.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Union
from urllib.parse import unquote

import botocore.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from botocore.validate import ParamValidator

ACCESS_KEY = "AKIAFAKEE2EBEDROCK01"
SECRET_KEY = "fake/e2e/secret/never+real/0000000000000"
REGION = "us-east-1"

_SERVICE = botocore.session.get_session().get_service_model("bedrock-runtime")
_ROUTE_RE = re.compile(r"^/model/(?P<model>[^/]+)/(?P<op>converse|converse-stream)$")
_OPS = {"converse": "Converse", "converse-stream": "ConverseStream"}
_AUTH_RE = re.compile(
    r"^AWS4-HMAC-SHA256 Credential=(?P<key>[^/]+)/(?P<date>\d{8})/(?P<region>[a-z0-9-]+)/"
    r"(?P<service>[a-z0-9-]+)/aws4_request, ?SignedHeaders=(?P<signed>[a-z0-9;-]+), ?"
    r"Signature=(?P<sig>[0-9a-f]{64})$")
# HTTP status + x-amzn-ErrorType per Bedrock Runtime error (botocore service model metadata).
ERROR_STATUS = {
    name: shape.metadata["error"]["httpStatusCode"]
    for name in _SERVICE.shape_names
    if (shape := _SERVICE.shape_for(name)).metadata.get("exception")
}


# --------------------------------------------------------------------------------------------------
# Scripted content (Converse ContentBlock shapes)
# --------------------------------------------------------------------------------------------------


@dataclass
class Text:
    text: str
    chunk: int = 12


@dataclass
class Reasoning:
    """``reasoningContent.reasoningText``; the signature is issued by the fake and remembered."""

    text: str
    signature: str = ""


@dataclass
class ToolUse:
    name: str
    input: dict[str, Any]
    tool_use_id: str = ""


Block = Union[Text, Reasoning, ToolUse]


@dataclass
class Turn:
    """One assistant message (streamed or not) + usage the service reports."""

    blocks: list[Block]
    stop_reason: str = ""
    input_tokens: int = 120
    output_tokens: int = 30


@dataclass
class HttpError:
    """An AWS JSON error response before any stream starts (e.g. 429 ThrottlingException)."""

    code: str
    message: str


@dataclass
class StreamException:
    """ConverseStream: emit ``after`` events of ``turn`` then a ``:message-type exception`` frame."""

    turn: Turn
    exception: str  # event member name, e.g. "throttlingException"
    message: str
    after: int = 2


@dataclass
class Drop:
    """ConverseStream: emit ``after`` events of ``turn`` then close the socket mid-stream.

    ``clean=True`` ends the HTTP body properly instead (valid framing, but the event stream stops
    before ``messageStop``: the service never finished the message)."""

    turn: Turn
    after: int = 3
    clean: bool = False


Reply = Union[Turn, HttpError, StreamException, Drop]


# --------------------------------------------------------------------------------------------------
# Event-stream framing (application/vnd.amazon.eventstream)
# --------------------------------------------------------------------------------------------------


def _header(name: str, value: str) -> bytes:
    n, v = name.encode(), value.encode()
    return struct.pack("!B", len(n)) + n + b"\x07" + struct.pack("!H", len(v)) + v  # 7 = string


def encode_frame(headers: dict[str, str], payload: bytes) -> bytes:
    """One event-stream message: prelude(total, headers_len) + prelude CRC + headers + payload + CRC."""
    hdr = b"".join(_header(k, v) for k, v in headers.items())
    total = 12 + len(hdr) + len(payload) + 4
    prelude = struct.pack("!II", total, len(hdr))
    head = prelude + struct.pack("!I", binascii.crc32(prelude) & 0xFFFFFFFF) + hdr + payload
    return head + struct.pack("!I", binascii.crc32(head) & 0xFFFFFFFF)


def event_frame(event_type: str, body: dict[str, Any]) -> bytes:
    _validate_event(event_type, body)
    return encode_frame({":event-type": event_type, ":content-type": "application/json",
                         ":message-type": "event"}, json.dumps(body).encode())


def exception_frame(exception: str, message: str) -> bytes:
    return encode_frame({":exception-type": exception, ":content-type": "application/json",
                         ":message-type": "exception"}, json.dumps({"message": message}).encode())


def _validate_event(event_type: str, body: dict[str, Any]) -> None:
    """The fake only emits events valid for the ConverseStream output shape (self-check)."""
    shape = _SERVICE.operation_model("ConverseStream").output_shape.members["stream"].members[event_type]
    report = ParamValidator().validate(body, shape)
    if report.has_errors():
        raise AssertionError(f"fake built an invalid {event_type} event: {report.generate_report()}")


# --------------------------------------------------------------------------------------------------
# Request validation
# --------------------------------------------------------------------------------------------------


class Rejection(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _wire_to_params(shape: Any, value: Any) -> Any:
    """rest-json wire value -> botocore param value (blobs arrive base64-encoded)."""
    kind = shape.type_name
    if kind == "blob" and isinstance(value, str):
        try:
            return base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            raise Rejection("ValidationException", f"Invalid base64 blob for {shape.name}") from None
    if kind == "structure" and isinstance(value, dict) and not shape.metadata.get("document"):
        return {k: (_wire_to_params(shape.members[k], v) if k in shape.members else v) for k, v in value.items()}
    if kind == "list" and isinstance(value, list):
        return [_wire_to_params(shape.member, v) for v in value]
    if kind == "map" and isinstance(value, dict):
        return {k: _wire_to_params(shape.value, v) for k, v in value.items()}
    return value


def validate_shape(op: str, model_id: str, body: dict[str, Any]) -> None:
    shape = _SERVICE.operation_model(op).input_shape
    params = _wire_to_params(shape, {**body, "modelId": model_id})
    report = ParamValidator().validate(params, shape)
    if report.has_errors():
        raise Rejection("ValidationException", report.generate_report())


def _tool_ids(content: list[dict[str, Any]], kind: str) -> list[str]:
    return [b[kind]["toolUseId"] for b in content if kind in b]


def validate_conversation(body: dict[str, Any], signatures: dict[str, str], reasoned_tools: set[str]) -> None:
    """Converse-side rules beyond the schema (real ValidationException texts)."""
    messages = body.get("messages") or []
    if not messages or messages[0]["role"] != "user":
        raise Rejection("ValidationException", "A conversation must start with a user message. "
                        "Try again with a conversation that starts with a user message.")
    for i, msg in enumerate(messages):
        if i and msg["role"] == messages[i - 1]["role"]:
            raise Rejection("ValidationException", "A conversation must alternate between user and "
                            "assistant roles. Make sure the conversation alternates between user and "
                            "assistant roles and try again.")
        for j, block in enumerate(msg["content"]):
            if "text" in block and not block["text"].strip():
                raise Rejection("ValidationException", f"The text field in the ContentBlock object at "
                                f"messages.{i}.content.{j} is blank. Add text to the text field, and try again.")
            _check_reasoning(block.get("reasoningContent"), signatures, f"messages.{i}.content.{j}")
        expected = _tool_ids(messages[i - 1]["content"], "toolUse") if i and msg["role"] == "user" else []
        got = _tool_ids(msg["content"], "toolResult")
        if sorted(expected) != sorted(got):
            raise Rejection("ValidationException", f"Expected toolResult blocks at messages.{i}.content "
                            f"for the following Ids: {', '.join(expected) or '(none)'}; got {', '.join(got) or '(none)'}")
    _check_final_assistant_thinking(messages, reasoned_tools)


def _check_final_assistant_thinking(messages: list[dict[str, Any]], reasoned_tools: set[str]) -> None:
    """Signed-thinking models: while a tool loop is open (the request ends in toolResults), the final
    assistant turn must START with its reasoning block if it was issued with one."""
    if len(messages) < 2 or not _tool_ids(messages[-1]["content"], "toolResult"):
        return
    final = messages[-2]["content"]
    if set(_tool_ids(final, "toolUse")) & reasoned_tools and "reasoningContent" not in final[0]:
        first = next(iter(final[0]), "?")
        raise Rejection("ValidationException", f"messages.{len(messages) - 2}.content.0.type: Expected "
                        f"`thinking` or `redacted_thinking`, but found `{first}`. When `thinking` is enabled, "
                        "a final `assistant` message must start with a thinking block (preceeding the lastmost "
                        "set of `tool_use` and `tool_result` blocks).")


def _check_reasoning(reasoning: Any, signatures: dict[str, str], where: str) -> None:
    text_block = (reasoning or {}).get("reasoningText")
    if not text_block:
        return
    issued = signatures.get(text_block.get("text", ""))
    if issued is None or text_block.get("signature") != issued:
        raise Rejection("ValidationException", f"{where}: The reasoning block signature is missing or "
                        "invalid. Reasoning content must be passed back unmodified with its signature.")


def verify_sigv4(method: str, url: str, headers: dict[str, str], body: bytes) -> dict[str, str]:
    """Re-compute the SigV4 signature with the fake secret; return the parsed credential scope."""
    auth = headers.get("authorization", "")
    m = _AUTH_RE.match(auth)
    if not m:
        raise Rejection("MissingAuthenticationTokenException", f"Missing or malformed SigV4 Authorization: {auth!r}")
    if m["key"] != ACCESS_KEY or m["service"] != "bedrock":
        raise Rejection("UnrecognizedClientException", "The security token included in the request is invalid.")
    signed = {h: headers[h] for h in m["signed"].split(";") if h in headers}
    request = AWSRequest(method=method, url=url, data=body, headers=signed)
    request.context["timestamp"] = headers.get("x-amz-date", "")
    signer = SigV4Auth(Credentials(ACCESS_KEY, SECRET_KEY), "bedrock", m["region"])
    canonical = signer.canonical_request(request)
    expected = signer.signature(signer.string_to_sign(request, canonical), request)
    if expected != m["sig"]:
        raise Rejection("InvalidSignatureException", "The request signature we calculated does not match "
                        "the signature you provided.")
    return m.groupdict()


# --------------------------------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------------------------------


Responder = Callable[[dict[str, Any]], Reply]


@dataclass
class FakeBedrock:
    """Recording, validating Bedrock Runtime fake. ``responder(record) -> Reply`` scripts each call."""

    responder: Responder
    requests: list[dict[str, Any]] = field(default_factory=list)
    signatures: dict[str, str] = field(default_factory=dict)  # reasoning text -> issued signature
    reasoned_tools: set[str] = field(default_factory=set)  # toolUseIds issued in a turn with reasoning
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _ids: int = 0
    _httpd: ThreadingHTTPServer | None = None

    def __enter__(self) -> "FakeBedrock":
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self))
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()

    @property
    def endpoint(self) -> str:
        assert self._httpd is not None
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def client_env(self) -> dict[str, str]:
        """Env for a Hermes child: botocore endpoint override + fake static credentials."""
        return {"AWS_ENDPOINT_URL_BEDROCK_RUNTIME": self.endpoint, "AWS_ACCESS_KEY_ID": ACCESS_KEY,
                "AWS_SECRET_ACCESS_KEY": SECRET_KEY, "AWS_REGION": REGION}

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.requests)

    def next_id(self, prefix: str) -> str:
        with self._lock:
            self._ids += 1
            return f"{prefix}{self._ids:04d}{int(time.monotonic() * 1000) % 100000:05d}"

    def record(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self.requests.append(rec)

    def materialize(self, turn: Turn) -> list[dict[str, Any]]:
        """Resolve ids/signatures and return Converse ContentBlocks for the turn."""
        out: list[dict[str, Any]] = []
        for block in turn.blocks:
            if isinstance(block, Text):
                out.append({"text": block.text})
            elif isinstance(block, Reasoning):
                block.signature = block.signature or base64.b64encode(
                    f"sig:{self.next_id('r')}:{len(block.text)}".encode()).decode()
                with self._lock:
                    self.signatures[block.text] = block.signature
                out.append({"reasoningContent": {"reasoningText": {"text": block.text, "signature": block.signature}}})
            else:
                block.tool_use_id = block.tool_use_id or self.next_id("tooluse_")
                out.append({"toolUse": {"toolUseId": block.tool_use_id, "name": block.name, "input": block.input}})
        if any(isinstance(b, Reasoning) for b in turn.blocks):
            with self._lock:
                self.reasoned_tools.update(b.tool_use_id for b in turn.blocks if isinstance(b, ToolUse))
        if not turn.stop_reason:
            turn.stop_reason = "tool_use" if any(isinstance(b, ToolUse) for b in turn.blocks) else "end_turn"
        return out


def _usage(turn: Turn) -> dict[str, int]:
    return {"inputTokens": turn.input_tokens, "outputTokens": turn.output_tokens,
            "totalTokens": turn.input_tokens + turn.output_tokens}


def _pieces(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def _block_events(index: int, block: dict[str, Any], chunk: int) -> list[tuple[str, dict[str, Any]]]:
    """ContentBlockStart/Delta/Stop events for one block (text blocks get no start, as on AWS)."""
    events: list[tuple[str, dict[str, Any]]] = []
    if "toolUse" in block:
        tu = block["toolUse"]
        events.append(("contentBlockStart", {"contentBlockIndex": index, "start": {
            "toolUse": {"toolUseId": tu["toolUseId"], "name": tu["name"]}}}))
        deltas = [{"toolUse": {"input": p}} for p in _pieces(json.dumps(tu["input"]), chunk)]
    elif "reasoningContent" in block:
        rt = block["reasoningContent"]["reasoningText"]
        deltas = [{"reasoningContent": {"text": p}} for p in _pieces(rt["text"], chunk)]
        deltas.append({"reasoningContent": {"signature": rt["signature"]}})
    else:
        deltas = [{"text": p} for p in _pieces(block["text"], chunk)]
    events += [("contentBlockDelta", {"contentBlockIndex": index, "delta": d}) for d in deltas]
    events.append(("contentBlockStop", {"contentBlockIndex": index}))
    return events


def stream_events(blocks: list[dict[str, Any]], turn: Turn) -> list[bytes]:
    chunk = min((b.chunk for b in turn.blocks if isinstance(b, Text)), default=12)
    events: list[tuple[str, dict[str, Any]]] = [("messageStart", {"role": "assistant"})]
    for i, block in enumerate(blocks):
        events += _block_events(i, block, chunk)
    events.append(("messageStop", {"stopReason": turn.stop_reason}))
    events.append(("metadata", {"usage": _usage(turn), "metrics": {"latencyMs": 7}}))
    return [event_frame(name, body) for name, body in events]


def _handler_for(fake: FakeBedrock) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            headers = {k.lower(): v for k, v in self.headers.items()}
            route = _ROUTE_RE.match(self.path.split("?", 1)[0])
            rec: dict[str, Any] = {"path": self.path, "headers": headers, "raw": raw, "t": time.time(),
                                   "op": _OPS.get(route["op"]) if route else None,
                                   "model": unquote(route["model"]) if route else None}
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = None
            rec["body"] = body
            try:
                if not route:
                    raise Rejection("UnknownOperationException", f"No route for {self.path}")
                rec["auth"] = verify_sigv4("POST", f"http://{headers.get('host')}{self.path}", headers, raw)
                if not isinstance(body, dict):
                    raise Rejection("SerializationException", "Request body is not a JSON object")
                validate_shape(rec["op"], rec["model"], body)
                validate_conversation(body, dict(fake.signatures), set(fake.reasoned_tools))
            except Rejection as rej:
                rec["rejected"] = f"{rej.code}: {rej.message}"
                fake.record(rec)
                return self._error(rej.code, rej.message)
            fake.record(rec)
            reply = fake.responder(rec)
            rec["reply"] = type(reply).__name__
            if isinstance(reply, HttpError):
                return self._error(reply.code, reply.message)
            turn = reply if isinstance(reply, Turn) else reply.turn
            rec["emitted"] = fake.materialize(turn)
            if rec["op"] == "Converse":
                return self._converse(turn, rec["emitted"])
            return self._stream(reply, turn, rec["emitted"])

        def _send(self, status: int, headers: dict[str, str], body: bytes) -> None:
            self.send_response(status)
            for k, v in {**headers, "Content-Length": str(len(body)),
                         "x-amzn-RequestId": fake.next_id("req-")}.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def _error(self, code: str, message: str) -> None:
            status = ERROR_STATUS.get(code, 403 if "Signature" in code or "Token" in code else 400)
            self._send(status, {"Content-Type": "application/json", "x-amzn-ErrorType": f"{code}:http://internal.amazon.com/coral/com.amazon.bedrock/"},
                       json.dumps({"message": message}).encode())

        def _converse(self, turn: Turn, blocks: list[dict[str, Any]]) -> None:
            body = {"output": {"message": {"role": "assistant", "content": blocks}},
                    "stopReason": turn.stop_reason, "usage": _usage(turn), "metrics": {"latencyMs": 9}}
            self._send(200, {"Content-Type": "application/json"}, json.dumps(body).encode())

        def _stream(self, reply: Reply, turn: Turn, blocks: list[dict[str, Any]]) -> None:
            frames = stream_events(blocks, turn)
            cut = len(frames) if isinstance(reply, Turn) else min(reply.after, len(frames))
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.amazon.eventstream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("x-amzn-RequestId", fake.next_id("req-"))
            self.end_headers()
            for frame in frames[:cut]:
                self._chunk(frame)
            if isinstance(reply, Drop) and not reply.clean:
                self.close_connection = True
                self.wfile.flush()
                self.connection.shutdown(2)  # socket.SHUT_RDWR: no terminating chunk
                return
            if isinstance(reply, StreamException):
                self._chunk(exception_frame(reply.exception, reply.message))
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def _chunk(self, data: bytes) -> None:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

    return Handler


def seq(*replies: Reply | Callable[[dict[str, Any]], Reply]) -> Responder:
    """Responder answering the Nth call with ``replies[N]`` (the last one repeats); callables get the record."""
    calls: list[int] = []
    lock = threading.Lock()

    def respond(rec: dict[str, Any]) -> Reply:
        with lock:
            calls.append(1)
            reply = replies[min(len(calls), len(replies)) - 1]
        return reply(rec) if callable(reply) else reply

    return respond
