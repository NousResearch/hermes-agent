"""Loopback OAuth authorization server + bearer-checking inference server for the OAuth E2E cells.

One real HTTP server on 127.0.0.1 that plays the vendor side of the OAuth providers:

* Nous Portal (RFC 8628 device flow + single-use refresh-token rotation):
  ``POST /api/oauth/device/code`` and ``POST /api/oauth/token`` (``grant_type`` device_code or
  refresh_token; the refresh token rides in the ``x-nous-refresh-token`` header). The device poll
  answers the scripted error codes in ``poll_script`` (one per poll) and then issues tokens; every
  poll's arrival time is recorded so a test can measure the client's real polling cadence.
* MiniMax OAuth refresh: ``POST /oauth/token`` (form ``refresh_token``), MiniMax's
  ``{"status": "success", "expired_in": ...}`` shape.
* Inference: ``POST …/chat/completions`` (JSON or SSE) and ``POST …/messages`` (Anthropic JSON or
  SSE). Every inference request is bearer-checked: a token in ``revoked`` gets HTTP 401 like a
  real gateway rejecting a revoked/expired key; any other token gets a plain answer ``reply``.

Refresh tokens are single-use, as at the real vendors: redeeming one retires it, and redeeming a
retired one returns ``invalid_grant``. Every request is recorded (method, path, headers, form or
JSON body, arrival time).
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

NOUS_INVOKE_SCOPE = "inference:invoke"


def b64url(obj: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


def make_jwt(tag: str, *, ttl_s: int = 3600, scope: str = NOUS_INVOKE_SCOPE) -> str:
    """Unsigned JWT-shaped bearer: the client only decodes claims (``scope``/``exp``)."""
    claims = {"sub": "oauth-e2e-user", "scope": scope, "exp": int(time.time()) + ttl_s, "tag": tag}
    return ".".join([b64url({"alg": "none", "typ": "JWT"}), b64url(claims), "sig"])


@dataclass
class Req:
    method: str
    path: str
    headers: dict[str, str]
    form: dict[str, str]
    body: Any
    t: float = field(default_factory=time.monotonic)

    @property
    def bearer(self) -> str:
        auth = self.headers.get("authorization", "")
        return auth[7:] if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")


class OAuthFake:
    """Threaded loopback vendor. Use as a context manager."""

    def __init__(self, *, reply: str = "OAUTH-TURN-COMPLETE", device_interval: int = 2,
                 poll_script: list[str] | None = None, valid_refresh: set[str] | None = None,
                 revoked: set[str] | None = None) -> None:
        self.reply = reply
        self.device_interval = device_interval
        self.poll_script = list(poll_script or [])
        self.valid_refresh = set(valid_refresh or ())
        self.revoked = set(revoked or ())
        self.requests: list[Req] = []
        self.issued: list[dict[str, Any]] = []  # every token response, in order
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None

    def __enter__(self) -> "OAuthFake":
        server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self))
        server.daemon_threads = True
        self._server = server
        threading.Thread(target=server.serve_forever, name="oauth-fake", daemon=True).start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    @property
    def origin(self) -> str:
        assert self._server is not None, "server not started"
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    # --- views -----------------------------------------------------------------------------
    def device_polls(self) -> list[Req]:
        return [r for r in self.requests
                if r.path == "/api/oauth/token" and r.form.get("grant_type", "").endswith("device_code")]

    def refreshes(self) -> list[Req]:
        return [r for r in self.requests if r.path in ("/api/oauth/token", "/oauth/token")
                and r.form.get("grant_type") == "refresh_token"]

    def inference(self) -> list[Req]:
        return [r for r in self.requests
                if r.method == "POST" and r.path.split("?")[0].endswith(("/chat/completions", "/messages"))]

    # --- token issuance --------------------------------------------------------------------
    def _issue(self, kind: str) -> dict[str, Any]:
        with self._lock:
            n = len(self.issued) + 1
            rt = f"rt-{kind}-{n}"
            self.valid_refresh.add(rt)
            tok = {"access_token": make_jwt(f"{kind}-{n}"), "refresh_token": rt, "token_type": "Bearer",
                   "expires_in": 3600, "scope": NOUS_INVOKE_SCOPE}
            self.issued.append(tok)
            return tok

    def _redeem(self, refresh_token: str) -> bool:
        with self._lock:
            if refresh_token not in self.valid_refresh:
                return False
            self.valid_refresh.discard(refresh_token)
            return True


def _handler_for(fake: OAuthFake) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a: object) -> None:
            pass

        def _json(self, status: int, payload: Any) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _sse(self, frames: list[tuple[str | None, Any]], done: bool) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            for event, data in frames:
                prefix = f"event: {event}\n" if event else ""
                self.wfile.write(f"{prefix}data: {json.dumps(data)}\n\n".encode())
            if done:
                self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def do_GET(self) -> None:  # noqa: N802
            headers = {k.lower(): v for k, v in self.headers.items()}
            with fake._lock:
                fake.requests.append(Req("GET", self.path, headers, {}, None))
            if self.path.split("?")[0].rstrip("/").endswith("/models"):
                self._json(200, {"object": "list", "data": [{"id": "oauth-e2e/model", "object": "model"}]})
                return
            self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8", "replace")
            headers = {k.lower(): v for k, v in self.headers.items()}
            ctype = headers.get("content-type", "")
            form = ({k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}
                    if "x-www-form-urlencoded" in ctype else {})
            try:
                body = json.loads(raw) if raw.startswith(("{", "[")) else None
            except json.JSONDecodeError:
                body = None
            req = Req("POST", self.path, headers, form, body)
            with fake._lock:
                fake.requests.append(req)
            path = self.path.split("?")[0].rstrip("/")
            route = _ROUTES.get(path) or next(
                (fn for suffix, fn in _SUFFIX_ROUTES if path.endswith(suffix)), None)
            if route is None:
                self._json(404, {"error": {"message": f"unsupported path {self.path}"}})
                return
            route(self, fake, req)

    return Handler


# --- routes -----------------------------------------------------------------------------------


def _device_code(h: Any, fake: OAuthFake, _req: Req) -> None:
    h._json(200, {"device_code": "dc-oauth-e2e", "user_code": "OAUT-HE2E",
                  "verification_uri": f"{fake.origin}/device",
                  "verification_uri_complete": f"{fake.origin}/device?code=OAUT-HE2E",
                  "expires_in": 120, "interval": fake.device_interval})


def _nous_token(h: Any, fake: OAuthFake, req: Req) -> None:
    grant = req.form.get("grant_type", "")
    if grant.endswith("device_code"):
        with fake._lock:
            step = fake.poll_script.pop(0) if fake.poll_script else None
        if step is not None:
            h._json(400, {"error": step, "error_description": f"scripted {step}"})
            return
        h._json(200, fake._issue("login"))
        return
    if grant == "refresh_token":
        if not fake._redeem(req.headers.get("x-nous-refresh-token", "")):
            h._json(400, {"error": "invalid_grant", "error_description": "refresh token already used or unknown"})
            return
        h._json(200, fake._issue("nous-rot"))
        return
    h._json(400, {"error": "unsupported_grant_type"})


def _minimax_token(h: Any, fake: OAuthFake, req: Req) -> None:
    if req.form.get("grant_type") != "refresh_token" or not fake._redeem(req.form.get("refresh_token", "")):
        h._json(400, {"status": "error", "error": "invalid_grant"})
        return
    tok = fake._issue("mm-rot")
    h._json(200, {"status": "success", "access_token": tok["access_token"],
                  "refresh_token": tok["refresh_token"], "expired_in": 3600, "token_type": "Bearer"})


def _reject_revoked(h: Any, fake: OAuthFake, req: Req) -> bool:
    if req.bearer and req.bearer not in fake.revoked:
        return False
    h._json(401, {"error": {"message": "invalid or revoked access token", "type": "authentication_error",
                            "code": "invalid_api_key"}})
    return True


def _chat(h: Any, fake: OAuthFake, req: Req) -> None:
    if _reject_revoked(h, fake, req):
        return
    body = req.body or {}
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    base = {"id": "chatcmpl-oauth", "created": int(time.time()), "model": body.get("model") or "m"}
    if not body.get("stream"):
        h._json(200, {**base, "object": "chat.completion", "usage": usage, "choices": [
            {"index": 0, "message": {"role": "assistant", "content": fake.reply}, "finish_reason": "stop"}]})
        return
    chunk = {**base, "object": "chat.completion.chunk"}
    h._sse([(None, {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": fake.reply},
                                          "finish_reason": None}]}),
            (None, {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage})],
           done=True)


def _messages(h: Any, fake: OAuthFake, req: Req) -> None:
    if _reject_revoked(h, fake, req):
        return
    body = req.body or {}
    msg = {"id": "msg_oauth", "type": "message", "role": "assistant", "model": body.get("model") or "m",
           "stop_sequence": None}
    if not body.get("stream"):
        h._json(200, {**msg, "content": [{"type": "text", "text": fake.reply}], "stop_reason": "end_turn",
                      "usage": {"input_tokens": 10, "output_tokens": 2}})
        return
    h._sse([
        ("message_start", {"type": "message_start", "message": {
            **msg, "content": [], "stop_reason": None, "usage": {"input_tokens": 10, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": fake.reply}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 2}}),
        ("message_stop", {"type": "message_stop"}),
    ], done=False)


_ROUTES = {
    "/api/oauth/device/code": _device_code,
    "/api/oauth/token": _nous_token,
    "/oauth/token": _minimax_token,
}
_SUFFIX_ROUTES = (("/chat/completions", _chat), ("/messages", _messages))
