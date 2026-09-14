#!/usr/bin/env python3
"""Fault-injecting stand-in for the two services behind the Nous free tier.

One process plays both the account service (NAS, ``/api/anonymous/*`` and the device-code
endpoints a sign-in touches) and the welcome inference host (``/v1/chat/completions``), answering
with the exact status codes, bodies and headers the real services send, so Hermes' failure
handling can be rehearsed end to end without touching production. Which failure it serves is a
live switch: change it from the desktop's JavaScript console, ``curl``, or a browser, and watch
the desktop react.

Run::

    python scripts/free_tier_fault_server.py            # 127.0.0.1:8765
    python scripts/free_tier_fault_server.py --port 9000 --inference rate_limited

then start Hermes Desktop pointed at it (the script prints the exact environment lines). See
``website/docs/developer-guide/free-tier-fault-rehearsal.md`` for the walkthrough.

Control surface (all CORS-open, so a renderer can call them)::

    GET  /__scenarios                 the catalogue: every scenario, per service, with what it does
    GET  /__scenario                  the switches as they stand
    POST /__scenario  {"nas": ..., "inference": ..., "once": bool, "retry_after": N}
    POST /__signin    {"status": "completed" | "voided", "reason": "..."}   settle a pending sign-in
    GET  /__log                       the last requests the server answered
    POST /__reset                     back to the happy path, log cleared

Stdlib only: no dependencies, nothing to install.
"""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

WELCOME_MODEL = "nous/welcome"
REAL_WELCOME_HOST = "welcome-api.nousresearch.com"
UPGRADE_URL = "https://portal.nousresearch.com/signup"

# --- Scenario catalogue --------------------------------------------------------------------------
#
# Names are what the control endpoint accepts. Each entry says what the real service does in that
# situation, so a rehearsal reads as the situation rather than the status code.

NAS_SCENARIOS: Dict[str, str] = {
    "ok": "The account service answers normally: sign-ups mint, exchanges return a JWT.",
    "not_enabled": "404 not_found on every /api/anonymous route: the surface is not enabled on this deployment.",
    "paused": "503 temporarily_disabled: the ops breaker is tripped (transient, no wait hinted).",
    "rate_limited": "429 temporarily_unavailable with Retry-After: too many sign-ups from this address.",
    "server_error": "500 with a non-JSON body on every /api/anonymous route.",
    "timeout": "Every /api/anonymous request hangs past the client's 5 s budget.",
    "pow": "Token exchange answers 428 pow_required: proof of work enforced (sign-ups still mint).",
    "locked": "Token exchange answers 403 account_locked: the account is locked.",
    "dead_once": "The NEXT token exchange answers 404 unknown_token (reaped / claimed); the replacement works.",
    "signin_busy": "Starting a sign-in (promotion-intent) answers 429 with Retry-After.",
    "signin_paused": "Starting a sign-in answers 503 temporarily_disabled.",
    "signin_server_error": "Starting a sign-in answers 500.",
}

INFERENCE_SCENARIOS: Dict[str, str] = {
    "ok": "Chat completions answer with a short canned reply (streaming or not).",
    "rate_limited": "429 reason=rate_limited, retry_after 600: the allowance is used up (long wait -> stop and say so).",
    "rate_limited_short": "429 reason=rate_limited, retry_after 5: a short wait the turn rides out quietly.",
    "at_capacity": "429 reason=at_capacity, retry_after 30: the tier is busy (retried in place, bounded).",
    "model_not_free": "429 reason=model_not_free, alternates=[nous/welcome]: the session asked for another model.",
    "tier_disabled": "403 with only the generic permission message: WELCOME_MODE=off on the gateway.",
    "wrong_host": "400 'Anonymous accounts must use https://welcome-api...': the route points at the paid host.",
    "bare_429": "429 with x-ratelimit-* headers showing an exhausted bucket and no reason field.",
    "upstream_503": "503 'The requested model is currently unavailable.': the upstream provider is down.",
    "upstream_500": "500 generic: an upstream 5xx surfaced by the gateway.",
    "invalid_token": "401 invalid_token / anonymous_credential_revoked: the session behind the JWT is gone.",
    "timeout": "The request hangs for two minutes.",
}


_FAIRSHARE_MESSAGE = ("You've reached this model's current fair-share rate limit. It adapts to demand — "
                      "retry after the indicated delay, or try an alternate model.")


def _fairshare(reason: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"status": 429, "message": message, "reason": reason, "alternates": [], "upgrade_url": UPGRADE_URL, **extra}


# Static inference answers: scenario -> (status, body, default Retry-After seconds). ``retry_after``
# in a fairshare body and every wait header are filled in per request from the live override.
INFERENCE_RESPONSES: Dict[str, tuple] = {
    "rate_limited": (429, _fairshare("rate_limited", _FAIRSHARE_MESSAGE), 600),
    "rate_limited_short": (429, _fairshare("rate_limited", _FAIRSHARE_MESSAGE), 5),
    "at_capacity": (429, _fairshare("at_capacity", "The free tier is at capacity and briefly paused. It reopens "
                                    "automatically — retry after the indicated delay."), 30),
    "model_not_free": (429, _fairshare("model_not_free", "This model isn't available on the free tier.",
                                       alternates=[WELCOME_MODEL]), 0),
    "tier_disabled": (403, {"status": 403, "message": "You tried to access something that you don't have permissions for."}, 0),
    "wrong_host": (400, {"status": 400, "message": "This request is not valid. Check the model name and other parameters. "
                         f"Additional info: Anonymous accounts must use https://{REAL_WELCOME_HOST} for inference."}, 0),
    "bare_429": (429, {"status": 429, "message": "Hold up for a bit, you've exceeded the rate limit on your API key."}, 600),
    "upstream_503": (503, {"status": 503, "message": "The requested model is currently unavailable."}, 0),
    "upstream_500": (500, {"status": 500, "message": "Something unexpected happened while processing your request. "
                           "Please try again in a moment, or contact us if the issue persists."}, 0),
    "invalid_token": (401, {"status": 401, "error": "invalid_token", "subcause": "anonymous_credential_revoked",
                            "message": "The anonymous account behind this token is gone"}, 0),
}


def _jwt(**claims: Any) -> str:
    def seg(obj: Any) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    payload = {"sub": "nas_user:rehearsal", "client_id": "nas-anonymous", "account_tier": "anonymous",
               "scope": "inference:invoke tool:invoke", "iss": "free-tier-fault-server",
               "iat": int(time.time()), "exp": int(time.time()) + 900, **claims}
    return f"{seg({'alg': 'RS256', 'typ': 'JWT'})}.{seg(payload)}.{seg({'sig': 'rehearsal'})}"


class State:
    """The switches, guarded by one lock (the server is threaded)."""

    def __init__(self, *, nas: str = "ok", inference: str = "ok", welcome_url: str = "") -> None:
        self.lock = threading.Lock()
        self.nas = nas
        self.inference = inference
        self.once = False
        self.retry_after: Optional[int] = None
        self.welcome_url = welcome_url
        self.minted = 0
        self.dead_tokens: set[str] = set()
        self.signin: Dict[str, Any] = {"status": "pending"}
        self.log: deque[Dict[str, Any]] = deque(maxlen=100)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {"nas": self.nas, "inference": self.inference, "once": self.once,
                    "retry_after": self.retry_after, "minted": self.minted, "signin": dict(self.signin),
                    "welcome_url": self.welcome_url}

    def set(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if "nas" in payload:
                name = str(payload["nas"] or "ok")
                if name not in NAS_SCENARIOS:
                    raise ValueError(f"unknown nas scenario {name!r}; one of {sorted(NAS_SCENARIOS)}")
                self.nas = name
            if "inference" in payload:
                name = str(payload["inference"] or "ok")
                if name not in INFERENCE_SCENARIOS:
                    raise ValueError(f"unknown inference scenario {name!r}; one of {sorted(INFERENCE_SCENARIOS)}")
                self.inference = name
            if "once" in payload:
                self.once = bool(payload["once"])
            if "retry_after" in payload:
                value = payload["retry_after"]
                self.retry_after = None if value in (None, "") else max(0, int(value))
        return self.snapshot()

    def consume_once(self, service: str) -> None:
        """After a one-shot failure was served, drop that service back to the happy path."""
        with self.lock:
            if self.once:
                setattr(self, service, "ok")

    def reset(self) -> None:
        with self.lock:
            self.nas = self.inference = "ok"
            self.once = False
            self.retry_after = None
            self.dead_tokens.clear()
            self.signin = {"status": "pending"}
            self.log.clear()


STATE = State()


class Handler(BaseHTTPRequestHandler):
    server_version = "free-tier-fault-server/1"

    # --- plumbing --------------------------------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter than the default
        pass

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type, authorization")

    def _send(self, status: int, body: Any = None, *, headers: Optional[Dict[str, str]] = None,
              raw: Optional[bytes] = None, content_type: str = "application/json") -> None:
        data = raw if raw is not None else json.dumps(body if body is not None else {}).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self._cors()
        self.end_headers()
        self.wfile.write(data)
        with STATE.lock:
            STATE.log.append({"at": time.time(), "method": self.command, "path": self.path, "status": status,
                              "scenario": {"nas": STATE.nas, "inference": STATE.inference}})

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            # Form-encoded (the device-code endpoints post forms).
            return {k: v[0] for k, v in parse_qs(raw.decode(errors="replace")).items()}

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/__scenarios":
            self._send(200, {"nas": NAS_SCENARIOS, "inference": INFERENCE_SCENARIOS})
        elif path == "/__scenario":
            query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            if query:
                self._control_set(query)
            else:
                self._send(200, STATE.snapshot())
        elif path == "/__log":
            with STATE.lock:
                entries = list(STATE.log)
            self._send(200, {"requests": entries})   # outside the lock: _send appends to the log
        elif path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": WELCOME_MODEL, "object": "model", "owned_by": "nous"}]})
        elif path.startswith("/__claim"):
            self._send(200, raw=(
                b"<html><body><h1>Free tier fault server</h1><p>This stands in for the sign-in page. "
                b"Settle the sign-in with <code>POST /__signin</code>.</p></body></html>"),
                content_type="text/html")
        elif path in ("/", "/healthcheck"):
            self._send(200, {"ok": True, "service": "free-tier-fault-server"})
        else:
            self._send(404, {"status": 404, "message": "Couldn't find that, sorry."})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._read_json()
        if path == "/__scenario":
            self._control_set(body)
        elif path == "/__signin":
            with STATE.lock:
                STATE.signin = {"status": str(body.get("status") or "pending"), **{
                    k: v for k, v in body.items() if k != "status"}}
            self._send(200, STATE.snapshot())
        elif path == "/__reset":
            STATE.reset()
            self._send(200, STATE.snapshot())
        elif path.startswith("/api/anonymous/"):
            self._nas(path, body)
        elif path == "/api/oauth/device/code":
            self._device_code(body)
        elif path == "/api/oauth/token":
            self._oauth_token(body)
        elif path in ("/v1/chat/completions", "/chat/completions"):
            self._inference(body)
        else:
            self._send(404, {"status": 404, "message": "Couldn't find that, sorry."})

    def _control_set(self, payload: Dict[str, Any]) -> None:
        try:
            self._send(200, STATE.set(payload))
        except ValueError as exc:
            self._send(400, {"error": str(exc)})

    # --- the account service (NAS) ---------------------------------------------------------------

    def _retry_after(self, default: int) -> int:
        with STATE.lock:
            return default if STATE.retry_after is None else STATE.retry_after

    def _nas_gate(self, scenario: str) -> bool:
        """The shared gate every /api/anonymous route runs first. True when a refusal was sent."""
        if scenario == "not_enabled":
            self._send(404, {"error": "not_found"})
        elif scenario == "paused":
            self._send(503, {"error": "temporarily_disabled",
                             "error_description": "The anonymous-account surface is currently switched off."})
        elif scenario == "server_error":
            self._send(500, raw=b"<html><body>Internal Server Error</body></html>", content_type="text/html")
        elif scenario == "timeout":
            time.sleep(12)
            return False
        else:
            return False
        STATE.consume_once("nas")
        return True

    def _nas(self, path: str, body: Dict[str, Any]) -> None:
        with STATE.lock:
            scenario = STATE.nas
        if self._nas_gate(scenario):
            return
        if path == "/api/anonymous/create":
            if scenario == "rate_limited":
                wait = self._retry_after(30)
                STATE.consume_once("nas")
                self._send(429, {"error": "temporarily_unavailable",
                                 "error_description": "Too many anonymous account creations from this address. Try again later."},
                           headers={"Retry-After": str(wait)})
                return
            with STATE.lock:
                STATE.minted += 1
                n = STATE.minted
            self._send(201, {"user_id": f"nas_user:rehearsal-{n}", "org_id": f"nas_org:rehearsal-{n}",
                             "token": f"anon_rehearsal_{n:04d}_{secrets.token_hex(4)}", "idle_ttl_days": 14})
            return
        if path == "/api/anonymous/token":
            token = str(body.get("token") or "")
            if not token.startswith("anon_"):
                self._send(400, {"error": "invalid_request", "error_description": 'Body must be { token: "anon_…" }.'})
                return
            with STATE.lock:
                dead = token in STATE.dead_tokens
                if scenario == "dead_once" and not dead:
                    # One-shot by definition: this credential is gone for good, the next one works.
                    STATE.dead_tokens.add(token)
                    STATE.nas = "ok"
                    dead = True
            if dead:
                self._send(404, {"error": "unknown_token", "error_description": "No anonymous account matches this token."})
                return
            if scenario == "rate_limited":
                STATE.consume_once("nas")
                self._send(429, {"error": "temporarily_unavailable",
                                 "error_description": "Too many token exchanges. Try again later."},
                           headers={"Retry-After": str(self._retry_after(30))})
                return
            if scenario == "pow":
                STATE.consume_once("nas")
                self._send(428, {"error": "pow_required", "pow": {
                    "challenge": secrets.token_hex(16), "alg": "blake2s-hashcash-v1", "bits": 31, "count": 16,
                    "expires_in": 1800}})
                return
            if scenario == "locked":
                STATE.consume_once("nas")
                self._send(403, {"error": "account_locked"})
                return
            with STATE.lock:
                welcome = STATE.welcome_url
            payload: Dict[str, Any] = {
                "access_token": _jwt(sid=token[-8:]), "token_type": "Bearer", "expires_in": 900,
                "user_id": "nas_user:rehearsal", "org_id": "nas_org:rehearsal"}
            if welcome:
                payload["inference_base_url"] = welcome
            self._send(200, payload)
            return
        if path == "/api/anonymous/promotion-intent":
            if scenario == "signin_busy":
                STATE.consume_once("nas")
                self._send(429, {"error": "temporarily_unavailable",
                                 "error_description": "Too many promotion requests. Try again later."},
                           headers={"Retry-After": str(self._retry_after(45))})
                return
            if scenario == "signin_paused":
                STATE.consume_once("nas")
                self._send(503, {"error": "temporarily_disabled",
                                 "error_description": "The anonymous-account surface is currently switched off."})
                return
            if scenario == "signin_server_error":
                STATE.consume_once("nas")
                self._send(500, {"error": "internal_error"})
                return
            code = f"{secrets.token_hex(2).upper()}-{secrets.token_hex(2).upper()}"
            with STATE.lock:
                STATE.signin = {"status": "pending"}
            host = self.headers.get("Host") or "127.0.0.1"
            self._send(200, {"claim_code": code, "claim_url": f"http://{host}/__claim?code={code}",
                             "expires_in": 900, "interval": 2})
            return
        if path == "/api/anonymous/promotion-status":
            with STATE.lock:
                outcome = dict(STATE.signin)
            self._send(200, outcome if outcome.get("status") != "pending" else {"status": "pending"})
            return
        if path == "/api/anonymous/claim":
            self._send(401, {"error": "unauthorized"})
            return
        self._send(404, {"error": "not_found"})

    def _device_code(self, body: Dict[str, Any]) -> None:
        host = self.headers.get("Host") or "127.0.0.1"
        code = secrets.token_hex(8)
        self._send(200, {"device_code": f"dc_{code}", "user_code": f"{code[:4].upper()}-{code[4:8].upper()}",
                         "verification_uri": f"http://{host}/__claim",
                         "verification_uri_complete": f"http://{host}/__claim?device={code}",
                         "expires_in": 900, "interval": 2})

    def _oauth_token(self, body: Dict[str, Any]) -> None:
        # A completed sign-in is not something this stand-in can finish honestly (the real
        # portal issues signed tokens the agent then verifies), so the token poll stays pending.
        self._send(400, {"error": "authorization_pending",
                         "error_description": "The free tier fault server never completes a sign-in; rehearse the failure paths here and the happy path against staging."})

    # --- the welcome inference host --------------------------------------------------------------

    def _inference(self, body: Dict[str, Any]) -> None:
        with STATE.lock:
            scenario = STATE.inference
        if scenario == "timeout":
            STATE.consume_once("inference")
            time.sleep(120)
            self._send(504, {"status": 504, "message": "timed out"})
            return
        canned = INFERENCE_RESPONSES.get(scenario)
        if canned is None:
            self._reply(str(body.get("model") or WELCOME_MODEL), stream=bool(body.get("stream")))
            return
        status, payload, default_wait = canned
        wait = self._retry_after(default_wait)
        headers = {"Retry-After": str(wait)} if default_wait or "reason" in payload else {}
        if payload.get("reason") in ("rate_limited", "at_capacity"):
            headers["RateLimit-Policy"] = '"fairshare";q=0;qu="tokens";w=60'
            headers["RateLimit"] = f'"fairshare";r=0;t={wait}'
        if scenario == "bare_429":
            headers.update({"x-ratelimit-limit-requests": "30", "x-ratelimit-remaining-requests": "0",
                            "x-ratelimit-reset-requests": str(wait)})
        if "reason" in payload:
            payload = {**payload, "retry_after": wait}
        STATE.consume_once("inference")
        self._send(status, payload, headers=headers)

    def _reply(self, model: str, *, stream: bool) -> None:
        text = ("Hello from the free tier fault server. Everything is working; switch a scenario on "
                "POST /__scenario to rehearse a failure.")
        now = int(time.time())
        ident = f"chatcmpl-rehearsal-{secrets.token_hex(4)}"
        if not stream:
            self._send(200, {"id": ident, "object": "chat.completion", "created": now, "model": model,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                          "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 12, "completion_tokens": 24, "total_tokens": 36}})
            return
        chunks = []
        for i, piece in enumerate(text.split(" ")):
            delta = {"content": (" " if i else "") + piece}
            if i == 0:
                delta["role"] = "assistant"
            chunks.append({"id": ident, "object": "chat.completion.chunk", "created": now, "model": model,
                           "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
        chunks.append({"id": ident, "object": "chat.completion.chunk", "created": now, "model": model,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 12, "completion_tokens": 24, "total_tokens": 36}})
        raw = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        self._send(200, raw=raw.encode(), content_type="text/event-stream")


def make_server(host: str = "127.0.0.1", port: int = 8765, *, nas: str = "ok", inference: str = "ok",
                welcome_url: Optional[str] = None) -> ThreadingHTTPServer:
    """Build (not start) the server; tests bind port 0 and read ``server.server_address``."""
    global STATE
    STATE = State(nas=nas, inference=inference, welcome_url=welcome_url or "")
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    if not welcome_url:
        STATE.welcome_url = f"http://{server.server_address[0]}:{server.server_address[1]}/v1"
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--nas", default="ok", choices=sorted(NAS_SCENARIOS))
    parser.add_argument("--inference", default="ok", choices=sorted(INFERENCE_SCENARIOS))
    args = parser.parse_args()
    server = make_server(args.host, args.port, nas=args.nas, inference=args.inference)
    base = f"http://{args.host}:{args.port}"
    print(f"free tier fault server on {base}  (nas={args.nas}, inference={args.inference})")
    print("Point Hermes at it with:")
    print(f"  export HERMES_GUEST_ONBOARDING=1 HERMES_PORTAL_BASE_URL={base} "
          f"NOUS_INFERENCE_BASE_URL={base}/v1 HERMES_EXTRA_WELCOME_HOSTS={args.host}")
    print("Switch a scenario with:")
    print(f"  curl -s -X POST {base}/__scenario -d '{{\"inference\": \"rate_limited\"}}'")
    print(f"  fetch('{base}/__scenario', {{method: 'POST', body: JSON.stringify({{nas: 'paused'}})}})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
