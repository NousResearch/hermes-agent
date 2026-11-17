"""A loopback stand-in for the Nous Portal account read (``GET /api/oauth/account``).

The account body is configurable per test (``artifacts_access``, ``paid_service_access.paid_access``)
and every request is recorded with its bearer. A token refresh (``POST /api/oauth/token``) is
answered ``400 invalid_grant`` and recorded too, so a test seeding a long-lived access token can
assert that no refresh happened. Run it in-process with :class:`FakePortal` (a context manager);
``origin`` is the ``portal_base_url`` to seed into ``auth.json``.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

ACCOUNT_PATH = "/api/oauth/account"
TOKEN_PATH = "/api/oauth/token"


def account_body(*, artifacts_access: Any = True, paid_access: Any = True) -> dict[str, Any]:
    """An ``/api/oauth/account`` body for a signed-in user (``None`` omits a flag entirely)."""
    body: dict[str, Any] = {
        "user": {"id": "user-e2e", "email": "artifacts-e2e@example.invalid"},
        "organisation": {"id": "org-e2e", "slug": "e2e", "name": "E2E Org"},
        "paid_service_access": {"allowed": paid_access, "paid_access": paid_access, "organisation_id": "org-e2e"},
    }
    if artifacts_access is not None:
        body["artifacts_access"] = artifacts_access
    return body


@dataclass
class PortalRequest:
    method: str
    path: str
    bearer: str


@dataclass
class FakePortal:
    body: Optional[dict[str, Any]] = None
    requests: list[PortalRequest] = field(default_factory=list)
    _server: Optional[ThreadingHTTPServer] = None
    _thread: Optional[threading.Thread] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.body is None:
            self.body = account_body()

    @property
    def origin(self) -> str:
        assert self._server is not None, "FakePortal is not started"
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def account_reads(self) -> list[PortalRequest]:
        with self._lock:
            return [r for r in self.requests if r.method == "GET" and r.path == ACCOUNT_PATH]

    def refreshes(self) -> list[PortalRequest]:
        with self._lock:
            return [r for r in self.requests if r.path == TOKEN_PATH]

    def _record(self, method: str, path: str, headers: Any) -> None:
        auth = headers.get("Authorization", "") or ""
        bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        with self._lock:
            self.requests.append(PortalRequest(method, path, bearer))

    def __enter__(self) -> FakePortal:
        portal = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a: object) -> None:
                pass

            def _json(self, status: int, payload: Any) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                portal._record("GET", path, self.headers)
                if path == ACCOUNT_PATH:
                    self._json(200, portal.body)
                else:
                    self._json(404, {"error": "not_found"})

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                portal._record("POST", path, self.headers)
                self._json(400, {"error": "invalid_grant"} if path == TOKEN_PATH else {"error": "not_found"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="fake-portal", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=10)
