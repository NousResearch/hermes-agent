"""Recording loopback OAuth 2.0 authorization server for end-to-end tests.

One real HTTP server on 127.0.0.1 that plays the vendor side of two OAuth
flows Hermes drives but does not own:

* **Refresh grant with single-use rotating refresh tokens.** Every successful
  ``grant_type=refresh_token`` spends the presented token and issues a fresh
  pair; presenting a spent (or unknown) token answers ``400 invalid_grant``
  exactly like a vendor that detects refresh-token reuse. Every grant is
  recorded (:attr:`OAuthTokenServer.grants`) so a test can count refreshes and
  prove which token each one spent.
* **RFC 8628 device authorization grant.** ``POST /api/oauth/device/code``
  returns a device code with a configurable ``interval``; the token endpoint
  answers ``authorization_pending`` / ``slow_down`` from a script before
  approving, recording the arrival time of every poll so a test can assert the
  client honoured ``interval`` and the ``slow_down`` +5 s back-off (§3.5).

Both the Anthropic (``/v1/oauth/token``) and Nous Portal (``/api/oauth/token``)
paths are served; request bodies may be form-encoded or JSON.

:class:`TLSInterceptProxy` lets a client whose token URL is a hardcoded
``https://<vendor host>`` reach this server with no product override: it is an
``HTTPS_PROXY`` that terminates CONNECT tunnels for an allowlist of hosts with a
leaf certificate signed by a throwaway test CA (trust it via ``SSL_CERT_FILE``)
and serves the decrypted request with the same handler. CONNECTs to any other
host are refused (403) and recorded, so the child process cannot reach the real
internet through it.
"""

from __future__ import annotations

import base64
import datetime as _dt
import json
import secrets
import select
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs

DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
TOKEN_PATHS = frozenset({"/v1/oauth/token", "/api/oauth/token"})
DEVICE_CODE_PATH = "/api/oauth/device/code"


def unsigned_jwt(claims: dict[str, Any]) -> str:
    """An ``alg: none`` JWT (clients that only *decode* claims accept it)."""
    def seg(obj: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")
    return f"{seg({'alg': 'none', 'typ': 'JWT'})}.{seg(claims)}.sig"


@dataclass
class Grant:
    """One token-endpoint request as the server saw it."""

    at: float  # time.monotonic() at arrival
    grant_type: str
    presented: str  # refresh token or device code
    status: int
    error: str | None = None
    issued_refresh_token: str | None = None
    issued_access_token: str | None = None
    host: str = ""
    path: str = ""
    content_type: str = ""


@dataclass
class DeviceFlow:
    """Scripted device-code session: one entry per token poll, last one repeats.

    Entries are ``"authorization_pending"``, ``"slow_down"`` or ``"approve"``.
    """

    interval: int = 5
    expires_in: int = 600
    script: list[str] = field(default_factory=lambda: ["approve"])
    device_code: str = field(default_factory=lambda: f"dc_{secrets.token_hex(8)}")
    user_code: str = "FAKE-CODE"
    polls: list[float] = field(default_factory=list)  # monotonic arrival times


class OAuthTokenServer:
    """Threaded loopback OAuth server; see the module docstring for the contract."""

    def __init__(
        self,
        *,
        access_token_factory: Callable[[int], str] | None = None,
        refresh_token_prefix: str = "rt-fake",
        expires_in: int = 3600,
        extra_token_fields: dict[str, Any] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._counter = 0
        self._access_factory = access_token_factory or (lambda n: f"sk-ant-oat01-fake-{n}")
        self._rt_prefix = refresh_token_prefix
        self.expires_in = expires_in
        self.extra_token_fields = dict(extra_token_fields or {})
        self.live: set[str] = set()  # unspent refresh tokens
        self.spent: set[str] = set()
        self.grants: list[Grant] = []
        self.device: DeviceFlow | None = None
        self.device_requests = 0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ---- state -----------------------------------------------------------

    def seed_refresh_token(self, token: str | None = None) -> str:
        """Register a live refresh token (the one a fixture writes to disk)."""
        token = token or f"{self._rt_prefix}-seed-{secrets.token_hex(4)}"
        with self._lock:
            self.live.add(token)
        return token

    def is_live(self, token: str) -> bool:
        with self._lock:
            return token in self.live

    def refresh_grants(self) -> list[Grant]:
        with self._lock:
            return [g for g in self.grants if g.grant_type == "refresh_token"]

    def start_device_flow(self, **kwargs: Any) -> DeviceFlow:
        self.device = DeviceFlow(**kwargs)
        return self.device

    def _issue_pair(self) -> tuple[str, str]:
        self._counter += 1
        n = self._counter
        refresh = f"{self._rt_prefix}-{n}-{secrets.token_hex(4)}"
        self.live.add(refresh)
        return self._access_factory(n), refresh

    def _token_body(self, access: str, refresh: str) -> dict[str, Any]:
        return {"access_token": access, "refresh_token": refresh, "token_type": "Bearer",
                "expires_in": self.expires_in, **self.extra_token_fields}

    # ---- grant handlers (called with the parsed form) ---------------------

    def handle_token(self, form: dict[str, str], meta: dict[str, str]) -> tuple[int, dict[str, Any]]:
        grant_type = form.get("grant_type", "")
        handler = {"refresh_token": self._refresh, DEVICE_CODE_GRANT: self._device_poll}.get(grant_type)
        if handler is None:
            return 400, {"error": "unsupported_grant_type"}
        return handler(form, meta)

    def _refresh(self, form: dict[str, str], meta: dict[str, str]) -> tuple[int, dict[str, Any]]:
        presented = form.get("refresh_token", "")
        with self._lock:
            grant = Grant(at=time.monotonic(), grant_type="refresh_token", presented=presented, status=200, **meta)
            self.grants.append(grant)
            if presented not in self.live:
                grant.status, grant.error = 400, "invalid_grant"
                why = "refresh token already used" if presented in self.spent else "unknown refresh token"
                return 400, {"error": "invalid_grant", "error_description": why}
            self.live.discard(presented)
            self.spent.add(presented)
            access, refresh = self._issue_pair()
            grant.issued_access_token, grant.issued_refresh_token = access, refresh
        return 200, self._token_body(access, refresh)

    def _device_poll(self, form: dict[str, str], meta: dict[str, str]) -> tuple[int, dict[str, Any]]:
        flow = self.device
        now = time.monotonic()
        with self._lock:
            grant = Grant(at=now, grant_type=DEVICE_CODE_GRANT, presented=form.get("device_code", ""),
                          status=400, **meta)
            self.grants.append(grant)
            if flow is None or grant.presented != flow.device_code:
                grant.error = "invalid_grant"
                return 400, {"error": "invalid_grant"}
            flow.polls.append(now)
            step = flow.script[min(len(flow.polls) - 1, len(flow.script) - 1)]
            if step != "approve":
                grant.error = step
                return 400, {"error": step}
            access, refresh = self._issue_pair()
            grant.status = 200
            grant.issued_access_token, grant.issued_refresh_token = access, refresh
        return 200, self._token_body(access, refresh)

    def handle_device_code(self, form: dict[str, str]) -> tuple[int, dict[str, Any]]:
        flow = self.device
        with self._lock:
            self.device_requests += 1
        if flow is None:
            return 400, {"error": "invalid_request", "error_description": "no device flow armed"}
        base = f"{self.base_url}/device"
        return 200, {
            "device_code": flow.device_code, "user_code": flow.user_code,
            "verification_uri": base, "verification_uri_complete": f"{base}?user_code={flow.user_code}",
            "expires_in": flow.expires_in, "interval": flow.interval,
        }

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> "OAuthTokenServer":
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self))
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05},
                                        name="fake-oauth", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    @property
    def port(self) -> int:
        assert self._httpd is not None, "server not started"
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def handler_class(self) -> type[BaseHTTPRequestHandler]:
        return _make_handler(self)

    def __enter__(self) -> "OAuthTokenServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _parse_form(raw: bytes, content_type: str) -> dict[str, str]:
    if "json" in content_type:
        data = json.loads(raw.decode() or "{}")
        return {str(k): str(v) for k, v in data.items()}
    return {k: v[0] for k, v in parse_qs(raw.decode()).items()}


def _make_handler(server: OAuthTokenServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            ctype = self.headers.get("Content-Type", "")
            path = self.path.split("?", 1)[0]
            try:
                form = _parse_form(raw, ctype)
            except ValueError:
                return self._send(400, {"error": "invalid_request"})
            meta = {"host": self.headers.get("Host", ""), "path": path, "content_type": ctype}
            if path in TOKEN_PATHS:
                return self._send(*server.handle_token(form, meta))
            if path == DEVICE_CODE_PATH:
                return self._send(*server.handle_device_code(form))
            self._send(404, {"error": "not_found"})

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return Handler


# ---- TLS interception ------------------------------------------------------


@dataclass
class TestCA:
    """A throwaway CA plus one leaf certificate covering ``hosts``."""

    __test__ = False  # not a pytest class

    ca_pem: Path
    leaf_pem: Path
    leaf_key: Path


def make_test_ca(directory: Path, hosts: Iterable[str]) -> TestCA:
    """Generate a CA and a leaf cert for ``hosts`` (SAN) under ``directory``."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    directory.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "hermes e2e test CA")])
    ca_cert = (
        x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
        .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(minutes=5)).not_valid_after(now + _dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                     content_commitment=False, key_encipherment=False,
                                     data_encipherment=False, key_agreement=False,
                                     encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    hosts = list(hosts)
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hosts[0])]))
        .issuer_name(ca_name).public_key(leaf_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(minutes=5)).not_valid_after(now + _dt.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(h) for h in hosts]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    pem = serialization.Encoding.PEM
    out = TestCA(ca_pem=directory / "ca.pem", leaf_pem=directory / "leaf.pem", leaf_key=directory / "leaf.key")
    out.ca_pem.write_bytes(ca_cert.public_bytes(pem))
    out.leaf_pem.write_bytes(leaf.public_bytes(pem))
    out.leaf_key.write_bytes(leaf_key.private_bytes(
        pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return out


class TLSInterceptProxy:
    """``HTTPS_PROXY`` that serves CONNECTs to ``hosts`` with the OAuth server's handler."""

    def __init__(self, oauth: OAuthTokenServer, ca: TestCA, hosts: Iterable[str]) -> None:
        self.oauth = oauth
        self.hosts = frozenset(h.lower() for h in hosts)
        self.connects: list[str] = []
        self.refused: list[str] = []
        self._ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._ctx.load_cert_chain(str(ca.leaf_pem), str(ca.leaf_key))
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._sock is not None, "proxy not started"
        return f"http://127.0.0.1:{self._sock.getsockname()[1]}"

    def start(self) -> "TLSInterceptProxy":
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(64)
        self._thread = threading.Thread(target=self._accept_loop, name="fake-oauth-proxy", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _accept_loop(self) -> None:
        sock = self._sock
        while not self._stop.is_set() and sock is not None:
            try:
                ready, _, _ = select.select([sock], [], [], 0.1)
                if not ready:
                    continue
                conn, addr = sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn, addr), daemon=True).start()

    def _serve(self, conn: socket.socket, addr: Any) -> None:
        try:
            conn.settimeout(30)
            head = b""
            while b"\r\n\r\n" not in head and len(head) < 65536:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                head += chunk
            request_line = head.split(b"\r\n", 1)[0].decode("latin-1")
            method, target, _ = (request_line.split(" ") + ["", "", ""])[:3]
            host = target.rsplit(":", 1)[0].lower()
            if method != "CONNECT" or host not in self.hosts:
                self.refused.append(request_line)
                conn.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                return
            self.connects.append(host)
            conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            tls = self._ctx.wrap_socket(conn, server_side=True)
            self.oauth.handler_class()(tls, addr, None)  # type: ignore[arg-type]
        except (OSError, ssl.SSLError):
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def __enter__(self) -> "TLSInterceptProxy":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
