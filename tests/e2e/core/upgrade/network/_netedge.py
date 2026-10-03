"""The network edge for the hostile-network suites: every byte a sandboxed install/update sends
leaves through something this module owns.

* ``netns_usable()``: the sandbox gets its own network namespace (``bwrap --unshare-net``, only
  ``lo``), so there is no route, no DNS and no internet. Egress is what the test provides, and a
  cell that expected "nothing reached pypi.org" is proven by the namespace, not by trust.
* ``Bridge``: host-side TCP services (the proxy, a fake provider, a mirror) appear inside the
  namespace on the SAME loopback port. The host listens on a unix socket per port under the test's
  tmp dir; ``_nsbridge.py`` (the sandbox's first process) listens on ``127.0.0.1:<port>`` in the
  namespace and relays each connection to that socket.
* ``EdgeProxy``: an HTTP/HTTPS forward proxy (``CONNECT``) in front of named fake sites. For a
  routed host it terminates TLS with a leaf signed by the test's own root (a TLS-inspecting
  corporate proxy) and serves the site's app; every other host is refused with 403 and logged, so
  a cell can list every host the product tried to reach. Faults per host: ``eof`` (accept the
  tunnel, then drop it before TLS: the ``UNEXPECTED_EOF`` a cut direct connection shows) and
  optional proxy auth (407 without the right ``Proxy-Authorization``).
* Apps: ``git_app`` (git smart-HTTP via ``git http-backend``), ``static_app`` (fixed routes with
  scripted per-path fault sequences, e.g. 503 + Retry-After then 200).
"""

from __future__ import annotations

import base64
import datetime
import gzip
import http.server
import ipaddress
import os
import shutil
import socket
import socketserver
import ssl
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

from tests.e2e.core.upgrade import _helpers as H

_BRIDGE_SCRIPT = Path(__file__).with_name("_nsbridge.py")
SYSTEM_CA_BUNDLE = Path("/etc/ssl/certs/ca-certificates.crt")


def netns_usable() -> bool:
    """bwrap can give the sandbox a private network namespace with a working loopback."""
    if not H.BWRAP_OK:
        return False
    probe = "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); print('ok')"
    try:
        r = subprocess.run(
            [shutil.which("bwrap") or "bwrap", "--dev-bind", "/", "/", "--unshare-net", "--unshare-pid",
             "--proc", "/proc", "--die-with-parent", sys.executable, "-I", "-c", probe],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and r.stdout.strip() == "ok"


def netns_required_reason() -> str | None:
    if netns_usable():
        return None
    return "bwrap --unshare-net unavailable: the hostile-network cells need a private network namespace"


# ---------------------------------------------------------------------------
# Certificates: one test root, a leaf per routed host.
# ---------------------------------------------------------------------------

class TestCA:
    """A throwaway root CA (the "corporate TLS-inspection root") and leaves signed by it."""

    __test__ = False  # not a pytest class

    def __init__(self, directory: Path) -> None:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        self._x509, self._hashes, self._ser, self._ec, self._oid = x509, hashes, serialization, ec, NameOID
        self.dir = directory
        directory.mkdir(parents=True, exist_ok=True)
        self._key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Hermes E2E TLS Inspection Root")])
        now = datetime.datetime.now(datetime.timezone.utc)
        self._cert = (
            x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(self._key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=30))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                         content_commitment=False, key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(self._key.public_key()), critical=False)
            .sign(self._key, hashes.SHA256())
        )
        self.pem = self._cert.public_bytes(serialization.Encoding.PEM)
        self.cert_file = directory / "test-root.pem"
        self.cert_file.write_bytes(self.pem)
        self._contexts: dict[str, ssl.SSLContext] = {}
        self._lock = threading.Lock()

    def os_trust_store(self, dest: Path) -> list[tuple[Path, Path]]:
        """What ``update-ca-certificates`` leaves after an admin installs this root as the corporate
        root: ``/etc/ssl/certs`` with the root in ``ca-certificates.crt`` AND as a hashed link.
        Returns the ``ro_binds`` that overlay it on the sandbox's distro trust store."""
        certs = Path("/etc/ssl/certs")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(certs, dest, symlinks=True) if certs.is_dir() else dest.mkdir(parents=True)
        bundle = dest / "ca-certificates.crt"
        system = SYSTEM_CA_BUNDLE.read_bytes() if SYSTEM_CA_BUNDLE.is_file() else b""
        if bundle.is_symlink():
            bundle.unlink()
        bundle.write_bytes(system + b"\n" + self.pem)
        (dest / "hermes-e2e-root.pem").write_bytes(self.pem)
        openssl = shutil.which("openssl")
        if openssl:
            digest = subprocess.run([openssl, "x509", "-hash", "-noout", "-in", str(self.cert_file)],
                                    capture_output=True, text=True, check=True).stdout.strip()
            link = dest / f"{digest}.0"
            if not link.exists():
                link.symlink_to("hermes-e2e-root.pem")
        return [(dest, certs)]

    def server_context(self, host: str) -> ssl.SSLContext:
        with self._lock:
            ctx = self._contexts.get(host)
            if ctx is None:
                ctx = self._contexts[host] = self._leaf_context(host)
            return ctx

    def _leaf_context(self, host: str) -> ssl.SSLContext:
        x509, hashes, ser, ec, oid = self._x509, self._hashes, self._ser, self._ec, self._oid
        key = ec.generate_private_key(ec.SECP256R1())
        now = datetime.datetime.now(datetime.timezone.utc)
        try:
            san = x509.IPAddress(ipaddress.ip_address(host))
        except ValueError:
            san = x509.DNSName(host)
        cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(oid.COMMON_NAME, host)]))
            .issuer_name(self._cert.subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=7))
            .add_extension(x509.SubjectAlternativeName([san]), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(self._key.public_key()), critical=False)
            .sign(self._key, hashes.SHA256())
        )
        stem = self.dir / f"leaf-{host.replace(':', '_')}"
        cert_path, key_path = stem.with_suffix(".crt"), stem.with_suffix(".key")
        cert_path.write_bytes(cert.public_bytes(ser.Encoding.PEM) + self.pem)
        key_path.write_bytes(key.private_bytes(ser.Encoding.PEM, ser.PrivateFormat.PKCS8, ser.NoEncryption()))
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert_path), str(key_path))
        ctx.set_alpn_protocols(["http/1.1"])
        return ctx


# ---------------------------------------------------------------------------
# Apps: (request) -> Response. A request is what the site saw after TLS termination.
# ---------------------------------------------------------------------------

@dataclass
class Request:
    host: str
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


@dataclass
class Response:
    status: int
    body: bytes | Iterator[bytes] = b""
    headers: dict[str, str] = field(default_factory=dict)
    note: str = ""  # diagnostic for the edge log (e.g. a failing backend's stderr)


App = Callable[[Request], Response]


@dataclass
class Hit:
    host: str
    kind: str  # "request" | "refused" | "eof" | "auth-required" | "tls-rejected"
    method: str = ""
    path: str = ""
    status: int = 0
    note: str = ""

    def __str__(self) -> str:
        text = f"{self.kind} {self.host} {self.method} {self.path} -> {self.status}".strip()
        return f"{text}  [{self.note}]" if self.note else text


def static_app(routes: Mapping[str, bytes | Callable[[Request], Response]], *,
               faults: dict[str, list[Response]] | None = None,
               always: dict[str, Response] | None = None) -> App:
    """Exact-path routes (anything else 404s); a key ending in ``*`` matches every path with that
    prefix (``{"*": body}`` is a catch-all), checked after the exact routes. ``faults[path]`` is a queue of responses served
    before the real one, one per request (e.g. ``[Response(503, headers={"Retry-After": "1"})]``);
    ``always[path]`` replaces the route for every request (an outage that never recovers)."""
    queues = {k: list(v) for k, v in (faults or {}).items()}
    lock = threading.Lock()

    def app(req: Request) -> Response:
        path = req.path.split("?", 1)[0]
        if always and path in always:
            return always[path]
        with lock:
            queue = queues.get(path) or next(
                (q for k, q in queues.items() if k.endswith("*") and path.startswith(k[:-1]) and q), None)
            if queue:
                return queue.pop(0)
        route = routes.get(path)
        if route is None:
            route = next((r for k, r in routes.items() if k.endswith("*") and path.startswith(k[:-1])), None)
        if route is None:
            return Response(404, b"not found\n")
        if callable(route):
            return route(req)
        return Response(200, route, {"Content-Type": "application/octet-stream"})

    return app


def git_app(project_root: Path, *, faults: list[Response] | None = None) -> App:
    """git smart-HTTP (fetch/clone) over ``git http-backend`` for every bare repo under
    ``project_root`` (``/<Owner>/<repo>.git/...`` maps to ``<project_root>/<Owner>/<repo>.git``).
    ``faults`` are served first, one per request (a rate-limited or erroring forge)."""
    git = shutil.which("git") or "git"
    queue = list(faults or [])
    lock = threading.Lock()

    def app(req: Request) -> Response:
        with lock:
            if queue:
                return queue.pop(0)
        path, _, query = req.path.partition("?")
        body = req.body
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "GIT_PROJECT_ROOT": str(project_root), "GIT_HTTP_EXPORT_ALL": "1",
            "PATH_INFO": path, "QUERY_STRING": query, "REQUEST_METHOD": req.method,
            "CONTENT_TYPE": req.headers.get("content-type", ""), "CONTENT_LENGTH": str(len(body)),
            "GIT_PROTOCOL": req.headers.get("git-protocol", ""), "REMOTE_ADDR": "127.0.0.1",
            "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(project_root),
        }
        if req.headers.get("content-encoding", "").lower() == "gzip":
            body = gzip.decompress(body)
        cp = subprocess.run([git, "http-backend"], input=body, env=env, capture_output=True, timeout=300)
        head, _, payload = cp.stdout.partition(b"\r\n\r\n")
        if not _:
            head, _, payload = cp.stdout.partition(b"\n\n")
        status, headers = 200, {}
        for line in head.decode("latin-1").splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            if k.lower() == "status":
                status = int(v.strip().split()[0])
            else:
                headers[k.strip()] = v.strip()
        note = ""
        if cp.returncode != 0:
            # Surfaced in the edge log. ``unable to read <sha>`` means the bare origin (a ``--shared``
            # clone of the developer checkout) lacks an object: the checkout is a blob-filtered
            # partial clone. CI checkouts are full clones; locally, fetch the missing objects.
            status = 500 if status == 200 else status
            note = f"git http-backend rc={cp.returncode}: " + cp.stderr.decode("utf-8", "replace").strip()[-400:]
        return Response(status, payload, headers, note)

    return app


# ---------------------------------------------------------------------------
# Serving an app on a socket (TLS-terminated tunnel or plain HTTP).
# ---------------------------------------------------------------------------

class _SiteHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    app: App
    site_host: str
    hits: list[Hit]

    def log_message(self, format, *args) -> None:  # noqa: A002 - quiet
        pass

    def _read_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            chunks = []
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return b"".join(chunks)
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _dispatch(self) -> None:
        path = self.path
        if path.startswith(("http://", "https://")):  # absolute-form through a plain proxy
            path = "/" + path.split("://", 1)[1].partition("/")[2]
        req = Request(self.site_host, self.command, path,
                      {k.lower(): v for k, v in self.headers.items()}, self._read_body())
        try:
            resp = self.app(req)
        except Exception as exc:  # a broken fake must be visible, not a hang
            resp = Response(500, f"fake site error: {exc!r}\n".encode())
        self.hits.append(Hit(self.site_host, "request", self.command, path, resp.status, resp.note))
        self.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                self.send_header(k, v)
        if isinstance(resp.body, bytes):
            self.send_header("Content-Length", str(len(resp.body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(resp.body)
        else:
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in resp.body:
                self.wfile.write(chunk)
            self.close_connection = True

    do_GET = do_POST = do_HEAD = do_PUT = _dispatch


def _serve_app(sock: socket.socket, addr, host: str, app: App, hits: list[Hit]) -> None:
    handler = type("SiteHandler", (_SiteHandler,), {"app": staticmethod(app), "site_host": host, "hits": hits})
    try:
        handler(sock, addr, None)  # type: ignore[arg-type]
    except (OSError, ssl.SSLError):
        pass


class HttpSite:
    """A plain-HTTP server for one app on 127.0.0.1 (a mirror reached without the proxy)."""

    def __init__(self, app: App, name: str = "127.0.0.1") -> None:
        self.hits: list[Hit] = []
        self.name = name
        outer = self

        class _S(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True

            def finish_request(self, request, client_address):
                _serve_app(request, client_address, outer.name, app, outer.hits)  # type: ignore[arg-type]

        self._srv = _S(("127.0.0.1", 0), socketserver.BaseRequestHandler)
        self.port = self._srv.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()


class EdgeProxy:
    """HTTP(S) forward proxy with TLS inspection for routed hosts; refuses (and logs) the rest."""

    def __init__(self, ca: TestCA, routes: dict[str, App] | None = None, *,
                 eof_hosts: Iterable[str] = (), auth: tuple[str, str] | None = None) -> None:
        self.ca = ca
        self.routes: dict[str, App] = dict(routes or {})
        self.eof_hosts = set(eof_hosts)
        self.auth = auth
        self.hits: list[Hit] = []
        outer = self

        class _S(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True
            request_queue_size = 128

            def finish_request(self, request, client_address):
                outer._handle(request, client_address)  # type: ignore[arg-type]

        self._srv = _S(("127.0.0.1", 0), socketserver.BaseRequestHandler)
        self.port = self._srv.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def url_with_auth(self) -> str:
        assert self.auth is not None
        return f"http://{self.auth[0]}:{self.auth[1]}@127.0.0.1:{self.port}"

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()

    # -- queries ------------------------------------------------------------
    def hosts(self, kind: str | None = None) -> set[str]:
        return {h.host for h in list(self.hits) if kind is None or h.kind == kind}

    def requests(self, host: str) -> list[Hit]:
        return [h for h in list(self.hits) if h.host == host and h.kind == "request"]

    def transcript(self, limit: int = 60) -> str:
        rows = [str(h) for h in list(self.hits)]
        return "\n".join(rows[-limit:]) or "(proxy saw no traffic)"

    # -- protocol -----------------------------------------------------------
    def _authorized(self, headers: dict[str, str]) -> bool:
        if self.auth is None:
            return True
        want = "Basic " + base64.b64encode(f"{self.auth[0]}:{self.auth[1]}".encode()).decode()
        return headers.get("proxy-authorization", "") == want

    def _handle(self, sock: socket.socket, addr) -> None:
        try:
            sock.settimeout(120)
            rfile = sock.makefile("rb")
            line = rfile.readline(65536).decode("latin-1").strip()
            if not line:
                return
            headers: dict[str, str] = {}
            while True:
                h = rfile.readline(65536).decode("latin-1")
                if h in ("\r\n", "\n", ""):
                    break
                k, _, v = h.partition(":")
                headers[k.strip().lower()] = v.strip()
            method, target, _ = (line.split(" ", 2) + ["", ""])[:3]
            if method == "CONNECT":
                host = target.rsplit(":", 1)[0].strip("[]")
            else:
                host = target.split("://", 1)[-1].split("/", 1)[0].rsplit(":", 1)[0]
            if not self._authorized(headers):
                self.hits.append(Hit(host, "auth-required", method, target, 407))
                sock.sendall(b"HTTP/1.1 407 Proxy Authentication Required\r\n"
                             b"Proxy-Authenticate: Basic realm=\"corp\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                return
            if host in self.eof_hosts:
                self.hits.append(Hit(host, "eof", method, target, 0))
                if method == "CONNECT":
                    sock.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                return  # drop the tunnel before any TLS bytes: UNEXPECTED_EOF at the client
            app = self.routes.get(host)
            if app is None:
                self.hits.append(Hit(host, "refused", method, target, 403))
                sock.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                return
            if method != "CONNECT":
                # Every fake site is HTTPS; a plain-HTTP request through the proxy is a product
                # talking cleartext to a host that should be TLS. Refuse it visibly.
                self.hits.append(Hit(host, "refused", method, target, 403))
                sock.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                return
            sock.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            try:
                tls = self.ca.server_context(host).wrap_socket(sock, server_side=True)
            except (OSError, ssl.SSLError) as exc:
                # The client rejected the inspected certificate (or hung up): its trust store
                # does not hold the corporate root.
                self.hits.append(Hit(host, "tls-rejected", method, str(exc)[:120], 0))
                return
            _serve_app(tls, addr, host, app, self.hits)
        except (OSError, ssl.SSLError, ValueError):
            pass
        finally:
            try:
                sock.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Bridge: host TCP services -> the sandbox's private loopback, same port numbers.
# ---------------------------------------------------------------------------

class Bridge:
    """Expose host ``127.0.0.1:<port>`` services inside the sandbox namespace at the same port."""

    def __init__(self, directory: Path, ports: Iterable[int]) -> None:
        self.dir = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.ports = sorted(set(ports))
        # Unix socket paths are capped at 108 bytes; pytest tmp paths are longer. Bind through a
        # held directory fd (``/proc/self/fd/N/<port>``) on both sides.
        self._dirfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        self._servers: list[socket.socket] = []
        self._stop = threading.Event()
        for port in self.ports:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.bind(f"/proc/self/fd/{self._dirfd}/{port}")
            s.listen(128)
            self._servers.append(s)
            threading.Thread(target=self._accept, args=(s, port), daemon=True).start()

    def _accept(self, server: socket.socket, port: int) -> None:
        while not self._stop.is_set():
            try:
                client, _ = server.accept()
            except OSError:
                return
            threading.Thread(target=self._relay, args=(client, port), daemon=True).start()

    @staticmethod
    def _relay(client: socket.socket, port: int) -> None:
        try:
            upstream = socket.create_connection(("127.0.0.1", port), timeout=30)
        except OSError:
            client.close()
            return
        upstream.settimeout(None)
        _pipe_pair(client, upstream)

    def wrap(self, argv: list[str]) -> list[str]:
        """``argv`` preceded by the in-namespace relay (exits with argv's status)."""
        spec = ",".join(str(p) for p in self.ports)
        return [sys.executable, "-I", str(_BRIDGE_SCRIPT), str(self.dir), spec, "--", *argv]

    def close(self) -> None:
        self._stop.set()
        for s in self._servers:
            try:
                s.close()
            except OSError:
                pass
        try:
            os.close(self._dirfd)
        except OSError:
            pass


def _pipe_pair(a: socket.socket, b: socket.socket) -> None:
    def pump(src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    t = threading.Thread(target=pump, args=(b, a), daemon=True)
    t.start()
    pump(a, b)
    t.join(timeout=600)
    for s in (a, b):
        try:
            s.close()
        except OSError:
            pass


def proxy_env(proxy_url: str, *, no_proxy: str = "127.0.0.1,localhost,::1") -> dict[str, str]:
    """Every spelling of the proxy variables that curl, git, uv, npm, node and Python read."""
    return {
        "HTTPS_PROXY": proxy_url, "https_proxy": proxy_url,
        "HTTP_PROXY": proxy_url, "http_proxy": proxy_url,
        "NO_PROXY": no_proxy, "no_proxy": no_proxy,
    }


def wait_port_free(port: int, timeout: float = 5.0) -> None:  # pragma: no cover - diagnostics helper
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(0.1)
