"""A thin HTTP transport for the Control API.

Built on the standard library so the platform layer keeps its zero-dependency surface.
The handlers in :mod:`nova.control.api` hold all the behaviour; this module only moves
bytes, which is what makes swapping in a production server later a contained change
rather than a rewrite.

Safety properties, all tested:

**Writes are a declared surface, not an open door.** POST is served only for the routes in
``auth.WRITE_ROUTES``; PUT, PATCH and DELETE are refused with 405 before any handler runs.
An undeclared write path is a 404, not an admin-only 403 — forgetting to declare a read
exposes data, and forgetting to declare a write hands out an action.

**Cross-site requests cannot act.** A loopback caller is a local admin, which was harmless
while everything was a read and is not harmless now: any page in the operator's browser can
make their browser POST to ``127.0.0.1``. Three things stop it, and each blocks a different
technique — a JSON content type (an HTML form cannot send one), a same-origin check on any
``Origin`` header, and no CORS preflight answer at all (so a scripted cross-origin fetch
never gets to send the real request).

**No accidental exposure.** Binding to anything other than loopback requires both
authentication and transport security. A control plane that lists a customer's agents and
work must not become world-readable because someone passed ``--host 0.0.0.0`` while
debugging, and a bearer token crossing a VPC in cleartext is not much better than no token.

**Every caller is a principal.** Authentication resolves a named :class:`~nova.control.auth.Principal`
with a role, and each route declares the minimum role it needs, so the access log records
*who* read what rather than only that someone did.
"""

from __future__ import annotations

import hmac
import json
import logging
import mimetypes
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from nova.control.api import ControlAPI
from nova.control.auth import LOCAL_ADMIN, API_PREFIX_LEN, Principal, PrincipalStore
from nova.errors import NovaError

#: Largest write body accepted. A decision is a few hundred bytes; a note that needs more
#: than this is a document, and belongs in the knowledge base rather than a comment field.
#: Checked from the header before anything is read, so an oversized body costs one refusal
#: rather than a buffer.
MAX_BODY_BYTES = 64 * 1024

logger = logging.getLogger("nova.control")

STATIC_DIR = Path(__file__).parent / "static"
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

#: Served on every response. The dashboard is same-origin and needs nothing exotic; a
#: strict policy here keeps an injected string in a task title from becoming script.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "Cache-Control": "no-store",
}


class _Handler(BaseHTTPRequestHandler):
    server_version = "nova-control"
    sys_version = ""

    # Injected by :func:`serve`.
    api: ControlAPI
    principals: PrincipalStore = PrincipalStore()
    #: True when TLS is terminated by a proxy in front of this server. Only affects what
    #: the operator was required to acknowledge; the socket here is plain either way.
    behind_tls_proxy: bool = False

    # -- plumbing -------------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 — stdlib signature
        logger.info("%s %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in SECURITY_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload) -> None:
        self._send(status, json.dumps(payload, indent=2).encode("utf-8"), "application/json")

    def _is_loopback_client(self) -> bool:
        """Whether this request came from this host.

        **Not trusted when a TLS proxy sits in front.** A proxy terminating TLS on the same
        machine makes every forwarded request look local, which would turn loopback trust
        into "anyone on the internet is a local admin". The operator states the proxy is
        there with ``--behind-tls-proxy``, and that statement disables this.
        """
        if self.behind_tls_proxy:
            return False
        return (self.client_address[0] if self.client_address else "") in LOOPBACK_HOSTS

    def _principal(self) -> Optional[Principal]:
        """The authenticated caller, or None.

        A loopback caller is a local admin, whether or not principals are configured. That
        is not a shortcut: someone on this host can already read the principals file, the
        tenant bundle, the audit log and every profile directory straight off disk.
        Demanding a bearer token from them protects nothing and would break the dashboard,
        which is a browser and cannot send one.

        So principals gate *remote* access, which is what they are for. The distinction is
        recorded in the access log — ``via=loopback`` versus a principal's name — so an
        operator can always tell how a request was authorised.
        """
        if self._is_loopback_client():
            return LOCAL_ADMIN
        if not self.principals.configured:
            return None
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not header.startswith(prefix):
            return None
        return self.principals.authenticate(header[len(prefix) :])

    # -- methods --------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 — stdlib signature
        self._handle()

    def do_HEAD(self) -> None:  # noqa: N802 — stdlib signature
        self._handle()

    def _reject_method(self) -> None:
        self._send_json(
            405,
            {
                "error": {
                    "status": 405,
                    "message": "only GET, HEAD and POST are served by the control API",
                }
            },
        )

    do_PUT = do_PATCH = do_DELETE = _reject_method  # noqa: N815 — stdlib names

    def do_OPTIONS(self) -> None:  # noqa: N802 — stdlib signature
        """Deliberately unhelpful.

        Answering a CORS preflight is what would let a page on another origin send a real
        cross-site POST. There is no browser client for this API other than the dashboard,
        which is same-origin and needs no preflight, so the correct answer is 405.
        """
        self._reject_method()

    def _cross_site(self) -> str:
        """Why this request looks cross-site, or "" if it does not.

        Only the ``Origin`` header is consulted, and only when present. ``Referer`` is
        stripped by privacy tooling often enough that requiring it would break real
        operators, and a missing ``Origin`` on a same-origin non-form request is normal.
        The content-type requirement below is what covers the form case, where ``Origin``
        is sent but the request is one a form could have made.
        """
        origin = self.headers.get("Origin", "")
        if not origin:
            return ""
        host = self.headers.get("Host", "")
        for scheme in ("http://", "https://"):
            if origin == f"{scheme}{host}":
                return ""
        return f"Origin {origin!r} does not match Host {host!r}"

    def do_POST(self) -> None:  # noqa: N802 — stdlib signature
        parsed = urlparse(self.path)
        path = parsed.path

        if not path.startswith("/platform/"):
            self._send_json(404, {"error": {"status": 404, "message": "not found"}})
            return

        principal = self._principal()
        if principal is None:
            self._send_json(
                401, {"error": {"status": 401, "message": "missing or invalid token"}}
            )
            return

        reason = self._cross_site()
        if reason:
            logger.warning("refused cross-site write from %s: %s", principal.name, reason)
            self._send_json(
                403,
                {
                    "error": {
                        "status": 403,
                        "message": f"refusing a cross-site write: {reason}",
                    }
                },
            )
            return

        # An HTML form can only send urlencoded, multipart or text/plain, so requiring JSON
        # is what stops a form on another page from driving this API through the operator's
        # own browser and their own loopback admin rights.
        content_type = (self.headers.get("Content-Type", "").split(";")[0] or "").strip()
        if content_type != "application/json":
            self._send_json(
                415,
                {
                    "error": {
                        "status": 415,
                        "message": "writes must be sent as application/json",
                    }
                },
            )
            return

        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            length = -1
        if length < 0:
            self._send_json(
                400, {"error": {"status": 400, "message": "a malformed Content-Length"}}
            )
            return
        if length > MAX_BODY_BYTES:
            self._send_json(
                413,
                {
                    "error": {
                        "status": 413,
                        "message": f"a write body may not exceed {MAX_BODY_BYTES} bytes",
                    }
                },
            )
            return

        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(
                400, {"error": {"status": 400, "message": f"body is not valid JSON: {exc}"}}
            )
            return
        if not isinstance(payload, dict):
            self._send_json(
                400, {"error": {"status": 400, "message": "body must be a JSON object"}}
            )
            return

        logger.info("%s POST %s", principal.name, path)
        try:
            response = self.api.write(path, principal, payload)
        except NovaError as exc:
            self._send_json(500, {"error": {"status": 500, "message": str(exc)}})
            return
        self._send_json(response.status, response.body)

    def _handle(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        principal = self._principal()
        if principal is None:
            self._send_json(
                401, {"error": {"status": 401, "message": "missing or invalid token"}}
            )
            return

        if path.startswith("/platform/") and not principal.may(path[API_PREFIX_LEN:] or "/"):
            # 403, not 404: the caller is authenticated and the route exists. Pretending
            # otherwise would make a permissions problem look like a bug and send an
            # operator debugging the wrong thing.
            logger.warning(
                "denied %s -> %s (role=%s)", principal.name, path, principal.role
            )
            self._send_json(
                403,
                {
                    "error": {
                        "status": 403,
                        "message": f"role {principal.role!r} may not read this route",
                    }
                },
            )
            return

        logger.info("%s %s %s", principal.name, self.command, path)

        if path.startswith("/platform/"):
            query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
            try:
                response = self.api.handle(path, query)
            except NovaError as exc:
                self._send_json(500, {"error": {"status": 500, "message": str(exc)}})
                return
            if response.raw is not None:
                self._send(
                    response.status,
                    response.raw,
                    response.content_type or "application/octet-stream",
                )
                return
            self._send_json(response.status, response.body)
            return

        self._serve_static(path)

    def _serve_static(self, path: str) -> None:
        """Serve the dashboard. Paths are resolved and confined to the static directory."""
        if path == "/favicon.ico":
            # Browsers request this unprompted. A tenant favicon is applied by the page
            # itself from identity; answering 204 keeps the access log honest.
            self._send(204, b"", "image/x-icon")
            return
        relative = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_json(404, {"error": {"status": 404, "message": "not found"}})
            return
        if not target.is_file():
            self._send_json(404, {"error": {"status": 404, "message": "not found"}})
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type += "; charset=utf-8"
        self._send(200, target.read_bytes(), content_type)


def build_server(
    api: ControlAPI,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    principals: Optional[PrincipalStore] = None,
    tls_certfile: str = "",
    tls_keyfile: str = "",
    behind_tls_proxy: bool = False,
) -> ThreadingHTTPServer:
    """Construct the server without starting it. Refuses unsafe binds.

    Two refusals, both before the socket accepts anything:

    **No anonymous exposure.** A non-loopback bind needs a principals file. Without one
    every caller would be the local admin, which is correct on loopback and catastrophic on
    an interface.

    **No cleartext exposure.** A non-loopback bind needs TLS — either terminated here with
    a certificate, or terminated by a proxy the operator explicitly says is in front. The
    acknowledgement is required rather than assumed because "there is probably a load
    balancer" is exactly the assumption that ships a bearer token in cleartext across a
    VPC.
    """
    principals = principals or PrincipalStore()

    if host not in LOOPBACK_HOSTS:
        if not principals.configured:
            raise NovaError(
                f"refusing to bind {host} with no principals file: every caller would be "
                "treated as a local admin. Create one with `nova token new`, or bind "
                "127.0.0.1 and use an SSH tunnel"
            )
        if not tls_certfile and not behind_tls_proxy:
            raise NovaError(
                f"refusing to bind {host} without TLS: bearer tokens and this tenant's "
                "agents, work and policy would cross the network in cleartext. Pass "
                "--tls-cert/--tls-key, or --behind-tls-proxy if TLS terminates at a load "
                "balancer in front of this process"
            )

    handler = type(
        "_BoundHandler",
        (_Handler,),
        {"api": api, "principals": principals, "behind_tls_proxy": behind_tls_proxy},
    )
    server = ThreadingHTTPServer((host, port), handler)

    if tls_certfile:
        import ssl

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        # TLS 1.2 floor: 1.0 and 1.1 are deprecated and a control plane is new enough to
        # have no legacy client to accommodate.
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        try:
            context.load_cert_chain(tls_certfile, tls_keyfile or None)
        except (OSError, ssl.SSLError) as exc:
            server.server_close()
            raise NovaError(f"could not load the TLS certificate: {exc}") from exc
        server.socket = context.wrap_socket(server.socket, server_side=True)

    return server


def serve(
    api: ControlAPI,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    principals: Optional[PrincipalStore] = None,
    tls_certfile: str = "",
    tls_keyfile: str = "",
    behind_tls_proxy: bool = False,
    ready: Optional[callable] = None,
) -> None:
    """Serve until interrupted. ``ready`` is called with the bound address once listening."""
    server = build_server(
        api,
        host=host,
        port=port,
        principals=principals,
        tls_certfile=tls_certfile,
        tls_keyfile=tls_keyfile,
        behind_tls_proxy=behind_tls_proxy,
    )
    bound_host, bound_port = server.server_address[:2]
    if ready is not None:
        scheme = "https" if tls_certfile else "http"
        ready(f"{scheme}://{bound_host}:{bound_port}/")
    restore = _install_shutdown_handlers(server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        restore()
        server.server_close()


def _install_shutdown_handlers(server: ThreadingHTTPServer):
    """Stop serving on SIGTERM, and return a callable that undoes this.

    Ctrl-C was already handled — ``serve_forever`` raises ``KeyboardInterrupt`` — but
    SIGTERM was not, and SIGTERM is how every process supervisor asks a service to stop:
    systemd, Kubernetes, ``docker stop``. Worse, a container's main process runs as PID 1,
    and the kernel does not apply default signal dispositions to PID 1. Without an explicit
    handler the signal is *ignored*, the supervisor waits out its grace period, and the
    process is killed with SIGKILL — in-flight requests dropped, sockets never closed.
    That was observable: ``docker stop`` took the full 30-second timeout and reported exit
    code 137 every time.

    ``shutdown()`` blocks until the serving loop stops, so calling it from the handler —
    which runs on the main thread, the thread inside ``serve_forever`` — would deadlock.
    It goes on a thread of its own instead, which is the documented way to do this.

    Handlers are only installed when this is the main thread; ``signal.signal`` raises
    anywhere else, and a test or an embedding application that serves on a worker thread
    should keep the host process's signal handling rather than have it taken over.
    """
    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    previous = {}

    def _stop(signum, _frame):
        logger.info("received %s — shutting down", signal.Signals(signum).name)
        threading.Thread(target=server.shutdown, name="nova-shutdown", daemon=True).start()

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[signum] = signal.signal(signum, _stop)
        except (OSError, ValueError):  # pragma: no cover — platform without this signal
            pass

    def _restore() -> None:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):  # pragma: no cover
                pass

    return _restore
