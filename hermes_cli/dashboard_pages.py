"""Plugin-contributed dashboard pages served under ``/p/<slug>`` on the dashboard's own port.

A plugin that needs a web surface used to have to run a second HTTP server: its own port, its
own supervision, its own token check, its own proxy publication. Everything in that list already
exists once in the dashboard, and the duplicate copy is where the failures live (a dead second
service behind a healthy-looking proxy, a hand-rolled ``compare_digest`` that raises on non-ASCII,
a spoofable identity header on a TCP port).

``ctx.register_dashboard_page(slug=..., title=..., router=...)`` puts the plugin's own router on
the dashboard port instead. The registry here is the process-global store behind that call; the
ASGI dispatch app (:func:`build_pages_asgi_app`) is mounted by ``web_server`` at ``/p`` *before*
``mount_spa()``, because the SPA's ``/{full_path:path}`` catch-all would otherwise swallow every
plugin path.

Two properties are deliberate:

* **Lookup happens per request, not at mount time.** One mount exists for the lifetime of the
  process and resolves the slug against this registry on each call, so a plugin may register
  after the server booted and an unloaded plugin's page disappears immediately (no restart, no
  route-table surgery).
* **The dispatcher authenticates before it dispatches.** ``/p/`` is NOT under ``/api/``, so the
  legacy loopback ``auth_middleware`` would not cover it; the gate is applied here for HTTP and
  for WebSocket upgrades (Starlette runs no HTTP middleware for WS scopes), using exactly the
  credentials the dashboard already issues. ``PUBLIC_API_PATHS`` is not consulted and must never
  be widened for plugin paths.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

_log = logging.getLogger(__name__)

#: ``/p/<slug>`` — lowercase, digit/letter bounded, hyphen-separated. The character class is the
#: path-traversal defence (no ``/``, no ``.``, no ``%``), so the slug can never climb out of its
#: own prefix or address another mount.
_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$")

#: Reserved first path segments. ``/p/<slug>`` cannot structurally collide with ``/api`` or
#: ``/auth``, but a slug that READS like a core route invites a plugin to publish a link that
#: looks like a Hermes URL, so the obvious ones are refused up front.
RESERVED_SLUGS: frozenset[str] = frozenset({
    "api", "auth", "chat", "login", "logout", "assets", "static", "ws", "p", "health", "favicon",
})


class PageSlugError(ValueError):
    """Raised for a slug that is malformed or reserved."""


def validate_slug(slug: Any) -> str:
    """Return the validated slug or raise :class:`PageSlugError`."""
    if not isinstance(slug, str) or not slug:
        raise PageSlugError("dashboard page slug must be a non-empty string")
    if not _SLUG_RE.match(slug):
        raise PageSlugError(
            f"invalid dashboard page slug {slug!r}: use lowercase letters, digits and "
            "hyphens (1-32 chars, must start and end alphanumeric)")
    if slug in RESERVED_SLUGS:
        raise PageSlugError(f"dashboard page slug {slug!r} is reserved by the dashboard")
    return slug


@dataclass(frozen=True)
class DashboardPage:
    """One plugin page: ``slug`` -> ``/p/<slug>``, ``router`` is any ASGI-callable
    (``fastapi.APIRouter``, a ``FastAPI`` sub-app, a Starlette app)."""

    slug: str
    title: str
    router: Any
    plugin: str = ""

    @property
    def prefix(self) -> str:
        return f"/p/{self.slug}"


_lock = threading.Lock()
_pages: Dict[str, DashboardPage] = {}


def _is_asgi_app(obj: Any) -> bool:
    """Whether ``obj`` can be called as an ASGI app (``APIRouter`` and ``FastAPI`` both can)."""
    return callable(obj) and hasattr(obj, "routes")


def register_page(page: DashboardPage) -> None:
    """Register (or rotate in place) one page.

    Upsert rather than raise on a repeat registration: a forced plugin re-discovery re-runs
    ``register()`` in the same process, and the old behaviour of a hard duplicate error would
    leave the page dead until restart. A DIFFERENT plugin claiming a live slug is refused.
    """
    validate_slug(page.slug)
    if not _is_asgi_app(page.router):
        raise TypeError(
            "dashboard page router must be an ASGI app exposing .routes "
            "(fastapi.APIRouter, FastAPI, or a Starlette app)")
    with _lock:
        existing = _pages.get(page.slug)
        if existing is not None and existing.plugin and page.plugin and existing.plugin != page.plugin:
            raise ValueError(
                f"dashboard page slug {page.slug!r} is already registered by "
                f"plugin {existing.plugin!r}")
        _pages[page.slug] = page
    _log.info("dashboard page registered: /p/%s (%s) by %s", page.slug, page.title,
              page.plugin or "?")


def unregister_page(slug: str, page: DashboardPage) -> bool:
    """Remove ``slug`` only while ``page`` is still the current registration (identity-conditional,
    so an older generation's teardown cannot evict a newer page)."""
    with _lock:
        if _pages.get(slug) is not page:
            return False
        _pages.pop(slug, None)
    _log.info("dashboard page unregistered: /p/%s", slug)
    return True


def get_page(slug: str) -> Optional[DashboardPage]:
    with _lock:
        return _pages.get(slug)


def list_pages() -> List[DashboardPage]:
    with _lock:
        return list(_pages.values())


def clear_pages() -> None:
    """Drop every registration (test clean-slate / plugin registry reset)."""
    with _lock:
        _pages.clear()


# ---------------------------------------------------------------------------
# ASGI dispatch
# ---------------------------------------------------------------------------

def _headless() -> bool:
    """``hermes serve --headless`` exposes JSON-RPC/WS only — never a plugin page."""
    return os.environ.get("HERMES_SERVE_HEADLESS") == "1"


async def _send_json(send: Callable, status: int, payload: bytes) -> None:
    await send({
        "type": "http.response.start", "status": status,
        "headers": [(b"content-type", b"application/json"),
                    (b"cache-control", b"no-store")],
    })
    await send({"type": "http.response.body", "body": payload})


#: Loopback-mode cookie that lets a page's SUBRESOURCES authenticate.
#:
#: A page is opened by navigation, which can carry ``?token=`` but cannot set a header — and the
#: ``<script>``/``<link>``/``<img>`` requests that follow carry neither. Without this the page
#: would load and every asset under it would 401, which is exactly the kind of hole a plugin
#: would otherwise patch with a hand-rolled token of its own. So core issues one: the value is
#: the dashboard's OWN session token (no new secret), httpOnly, SameSite=Strict, scoped to
#: ``/p``. Gated deployments never use it — there the real auth cookies already cover
#: subresources.
_PAGE_COOKIE = "hermes_page_session"


def _gate_is_on(scope: dict) -> bool:
    """Whether the dashboard auth gate is enforced for this request.

    Read off the app the request was routed through (Starlette sets ``scope["app"]`` for HTTP and
    WebSocket alike), never off an imported module global: a test harness mounts the dispatcher on
    its own app, and reading a global would answer for the wrong one.
    """
    return bool(getattr(scope["app"].state, "auth_required", False))


def _http_authorized(scope: dict) -> tuple[bool, bool]:
    """``(authorized, issue_cookie)`` for an HTTP request to a plugin page.

    Gated deployments: the dashboard auth gate ran already (it covers every non-public path) and
    left a verified session on the connection state — trust that and nothing else. Loopback
    deployments: the gate is off and ``auth_middleware`` only covers ``/api/``, so the dashboard
    session token is checked here — from the header, the ``?token=`` of a navigation, or the
    page cookie a previous navigation was issued.
    """
    from starlette.requests import HTTPConnection

    from hermes_cli.web_server import _has_valid_session_token

    conn = HTTPConnection(scope)
    if _gate_is_on(scope):
        state = scope.get("state") or {}
        ok = state.get("session") is not None or bool(state.get("token_authenticated"))
        return ok, False
    if _has_valid_session_token(conn):  # type: ignore[arg-type]
        return True, False
    if _matches_session_token(conn.cookies.get(_PAGE_COOKIE, "")):
        return True, False
    # Only a navigation's ``?token=`` earns the cookie; a header-authenticated API call has no
    # use for one.
    if _matches_session_token(conn.query_params.get("token", "")):
        return True, True
    return False, False


def _matches_session_token(value: str) -> bool:
    """Constant-time comparison against the dashboard session token.

    ``compare_digest`` raises on non-ASCII, and a pasted ``?token=abc…`` is a routine way for one
    to arrive — encoding first turns that into a wrong token (401) instead of a 500.
    """
    from hermes_cli.web_server import _SESSION_TOKEN

    return bool(value) and hmac.compare_digest(value.encode(), _SESSION_TOKEN.encode())


def _cookie_issuing_send(send: Callable, secure: bool) -> Callable:
    """Wrap ``send`` so the first response start carries the page cookie."""
    from hermes_cli.web_server import _SESSION_TOKEN

    attrs = f"{_PAGE_COOKIE}={_SESSION_TOKEN}; Path=/p; HttpOnly; SameSite=Strict"
    if secure:
        attrs += "; Secure"

    async def _send(message: dict) -> None:
        if message.get("type") == "http.response.start":
            message = dict(message)
            headers = list(message.get("headers") or [])
            headers.append((b"set-cookie", attrs.encode("latin-1")))
            message["headers"] = headers
        await send(message)

    return _send


def _ws_authorized(scope: dict) -> bool:
    """Whether this WebSocket upgrade may reach a plugin page.

    HTTP middleware does not run for WS scopes, so the credential check, the DNS-rebinding
    Host/Origin guard and the peer check are applied explicitly — the same three the dashboard's
    own ``/api/pty`` runs, reused rather than re-implemented (``?ticket=`` in gated mode,
    ``?token=`` on loopback).
    """
    from starlette.websockets import WebSocket

    from hermes_cli.web_server_chat import _ws_auth_ok, _ws_request_is_allowed

    ws = WebSocket(scope, receive=_noop_receive, send=_noop_send)
    return _ws_auth_ok(ws) and _ws_request_is_allowed(ws)


async def _noop_receive() -> dict:  # pragma: no cover - never awaited by the gate helpers
    return {"type": "websocket.disconnect", "code": 1000}


async def _noop_send(_message: dict) -> None:  # pragma: no cover - same
    return None


def _route_path(scope: dict) -> str:
    """The path below the mount point.

    Starlette's ``Mount`` does NOT rewrite ``scope["path"]``: it extends ``root_path`` and every
    router subtracts it again. Reading ``path`` directly would make the slug come out as ``"p"``
    and every plugin page 404. This reproduces ``starlette._utils.get_route_path`` rather than
    importing a private helper.
    """
    path = scope.get("path", "/")
    root_path = scope.get("root_path", "")
    if root_path and path.startswith(root_path):
        return path[len(root_path):] or "/"
    return path


def _split_slug(route_path: str) -> tuple[str, str]:
    """``/i3/ws`` -> ``("i3", "/ws")``; ``/i3`` -> ``("i3", "/")``."""
    trimmed = route_path.lstrip("/")
    slug, _, rest = trimmed.partition("/")
    return slug, ("/" + rest if rest else "/")


def build_pages_asgi_app() -> Callable:
    """The ASGI app mounted at ``/p``: resolve the slug per request, gate, then dispatch."""

    async def pages_app(scope: dict, receive: Callable, send: Callable) -> None:
        kind = scope.get("type")
        if kind not in ("http", "websocket"):  # pragma: no cover - lifespan is not routed here
            return
        slug, _rest = _split_slug(_route_path(scope))
        page = None if _headless() else get_page(slug)
        if page is None:
            if kind == "websocket":
                await send({"type": "websocket.close", "code": 4404})
                return
            await _send_json(send, 404, b'{"detail":"Plugin page not found"}')
            return

        if kind == "http":
            authorized, issue_cookie = _http_authorized(scope)
        else:
            authorized, issue_cookie = _ws_authorized(scope), False
        if not authorized:
            if kind == "websocket":
                await send({"type": "websocket.close", "code": 4401})
                return
            await _send_json(send, 401, b'{"detail":"Unauthorized"}')
            return
        if issue_cookie:
            send = _cookie_issuing_send(send, secure=scope.get("scheme") in ("https", "wss"))

        # Hand the slug segment to the plugin the same way Mount does: consumed into
        # ``root_path`` so the plugin's own paths are relative to ``/p/<slug>`` while
        # ``url_for``/redirects still produce the full external URL.
        child = dict(scope)
        child["root_path"] = scope.get("root_path", "") + "/" + slug
        await page.router(child, receive, send)

    return pages_app


def mount_pages(application) -> None:
    """Mount the ``/p`` dispatcher. MUST run before ``mount_spa()``.

    ``mount_spa`` registers ``/{full_path:path}`` for client-side routing, which matches every
    otherwise-unclaimed path; anything mounted after it is unreachable. Headless backends get no
    mount at all (the dispatcher also refuses at request time, so a late env flip cannot leak a
    page).
    """
    if _headless():
        _log.debug("headless backend: plugin dashboard pages not mounted")
        return
    application.mount("/p", build_pages_asgi_app())


# ---------------------------------------------------------------------------
# Client bootstrap
# ---------------------------------------------------------------------------

def page_base_path(request_or_scope: Any) -> str:
    """External base path of the page (``/p/<slug>``, prefixed when behind a proxy).

    A plugin's HTML must not hardcode ``/p/<slug>``: the same page is reachable under an
    ``X-Forwarded-Prefix`` deployment, where every relative URL needs that prefix too.
    """
    scope = getattr(request_or_scope, "scope", request_or_scope)
    root_path = scope.get("root_path", "") or ""
    return f"{_proxy_prefix(scope)}{root_path}"


def _proxy_prefix(scope: dict) -> str:
    """Normalised ``X-Forwarded-Prefix`` for this request, or ``""``.

    ``normalise_prefix`` is the dashboard's single validator for this header (it rejects ``..``,
    ``//`` and injection characters) — reused rather than re-derived, so a plugin page and the
    SPA agree on what the prefix is.
    """
    from hermes_cli.dashboard_auth.prefix import normalise_prefix

    headers = dict(scope.get("headers") or [])
    return normalise_prefix(headers.get(b"x-forwarded-prefix", b"").decode("latin-1"))


def bootstrap_script(request_or_scope: Any) -> str:
    """``<script>`` block a plugin page injects into its own HTML.

    Publishes exactly what a page needs to open its own authenticated WebSockets, and nothing a
    plugin should be deriving itself:

    * ``__HERMES_PAGE_BASE__`` — the page's external base path, so relative URLs survive a proxy
      prefix.
    * ``__HERMES_DASHBOARD_BASE__`` — the dashboard's own base, for calling core endpoints such
      as ``POST /api/auth/ws-ticket``.
    * ``__HERMES_AUTH_REQUIRED__`` — which credential scheme applies.
    * ``__HERMES_SESSION_TOKEN__`` — loopback only. Under the auth gate the token is NOT
      published (same rule as the SPA's index.html): the page mints a per-socket ticket via
      ``POST /api/auth/ws-ticket`` instead.
    """
    from hermes_cli.web_server import _SESSION_TOKEN

    scope = getattr(request_or_scope, "scope", request_or_scope)
    gated = _gate_is_on(scope)
    token_js = "" if gated else f"window.__HERMES_SESSION_TOKEN__={json.dumps(_SESSION_TOKEN)};"
    return (
        "<script>"
        f"window.__HERMES_PAGE_BASE__={json.dumps(page_base_path(scope))};"
        f"window.__HERMES_DASHBOARD_BASE__={json.dumps(_proxy_prefix(scope))};"
        f"window.__HERMES_AUTH_REQUIRED__={'true' if gated else 'false'};"
        f"{token_js}"
        "</script>"
    )
