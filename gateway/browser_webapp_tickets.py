"""One-use Webapp tickets for the paired Hermes Browser extension.

The gateway's own HTTP listener serves the Webapp (``gateway/webapp_listener.py``) and signs a browser
in with a credential only a launch ticket (``hermes webapp``) or a window ticket
(``hermes_cli.web_routers.webapp``) can hand over. The paired extension holds a scoped bearer token for
the API server, never that credential, so it had no way in without a terminal.

This module is what the API server's two loopback routes call to mint, in this process and through the
Webapp's own stores, the tickets the extension needs:

* :func:`mint_window_ticket` is a one-use *window* ticket (``launch=False``) plus the Webapp's own
  origin. ``<webapp_url>/webapp/launch#ticket=<ticket>&query=profile%3D<name>`` redeems it into a tab
  that is signed in, exactly as the Webapp's child window does.
* :func:`mint_socket_ticket` is a one-use ticket for the Webapp's ``/api/ws`` gateway socket, bound to
  the profile like the Webapp's own ``POST /api/webapp/socket-ticket``. It gives the extension a
  session-authority socket without holding the long-lived Webapp credential.

The two routes are registered by ``connect()`` (:func:`http_routes`) beside the pairing routes and answer
only on loopback, to the paired extension token or the API key.

Both are minted under the browser credential in use now (``hermes_cli.browser_credential``), so
``hermes webapp --sign-out`` revokes them and the sockets they opened.

Only a *private* Webapp qualifies: on a loopback bind with no sign-in gate and no public URL. A Webapp
that is off, still starting, unavailable, behind sign-in or served at a public address refuses, and each
refusal carries a ``code`` the extension reads to choose what to do next. Nothing here imports the
Webapp at module load, so a gateway without it answers ``webapp_unavailable``.
"""

from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import urlsplit

try:
    from aiohttp import web
except ImportError:  # the API server itself needs aiohttp; this module only imports without it
    web = None  # type: ignore[assignment]

# Same shape as ``hermes_constants.PROFILE_ID_RE``, which the Webapp's launch verb checks.
PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
WEBAPP_SOCKET_PATH = "/api/ws"

# Error codes (the ``error.code`` of a refusal).
CODE_UNAVAILABLE = "webapp_unavailable"
CODE_OFF = "webapp_off"
CODE_STARTING = "webapp_starting"
CODE_NOT_PRIVATE = "webapp_not_private"
CODE_INVALID_PROFILE = "invalid_profile"
CODE_PROFILE_NOT_FOUND = "profile_not_found"
CODE_TOO_MANY = "webapp_tickets_exhausted"


class WebappTicketRefused(Exception):
    """A ticket was refused: *status* is the HTTP status, *code* the machine-readable reason."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class _Webapp:
    """The serving Webapp this process hosts: its origin and the dashboard app's state."""

    __slots__ = ("url", "state")

    def __init__(self, url: str, state: Any):
        self.url = url
        self.state = state


def _clean_profile(profile: Optional[str]) -> Optional[str]:
    """*profile* as a profile name, ``None`` when none was asked for; 400 when it is not a name."""
    name = (profile or "").strip()
    if not name:
        return None
    if not PROFILE_NAME_RE.match(name):
        raise WebappTicketRefused(400, CODE_INVALID_PROFILE, "profile must be a profile name")
    return name


def _serving_webapp(runner: Any) -> _Webapp:
    """The private Webapp *runner* serves right now, else a :class:`WebappTicketRefused`."""
    status = getattr(runner, "webapp_status", None)
    if not isinstance(status, dict):
        raise WebappTicketRefused(404, CODE_UNAVAILABLE, "This gateway does not serve the Hermes Webapp.")
    state = status.get("state")
    if state == "off":
        raise WebappTicketRefused(
            404, CODE_OFF,
            "The Webapp is off. Set webapp.enabled: true in config.yaml and restart the gateway.")
    if state == "starting":
        raise WebappTicketRefused(503, CODE_STARTING, "The Webapp is still starting; try again in a moment.")
    if state != "serving" or not status.get("url"):
        raise WebappTicketRefused(
            404, CODE_UNAVAILABLE, f"The Webapp is unavailable: {status.get('reason') or 'it is not serving'}")
    mount = getattr(getattr(runner, "session_api", None), "app", None)
    web = getattr(mount, "web", None)
    app_state = getattr(getattr(web, "app", None), "state", None)
    if app_state is None:
        raise WebappTicketRefused(503, CODE_STARTING, "The Webapp is still starting; try again in a moment.")
    try:
        from gateway.webapp_listener import bundle_missing_reason
        from hermes_cli.web_server_surface import private_launch
    except ImportError as exc:  # a gateway whose Webapp code is absent or older than these hooks
        raise WebappTicketRefused(404, CODE_UNAVAILABLE, "This gateway does not serve the Hermes Webapp.") from exc
    dist = getattr(app_state, "web_dist", None)
    missing = None if dist is None else bundle_missing_reason(dist)
    if missing:
        raise WebappTicketRefused(404, CODE_UNAVAILABLE, f"The Webapp is unavailable: {missing}")
    if not private_launch(app_state):
        raise WebappTicketRefused(
            404, CODE_NOT_PRIVATE,
            "The Webapp is behind sign-in or served at a public address; open it and sign in there.")
    return _Webapp(str(status["url"]).rstrip("/"), app_state)


def mint_window_ticket(runner: Any, profile: Optional[str] = None) -> dict:
    """``{"ticket", "webapp_url", "expires_in_seconds"}``: a one-use, non-launch window ticket for the
    Webapp *runner* serves. *profile* is validated as a name only (the launch page opens on it)."""
    _clean_profile(profile)
    webapp = _serving_webapp(runner)
    try:
        from hermes_cli.web_routers.webapp import WINDOW_TICKET_TTL_SECONDS, _mint_handoff_ticket
    except ImportError as exc:
        raise WebappTicketRefused(404, CODE_UNAVAILABLE, "This gateway does not serve the Hermes Webapp.") from exc
    try:
        ticket = _mint_handoff_ticket(webapp.state, WINDOW_TICKET_TTL_SECONDS, launch=False)
    except Exception as exc:  # fastapi.HTTPException(429): too many windows pending
        if getattr(exc, "status_code", None) == 429:
            raise WebappTicketRefused(
                429, CODE_TOO_MANY, "Too many pending Webapp windows; try again shortly.") from exc
        raise
    return {"ticket": ticket, "webapp_url": webapp.url, "expires_in_seconds": WINDOW_TICKET_TTL_SECONDS}


def _websocket_url(webapp_url: str) -> str:
    parts = urlsplit(webapp_url)
    scheme = "wss" if parts.scheme == "https" else "ws"
    return f"{scheme}://{parts.netloc}{parts.path.rstrip('/')}{WEBAPP_SOCKET_PATH}"


def mint_socket_ticket(runner: Any, profile: Optional[str] = None) -> dict:
    """``{"ticket", "ws_url", "ttl_seconds"}``: a one-use ticket for one dial of the Webapp's ``/api/ws``
    gateway socket. Like the Webapp's own ``socket-ticket``, a ticket minted with *profile* answers for
    that profile when the socket's requests name none (it never falls back to the launch profile)."""
    name = _clean_profile(profile)
    webapp = _serving_webapp(runner)
    try:
        from fastapi import HTTPException
        from hermes_cli.browser_credential import current
        from hermes_cli.dashboard_auth.ws_tickets import TTL_SECONDS
        from hermes_cli.web_routers.webapp import SESSION_TOKEN_OWNER, mint_socket_ticket as mint
        from hermes_cli.web_server_cron import _cron_profile_home
        from hermes_cli.web_server_surface import policy
        from hermes_cli.web_session_sockets import GATEWAY_SOCKET
    except ImportError as exc:
        raise WebappTicketRefused(404, CODE_UNAVAILABLE, "This gateway does not serve the Hermes Webapp.") from exc
    if not policy(webapp.state).gateway_ticket:
        raise WebappTicketRefused(
            404, CODE_UNAVAILABLE, "Webapp gateway tickets are unavailable on this surface.")
    try:
        _canonical, home = _cron_profile_home(name)
    except HTTPException as exc:
        if exc.status_code == 404:
            raise WebappTicketRefused(404, CODE_PROFILE_NOT_FOUND, str(exc.detail)) from exc
        raise WebappTicketRefused(400, CODE_INVALID_PROFILE, str(exc.detail)) from exc
    ticket = mint(dict(SESSION_TOKEN_OWNER), GATEWAY_SOCKET, session_token=current(),
                  profile_home=str(home) if name else None)
    return {"ticket": ticket, "ws_url": _websocket_url(webapp.url), "ttl_seconds": TTL_SECONDS}


async def _respond(adapter: Any, request: "web.Request", mint) -> "web.Response":
    """The ladder both routes share: loopback peers only (403), then the paired token or the API key
    (401), then *mint* for the Webapp this gateway serves. A refusal keeps its own status and
    ``error.code`` (404 off or not private, 503 starting); every reply is uncached."""
    if not adapter._request_is_loopback(request):
        return web.json_response(
            {"error": {"message": "Webapp tickets are available only on the local machine.",
                       "code": "webapp_ticket_loopback_only"}}, status=403)
    # A route that hands out a sign-in never takes ``_check_auth``'s shortcut for a server with no key:
    # with none, only a paired extension's token is a credential.
    bearer = request.headers.get("Authorization", "")
    token = bearer[7:].strip() if bearer.startswith("Bearer ") else ""
    if not token or not (adapter._expected_api_key() or adapter._browser_pairing.is_valid_token(token)):
        return adapter._auth_failed_response()
    auth_error = adapter._check_auth(request)
    if auth_error:
        return auth_error
    runner = adapter.gateway_runner or request.app.get("gateway_runner")
    try:
        body = mint(runner, request.query.get("profile"))
    except WebappTicketRefused as exc:
        return web.json_response(
            {"error": {"message": exc.message, "code": exc.code}}, status=exc.status,
            headers={"Cache-Control": "no-store"})
    return web.json_response(body, headers={"Cache-Control": "no-store"})


def http_routes(adapter: Any) -> list[tuple]:
    """``(method, path, handler)`` rows for ``connect()``.

    ``POST /api/browser-extension/webapp-ticket?profile=`` answers ``{ticket, webapp_url}``: the
    extension opens ``<webapp_url>/webapp/launch#ticket=<ticket>`` (and ``&query=profile%3D<name>``) in
    a tab, which the Webapp signs in. ``POST /api/browser-extension/webapp-socket-ticket?profile=``
    answers ``{ticket, ws_url}``: a one-use ticket for ``ws_url``, offered as the
    ``hermes-gateway-ticket.<ticket>`` subprotocol beside ``hermes-gateway-v1``."""

    async def webapp_ticket(request: "web.Request") -> "web.Response":
        return await _respond(adapter, request, mint_window_ticket)

    async def webapp_socket_ticket(request: "web.Request") -> "web.Response":
        return await _respond(adapter, request, mint_socket_ticket)

    return [
        ("POST", "/api/browser-extension/webapp-ticket", webapp_ticket),
        ("POST", "/api/browser-extension/webapp-socket-ticket", webapp_socket_ticket)]
