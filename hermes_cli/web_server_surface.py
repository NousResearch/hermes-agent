"""What each launch surface of the one web-server stack may do.

The host serving the app records its surface once on ``app.state.ui_surface``; unset, it reads
as ``gateway`` on the gateway's own listener for native clients (the app has a
``gateway_runner``, as the lifespan choice reads) and ``dashboard`` elsewhere (the standalone
``hermes dashboard`` and ``hermes serve``). Every surface-dependent decision reads this table
instead of re-deriving it, and :func:`route_on` keeps a route off the surfaces that do not use
it. ``auth_required`` is decided at runtime by the auth gate, so the private-launch predicate is
a function over state, not a table field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class SurfacePolicy:
    #: Mounts a browser UI; ``serve`` is JSON-RPC/WS only.
    serves_spa: bool
    #: Public HTML may carry the session token. Where it may not, every HTML entry
    #: (``/index.html`` included) gets the token-free bootstrap instead of the raw file.
    html_token: bool
    #: ``/api/host-terminal`` may open host shells (still gated per request).
    host_terminal: bool
    #: An unauthenticated start hands the operator a private launch link.
    launch_link: bool
    #: ``/api/webapp/socket-ticket`` mints the browser's one-use tickets for ``/api/ws`` and the voice
    #: streams on this same origin (accepted by the loopback and gated upgrade gates alike).
    gateway_ticket: bool
    #: Behind the sign-in gate, a write that carries the session cookie (``/auth/logout`` included)
    #: must come from this listener's own origin. Elsewhere the gate keeps main's rule.
    own_origin_cookie_writes: bool
    #: ``/api/webapp-access/*`` serves: native Desktop's Web Access pane calls the gateway's own
    #: listener (the private one, or the Webapp's when it is on).
    web_access: bool
    #: The browser's file routes serve: ``/api/chat/file-upload`` stages a picked file,
    #: ``/api/files/ticket`` mints the path-bound tickets for downloads and media links, and chat
    #: upload bodies are capped before they are parsed (``web_upload_limit``).
    browser_files: bool
    #: ``/api/desktop-plugins/*`` serves this host's desktop plugins to the browser's plugin loader;
    #: native Desktop reads its own disk.
    desktop_plugins: bool


POLICIES: dict[str, SurfacePolicy] = {
    "serve": SurfacePolicy(
        serves_spa=False, html_token=True, host_terminal=False, launch_link=False, gateway_ticket=False,
        own_origin_cookie_writes=False, web_access=False, browser_files=False,
        desktop_plugins=False),
    "dashboard": SurfacePolicy(
        serves_spa=True, html_token=True, host_terminal=False, launch_link=False, gateway_ticket=False,
        own_origin_cookie_writes=False, web_access=False, browser_files=False,
        desktop_plugins=False),
    "gateway": SurfacePolicy(
        serves_spa=True, html_token=True, host_terminal=False, launch_link=False, gateway_ticket=False,
        own_origin_cookie_writes=False, web_access=True, browser_files=False,
        desktop_plugins=False),
    "webapp": SurfacePolicy(
        serves_spa=True, html_token=False, host_terminal=True, launch_link=True, gateway_ticket=True,
        own_origin_cookie_writes=True, web_access=True, browser_files=True,
        desktop_plugins=True),
}


def surface(state: Any) -> str:
    """The surface *state*'s app serves (see the module docstring)."""
    recorded = getattr(state, "ui_surface", None)
    if recorded:
        return recorded
    return "gateway" if getattr(state, "gateway_runner", None) is not None else "dashboard"


def policy(state: Any) -> SurfacePolicy:
    return POLICIES[surface(state)]


def route_on(field: str) -> Callable[[Request], None]:
    """A route dependency answering 404 where the serving surface's policy leaves *field* off, so
    the route does not exist there for any caller."""
    def surface_serves(request: Request) -> None:
        if not getattr(policy(request.app.state), field):
            raise HTTPException(status_code=404, detail="Not Found")

    return surface_serves


def private_launch(state: Any) -> bool:
    """Webapp on an unauthenticated bind: the launch link is the only way in."""
    return policy(state).launch_link and not getattr(state, "auth_required", False)
