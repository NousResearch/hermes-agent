"""What each launch surface of the one web-server stack may do.

The host serving the app records its surface once on ``app.state.ui_surface`` (unset
reads as ``dashboard``); every surface-dependent decision reads this table instead of
re-deriving it. ``auth_required`` is decided at runtime by the auth gate, so the
private-launch predicate is a function over state, not a table field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    #: ``/api/webapp/gateway-ticket`` mints the browser's one-use ticket for ``/api/ws`` on this
    #: same origin (accepted by the loopback and gated upgrade gates alike).
    gateway_ticket: bool


POLICIES: dict[str, SurfacePolicy] = {
    "serve": SurfacePolicy(
        serves_spa=False, html_token=True, host_terminal=False, launch_link=False, gateway_ticket=False),
    "dashboard": SurfacePolicy(
        serves_spa=True, html_token=True, host_terminal=False, launch_link=False, gateway_ticket=False),
    "webapp": SurfacePolicy(
        serves_spa=True, html_token=False, host_terminal=True, launch_link=True, gateway_ticket=True),
}


def policy(state: Any) -> SurfacePolicy:
    return POLICIES[getattr(state, "ui_surface", "dashboard")]


def private_launch(state: Any) -> bool:
    """Webapp on an unauthenticated bind: the launch link is the only way in."""
    return policy(state).launch_link and not getattr(state, "auth_required", False)
