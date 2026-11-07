"""Tailnet sign-in for the gateway's Webapp behind ``tailscale serve`` (``webapp.tailscale_logins``).

``tailscale serve`` proxies the machine's tailnet HTTPS name (``https://<machine>.<tailnet>.ts.net``) to
a loopback backend and sets ``Tailscale-User-Login`` on every proxied request to the login of the
person behind the device, replacing any value the client sent. On a Webapp listener without the
sign-in gate (loopback ``host``, no ``public_url``), a request carrying one of the listed logins is
treated like a one-use launch link: its ts.net Host is accepted, and the page signs the tab in with the
Webapp's browser credential. Every later request authenticates as usual (session header, socket
tickets), so signing out still ends every tab's access until it loads the page again.

A request counts as proxied by ``tailscale serve`` only when all of these hold:

* the TCP peer is loopback (Serve connects from it, and the listener binds nothing else);
* the Host is a ``*.ts.net`` name, the only names Serve answers for. A web page cannot point one at
  127.0.0.1, so a DNS-rebinding page (same-origin with itself, free to set the header from script)
  is refused;
* on Linux, the peer's socket belongs to root (``tailscaled``) or to the gateway's own user, who can
  read the credential file anyway, so another account on the machine cannot claim a login;
* the request is first-party (:func:`_first_party`): a browser sends the identity on every request
  through the proxy, so a page on another site must not reach the credential through it.

Elsewhere than Linux the peer's account cannot be checked and any local account can claim a listed
login, as Tailscale's documentation says of identity headers: use the setting only on a machine whose
accounts are already trusted with Hermes. The list is read once per process; restart the gateway
after changing it. Empty (the default) turns the feature off.
"""
from __future__ import annotations

import functools
import ipaddress
import logging
import os
import sys
from typing import Any, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

LOGIN_HEADER = "tailscale-user-login"
_TAILNET_SUFFIX = ".ts.net"
_FIRST_PARTY_FETCH = frozenset({"", "same-origin", "none"})

Endpoint = tuple[str, int]
_REFUSED_UIDS: set[Optional[int]] = set()


@functools.lru_cache(maxsize=1)
def allowed_logins() -> frozenset[str]:
    """``webapp.tailscale_logins``, lower-cased; read once per process, empty when unset or invalid."""
    try:
        from hermes_cli.config import load_config

        raw = (load_config().get("webapp") or {}).get("tailscale_logins") or []
    except Exception:
        logger.warning("webapp.tailscale_logins could not be read; tailnet sign-in is off", exc_info=True)
        return frozenset()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        logger.warning("webapp.tailscale_logins must be a list of logins; tailnet sign-in is off")
        return frozenset()
    return frozenset(str(item).strip().lower() for item in raw if str(item).strip())


def _address(text: str) -> Optional[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """*text* as an IP address (an IPv4-mapped IPv6 address as its IPv4 one), None when it is not one."""
    try:
        address = ipaddress.ip_address(str(text).split("%", 1)[0])
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _serve_host(headers: Any) -> bool:
    """True when the Host names a machine under ``.ts.net`` (port optional)."""
    host = str(headers.get("host") or "").strip().lower()
    name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    label = name[: -len(_TAILNET_SUFFIX)]
    return name.endswith(_TAILNET_SUFFIX) and bool(label) and not label.startswith(".")


def _first_party(headers: Any, method: str) -> bool:
    """True when another site did not make the request: no cross-site ``Sec-Fetch-Site`` (browsers send
    it on navigations, sub-resources, fetches and socket upgrades alike), and no ``Origin`` or one that
    names this Host. A top-level GET navigation passes too (a link to the Webapp in a mail): it changes
    nothing and the linking site cannot read the page. A cross-site form POST is a navigation as well,
    hence the GET; a framed page is not a top-level document."""
    site = str(headers.get("sec-fetch-site") or "").strip().lower()
    if site not in _FIRST_PARTY_FETCH:
        return (method.upper() in ("GET", "HEAD")
                and str(headers.get("sec-fetch-mode") or "").strip().lower() == "navigate"
                and str(headers.get("sec-fetch-dest") or "").strip().lower() == "document")
    origin = str(headers.get("origin") or "").strip()
    if not origin:
        return True
    parts = urlsplit(origin)
    host = str(headers.get("host") or "").strip().lower()
    return parts.scheme in ("https", "http") and bool(host) and parts.netloc.lower() == host


def _peer_is_trusted(peer: Endpoint, server: Optional[Endpoint], platform: str = sys.platform) -> bool:
    """On Linux (*platform*, a ``sys.platform`` value), the peer's socket belongs to root (``tailscaled``)
    or to this process's user. Elsewhere the account cannot be read and any loopback peer passes (see
    the module docstring)."""
    if platform != "linux":
        return True
    from hermes_cli.web_routers.webapp import _loopback_peer_uid  # FastAPI-side: only reached on the Webapp

    uid = _loopback_peer_uid(peer, server) if server else None
    if uid is not None and uid in (0, os.getuid()):  # windows-footgun: ok — Linux-only branch
        return True
    if uid not in _REFUSED_UIDS:  # once per account: a misconfigured tailscaled would log every request
        _REFUSED_UIDS.add(uid)
        logger.warning("tailnet sign-in refused: a loopback peer belongs to uid %s, neither root (tailscaled) "
                       "nor this gateway's user", "unknown" if uid is None else uid)
    return False


def tailnet_login(headers: Any, peer: Optional[Endpoint], method: str = "GET",
                  server: Optional[Endpoint] = None) -> Optional[str]:
    """The listed tailnet login of a first-party request ``tailscale serve`` proxied, else None. *peer*
    and *server* are the connection's two ends, as in an ASGI scope's ``client`` and ``server``."""
    allowed = allowed_logins()
    if not allowed or not peer:
        return None
    address = _address(peer[0])
    if address is None or not address.is_loopback:
        return None
    login = str(headers.get(LOGIN_HEADER) or "").strip().lower()
    if login not in allowed or not _serve_host(headers) or not _first_party(headers, method):
        return None
    if not _peer_is_trusted((str(peer[0]), int(peer[1])), (str(server[0]), int(server[1])) if server else None):
        return None
    return login


def connection_login(conn: Any, state: Any = None) -> Optional[str]:
    """:func:`tailnet_login` of a Starlette ``Request`` or ``WebSocket`` on the gateway's Webapp without
    the sign-in gate. *state* is the listener's app state (default: the connection's app). The gated
    Webapp and the standalone dashboard keep their own sign-in."""
    if state is None:
        state = getattr(conn.scope.get("app"), "state", None)
    if getattr(state, "ui_surface", None) != "webapp" or getattr(state, "auth_required", True):
        return None
    scope = conn.scope
    method = scope.get("method", "GET") if scope.get("type") == "http" else "GET"
    return tailnet_login(conn.headers, scope.get("client"), method, scope.get("server"))
