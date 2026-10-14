"""Listener boundary primitives shared by the dashboard app and the gateway's session listener:
Host validation, the auth-gate decision and the native WebSocket ticket subprotocol.

Import-light on purpose (no FastAPI): ``gateway.run_api`` serves the native session WebSocket
without importing the dashboard app (about 690 routes, ~0.6 s), and must apply the same
DNS-rebinding Host check and gate decision. ``hermes_cli.web_server`` and
``hermes_cli.web_server_chat`` re-export these names, so ``web_server.<name>`` keeps working.
"""

import os
import re
import urllib.parse
from typing import Optional

# Accepted Host values for loopback binds. DNS rebinding TTL-flips an attacker
# hostname to 127.0.0.1 so the browser treats it as same-origin; validating Host
# at the app layer rejects it. See GHSA-ppp5-vxwm-4cf7.
_LOOPBACK_HOST_VALUES: frozenset = frozenset({"localhost", "127.0.0.1", "::1"})

# Starlette's TestClient reports the peer as "testclient"; treat it as
# loopback so tests don't need to rewrite request scope.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})

# Desktop file.attach sends a whole base64 data URL in one JSON-RPC frame;
# uvicorn's 16 MiB default rejects files under the 256 MiB raw attach cap.
_DESKTOP_ATTACHMENT_WS_MAX_BYTES = 384 * 1024 * 1024


def _dashboard_public_hosts() -> frozenset[str]:
    """Return the exact hostname declared by ``dashboard.public_url``.

    One source of truth for OAuth redirects, Host and WS Origin validation.
    Malformed or unset values fail closed as an empty set.
    """
    from hermes_cli.dashboard_auth.prefix import resolve_public_url

    public_url = resolve_public_url()
    try:
        hostname = urllib.parse.urlparse(public_url).hostname if public_url else None
    except ValueError:
        hostname = None
    return frozenset({hostname.lower()}) if hostname else frozenset()


def _host_header_hostname(host_header: str) -> str:
    """Return a normalized hostname from a valid HTTP Host authority.

    Host headers are authorities, not full URLs. Reject ambiguous ports,
    malformed IPv6 brackets, and URL syntax so validation always fails closed.
    """
    value = (host_header or "").strip()
    if not value or "://" in value or any(c in value for c in '"\'<> \n\r\t/?#@'):
        return ""

    if value.startswith("["):
        close = value.find("]")
        if close == -1:
            return ""
        hostname = value[1:close]
        # Bracket notation is reserved for IPv6 literals.
        if ":" not in hostname:
            return ""
        suffix = value[close + 1:]
        if suffix and not re.fullmatch(r":\d+", suffix):
            return ""
        return hostname.lower()

    # Unbracketed IPv6 authorities are ambiguous with a port separator.
    if value.count(":") > 1:
        return ""
    if ":" in value:
        hostname, port = value.rsplit(":", 1)
        if not hostname or not port.isdigit():
            return ""
        return hostname.lower()
    return value.lower()


def _is_accepted_host(
    host_header: str,
    bound_host: str,
    trusted_public_hosts: frozenset[str] = frozenset(),
) -> bool:
    """True if the Host header targets the interface we bound to.

    Accepts:
    - Exact bound host (with or without port suffix)
    - Loopback aliases when bound to loopback
    - Exact operator-declared public hosts (with or without port suffix)
    - Any host when bound to 0.0.0.0 (explicit opt-in to non-loopback,
      no protection possible at this layer)
    """
    host_only = _host_header_hostname(host_header)
    if not host_only:
        return False
    # All-interfaces bind: no Host-layer defence is possible; rely on operator
    # network controls.
    if host_only in trusted_public_hosts or bound_host in {"0.0.0.0", "::"}:
        return True
    bound_lc = bound_host.lower()
    if bound_lc in _LOOPBACK_HOST_VALUES:
        return host_only in _LOOPBACK_HOST_VALUES
    return host_only == bound_lc


def should_require_auth(host: str, allow_public: bool = False) -> bool:
    """True iff the auth gate must be active: any non-loopback bind.

    RFC1918 / CGNAT / link-local are deliberately PUBLIC — a hostile LAN device
    is the threat model. ``allow_public`` (legacy ``--insecure``) is accepted for
    old launch scripts but IGNORED since the June 2026 hermes-0day campaign.
    """
    return host not in _LOOPBACK_HOST_VALUES


def public_hosts_engage_gate(trusted_public_hosts: frozenset[str]) -> bool:
    """A non-loopback ``dashboard.public_url`` host engages the auth gate even on a loopback bind."""
    return any(h not in _LOOPBACK_HOST_VALUES for h in trusted_public_hosts)


def _desktop_loopback_auth_exempt(
    host: str,
    ssh_session_token: Optional[str] = None,
    ssh_owner_nonce: Optional[str] = None,
) -> bool:
    """True for a Desktop-owned loopback backend (#96490).

    A non-loopback ``dashboard.public_url`` would otherwise engage the
    ticket-only gate for the private loopback backends Desktop spawns, whose
    per-spawn session token the gate's WS path refuses — Desktop could not boot.
    The public dashboard is a separate non-loopback process that stays gated, so
    this never opens the public surface. Requires ALL of: loopback bind,
    ``HERMES_DESKTOP=1``, and an operator-minted credential (env token, SSH
    session token, or owner nonce).
    """
    return (
        host in _LOOPBACK_HOST_VALUES
        and os.environ.get("HERMES_DESKTOP") == "1"
        and bool(os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN") or ssh_session_token or ssh_owner_nonce)
    )


def listener_auth_required(host: str, trusted_public_hosts: frozenset[str]) -> bool:
    """The ``app.state.auth_required`` value ``web_server._configure_auth_gate(host, False, None,
    None)`` resolves for a listener with no SSH credentials (the gateway's session listener)."""
    if _desktop_loopback_auth_exempt(host):
        return should_require_auth(host)
    return should_require_auth(host) or public_hosts_engage_gate(trusted_public_hosts)


_GATEWAY_WS_PROTOCOL = "hermes-gateway-v1"
_GATEWAY_WS_TICKET_PROTOCOL_PREFIX = "hermes-gateway-ticket."


def _gateway_ws_ticket_from_subprotocol(ws) -> tuple[str, str]:
    """Return ``(ticket, reason)`` from an unambiguous gateway protocol set."""
    raw = str(ws.headers.get("sec-websocket-protocol", "") or "")
    protocols = [value.strip() for value in raw.split(",") if value.strip()]
    ticket_protocols = [
        value for value in protocols if value.startswith(_GATEWAY_WS_TICKET_PROTOCOL_PREFIX)]
    if not ticket_protocols:
        return "", "none"
    if _GATEWAY_WS_PROTOCOL not in protocols or len(ticket_protocols) != 1:
        return "", "invalid"
    ticket = ticket_protocols[0][len(_GATEWAY_WS_TICKET_PROTOCOL_PREFIX):]
    return (ticket, "ok") if ticket else ("", "invalid")


def ws_client_reason(ws, state) -> Optional[str]:
    """Rejection reason token for the WebSocket peer IP under listener *state*
    (``auth_required`` / ``bound_host``), or None when allowed.

    Loopback bind: only loopback peers (the legacy ``?token=`` is the only auth,
    LAN hosts must not get to guess it); an empty peer fails closed.  Explicit
    non-loopback bind (``--insecure``) or gated mode: any peer — DNS-rebinding is
    blocked by :func:`ws_host_origin_reason`, and in gated mode
    ``ws.client.host`` is the X-Forwarded-For value anyway.
    """
    if getattr(state, "auth_required", False):
        return None
    bound_host = (getattr(state, "bound_host", "") or "").strip().lower()
    if bound_host and bound_host not in _LOOPBACK_HOSTS:
        return None
    client_host = ws.client.host if ws.client else ""
    if not client_host:
        return f"missing_or_empty_peer bound={bound_host or '?'}"
    if client_host in _LOOPBACK_HOSTS:
        return None
    return f"peer_not_loopback peer={client_host} bound={bound_host or '?'}"


def ws_host_origin_reason(ws, state) -> Optional[str]:
    """``host_mismatch …`` / ``origin_mismatch …`` under listener *state*, or None when allowed.

    HTTP middleware does not run for WebSocket routes, so the DNS-rebinding
    Host check is repeated here; an Origin header, when present, must target the
    bound host.  Non-web origins (packaged Electron: file://, null, app://) are
    trusted — the credential check is the real auth boundary there.
    """
    bound_host = getattr(state, "bound_host", None)
    if not bound_host:
        return None
    trusted_public_hosts = getattr(state, "trusted_public_hosts", frozenset())
    host_header = ws.headers.get("host", "")
    if not _is_accepted_host(host_header, bound_host, trusted_public_hosts):
        return f"host_mismatch host={host_header or '?'} bound={bound_host}"
    origin = ws.headers.get("origin", "")
    if not origin:
        return None
    try:
        parsed = urllib.parse.urlparse(origin)
    except ValueError:  # malformed authority, e.g. "http://[::1" — fail closed
        parsed = None
    if parsed is not None and parsed.scheme not in {"http", "https"}:
        return None
    if parsed is None or not parsed.netloc or not _is_accepted_host(parsed.netloc, bound_host, trusted_public_hosts):
        return f"origin_mismatch origin={origin} bound={bound_host}"
    return None
