"""Who holds the address a ``hermes dashboard`` / ``hermes serve`` was asked to bind, when it is this
home's own gateway.

A host moved to the gateway serves the Webapp from the gateway's listener, often on the address the
old backend used (9119). A launcher left behind (a hand-written unit with ``Restart=always``) then
fails its bind with the generic port-in-use hint, exit 75 (``EX_TEMPFAIL``: try again).
When the holder is this home's gateway, retrying never helps: :func:`gateway_holding_port` proves
it from the gateway's own answer on its private control socket (same OS user only, its ``identify``
names the listener's port) and from the OS's socket table of that gateway process (its listener takes
the requested address), never from an argv listing, and the caller says so and exits
``EX_CONFIG`` (78), the code a supervisor can be told to stop on (``RestartPreventExitStatus=78``).

Callers ask only once the port is known to be held (a failed bind, or a held port the status listing
could not attribute to a listed backend): a free port never reaches the gateway's control socket.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit


@dataclass(frozen=True)
class GatewayHolder:
    """This home's gateway, whose listener holds the requested port."""

    control_home: Path
    api_origin: str  # the listener's loopback dial origin, from the gateway's ``identify``
    pid: Optional[int]
    webapp_state: str  # serving | unavailable | off | starting | unknown
    webapp_reason: Optional[str]
    public_url: str


def _webapp_status(control_home: Path) -> dict:
    """The gateway's own ``webapp-status`` answer, ``{}`` when it gives none."""
    from hermes_cli.webapp_launcher import GatewayUnreachable, _control
    try:
        response = _control(control_home, "webapp-status")
    except GatewayUnreachable:
        return {}
    result = response.get("result") if response.get("ok") else None
    return result if isinstance(result, dict) else {}


def _gateway_pid(control_home: Path) -> Optional[int]:
    from gateway.status import live_gateway_pid_for_home
    try:
        return live_gateway_pid_for_home(control_home)
    except (OSError, ValueError):  # an unreadable record only drops the PID from the message
        return None


def _live_endpoint(target: Path):
    """The endpoint the live gateway serving *target* names in its ``identify``, or None.

    Read-only, unlike ``discover_gateway_endpoint``: when no control socket answers, that call judges
    (and narrows the mode of) ``gateway.lock``; this one only connects. A named profile without a
    socket of its own is asked of the default multiplexer, as discovery does. An overlay home or an
    owner that is starting, draining or refusing is None.
    """
    from hermes_cli.gateway_runtime import _canonical_home, _endpoint
    from hermes_cli.gateway_runtime_discovery import DiscoveryError, query_identify
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for
    home = Path(_canonical_home(target))
    for control_home in (home, multiplexer_root_for(home)):
        if control_home is None:
            return None
        try:
            payload = query_identify(control_home, timeout=2.0)
        except (FileNotFoundError, ConnectionRefusedError):
            continue
        except (OSError, ValueError, TypeError, DiscoveryError):
            return None
        return _endpoint(payload, home, control_home=None if control_home == home else control_home).endpoint
    return None


def _listening_addresses(pid: Optional[int], port: int) -> list[str]:
    """The addresses *pid* holds a TCP listener on at *port*, from the OS; [] when it will not say."""
    if pid is None:
        return []
    try:
        import psutil
    except ImportError:
        return []
    try:
        connections = psutil.Process(pid).net_connections(kind="inet")
    except (psutil.Error, OSError):
        return []
    return [c.laddr.ip for c in connections if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port]


def _requested_address(host: str, port: int):
    """The address a bind of *host* (as ``_port_bind_conflict`` probes it) would take, or None."""
    import ipaddress
    import socket
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        return ipaddress.ip_address(socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)[0][4][0])
    except (OSError, ValueError, IndexError):
        return None


def _binds_overlap(held: str, host: str, port: int) -> bool:
    """Whether a listener on address *held* takes the address a bind of *host* asks for, proven in
    the probe's own address family: the same address, or a wildcard of that family on either side.
    Across families nothing is proven (the old command's bind probe sets no dual-stack option, and
    IPv6 binds IPv6 only on Windows), so it is no overlap; so is anything that cannot be parsed."""
    import ipaddress
    wanted = _requested_address(host, port)
    try:
        bound = ipaddress.ip_address(held.split("%", 1)[0])
    except ValueError:
        return False
    if wanted is None or bound.version != wanted.version:
        return False
    return bound == wanted or bound.is_unspecified or wanted.is_unspecified


def running_gateway(home: Optional[Path] = None, *, port: Optional[int] = None,
                    host: Optional[str] = None) -> Optional[GatewayHolder]:
    """This home's running gateway and what it says about the Webapp, or None.

    *home* defaults to the active Hermes home; a named profile served by the default multiplexer
    resolves to that gateway. With *port*, a gateway whose listener is bound elsewhere is None, decided
    from its ``identify`` alone before anything else is asked. With *host* as well, the gateway process
    must also hold a listener on *port* whose address takes *host*'s in the same address family (the
    OS's own socket table);
    without that proof it is None. Never starts a gateway and never touches the home's files: anything
    it cannot prove (no gateway, one still starting, a refused or slow socket) is None.
    """
    from gateway.webapp_listener import webapp_settings
    from hermes_constants import get_hermes_home
    target = Path(home) if home is not None else get_hermes_home()
    endpoint = _live_endpoint(target)
    if endpoint is None or (port is not None and urlsplit(endpoint.api_origin).port != port):
        return None
    control_home = Path(endpoint.control_home or target)
    pid = _gateway_pid(control_home)
    if host is not None and port is not None and not any(
            _binds_overlap(held, host, port) for held in _listening_addresses(pid, port)):
        return None
    status = _webapp_status(control_home)
    return GatewayHolder(control_home=control_home, api_origin=endpoint.api_origin, pid=pid,
                         webapp_state=str(status.get("state") or "unknown"),
                         webapp_reason=status.get("reason"), public_url=webapp_settings(control_home).public_url)


def gateway_holding_port(port: int, home: Optional[Path] = None, *,
                         host: Optional[str] = None) -> Optional[GatewayHolder]:
    """This home's running gateway when its listener is bound to *port* (the listener's own bound
    origin, from the gateway's ``identify``), else None. With *host*, only when the gateway's live
    listener on *port* also takes *host*'s address (see :func:`running_gateway`). Port 0 (an
    ephemeral bind) is never held."""
    return running_gateway(home, port=port, host=host) if port else None


def _listener_serves_webapp(holder: GatewayHolder) -> bool:
    """The Webapp's listener is up: serving, or up with its bundle not built yet (the gateway reports
    that as ``unavailable`` but keeps the address)."""
    from gateway.webapp_listener import NOT_BUILT
    return holder.webapp_state == "serving" or (
        holder.webapp_state == "unavailable" and str(holder.webapp_reason or "").startswith(NOT_BUILT))


def gateway_webapp_summary(host: str, port: int, home: Optional[Path] = None) -> Optional[str]:
    """One line for ``hermes dashboard --status`` when this home's gateway serves the Webapp on the
    concrete address *host*:*port* the status probe reached (its live listener takes that address),
    else None."""
    holder = gateway_holding_port(port, home, host=host)
    if holder is None or not _listener_serves_webapp(holder):
        return None
    pid = f" (PID {holder.pid})" if holder.pid else ""
    line = (f"The Hermes gateway{pid} serves the Webapp for this profile at "
            f"{holder.public_url or holder.api_origin} (hermes webapp opens it).")
    if holder.webapp_state != "serving":
        line += f" {holder.webapp_reason}"
    return line


def launcher_examples(platform: str) -> str:
    """What usually starts an old backend on *platform* (``sys.platform``), for the hint above."""
    return "Task Scheduler task or Startup-folder entry" if platform == "win32" else "systemd unit or launchd job"


def _command_name(headless: Optional[bool]) -> str:
    if headless is None:
        return "hermes dashboard / hermes serve"
    return "hermes serve" if headless else "hermes dashboard"


def describe_gateway_holder(holder: GatewayHolder, host: str, port: int, *,
                            headless: Optional[bool] = None) -> list[str]:
    """What to tell the person (or the log of the supervisor) that ran the old command."""
    command = _command_name(headless)
    pid = f" (PID {holder.pid})" if holder.pid else ""
    if _listener_serves_webapp(holder):
        lines = [f"  Port {port} is held by this profile's Hermes gateway{pid}: the gateway serves the "
                 f"Webapp (the dashboard and its chats) at this address now."]
        if holder.webapp_state != "serving":
            lines.append(f"  {holder.webapp_reason}")
        browser = holder.public_url or holder.api_origin
        lines.append(f"  Open it in a browser at {browser}/ , or run: hermes webapp")
        if host in ("0.0.0.0", "::"):
            lines.append(f"  From another device, use this machine's address on port {port}.")
        lines.append(f"  `{command}` is no longer needed here. Stop whatever starts it (for example its "
                     f"{launcher_examples(sys.platform)}), or pass --port <other> to run it beside the gateway.")
        return lines
    return [f"  Port {port} is held by this profile's Hermes gateway{pid} (its listener at "
            f"{holder.api_origin}). Retrying will not free it: pass --port <other> "
            "(--port 0 picks a free ephemeral port), or stop the gateway (hermes gateway stop)."]
