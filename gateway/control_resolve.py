"""``resolve-profile``: the control socket names the gateway that serves a profile home and how to dial it.

Part of the public local control-socket contract (``website/docs/developer-guide/gateway-internals.md``,
"Local control socket"). The answer is built from the discovery primitives ``hermes gateway attach``
runs, applied to the descriptor this gateway's ``identify`` verb answers with, so a client that asks
here gets the verdict ``discover_gateway_endpoint`` gives. It carries no credential: the ticket comes
from ``session-ticket`` on the same socket.
"""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any, Callable, Optional

RESOLVE_VERB = "resolve-profile"
RESOLVE_PROTOCOL = 1
_ENVELOPE_KEYS = frozenset({"protocol", "verb", "id", "params"})
_MAX_HOME_CHARS = 4096
_MAX_ID_CHARS = 128


def _requested_home(request: dict) -> str:
    """The ``home`` of a strict ``resolve-profile`` envelope; ``PermissionError`` otherwise."""
    rid = request.get("id")
    if (set(request) - _ENVELOPE_KEYS or type(request.get("protocol")) is not int or request["protocol"] != 1
            or not isinstance(request.get("params"), dict)
            or ("id" in request and not (type(rid) is int or (isinstance(rid, str) and len(rid) <= _MAX_ID_CHARS)))):
        raise PermissionError("invalid resolve envelope")
    params = request["params"]
    home = params.get("home")
    if (set(params) != {"home"} or not isinstance(home, str) or not 0 < len(home) <= _MAX_HOME_CHARS
            or "\0" in home or not os.path.isabs(home)):
        raise PermissionError("invalid resolve parameters")
    return home


def _install_code_sha() -> Optional[str]:
    """The commit installed on disk now (what ``attach --json`` reports as ``client_code_sha``).

    Read without touching the process's cached identity: ``identify``'s ``code_sha`` must keep naming
    the code this gateway booted from, which is how a client spots an owner that outlived an update."""
    from hermes_cli.version_info import _resolve_repo_dir, _run_git, _stamp_or_none
    stamp = _stamp_or_none()
    if stamp is not None:
        return stamp.commit
    repo_dir = _resolve_repo_dir()
    return _run_git(repo_dir, "rev-parse", "HEAD") if repo_dir is not None else None


def _verdict(payload: dict, requested: Path, own: Path):
    """``(control, verdict)`` exactly as discovery would reach this socket for *requested*.

    ``self``: this is the profile's own socket (discovery's first ``identify``). ``host``: this
    gateway's home is the multiplexer root of *requested*, the socket discovery asks when the
    profile has none of its own (``_served_by_multiplexer``, then the lock verdict). Any other home
    is not this gateway's to judge."""
    from hermes_cli import gateway_runtime as runtime
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for
    if requested == own:
        return "self", runtime._endpoint(payload, requested)
    if multiplexer_root_for(requested) != own:
        return None, runtime.GatewayDiscovery("inaccessible", reason_code="profile_mismatch")
    verdict = runtime._endpoint(payload, requested, control_home=own)
    if verdict.state == "inaccessible" and verdict.reason_code == "profile_mismatch":
        verdict = runtime._parked_by_multiplexer(payload, requested) or runtime._missing_owner_verdict(requested)
    return "host", verdict


def resolve_profile(home: str, *, identify: Callable[[], dict[str, Any]], own_home: Path) -> dict[str, Any]:
    """The ``resolve-profile`` result for the client's profile *home* (any spelling)."""
    from hermes_cli import gateway_runtime as runtime
    from hermes_cli.gateway_client import GATEWAY_WS_PROTOCOL, GATEWAY_WS_TICKET_PREFIX, gateway_ws_target
    from hermes_cli.gateway_runtime_attach import failure_message
    from hermes_constants import get_default_hermes_root

    requested = runtime._overlay_owner_home(Path(runtime._canonical_home(home)))
    own = Path(runtime._canonical_home(own_home))
    control, verdict = _verdict(identify(), requested, own)
    # The set ``ensure`` checks for this client ({its home, its process home, the default root}).
    fence_homes = {own, get_default_hermes_root()} | ({requested} if control else set())
    ok = runtime.attachable(verdict)
    endpoint = None
    if verdict.endpoint is not None:
        endpoint = {**asdict(verdict.endpoint), "capabilities": sorted(verdict.endpoint.capabilities),
                    "ws_url": gateway_ws_target(verdict.endpoint, "")[0],
                    "ws_protocol": GATEWAY_WS_PROTOCOL, "ticket_protocol_prefix": GATEWAY_WS_TICKET_PREFIX}
    return {"resolve_protocol": RESOLVE_PROTOCOL, "state": verdict.state, "reason_code": verdict.reason_code,
            "detail": verdict.detail, "attachable": ok, "update_fenced": runtime._update_fenced(fence_homes),
            "control": control, "message": None if ok else failure_message(verdict), "endpoint": endpoint,
            "install_code_sha": _install_code_sha()}


def resolve_profile_handler(identify: Callable[[], dict[str, Any]]) -> Callable[[dict, Path], dict]:
    """The ``GatewayControlServer.resolve_handler`` for a gateway answering ``identify`` from *identify*;
    the server passes the full request and its own home."""
    def handle(request: dict, own_home: Path) -> dict:
        return resolve_profile(_requested_home(request), identify=identify, own_home=own_home)
    return handle
