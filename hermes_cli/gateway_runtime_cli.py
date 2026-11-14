"""Machine-readable ensure/attach boundary. Only ``attach`` prints a credential: its one ticket."""
from dataclasses import asdict
import json
import math
import sys
from contextlib import redirect_stdout

INVALID_INVOCATION = {"state": "inaccessible", "reason_code": "invalid_invocation", "endpoint": None}


def ensure_exit_code(result) -> int:
    if result.reason_code == "deadline":
        return 5
    if result.reason_code in {"authorization", "profile_mismatch", "gateway_elevated", "gateway_other_session"}:
        return 4
    return {"ready": 0, "incompatible": 3, "draining": 6,
            "inaccessible": 7, "conflict": 7, "starting": 5, "absent": 5}[result.state]


def _verdict_payload(result) -> dict:
    payload = asdict(result)
    if result.endpoint:
        payload["endpoint"]["capabilities"] = sorted(result.endpoint.capabilities)
    # The code THIS client (and any gateway it starts) runs, resolved exactly like the owner's
    # ``endpoint.code_sha``: a client that sees them differ is attached to a gateway that
    # outlived ``hermes update`` and must restart it rather than re-attach.
    from hermes_cli.version_info import get_code_identity
    payload["client_code_sha"] = get_code_identity().get("sha")
    return payload


def _emit(payload, code) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    raise SystemExit(code)


def cmd_gateway_ensure(args) -> None:
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import ensure_gateway_runtime

    try:
        # Legacy imported utilities can print; protocol stdout belongs only to us.
        with redirect_stdout(sys.stderr):
            result = ensure_gateway_runtime(get_hermes_home(), timeout=float(args.timeout))
        payload, code = _verdict_payload(result), ensure_exit_code(result)
    except (ValueError, TypeError):
        payload, code = INVALID_INVOCATION, 2
    _emit(payload, code)


def cmd_gateway_attach(args) -> None:
    """``ensure``'s verdict plus, once attachable, the owner's WebSocket ``url``, its ``protocol``,
    the dial ``protocols`` and one single-use ``ticket`` for this profile. A refusal adds the
    ``message`` Ink's bootstrap shows (``gateway <state>: <reason>`` lines) and exits like ensure."""
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_client import GATEWAY_WS_PROTOCOL, gateway_ws_target
    from hermes_cli.gateway_runtime import GatewayDiscovery
    from hermes_cli.gateway_runtime_attach import (GatewayAttach, TicketUnavailable, attach_gateway,
                                                   failure_message, scrub_client_launch_env)

    try:
        timeout = float(args.timeout)
    except ValueError:
        timeout = math.nan
    if not math.isfinite(timeout) or timeout <= 0:
        _emit(INVALID_INVOCATION, 2)
    scrub_client_launch_env()
    with redirect_stdout(sys.stderr):
        try:
            attached = attach_gateway(get_hermes_home(), start=args.start, recover=args.recover, timeout=timeout)
        except TicketUnavailable as exc:
            # Attachable at discovery, then no ticket (the owner went away or refused the grant).
            attached = GatewayAttach(GatewayDiscovery("inaccessible", reason_code="ticket_unavailable",
                                                      detail=exc.detail or exc.reason))
        payload = _verdict_payload(attached.receipt)
    if attached.ticket is None:
        message = failure_message(attached.receipt)
        print(message, file=sys.stderr)
        _emit({**payload, "message": message}, ensure_exit_code(attached.receipt))
    url, protocols = gateway_ws_target(attached.receipt.endpoint, attached.ticket)
    _emit({**payload, "url": url, "protocol": GATEWAY_WS_PROTOCOL, "protocols": protocols,
           "ticket": attached.ticket}, 0)
