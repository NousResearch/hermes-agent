"""``hermes gateway tickets``: REST tickets for a client that reaches the gateway over SSH.

``--probe`` says what this host can mint and touches nothing. ``--serve`` stays open on one channel:
a hello line, then one reply line per request line, so a client does not pay an SSH exec per HTTP
call. It never starts a gateway (that is ``hermes gateway attach --start``). A ticket is a
credential for the process that asked: it goes to stdout only, never to stderr or a log.

A request may name another profile of the same gateway (``"profile"``), as Desktop's local
connection does for a ``?profile=`` route (apps/desktop/electron/local-gateway.ts,
``routedGatewayEndpoint``): the ticket is asked of the same control socket for that profile's home,
and the gateway refuses a home it does not serve.
"""
import json
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

TICKETS_PROTOCOL = 1
# What ``--serve`` mints. The WebSocket's ``interactive`` ticket stays with ``hermes gateway attach``.
# The two profile purposes are the grants Desktop's Profiles screen uses: ``native-profile-create``
# for ``POST /api/profiles``, ``native-profile-registry`` for ``PATCH``/``DELETE /api/profiles/<name>``
# (hermes_cli/dashboard_auth/native_http.py decides what each one may do).
TICKET_PURPOSES = ("native-http", "native-profile-create", "native-profile-registry")
# Longest request line ``--serve`` reads, newline excluded. A longer one is refused unread.
MAX_REQUEST_LINE_BYTES = 64 * 1024
PROFILE_NOT_SERVED = "profile_not_served"


def _write(out, payload) -> None:
    out.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    out.flush()


def _request_lines(stream):
    """Each stdin line; None stands for one over the limit, dropped as it is read, never held."""
    limit = MAX_REQUEST_LINE_BYTES + 1  # room for the newline
    while line := stream.readline(limit):
        if len(line) < limit or line.endswith(b"\n"):
            yield line
            continue
        while (rest := stream.readline(limit)) and not rest.endswith(b"\n"):
            pass
        yield None


def _profile_home(name: str) -> Path | None:
    """The home ``?profile=<name>`` selects on the gateway (``_resolve_profile_dir`` there), or None
    when no such profile exists. ``ValueError``: not a profile name."""
    from hermes_cli import profiles
    name = name.strip()
    profiles.validate_profile_name(name)
    return profiles.get_profile_dir(name).resolve() if profiles.profile_exists(name) else None


def _gateway_lacks(control_home: str, instance_id: str, profile_home: Path) -> bool:
    """The same gateway instance is still answering and does not list *profile_home* as served.
    Any doubt is False: the refusal then stays the generic ``ticket_unavailable``."""
    from hermes_cli.gateway_runtime_attach import DISCOVER_TIMEOUT
    from hermes_cli.gateway_runtime_discovery import query_identify
    try:
        payload = query_identify(Path(control_home), timeout=DISCOVER_TIMEOUT)
        served = {entry["profile_id"] for entry in payload["served_profiles"]}
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return payload.get("instance_id") == instance_id and str(profile_home) not in served


def _reply(line: bytes | None, mint) -> dict:
    """The one reply to a request line; ``mint(purpose, profile_home)`` returns a ticket or raises
    TicketUnavailable (``profile_home`` None: the profile the command runs in)."""
    from gateway.runtime_bootstrap import TicketStore
    from hermes_cli.gateway_runtime_attach import TicketUnavailable

    def refuse(request_id, code, message):
        return {"id": request_id, "ok": False, "error": {"code": code, "message": message}}

    if line is None:
        return refuse(None, "invalid_request", f"request line is longer than {MAX_REQUEST_LINE_BYTES} bytes")
    try:
        request = json.loads(line)
    except (ValueError, RecursionError):  # RecursionError: nesting too deep for the decoder
        return refuse(None, "invalid_request", "request line is not JSON")
    if not isinstance(request, dict):
        return refuse(None, "invalid_request", "request must be a JSON object")
    request_id = request.get("id")
    if isinstance(request_id, bool) or not isinstance(request_id, (int, str)):
        return refuse(None, "invalid_request", "id must be an integer or a string")
    purpose = request.get("purpose")
    if not isinstance(purpose, str) or purpose not in TICKET_PURPOSES:
        return refuse(request_id, "unsupported_purpose", "purpose must be one of: " + ", ".join(TICKET_PURPOSES))
    profile, profile_home = request.get("profile"), None
    if profile is not None:
        try:
            if not isinstance(profile, str):
                raise ValueError
            profile_home = _profile_home(profile)
        except ValueError:
            return refuse(request_id, "invalid_request", "profile must be a profile name")
        if profile_home is None:
            return refuse(request_id, "unknown_profile", "no such profile on this host")
    try:
        ticket = mint(purpose, profile_home)
    except TicketUnavailable as exc:
        if exc.reason == PROFILE_NOT_SERVED:
            return refuse(request_id, PROFILE_NOT_SERVED, "the running gateway does not serve this profile")
        detail = f" ({exc.detail})" if exc.detail else ""
        return refuse(request_id, "ticket_unavailable", f"gateway refused the ticket: {exc.reason}{detail}")
    return {"id": request_id, "ok": True, "ticket": ticket, "expires_in": TicketStore.TTL_SECONDS}


def cmd_gateway_tickets(args) -> None:
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import attachable, discover_gateway_endpoint, gateway_profile_home
    from hermes_cli.gateway_runtime_attach import DISCOVER_TIMEOUT, TicketUnavailable, failure_message, mint_ticket
    from hermes_cli.gateway_runtime_cli import _verdict_payload, ensure_exit_code

    out = sys.stdout
    if args.probe:
        from hermes_cli.gateway_runtime_attach import windows_service_session
        _write(out, {"protocol": TICKETS_PROTOCOL, "purposes": list(TICKET_PURPOSES),
                     "windows_service_session": windows_service_session()})
        return
    # Legacy imported utilities can print; protocol stdout belongs only to us.
    with redirect_stdout(sys.stderr):
        home = gateway_profile_home(get_hermes_home())
        receipt = discover_gateway_endpoint(home, timeout=DISCOVER_TIMEOUT)
        if not attachable(receipt):
            _write(out, {"type": "error", "error": {"code": receipt.state, "reason_code": receipt.reason_code,
                                                    "message": failure_message(receipt)}})
            raise SystemExit(ensure_exit_code(receipt) or 7)
        endpoint = receipt.endpoint

        def mint(purpose, profile_home):
            if profile_home is None or str(profile_home) == endpoint.profile_id:
                return mint_ticket(home, endpoint, purpose=purpose)
            # Same gateway, the sibling's home: its control socket mints for every home it serves.
            routed = replace(endpoint, profile_id=str(profile_home),
                             control_home=endpoint.control_home or endpoint.profile_id)
            try:
                return mint_ticket(home, routed, purpose=purpose)
            except TicketUnavailable as exc:
                if _gateway_lacks(routed.control_home, endpoint.instance_id, profile_home):
                    raise TicketUnavailable(PROFILE_NOT_SERVED) from exc
                raise

        try:
            _write(out, {"type": "hello", "protocol": TICKETS_PROTOCOL,
                         "endpoint": _verdict_payload(receipt)["endpoint"]})
            for line in _request_lines(sys.stdin.buffer):
                _write(out, _reply(line, mint))
        except BrokenPipeError:
            # The client closed the channel; nothing is left to answer.
            raise SystemExit(0) from None
