"""``hermes gateway tickets``: REST tickets for a client that reaches the gateway over SSH.

``--probe`` says what this host can mint and touches nothing. ``--serve`` stays open on one channel:
a hello line, then one reply line per request line, so a client does not pay an SSH exec per HTTP
call. It never starts a gateway (that is ``hermes gateway attach --start``). A ticket is a
credential for the process that asked: it goes to stdout only, never to stderr or a log.
"""
import json
import sys
from contextlib import redirect_stdout

TICKETS_PROTOCOL = 1
# What ``--serve`` mints. The WebSocket's ``interactive`` ticket stays with ``hermes gateway attach``.
TICKET_PURPOSES = ("native-http",)
# Longest request line ``--serve`` reads, newline excluded. A longer one is refused unread.
MAX_REQUEST_LINE_BYTES = 64 * 1024


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


def _reply(line: bytes | None, mint) -> dict:
    """The one reply to a request line; ``mint(purpose)`` returns a ticket or raises TicketUnavailable."""
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
    try:
        ticket = mint(purpose)
    except TicketUnavailable as exc:
        detail = f" ({exc.detail})" if exc.detail else ""
        return refuse(request_id, "ticket_unavailable", f"gateway refused the ticket: {exc.reason}{detail}")
    return {"id": request_id, "ok": True, "ticket": ticket, "expires_in": TicketStore.TTL_SECONDS}


def cmd_gateway_tickets(args) -> None:
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import attachable, discover_gateway_endpoint, gateway_profile_home
    from hermes_cli.gateway_runtime_attach import DISCOVER_TIMEOUT, failure_message, mint_ticket
    from hermes_cli.gateway_runtime_cli import _verdict_payload, ensure_exit_code

    out = sys.stdout
    if args.probe:
        _write(out, {"protocol": TICKETS_PROTOCOL, "purposes": list(TICKET_PURPOSES)})
        return
    # Legacy imported utilities can print; protocol stdout belongs only to us.
    with redirect_stdout(sys.stderr):
        home = gateway_profile_home(get_hermes_home())
        receipt = discover_gateway_endpoint(home, timeout=DISCOVER_TIMEOUT)
        if not attachable(receipt):
            _write(out, {"type": "error", "error": {"code": receipt.state, "reason_code": receipt.reason_code,
                                                    "message": failure_message(receipt)}})
            raise SystemExit(ensure_exit_code(receipt) or 7)
        try:
            _write(out, {"type": "hello", "protocol": TICKETS_PROTOCOL,
                         "endpoint": _verdict_payload(receipt)["endpoint"]})
            for line in _request_lines(sys.stdin.buffer):
                _write(out, _reply(line, lambda purpose: mint_ticket(home, receipt.endpoint, purpose=purpose)))
        except BrokenPipeError:
            # The client closed the channel; nothing is left to answer.
            raise SystemExit(0) from None
