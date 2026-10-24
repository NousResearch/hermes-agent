"""One client attach: find (or start) the profile's owner, then mint its single-use session ticket.

Shared by ``hermes gateway attach`` (any local client) and Ink's ``ui-tui/scripts/gateway_bootstrap.py``.
The ticket is a credential for the one process that dials: never log it, never put it in a repr.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.gateway_runtime import GatewayDiscovery

# Launch policy travels in session.create, not into daemon-wide defaults. That includes the Kanban
# board `hermes --tui` pins at launch (_pin_kanban_board_env): a daemon started from here would
# otherwise resolve every later chat's, Desktop window's and cron job's board to this terminal's
# launch-time board instead of the profile's `kanban/current`.
CLIENT_LAUNCH_ENV = (
    "HERMES_MODEL", "HERMES_INFERENCE_MODEL", "HERMES_TUI_PROVIDER", "HERMES_INFERENCE_PROVIDER",
    "HERMES_TUI_TOOLSETS", "HERMES_TUI_SKILLS", "HERMES_CWD", "TERMINAL_CWD", "HERMES_YOLO",
    "HERMES_ACCEPT_HOOKS", "HERMES_TUI_CHECKPOINTS", "HERMES_TUI_PASS_SESSION_ID", "HERMES_KANBAN_BOARD",
    "HERMES_TUI_CLIENT_ENV",
)
# A plain attach only asks the control socket; it never waits for a start.
DISCOVER_TIMEOUT = 5.0


def scrub_client_launch_env() -> None:
    """Drop ``CLIENT_LAUNCH_ENV`` from this process before an attach that may start a daemon (it
    inherits ``os.environ``). Only for a process that exists to attach: the CLI command and Ink's
    bootstrap."""
    for key in CLIENT_LAUNCH_ENV:
        os.environ.pop(key, None)


class TicketUnavailable(RuntimeError):
    """The owner was attachable but no ticket was minted (it went away, or refused the grant).
    ``reason`` is bounded; ``detail`` is a local repair hint when there is one."""

    def __init__(self, reason: str, detail: str | None = None):
        super().__init__(f"ticket_unavailable: {reason}")
        self.reason, self.detail = reason, detail


@dataclass(frozen=True)
class GatewayAttach:
    """``receipt`` is the discovery/ensure verdict; ``ticket`` is set only when it was attachable."""
    receipt: GatewayDiscovery
    ticket: str | None = field(default=None, repr=False)


def failure_message(receipt: GatewayDiscovery) -> str:
    """The verdict a client shows when there is no owner to attach: ``gateway <state>: <reason>``,
    then its detail (a refusal's remedy, or a start that died's own redacted output), each line
    still a verdict so a client that echoes verdict lines shows all of them."""
    head = f"gateway {receipt.state}: {receipt.reason_code or 'not ready'}"
    detail = receipt.detail
    if not detail:
        return head
    if receipt.reason_code == "runtime_exited":
        return f"{head}\n{detail}"
    lines = [line.strip() for line in str(detail).splitlines() if line.strip()]
    return "\n".join([head, *(f"gateway {receipt.state}: {line}" for line in lines)])


def attach_gateway(home: str | Path, *, start: bool, recover: bool = False, timeout: float) -> GatewayAttach:
    """Attach a local client to the owner serving ``home`` (the caller's profile home).

    ``start`` ensures the owner (a client's first launch); otherwise this only discovers it, except
    ``recover``, which re-ensures whenever the owner is not ready: crashed, ended by its idle exit,
    stopped with ``hermes gateway stop``, or an older Hermes's gateway (ensure replaces that one). A
    restart's after-turn wait is attachable as it is (#139041). ``timeout`` bounds an ensure; a
    discovery waits at most ``DISCOVER_TIMEOUT`` of it. A daemon this starts ends itself once idle.
    A failed mint (control socket I/O, a refused or changed grant) raises ``TicketUnavailable``.
    """
    from hermes_cli.gateway_client import _session_ticket
    from hermes_cli.gateway_runtime import (attachable, discover_gateway_endpoint, ensure_gateway_runtime,
                                           gateway_profile_home)
    from hermes_cli.gateway_runtime_discovery import DiscoveryError

    home = gateway_profile_home(home)
    receipt = (ensure_gateway_runtime(home, timeout=timeout, idle_exit=True) if start
               else discover_gateway_endpoint(home, timeout=min(timeout, DISCOVER_TIMEOUT)))
    if not start and recover and not attachable(receipt):
        receipt = ensure_gateway_runtime(home, timeout=timeout, idle_exit=True)
    if not attachable(receipt):
        return GatewayAttach(receipt)
    try:
        # A profile served by the default multiplexer has no socket of its own; the host mints its ticket.
        return GatewayAttach(receipt, _session_ticket(home, receipt.endpoint))
    except DiscoveryError as exc:
        raise TicketUnavailable(exc.reason, exc.bounded_detail(500)) from exc
    except (OSError, ValueError) as exc:
        # Raw socket errors and replies are not echoed (paths, peer data): a bounded class name only.
        raise TicketUnavailable(type(exc).__name__) from exc
