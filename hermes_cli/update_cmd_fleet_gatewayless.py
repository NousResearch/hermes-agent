"""Gateway-less host evidence for the update-restart obligation (``update_cmd_fleet`` sibling).

An inventory-less obligation (the pre-inventory writer, or a tail that died before recording one)
has no owed set, so the live gateway matrix is its whole evidence. On a host that runs no gateway
(the Desktop app alone, #118742) that matrix is empty forever and the obligation could never
settle. An empty probe cannot tell that host from one whose gateway the dying update stopped, so
these readers ask the live host directly. A historical receipt is never consulted: it can belong
to an older update and says nothing about what runs now.
"""

from __future__ import annotations

from dataclasses import asdict


def runtime_outside_gateway_evidence(runtime: dict) -> bool:
    """A serve/dashboard row the gateway matrix neither covers nor needs to.

    Its supervisor owns the restart (Desktop backend, launchd/systemd unit, Windows service), or it
    is a manual serve whose restart ``defer_manual_serve`` has handed to its own durable reminder.
    Unclassified backends and failed transfers stay evidence against settlement (#115090, #111494).
    """
    from hermes_cli.update_cmd_fleet import _SUPERVISOR_OWNED_SERVE_BACKENDS
    from hermes_cli.update_serve_obligations import defer_manual_serve

    return runtime.get("kind") in ("serve", "dashboard") and (
        defer_manual_serve(runtime) or runtime.get("supervisor") in _SUPERVISOR_OWNED_SERVE_BACKENDS
    )


def host_owes_no_gateway_restart() -> bool:
    """True when no profile expects a gateway to be running and every live runtime is outside the matrix.

    A ``gateway_state.json`` that does not say ``stopped``/``startup_failed`` belongs to a gateway
    that went away without a clean stop, which is what an update that died mid-restart leaves
    behind; it keeps the obligation. Profiles that never ran a gateway have no record at all.
    """
    from gateway.status import read_runtime_status
    from hermes_cli.update_inventory import collect_runtime_inventory
    from hermes_cli.update_receipt import _NOT_EXPECTED_STATES, _profile_homes

    for _profile, home in _profile_homes():
        record = read_runtime_status(home / "gateway_state.json")
        if record is None:
            continue
        state = record.get("gateway_state") if isinstance(record, dict) else None
        if not (isinstance(state, str) and state in _NOT_EXPECTED_STATES):
            return False
    return all(runtime_outside_gateway_evidence(asdict(runtime)) for runtime in collect_runtime_inventory().runtimes)
