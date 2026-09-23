"""Which runtimes sit outside the gateway matrix's evidence (``update_cmd_fleet`` sibling)."""

from __future__ import annotations


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
