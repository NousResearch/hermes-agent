"""#125390: a default-profile gateway supervised by s6 (container) reports as running.

``_cmd_status``'s final ``else`` branch decided liveness from ``snapshot.gateway_pids``
alone. In the s6-overlay container the gateway runs as a ``python -c`` wrapper argv the
process scan deliberately refuses to match (#123881) and there is no ``gateway.pid``
file, so a healthy ``gateway-default`` service printed a false
"✗ Gateway is not running". The branch now honors ``snapshot.service_running``.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from types import SimpleNamespace


def _run_status(monkeypatch, snapshot) -> str:
    import hermes_cli.gateway_profile_lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "print_parked_status", lambda: False)
    from hermes_cli import gateway as gw
    monkeypatch.setattr(gw, "get_gateway_runtime_snapshot", lambda system=False: snapshot)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda probe: None)
    monkeypatch.setattr(gw, "named_profile_served_by_running_multiplexer", lambda: False)
    for name in (
        "_print_runtime_health",
        "_print_multiplex_standalone_reason",
        "_print_served_ingress_urls",
        "_print_duplicate_credential_warnings",
        "_print_other_profiles_gateway_status",
        "_print_standalone_by_config",
    ):
        monkeypatch.setattr(gw, name, lambda *a, **k: None)

    buf = io.StringIO()
    with redirect_stdout(buf):
        gw._cmd_status(SimpleNamespace(deep=False, full=False, system=False))
    return buf.getvalue()


def test_s6_supervised_gateway_without_scannable_pid_reports_running(monkeypatch):
    """Service up, process scan empty (s6 container): must not report a false outage."""
    from hermes_cli.gateway import GatewayRuntimeSnapshot

    snapshot = GatewayRuntimeSnapshot(
        manager="s6 (container supervisor)",
        service_installed=True,
        service_running=True,
        gateway_pids=(),  # `python -c` launcher argv unmatched (#123881), no gateway.pid
    )
    out = _run_status(monkeypatch, snapshot)
    assert out.startswith("✓ Gateway is running (supervised by s6 (container supervisor))")
    assert "not running" not in out


def test_manual_gateway_without_pids_still_reports_stopped(monkeypatch):
    """No service, no PIDs (plain manual host): the stopped message and hints stay."""
    from hermes_cli.gateway import GatewayRuntimeSnapshot

    snapshot = GatewayRuntimeSnapshot(manager="manual process")
    out = _run_status(monkeypatch, snapshot)
    assert out.startswith("✗ Gateway is not running")
