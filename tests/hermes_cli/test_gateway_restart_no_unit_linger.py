"""``hermes gateway restart`` without an installed service unit must actually restart.

Linger is a systemd-unit concern. With no unit installed the detached ``run_gateway`` IS the
restart, yet the linger check ran unconditionally and bailed with exit 0 on any Linux login
session whose linger is off — the Desktop/dashboard "Restart gateway" action then reported
success while nothing restarted and the saved credentials never loaded.
"""
from types import SimpleNamespace

import pytest


@pytest.mark.linux_only
def test_restart_without_service_unit_runs_gateway_even_without_linger(monkeypatch):
    from hermes_cli import gateway as gw

    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: None)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gw, "get_systemd_linger_status", lambda *a, **k: (False, "Linger=no"))
    monkeypatch.setattr(gw, "stop_profile_gateway", lambda: True)
    monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **k: None)
    started = []
    monkeypatch.setattr(gw, "run_gateway", lambda **k: started.append(k))

    gw._cmd_restart(SimpleNamespace(system=False, all=False))

    assert started, "no unit installed: restart must fall through to run_gateway, linger or not"


@pytest.mark.linux_only
def test_restart_with_systemd_unit_still_reports_missing_linger(monkeypatch):
    from hermes_cli import gateway as gw

    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: "systemd")

    def _failing_service_call(*a, **k):
        import subprocess
        raise subprocess.CalledProcessError(1, ["systemctl"])

    monkeypatch.setattr(gw, "_service_call", _failing_service_call)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gw, "get_systemd_linger_status", lambda *a, **k: (False, "Linger=no"))
    started = []
    monkeypatch.setattr(gw, "run_gateway", lambda **k: started.append(k))

    gw._cmd_restart(SimpleNamespace(system=False, all=False))

    assert not started, "a broken unit must surface the linger hint, not double-start a detached gateway"
