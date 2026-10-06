"""A pending restart is discharged by supervisor evidence, not an empty PID scan."""
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import gateway, main, update_cmd_fleet as fleet, update_receipt


@pytest.mark.parametrize("failure", ["listing", "restart", "inactive", "unloaded", None])
def test_pending_launchd_requires_complete_supervision(monkeypatch, tmp_path, failure):
    # Host-independent subprocess-boundary fixture, not native launchd validation.
    current, sibling = "ai.hermes.gateway", "ai.hermes.gateway-two"
    (tmp_path / f"{sibling}.plist").touch()
    monkeypatch.setattr(gateway, "get_launchd_label", lambda: current)
    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: tmp_path / f"{current}.plist")
    monkeypatch.setattr(gateway, "launchd_gateway_labels_for_install", lambda: [current, sibling])
    monkeypatch.setattr(fleet, "_restart_launchd_gateway_after_update", lambda **kw: ([], []))
    monkeypatch.setattr(gateway, "_locate_launchd_gateway_service", lambda _: (None, None) if failure == "unloaded" else ("gui/501", None))
    monkeypatch.setattr(gateway, "_wait_for_launchd_service_pid", lambda *a, **kw: None if failure == "inactive" else 42)

    def kickstart(*args):
        if failure == "restart":
            raise subprocess.CalledProcessError(1, "launchctl")

    monkeypatch.setattr(gateway, "_launchd_kickstart", kickstart)
    monkeypatch.setattr(fleet.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=int(failure == "listing"), stdout="", stderr=""))
    restarted, failed = [], []
    fleet._restart_macos_launchd_gateways(restarted, failed, 0, require_supervision=True)
    assert bool(failed) is bool(failure)
    assert (sibling in restarted) is (failure is None)
