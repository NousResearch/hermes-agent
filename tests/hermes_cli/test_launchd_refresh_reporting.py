"""launchd plist refresh must not report success when launchd never registered the service (#12866)."""
import subprocess
from unittest.mock import MagicMock

import pytest

from hermes_cli import gateway as gw
from hermes_cli.gateway_command_errors import explain_service_failure


def _stale_plist(tmp_path, monkeypatch, *, registered: bool):
    plist_path = tmp_path / "com.hermes.plist"
    plist_path.write_text("<old/>", encoding="utf-8")
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gw, "launchd_plist_is_current", lambda: False)
    monkeypatch.setattr(gw, "generate_launchd_plist", lambda: "<new/>")
    monkeypatch.setattr(gw, "_refuse_temp_home_service_write", lambda *a: False)
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "com.hermes.agent")
    monkeypatch.setattr(gw, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(gw, "_append_launchd_reload_log", lambda msg: None)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr(gw, "_retry_launchctl_bootstrap_until_registered", lambda *a, **k: registered)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: MagicMock(returncode=0))


def test_refresh_reports_registration_outcome(tmp_path, monkeypatch, capsys):
    _stale_plist(tmp_path, monkeypatch, registered=False)
    assert gw.refresh_launchd_plist_if_needed() is False
    assert "Updated" not in capsys.readouterr().out

    _stale_plist(tmp_path, monkeypatch, registered=True)
    assert gw.refresh_launchd_plist_if_needed() is True
    assert "Updated" in capsys.readouterr().out


def test_install_repair_warns_instead_of_claiming_success(tmp_path, monkeypatch, capsys):
    plist_path = tmp_path / "com.hermes.plist"
    plist_path.write_text("<old/>", encoding="utf-8")
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gw, "launchd_plist_is_current", lambda: False)
    monkeypatch.setattr(gw, "refresh_launchd_plist_if_needed", lambda: False)
    monkeypatch.setattr("hermes_constants.display_hermes_home", lambda: "~/.hermes-work")

    gw.launchd_install(force=False)

    out = capsys.readouterr().out
    assert "Service definition updated" not in out
    assert "could not be reloaded" in out
    assert "~/.hermes-work/logs/launchd-reload.log" in out


@pytest.mark.parametrize("registered", [True, False])
def test_forced_install_waits_for_old_job_and_reports_actual_registration(
    tmp_path, monkeypatch, capsys, registered
):
    """Regression for #120629: EIO during drain must not imply the old job survived."""
    _stale_plist(tmp_path, monkeypatch, registered=registered)
    calls = []
    monkeypatch.setattr(gw, "_launchctl_supervised_pid", lambda label: 417)
    monkeypatch.setattr(gw, "_wait_for_pid_exit", lambda pid, budget: calls.append(("wait", pid)) or True)
    monkeypatch.setattr(gw, "_retry_launchctl_bootstrap_until_registered",
                        lambda *a, **k: calls.append(("retry", a[2])) or registered)
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: calls.append(("bootout", cmd)) or MagicMock())
    monkeypatch.setattr(gw, "_clear_launchd_unsupported_marker", lambda: None)

    if registered:
        gw.launchd_install(force=True)
    else:
        with pytest.raises(SystemExit) as exc:
            gw.launchd_install(force=True)
        assert exc.value.code == 1
    out = capsys.readouterr().out
    assert [call[0] for call in calls] == ["bootout", "wait", "retry"]
    assert ("wait", 417) in calls
    assert ("retry", "com.hermes.agent") in calls
    assert ("Service installed and loaded!" in out) is registered
    if not registered:
        assert "may be unloaded" in out
        assert "launchctl bootstrap gui/501" in out


def test_launchctl_failure_guidance_does_not_claim_systemd():
    lines = explain_service_failure(subprocess.CalledProcessError(5, ["launchctl", "bootstrap"]))
    assert "launchd" in " ".join(lines)
    assert "systemd" not in " ".join(lines)
    assert "journalctl" not in " ".join(lines)
