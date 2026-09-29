"""launchd plist refresh must not report success when launchd never registered the service (#12866)."""
import subprocess
from unittest.mock import MagicMock

from hermes_cli import gateway as gw


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
