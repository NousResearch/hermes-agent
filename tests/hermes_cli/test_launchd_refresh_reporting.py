"""launchd plist refresh must not report success when launchd never registered the service (#12866)."""
from gateway import launchd_service
import subprocess
from unittest.mock import MagicMock



def _stale_plist(tmp_path, monkeypatch, *, registered: bool):
    plist_path = tmp_path / "com.hermes.plist"
    plist_path.write_text("<old/>", encoding="utf-8")
    monkeypatch.setattr(launchd_service, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(launchd_service, "launchd_plist_is_current", lambda: False)
    monkeypatch.setattr(launchd_service, "generate_launchd_plist", lambda: "<new/>")
    monkeypatch.setattr(launchd_service, "refuse_temp_home_write", lambda *a: False)
    monkeypatch.setattr(launchd_service, "get_launchd_label", lambda: "com.hermes.agent")
    monkeypatch.setattr(launchd_service, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(launchd_service, "_append_launchd_reload_log", lambda msg: None)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr(launchd_service, "_retry_launchctl_bootstrap_until_registered", lambda *a, **k: registered)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: MagicMock(returncode=0))


def test_refresh_reports_registration_outcome(tmp_path, monkeypatch, capsys):
    _stale_plist(tmp_path, monkeypatch, registered=False)
    assert launchd_service.refresh_launchd_plist_if_needed() is False
    assert "Updated" not in capsys.readouterr().out

    _stale_plist(tmp_path, monkeypatch, registered=True)
    assert launchd_service.refresh_launchd_plist_if_needed() is True
    assert "Updated" in capsys.readouterr().out


