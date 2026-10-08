"""launchd plist refresh must not report success when launchd never registered the service (#12866)."""
import subprocess
from unittest.mock import MagicMock

from hermes_cli import gateway as gw
from hermes_cli import gateway_launchd as launchd

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
    assert gw.refresh_launchd_plist_if_needed() is gw.LaunchdReload.FAILED
    failed = capsys.readouterr().out
    assert "Updated gateway launchd service definition" not in failed
    assert "did not re-register" in failed

    _stale_plist(tmp_path, monkeypatch, registered=True)
    assert gw.refresh_launchd_plist_if_needed() is gw.LaunchdReload.REGISTERED
    assert "Updated gateway launchd service definition" in capsys.readouterr().out


def _stale_for_start(tmp_path, monkeypatch, *, deferred, registered):
    """Wire refresh_launchd_plist_if_needed's real internals so launchd_start()
    exercises the true deferred-vs-in-process branch instead of a bare bool stub.

    deferred=True  -> _spawn_deferred_launchd_reload succeeds (spawned only, NOT
                      a confirmed supervising PID).
    deferred=False -> in-process bootstrap path; `registered` is what
                      _retry_launchctl_bootstrap_until_registered reports.
    """
    plist_path = tmp_path / "com.hermes.plist"
    plist_path.write_text("<old/>", encoding="utf-8")
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gw, "launchd_plist_is_current", lambda: False)
    monkeypatch.setattr(gw, "generate_launchd_plist", lambda: "<new/>")
    monkeypatch.setattr(gw, "_refuse_temp_home_service_write", lambda *a: False)
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "com.hermes.agent")
    monkeypatch.setattr(gw, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(gw, "_append_launchd_reload_log", lambda msg: None)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 4242)
    monkeypatch.setattr(gw, "_spawn_deferred_launchd_reload", lambda **k: deferred)
    monkeypatch.setattr(gw, "_wait_for_pid_exit", lambda *a, **k: True)
    monkeypatch.setattr(
        gw, "_retry_launchctl_bootstrap_until_registered", lambda *a, **k: registered
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: MagicMock(returncode=0))
    return plist_path


def test_launchd_start_kickstarts_after_deferred_refresh(tmp_path, monkeypatch):
    """A deferred helper reload only means the reload was *spawned*, not that a
    supervising PID is up. launchd_start must still kickstart so the service is
    actually running; skipping it (and claiming "Service started") on a deferred
    True is the #121154 review bug."""
    _stale_for_start(tmp_path, monkeypatch, deferred=True, registered=False)

    kickstarts = []
    monkeypatch.setattr(launchd, "_launchctl_kickstart_current", lambda label: kickstarts.append(label))
    monkeypatch.setattr(launchd, "_launchd_bootstrap_and_kickstart", lambda *a, **k: True)
    oks = []
    monkeypatch.setattr(launchd, "_launchd_ok", lambda msg: oks.append(msg))

    gw.launchd_start()

    assert kickstarts == ["com.hermes.agent"]


def test_launchd_start_skips_kickstart_after_inprocess_registered_refresh(tmp_path, monkeypatch):
    """The in-process path returns only after a confirmed supervising PID. The
    plist has RunAtLoad/KeepAlive, so a second kickstart races another local
    gateway into existence (#5109) — skip it only here."""
    _stale_for_start(tmp_path, monkeypatch, deferred=False, registered=True)

    kickstarts = []
    monkeypatch.setattr(launchd, "_launchctl_kickstart_current", lambda label: kickstarts.append(label))
    monkeypatch.setattr(launchd, "_launchd_ok", lambda msg: None)

    gw.launchd_start()

    assert kickstarts == []


def test_launchd_start_kickstarts_when_refresh_is_noop(tmp_path, monkeypatch):
    plist_path = tmp_path / "com.hermes.plist"
    plist_path.write_text("<current/>", encoding="utf-8")
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "com.hermes.agent")
    monkeypatch.setattr(gw, "launchd_plist_is_current", lambda: True)
    kickstarts = []
    monkeypatch.setattr(launchd, "_launchctl_kickstart_current", lambda label: kickstarts.append(label))
    monkeypatch.setattr(launchd, "_launchd_ok", lambda msg: None)

    gw.launchd_start()

    assert kickstarts == ["com.hermes.agent"]


def test_install_repair_warns_instead_of_claiming_success(tmp_path, monkeypatch, capsys):
    plist_path = tmp_path / "com.hermes.plist"
    plist_path.write_text("<old/>", encoding="utf-8")
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gw, "launchd_plist_is_current", lambda: False)
    monkeypatch.setattr(gw, "refresh_launchd_plist_if_needed", lambda: gw.LaunchdReload.FAILED)
    monkeypatch.setattr("hermes_constants.display_hermes_home", lambda: "~/.hermes-work")

    gw.launchd_install(force=False)
    out = capsys.readouterr().out
    assert "Service definition updated" not in out
    assert "could not be reloaded" in out
    assert "~/.hermes-work/logs/launchd-reload.log" in out
