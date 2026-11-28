"""Linux detached fallback when user-scope systemd is unreachable (port of
qwibitai/nanoclaw#3768/#3760): ``systemd_start`` degrades to a CLI-managed detached
gateway instead of aborting, and ``status`` labels it instead of calling it a manual run."""

from types import SimpleNamespace

import pytest

from hermes_cli import gateway as gw
from hermes_cli import gateway_systemd_fallback as fb


@pytest.fixture
def user_systemd_down(monkeypatch, tmp_path):
    monkeypatch.setattr(gw, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(fb, "get_hermes_home", lambda: tmp_path)
    unit = tmp_path / "hermes-gateway.service"
    unit.write_text("[Unit]\n", encoding="utf-8")
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit if not system else tmp_path / "none")

    def preflight(**_):
        raise gw.UserSystemdUnavailableError("User D-Bus session is not available (linger disabled).\n  more")

    monkeypatch.setattr(gw, "_preflight_user_systemd", preflight)
    monkeypatch.setattr(gw, "_run_systemctl", lambda *a, **k: pytest.fail("systemctl must not run without a user bus"))
    return tmp_path


def test_start_falls_back_to_detached_and_marks_it(user_systemd_down, monkeypatch, capsys):
    spawned = []
    monkeypatch.setattr(gw, "_spawn_detached_gateway", lambda: spawned.append(1) or True)
    monkeypatch.setattr(fb, "wait_for_detached_gateway", lambda timeout=0: 4242)

    gw.systemd_start()

    out = capsys.readouterr().out
    assert spawned == [1]
    assert "PID 4242" in out and "NOT auto-start" in out
    assert fb.marker_exists()
    # ``status`` explains the inactive unit instead of calling it a manual foreground/nohup run.
    monkeypatch.setattr("gateway.status.get_running_pid", lambda cleanup_stale=True: 4242)
    snapshot = SimpleNamespace(has_process_service_mismatch=True, gateway_pids=(4242,))
    gw._print_gateway_process_mismatch(snapshot)
    assert "user systemd cannot supervise it" in capsys.readouterr().out


def test_detached_child_that_dies_is_reported_not_claimed(user_systemd_down, monkeypatch, capsys):
    monkeypatch.setattr(gw, "_spawn_detached_gateway", lambda: True)
    monkeypatch.setattr(fb, "wait_for_detached_gateway", lambda timeout=0: None)

    gw.systemd_start()

    out = capsys.readouterr().out
    assert "exited before claiming its pid file" in out
    assert not fb.marker_exists()
