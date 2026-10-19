"""Interactive ``hermes gateway restart`` refuses while the gateway is busy or recently active."""
import time

import pytest

from hermes_cli import gateway as gw


@pytest.fixture
def live(monkeypatch):
    """Interactive terminal, fresh live snapshot; tests tweak ``record`` and the last-inbound age."""
    record = {"gateway_state": "running", "active_agents": 0, "updated_at": "now"}
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.delenv("HERMES_RESTART_FORCE", raising=False)
    monkeypatch.setattr("gateway.status.read_runtime_status", lambda *a, **k: record)
    monkeypatch.setattr("gateway.status.runtime_status_is_stale", lambda r: False)
    monkeypatch.setattr("gateway.status.runtime_status_pid_is_live", lambda r: True)
    monkeypatch.setattr(gw, "_last_inbound_age_s", lambda: None)
    return record


def test_refuses_when_agents_active(live, capsys):
    live["active_agents"] = 2
    with pytest.raises(SystemExit) as exc:
        gw._refuse_restart_while_busy(False)
    assert exc.value.code == 2
    assert "2 active work unit(s)" in capsys.readouterr().out


def test_refuses_when_recently_active(live, monkeypatch):
    monkeypatch.setattr(gw, "_last_inbound_age_s", lambda: 30.0)
    with pytest.raises(SystemExit) as exc:
        gw._refuse_restart_while_busy(False)
    assert exc.value.code == 2


def test_force_flag_and_env_bypass(live, monkeypatch):
    live["active_agents"] = 2
    gw._refuse_restart_while_busy(True)
    monkeypatch.setenv("HERMES_RESTART_FORCE", "1")
    gw._refuse_restart_while_busy(False)


def test_idle_and_quiet_passes(live, monkeypatch):
    gw._refuse_restart_while_busy(False)
    monkeypatch.setattr(gw, "_last_inbound_age_s", lambda: gw.RESTART_QUIET_S + 1)
    gw._refuse_restart_while_busy(False)


def test_non_interactive_caller_never_blocked(live, monkeypatch):
    live["active_agents"] = 2
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    gw._refuse_restart_while_busy(False)


def test_dead_snapshot_never_blocks(live, monkeypatch):
    live["active_agents"] = 2
    monkeypatch.setattr("gateway.status.runtime_status_pid_is_live", lambda r: False)
    gw._refuse_restart_while_busy(False)


def test_last_inbound_age_parses_log_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(gw, "get_hermes_home", lambda: tmp_path)
    (tmp_path / "logs").mkdir()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 60))
    (tmp_path / "logs" / "gateway.log").write_text(
        f"{stamp},123 INFO gateway.run: inbound message: platform=slack user=u\n"
        f"{stamp},456 INFO gateway.run: other line\n")
    assert 55 < gw._last_inbound_age_s() < 70
    (tmp_path / "logs" / "gateway.log").write_text("nothing\n")
    assert gw._last_inbound_age_s() is None
