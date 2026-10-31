"""``hermes update --when-idle`` at the commit point: a turn admitted after the last idle poll is never
put behind the update's drain. The pause asks ``pause-if-idle`` (checked and paused in one gateway
loop step); a busy answer calls the pause off, waits within the bound, and retries."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from hermes_cli import update_cmd_posix_pause as pause
from hermes_cli import update_cmd_when_idle as when_idle
from hermes_cli import update_pause_record as pause_record


@pytest.fixture
def one_bare_gateway(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    gateway = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)
    entry = {"pid": gateway.pid, "argv": ["hermes", "gateway", "run"], "home": str(home)}
    monkeypatch.setattr(pause, "_discover_systemd", list)
    monkeypatch.setattr(pause, "_discover_launchd", list)
    monkeypatch.setattr(pause, "_discover_bare", lambda service_pids: ([dict(entry)], []))
    monkeypatch.setattr(pause, "_drain_budget", lambda: 5.0)
    monkeypatch.setattr(pause, "_RUN", pause._empty_token())
    yield gateway
    gateway.kill()
    gateway.wait(timeout=10)


def _gateway_answering(monkeypatch, gateway, pause_if_idle_answers):
    """The gateway's control socket: ``pause-if-idle`` answers in turn; a drain it accepts exits it."""
    from gateway import control_socket
    asked, answers = [], iter(pause_if_idle_answers)

    def query(home, verb, *, params=None, timeout=0):
        asked.append(verb)
        answer = next(answers) if verb == "pause-if-idle" else {"pausing": True}
        if answer.get("pausing"):
            gateway.kill()
        return answer
    monkeypatch.setattr(control_socket, "query_gateway_control", query)
    return asked


@pytest.mark.platforms("posix")
def test_a_turn_admitted_before_the_pause_calls_it_off_until_the_bound(one_bare_gateway, monkeypatch, capsys):
    busy = {"busy": True, "active_work": [{"kind": "chat"}]}
    asked = _gateway_answering(monkeypatch, one_bare_gateway, [busy] * 50)
    monkeypatch.setattr(when_idle, "gateway_busy", lambda: "1 chat turn")
    monkeypatch.setattr(when_idle, "_pause_deadline", time.monotonic() + 0.3, raising=False)

    assert pause.pause_at_commit_point() == when_idle.BUSY_AT_PAUSE
    assert "pause-for-update" not in asked, f"the busy gateway was drained anyway: {asked}"
    assert one_bare_gateway.poll() is None, "the busy gateway was stopped"
    assert pause_record.read() is None, "the refused attempt left a pause record behind"
    assert "1 chat turn" in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_the_pause_retries_once_the_late_turn_finishes(one_bare_gateway, monkeypatch):
    asked = _gateway_answering(monkeypatch, one_bare_gateway,
                               [{"busy": True, "active_work": [{"kind": "chat"}]}, {"pausing": True}])
    monkeypatch.setattr(when_idle, "gateway_busy", lambda: "")
    monkeypatch.setattr(when_idle, "_pause_deadline", time.monotonic() + 30, raising=False)

    assert pause.pause_at_commit_point() is None
    # The accepted pause-if-idle already began the drain: no second (pause-for-update) request.
    assert asked == ["pause-if-idle", "pause-if-idle"], asked
    assert one_bare_gateway.wait(timeout=5) is not None
