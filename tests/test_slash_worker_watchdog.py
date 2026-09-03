import inspect
import io
import json
import threading
import time

from tui_gateway import slash_worker


def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to a subreaper/init.
    assert slash_worker._is_orphaned(1234, getppid=lambda: 999999) is True


def test_is_orphaned_false_when_direct_parent_is_unchanged():
    original_ppid = 1234
    assert slash_worker._is_orphaned(original_ppid, getppid=lambda: original_ppid) is False


def test_parent_death_watchdog_contract_has_no_create_time_plumbing():
    assert list(inspect.signature(slash_worker._is_orphaned).parameters) == [
        "original_ppid",
        "getppid",
    ]
    assert list(inspect.signature(slash_worker._start_parent_death_watchdog).parameters) == [
        "original_ppid",
    ]


def test_heartbeat_interval_is_half_the_slash_timeout_capped(monkeypatch):
    # Default: half of 45s, capped at 30s.
    monkeypatch.delenv("HERMES_TUI_SLASH_TIMEOUT_S", raising=False)
    monkeypatch.delenv("HERMES_SLASH_HEARTBEAT_S", raising=False)
    assert slash_worker._resolve_heartbeat_s() == 22.5

    # Small timeout: interval scales down but never below 0.5s.
    monkeypatch.setenv("HERMES_TUI_SLASH_TIMEOUT_S", "4")
    assert slash_worker._resolve_heartbeat_s() == 2.0
    monkeypatch.setenv("HERMES_TUI_SLASH_TIMEOUT_S", "0.2")
    assert slash_worker._resolve_heartbeat_s() == 0.5

    # Large timeout: capped at 30s.
    monkeypatch.setenv("HERMES_TUI_SLASH_TIMEOUT_S", "600")
    assert slash_worker._resolve_heartbeat_s() == 30.0

    # Explicit heartbeat override wins over the derived value.
    monkeypatch.setenv("HERMES_SLASH_HEARTBEAT_S", "1.5")
    assert slash_worker._resolve_heartbeat_s() == 1.5


def test_emit_heartbeats_writes_while_running_and_stops_on_done(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(slash_worker.sys, "stdout", buf)
    monkeypatch.setattr(slash_worker, "_HEARTBEAT_S", 0.02)

    done = threading.Event()
    t = threading.Thread(target=slash_worker._emit_heartbeats, args=(7, done), daemon=True)
    t.start()

    # Let at least two heartbeat intervals elapse, then stop the emitter.
    time.sleep(0.07)
    done.set()
    t.join(timeout=2.0)
    assert not t.is_alive()

    lines = [json.loads(line) for line in buf.getvalue().splitlines() if line]
    assert lines, "expected at least one heartbeat line while the command ran"
    assert all(line == {"id": 7, "heartbeat": True} for line in lines)

    # No further writes after done was observed.
    count_after = len(lines)
    time.sleep(0.06)
    lines_now = [line for line in buf.getvalue().splitlines() if line]
    assert len(lines_now) == count_after


def test_emit_heartbeats_broken_pipe_ends_quietly(monkeypatch):
    class _ClosedPipe:
        def write(self, _data):
            raise BrokenPipeError("closed")

        def flush(self):
            pass

    monkeypatch.setattr(slash_worker.sys, "stdout", _ClosedPipe())
    monkeypatch.setattr(slash_worker, "_HEARTBEAT_S", 0.01)

    done = threading.Event()
    t = threading.Thread(target=slash_worker._emit_heartbeats, args=(3, done), daemon=True)
    t.start()
    # The write failure must end the thread instead of raising from it.
    t.join(timeout=2.0)
    assert not t.is_alive()
    done.set()
