"""Tests for the persistent claude-cli live-session manager.

The real ``claude -p`` subprocess is never spawned; a scripted FakeProc feeds
stream-json events so lifecycle, watchdog, fingerprint respawn, orphaned
tool_use, and usage parsing are all exercised deterministically.
"""

import json
import threading

import pytest

from agent import claude_live_session as s


# ---------------------------------------------------------------------------
# Fake subprocess
# ---------------------------------------------------------------------------


class _FakeStream:
    """Blocking readline over a scripted list of lines, then EOF ("")."""

    def __init__(self, lines):
        self._lines = list(lines)
        self._idx = 0
        self._cv = threading.Condition()
        self._closed = False

    def readline(self):
        with self._cv:
            while self._idx >= len(self._lines) and not self._closed:
                self._cv.wait(timeout=0.05)
                if self._idx >= len(self._lines) and not self._closed:
                    return ""  # nothing more scripted → EOF-like for the reader
            if self._idx < len(self._lines):
                line = self._lines[self._idx]
                self._idx += 1
                return line
            return ""

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()


class _FakeStdin:
    def __init__(self):
        self.written = []
        self.closed = False

    def write(self, data):
        self.written.append(data)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeProc:
    def __init__(self, argv=None, stdout_lines=None, **kwargs):
        self.argv = argv or []
        self.pid = 4242
        self.returncode = None
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(stdout_lines or [])
        self.stderr = _FakeStream([])
        self.signals = []
        self._alive = True

    def poll(self):
        return None if self._alive else (self.returncode or 0)

    def wait(self, timeout=None):
        self._alive = False
        self.returncode = 0
        return 0

    def send_signal(self, sig):
        self.signals.append(sig)
        self._alive = False

    def kill(self):
        self._alive = False


def _line(obj):
    return json.dumps(obj) + "\n"


def _make_config(**overrides):
    base = dict(
        command="claude",
        argv=("claude", "-p"),
        cwd="/tmp",
        env={"HOME": "/home/x"},
        model="sonnet",
        effort="low",
        system_prompt_hash="sys1",
        mcp_config_hash="mcp1",
        auth_identity="oauth:abc",
    )
    base.update(overrides)
    return s.LiveSessionConfig(**base)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_stable_and_sensitive():
    a = _make_config()
    b = _make_config()
    assert a.fingerprint == b.fingerprint
    assert _make_config(model="opus").fingerprint != a.fingerprint
    assert _make_config(effort="high").fingerprint != a.fingerprint
    assert _make_config(system_prompt_hash="sys2").fingerprint != a.fingerprint
    assert _make_config(auth_identity="oauth:zzz").fingerprint != a.fingerprint


# ---------------------------------------------------------------------------
# Usage parsing
# ---------------------------------------------------------------------------


def test_usage_from_message_extracts_counters():
    row = s.usage_from_message(
        {
            "input_tokens": 10,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 900,
            "output_tokens": 20,
        }
    )
    assert row == {
        "input_tokens": 10,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 900,
        "output_tokens": 20,
    }


def test_usage_accumulates_across_assistant_events():
    lines = [
        _line({"type": "system", "subtype": "init", "session_id": "sess-1"}),
        _line({"type": "assistant", "message": {"usage": {"input_tokens": 4, "output_tokens": 1}, "content": [{"type": "text", "text": "hi"}]}}),
        _line({"type": "assistant", "message": {"usage": {"input_tokens": 2, "output_tokens": 3}, "content": [{"type": "text", "text": "there"}]}}),
        _line({"type": "result", "subtype": "success", "is_error": False}),
    ]
    session = s.LiveSession(_make_config(), popen=lambda *a, **k: FakeProc(stdout_lines=lines))
    session.spawn()
    result = session.send_turn("go", fresh=True, quiet_budget=2.0, hard_deadline=10.0)
    assert result.session_id == "sess-1"
    assert result.text == "hi\nthere"
    assert result.usage["input_tokens"] == 6
    assert result.usage["output_tokens"] == 4
    assert not result.orphaned_tool_use
    session.teardown()


# ---------------------------------------------------------------------------
# Tool_use / tool_result tracking + orphan detection
# ---------------------------------------------------------------------------


def test_matched_tool_use_and_result_not_orphaned():
    lines = [
        _line({"type": "system", "subtype": "init", "session_id": "sess-2"}),
        _line({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "get_x", "input": {"a": 1}}]}}),
        _line({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "42"}]}}),
        _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}}),
        _line({"type": "result", "subtype": "success"}),
    ]
    session = s.LiveSession(_make_config(), popen=lambda *a, **k: FakeProc(stdout_lines=lines))
    session.spawn()
    result = session.send_turn("go", fresh=False, quiet_budget=2.0, hard_deadline=10.0)
    assert result.text == "done"
    assert len(result.tool_uses) == 1
    assert result.orphaned_tool_use is False
    assert session._needs_fresh is False
    session.teardown()


def test_trailing_tool_use_without_result_is_orphaned():
    # No result event, tool_use never resolved, short quiet budget → watchdog
    # trips and the turn is flagged orphaned (needs fresh reseed, not resume).
    lines = [
        _line({"type": "system", "subtype": "init", "session_id": "sess-3"}),
        _line({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t9", "name": "slow", "input": {}}]}}),
    ]
    session = s.LiveSession(_make_config(), popen=lambda *a, **k: FakeProc(stdout_lines=lines))
    session.spawn()
    result = session.send_turn(
        "go", fresh=False, quiet_budget=0.2, tool_quiet_budget=0.2, hard_deadline=3.0
    )
    assert result.orphaned_tool_use is True
    assert session._needs_fresh is True
    session.teardown()


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


def test_send_turn_writes_user_envelope():
    proc = FakeProc(stdout_lines=[_line({"type": "result", "subtype": "success"})])
    session = s.LiveSession(_make_config(), popen=lambda *a, **k: proc)
    session.spawn()
    session.session_id = "sess-x"
    session.send_turn("hello world", fresh=True, quiet_budget=2.0, hard_deadline=5.0)
    written = json.loads(proc.stdin.written[0])
    assert written["type"] == "user"
    assert written["session_id"] == "sess-x"
    assert written["message"] == {"role": "user", "content": "hello world"}
    session.teardown()


# ---------------------------------------------------------------------------
# Registry: reuse / fingerprint respawn / LRU / idle
# ---------------------------------------------------------------------------


def test_registry_reuses_same_fingerprint():
    procs = []

    def factory(*a, **k):
        p = FakeProc(stdout_lines=[])
        procs.append(p)
        return p

    reg = s.LiveSessionRegistry(popen=factory)
    cfg = _make_config()
    a = reg.get_or_create("k1", cfg)
    b = reg.get_or_create("k1", cfg)
    assert a is b
    assert len(procs) == 1
    reg.shutdown()


def test_registry_respawns_on_fingerprint_change():
    procs = []

    def factory(*a, **k):
        p = FakeProc(stdout_lines=[])
        procs.append(p)
        return p

    reg = s.LiveSessionRegistry(popen=factory)
    first = reg.get_or_create("k1", _make_config(model="sonnet"))
    second = reg.get_or_create("k1", _make_config(model="opus"))
    assert first is not second
    assert len(procs) == 2
    assert first.proc is None  # old one torn down
    reg.shutdown()


def test_registry_lru_cap_evicts_least_recent():
    clock = {"t": 0.0}

    def now():
        clock["t"] += 1.0
        return clock["t"]

    reg = s.LiveSessionRegistry(
        popen=lambda *a, **k: FakeProc(stdout_lines=[]),
        lru_cap=2,
        clock=now,
    )
    reg.get_or_create("a", _make_config())
    reg.get_or_create("b", _make_config())
    reg.get_or_create("c", _make_config())  # evicts "a" (oldest activity)
    assert len(reg) == 2
    reg.shutdown()


def test_registry_reaps_idle():
    clock = {"t": 100.0}

    def now():
        return clock["t"]

    reg = s.LiveSessionRegistry(
        popen=lambda *a, **k: FakeProc(stdout_lines=[]),
        idle_timeout_s=10.0,
        clock=now,
    )
    reg.get_or_create("a", _make_config())
    clock["t"] = 1000.0  # far past idle timeout
    reg.get_or_create("b", _make_config())
    assert "a" not in reg._sessions
    reg.shutdown()


def test_registry_recover_uses_resume_when_not_orphaned():
    captured = {}

    def factory(argv=None, **k):
        captured.setdefault("argvs", []).append(list(argv or []))
        return FakeProc(argv=argv, stdout_lines=[])

    reg = s.LiveSessionRegistry(popen=factory)
    session = reg.get_or_create("k1", _make_config())
    session.session_id = "resume-me"
    session._needs_fresh = False
    reg.recover("k1")
    # Second spawn argv carries --resume resume-me
    assert "--resume" in captured["argvs"][1]
    assert "resume-me" in captured["argvs"][1]
    reg.shutdown()


def test_registry_recover_reseeds_fresh_when_orphaned():
    captured = {}

    def factory(argv=None, **k):
        captured.setdefault("argvs", []).append(list(argv or []))
        return FakeProc(argv=argv, stdout_lines=[])

    reg = s.LiveSessionRegistry(popen=factory)
    session = reg.get_or_create("k1", _make_config())
    session.session_id = "do-not-resume"
    session._needs_fresh = True  # orphaned tool_use
    reg.recover("k1")
    assert "--resume" not in captured["argvs"][1]
    reg.shutdown()


# ---------------------------------------------------------------------------
# Teardown signals
# ---------------------------------------------------------------------------


def test_teardown_signals_process():
    proc = FakeProc(stdout_lines=[])
    session = s.LiveSession(_make_config(), popen=lambda *a, **k: proc)
    session.spawn()
    session.teardown()
    assert proc.stdin.closed is True
    assert session.proc is None
