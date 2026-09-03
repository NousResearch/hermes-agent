"""Durable exact-call breaker prevents terminal timeout retry loops."""

import json
import os
import subprocess
import sys
from pathlib import Path

import agent.tool_timeout_circuit as circuit


def test_fingerprint_is_stable_and_sensitive():
    one = circuit.tool_call_fingerprint("terminal", {"command": "sleep 9", "timeout": 1}, "s1")
    reordered = circuit.tool_call_fingerprint("terminal", {"timeout": 1, "command": "sleep 9"}, "s1")
    changed = circuit.tool_call_fingerprint("terminal", {"command": "sleep 8", "timeout": 1}, "s1")
    other_session = circuit.tool_call_fingerprint("terminal", {"command": "sleep 9", "timeout": 1}, "s2")
    assert one == reordered
    assert one != changed
    assert one != other_session


def test_timeout_record_blocks_immediately_and_contains_no_plaintext(tmp_path, monkeypatch):
    path = tmp_path / "tool-timeout-circuit.json"
    monkeypatch.setattr(circuit, "_ledger_path", lambda: path)
    monkeypatch.setattr(circuit.time, "time", lambda: 1000.0)
    args = {"command": "SECRET_PAYLOAD", "timeout": 1}

    circuit.record_tool_timeout("terminal", args, "session-a")

    assert circuit.is_tool_timeout_blocked("terminal", args, "session-a") is True
    raw = path.read_text(encoding="utf-8")
    assert "SECRET_PAYLOAD" not in raw
    payload = json.loads(raw)
    assert payload["version"] == 1
    assert len(payload["entries"]) == 1
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert circuit._key_path().stat().st_mode & 0o777 == 0o600


def test_breaker_is_exact_session_scoped_and_expires(tmp_path, monkeypatch):
    path = tmp_path / "tool-timeout-circuit.json"
    monkeypatch.setattr(circuit, "_ledger_path", lambda: path)
    now = [1000.0]
    monkeypatch.setattr(circuit.time, "time", lambda: now[0])
    args = {"command": "sleep 9", "timeout": 1}
    circuit.record_tool_timeout("terminal", args, "session-a")

    assert circuit.is_tool_timeout_blocked("terminal", {**args, "timeout": 2}, "session-a") is True
    assert circuit.is_tool_timeout_blocked("terminal", {**args, "force": True}, "session-a") is True
    assert circuit.is_tool_timeout_blocked("terminal", args, "session-b") is False
    now[0] += circuit._TIMEOUT_RETRY_TTL_SECONDS + 1
    assert circuit.is_tool_timeout_blocked("terminal", args, "session-a") is False


def test_corrupt_or_unwritable_ledger_never_crashes(tmp_path, monkeypatch):
    path = tmp_path / "tool-timeout-circuit.json"
    path.write_text("bad json", encoding="utf-8")
    monkeypatch.setattr(circuit, "_ledger_path", lambda: path)
    assert circuit.is_tool_timeout_blocked("terminal", {}, "s") is False
    circuit.record_tool_timeout("terminal", {}, "s")


def test_fingerprint_is_keyed(monkeypatch):
    monkeypatch.setattr(circuit, "_fingerprint_key", lambda: b"a" * 32)
    first = circuit.tool_call_fingerprint("terminal", {"command": "x"}, "s")
    monkeypatch.setattr(circuit, "_fingerprint_key", lambda: b"b" * 32)
    second = circuit.tool_call_fingerprint("terminal", {"command": "x"}, "s")
    assert first != second


def test_two_process_records_do_not_lose_entries(tmp_path):
    root = tmp_path / "profile"
    marker = tmp_path / "go"
    script = """
import os
import time
from pathlib import Path
from agent.tool_timeout_circuit import record_tool_timeout

marker = Path(os.environ["START_MARKER"])
while not marker.exists():
    time.sleep(0.01)
record_tool_timeout(
    "terminal",
    {"command": os.environ["TEST_COMMAND"]},
    "s",
)
"""
    env = dict(os.environ)
    env["HERMES_HOME"] = str(root)
    env["START_MARKER"] = str(marker)
    children = []
    for command in ("one", "two"):
        child_env = {**env, "TEST_COMMAND": command}
        children.append(
            subprocess.Popen([sys.executable, "-c", script], env=child_env, cwd=str(Path(__file__).parents[2]))
        )
    marker.touch()
    for child in children:
        assert child.wait(timeout=20) == 0
    payload = json.loads((root / "cache" / "tool-timeout-circuit.json").read_text())
    assert len(payload["entries"]) == 2
