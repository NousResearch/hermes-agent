from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_hermes(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["HERMES_IGNORE_USER_CONFIG"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "runs", *args],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_runs_cli_json_commands(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.run_ledger import RunLedger

    ledger = RunLedger(run_id="run-cli", session_id="session-a", hermes_home=home)
    ledger.tool_started(tool_name="terminal", tool_call_id="call-a")
    ledger.tool_finished(tool_name="terminal", tool_call_id="call-a", status="ok")
    ledger.write_state_capsule(notes="operator note")

    result = _run_hermes(home, "list", "--json")
    assert result.returncode == 0, result.stderr
    listed = json.loads(result.stdout)
    assert listed["runs"][0]["run_id"] == "run-cli"
    assert listed["runs"][0]["event_count"] == 3

    result = _run_hermes(home, "events", "run-cli:1..2", "--json")
    assert result.returncode == 0, result.stderr
    events = json.loads(result.stdout)
    assert events["run_id"] == "run-cli"
    assert [event["event_seq"] for event in events["events"]] == [1, 2]
    assert events["truncated"] is False

    result = _run_hermes(home, "capsule", "run-cli", "--latest", "--json")
    assert result.returncode == 0, result.stderr
    capsule = json.loads(result.stdout)
    assert capsule["capsule"]["capsule_id"] == "cap_000000002"
    assert capsule["relative_path"] == "capsules/cap_000000002.json"

    result = _run_hermes(home, "capsule", "run-cli", "--capsule", capsule["relative_path"], "--json")
    assert result.returncode == 0, result.stderr
    capsule_by_relative_path = json.loads(result.stdout)
    assert capsule_by_relative_path["capsule"]["capsule_id"] == "cap_000000002"

    result = _run_hermes(home, "recover", "run-cli", "--json")
    assert result.returncode == 0, result.stderr
    recovery = json.loads(result.stdout)
    assert recovery["run_id"] == "run-cli"
    assert recovery["recovery"]["in_flight"] == {}
    assert recovery["recovery"]["recent_completed_tools"][0]["tool_call_id"] == "call-a"


def test_runs_cli_invalid_span_exits_nonzero_with_clear_stderr(tmp_path):
    home = tmp_path / "hermes-home"

    result = _run_hermes(home, "events", "../escape", "--json")

    assert result.returncode != 0
    assert "Error:" in result.stderr
    assert "run" in result.stderr or "span" in result.stderr
    assert not (home / "runs" / "escape").exists()


def test_runs_cli_capsule_rejects_absolute_path_with_parent_segment(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.run_ledger import RunLedger

    ledger = RunLedger(run_id="run-cli-safe", session_id="session-a", hermes_home=home)
    ledger.tool_started(tool_name="terminal", tool_call_id="call-a")
    ledger.write_state_capsule(notes="operator note")
    capsules_dir = home / "runs" / "run-cli-safe" / "capsules"
    (capsules_dir / "subdir").mkdir()
    unsafe_absolute = str(capsules_dir / "subdir" / ".." / "cap_000000001.json")

    result = _run_hermes(home, "capsule", "run-cli-safe", "--capsule", unsafe_absolute, "--json")

    assert result.returncode != 0
    assert "Error:" in result.stderr
    assert "unsafe" in result.stderr or "outside" in result.stderr
