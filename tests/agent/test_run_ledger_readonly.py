from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


def _events_path(home: Path, run_id: str) -> Path:
    return home / "runs" / run_id / "events.jsonl"


def _write_capsule(home: Path, run_id: str, seq: int, **extra) -> Path:
    capsules = home / "runs" / run_id / "capsules"
    capsules.mkdir(parents=True, exist_ok=True)
    path = capsules / f"cap_{seq:09d}.json"
    capsule = {
        "schema_version": 1,
        "capsule_id": f"cap_{seq:09d}",
        "run_id": run_id,
        "event_span": {"end_seq": seq},
        "in_flight": {},
        "recent_completed_tools": [],
        "artifact_refs": [],
        "blockers": [],
        "next_action": "continue",
        "notes": "",
        **extra,
    }
    path.write_text(json.dumps(capsule), encoding="utf-8")
    return path


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_run_listing_is_read_only_and_operator_useful(hermes_home):
    from agent.run_ledger import RunLedger
    from agent.run_ledger_reader import list_run_ledgers

    older = RunLedger(run_id="run-older", session_id="session-a", hermes_home=hermes_home)
    older.tool_started(tool_name="terminal", tool_call_id="call-live")
    older.tool_started(tool_name="read_file", tool_call_id="call-done")
    older.tool_finished(tool_name="read_file", tool_call_id="call-done", status="ok")
    _write_capsule(hermes_home, "run-older", 3)
    _events_path(hermes_home, "run-older").write_text(
        _events_path(hermes_home, "run-older").read_text(encoding="utf-8")
        + '{"event_seq":4,"type":"partial"',
        encoding="utf-8",
    )

    time.sleep(0.01)
    newer = RunLedger(run_id="run-newer", session_id="session-b", hermes_home=hermes_home)
    newer.tool_started(tool_name="search", tool_call_id="call-newer")
    _write_capsule(hermes_home, "run-newer", 1)

    summaries = list_run_ledgers(hermes_home=hermes_home)

    assert [summary["run_id"] for summary in summaries] == ["run-newer", "run-older"]
    older_summary = summaries[1]
    assert older_summary["event_count"] == 3
    assert older_summary["last_event_id"] == "evt_000000003"
    assert older_summary["last_event_type"] == "tool.finished"
    assert older_summary["latest_capsule"] == "capsules/cap_000000003.json"
    assert older_summary["in_flight_count"] == 1
    assert older_summary["corrupt_line_count"] == 1
    assert older_summary["run_root"] == str(hermes_home / "runs" / "run-older")

    missing_home = hermes_home.parent / "missing-home"
    assert list_run_ledgers(hermes_home=missing_home) == []
    assert not (missing_home / "runs").exists()


def test_event_span_parsing_supports_durable_handles():
    from agent.run_ledger_reader import RunLedgerReadError, parse_run_span

    span = parse_run_span("run-id:evt_000000002..evt_000000004")
    assert span.run_id == "run-id"
    assert span.start_seq == 2
    assert span.end_seq == 4

    span = parse_run_span("run-id:2..4")
    assert span.run_id == "run-id"
    assert span.start_seq == 2
    assert span.end_seq == 4

    span = parse_run_span("run-id")
    assert span.run_id == "run-id"
    assert span.start_seq is None
    assert span.end_seq is None

    for bad in ("../escape", "run/escape", "run-id:4..2", "run-id:abc..def"):
        with pytest.raises(RunLedgerReadError, match="run|range|span"):
            parse_run_span(bad)


def test_event_retrieval_redacts_span_filters_and_does_not_dereference_blobs(hermes_home):
    from agent.run_ledger import RunLedger
    from agent.run_ledger_reader import fetch_run_events

    ledger = RunLedger(
        run_id="run-events",
        session_id="session-a",
        hermes_home=hermes_home,
        config={"preview_chars": 32, "blob_threshold_chars": 40, "max_blob_bytes": 10_000},
    )
    ledger.tool_started(tool_name="terminal", tool_call_id="call-a", input={"cmd": "echo hi"})
    ledger.tool_finished(tool_name="terminal", tool_call_id="call-a", status="ok", output="x" * 200)
    ledger.tool_started(tool_name="read_file", tool_call_id="call-b")
    ledger.tool_failed(tool_name="terminal", tool_call_id="call-c", status="error")
    object_path = hermes_home / "runs" / "run-events" / ledger.read_events().events[1]["output"]["object_path"]
    object_path.write_text("SHOULD NOT BE READ", encoding="utf-8")

    result = fetch_run_events(
        "run-events:evt_000000002..evt_000000004",
        hermes_home=hermes_home,
        filters={"tool_name": "terminal"},
        limit=10,
    )

    assert [event["event_seq"] for event in result["events"]] == [2, 4]
    assert result["truncated"] is False
    assert result["next_start"] is None
    assert "SHOULD NOT BE READ" not in json.dumps(result)
    assert result["events"][0]["output"]["object_path"].startswith("objects/sha256/")


def test_active_torn_reads_are_safe_and_missing_reads_do_not_mutate(hermes_home):
    from agent.run_ledger import RunLedger
    from agent.run_ledger_reader import fetch_run_events, recover_run, RunLedgerReadError

    ledger = RunLedger(run_id="run-torn", session_id="session-a", hermes_home=hermes_home)
    ledger.tool_started(tool_name="terminal", tool_call_id="call-a")
    _events_path(hermes_home, "run-torn").write_text(
        _events_path(hermes_home, "run-torn").read_text(encoding="utf-8")
        + '{"event_seq":2,"type":"tool.finished"',
        encoding="utf-8",
    )

    events = fetch_run_events("run-torn", hermes_home=hermes_home)
    recovery = recover_run("run-torn", hermes_home=hermes_home)

    assert [event["event_seq"] for event in events["events"]] == [1]
    assert events["corrupt_lines"][0]["line_number"] == 2
    assert recovery["corrupt_lines"][0]["line_number"] == 2

    missing = hermes_home / "runs" / "missing-run"
    with pytest.raises(RunLedgerReadError, match="not found"):
        fetch_run_events("missing-run", hermes_home=hermes_home)
    assert not missing.exists()
    assert not (missing / "events.lock").exists()


def test_existing_lock_timeout_is_clear(hermes_home):
    from agent.run_ledger import RunLedger
    from agent.run_ledger_reader import RunLedgerLockTimeout, fetch_run_events

    pytest.importorskip("fcntl")
    import fcntl

    RunLedger(run_id="run-locked", session_id="session-a", hermes_home=hermes_home).tool_started(
        tool_name="terminal", tool_call_id="call-a"
    )
    lock_path = hermes_home / "runs" / "run-locked" / "events.lock"
    with lock_path.open("r", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(RunLedgerLockTimeout, match="timed out"):
                fetch_run_events("run-locked", hermes_home=hermes_home, lock_timeout_seconds=0.01)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def test_existing_lock_is_respected_when_events_file_is_absent(hermes_home):
    from agent.run_ledger_reader import RunLedgerLockTimeout, fetch_run_events, list_run_ledgers, recover_run

    pytest.importorskip("fcntl")
    import fcntl

    run_root = hermes_home / "runs" / "run-lock-no-events"
    run_root.mkdir(parents=True)
    lock_path = run_root / "events.lock"
    lock_path.write_text("", encoding="utf-8")

    with lock_path.open("r", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(RunLedgerLockTimeout, match="timed out"):
                fetch_run_events("run-lock-no-events", hermes_home=hermes_home, lock_timeout_seconds=0.01)
            with pytest.raises(RunLedgerLockTimeout, match="timed out"):
                recover_run("run-lock-no-events", hermes_home=hermes_home, lock_timeout_seconds=0.01)
            with pytest.raises(RunLedgerLockTimeout, match="timed out"):
                list_run_ledgers(hermes_home=hermes_home, lock_timeout_seconds=0.01)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def test_default_event_output_is_bounded_and_explicit(hermes_home):
    from agent.run_ledger import RunLedger
    from agent.run_ledger_reader import fetch_run_events

    ledger = RunLedger(run_id="run-many", session_id="session-a", hermes_home=hermes_home)
    for i in range(205):
        ledger.append_event("tool.started", tool_name="terminal", tool_call_id=f"call-{i}")

    result = fetch_run_events("run-many", hermes_home=hermes_home)

    assert len(result["events"]) == 200
    assert result["truncated"] is True
    assert result["next_start"] == "evt_000000201"
    assert result["matched_count"] == 200


def test_capsule_latest_and_explicit_resolution_are_safe(hermes_home, tmp_path):
    from agent.run_ledger import RunLedger
    from agent.run_ledger_reader import RunLedgerReadError, read_run_capsule

    RunLedger(run_id="run-capsules", session_id="session-a", hermes_home=hermes_home)
    cap3 = _write_capsule(hermes_home, "run-capsules", 3)
    cap7 = _write_capsule(hermes_home, "run-capsules", 7)

    latest = read_run_capsule("run-capsules", hermes_home=hermes_home, latest=True)
    assert latest["capsule"]["capsule_id"] == "cap_000000007"
    assert latest["relative_path"] == "capsules/cap_000000007.json"

    selected = read_run_capsule("run-capsules", hermes_home=hermes_home, capsule="cap_000000003")
    assert selected["capsule"]["capsule_id"] == "cap_000000003"

    selected_relative = read_run_capsule(
        "run-capsules",
        hermes_home=hermes_home,
        capsule=latest["relative_path"],
    )
    assert selected_relative["capsule"]["capsule_id"] == "cap_000000007"

    selected_path = read_run_capsule("run-capsules", hermes_home=hermes_home, capsule=str(cap3))
    assert selected_path["capsule"]["capsule_id"] == "cap_000000003"

    with pytest.raises(RunLedgerReadError, match="unsafe|outside"):
        read_run_capsule("run-capsules", hermes_home=hermes_home, capsule="../escape.json")

    inside_after_resolve = cap3.parent / "subdir" / ".." / cap3.name
    with pytest.raises(RunLedgerReadError, match="unsafe|outside"):
        read_run_capsule("run-capsules", hermes_home=hermes_home, capsule=str(inside_after_resolve))

    real_dir = cap3.parent / "real-dir"
    real_dir.mkdir()
    real_child = real_dir / "cap_000000011.json"
    real_child.write_text(json.dumps({"capsule_id": "cap_000000011"}), encoding="utf-8")
    linked_dir = cap3.parent / "linked-dir"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(RunLedgerReadError, match="symlink"):
        read_run_capsule(
            "run-capsules",
            hermes_home=hermes_home,
            capsule="linked-dir/cap_000000011.json",
        )

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"capsule_id": "outside"}), encoding="utf-8")
    symlink = hermes_home / "runs" / "run-capsules" / "capsules" / "cap_000000009.json"
    symlink.symlink_to(outside)
    with pytest.raises(RunLedgerReadError, match="symlink"):
        read_run_capsule("run-capsules", hermes_home=hermes_home, capsule="cap_000000009")

    assert cap7.exists()


def test_recovery_reconstructs_in_flight_completed_artifacts_and_corruption(hermes_home):
    from agent.run_ledger import RunLedger
    from agent.run_ledger_reader import recover_run

    ledger = RunLedger(run_id="run-recover", session_id="session-a", hermes_home=hermes_home)
    ledger.tool_started(tool_name="terminal", tool_call_id="call-live")
    ledger.tool_started(tool_name="read_file", tool_call_id="call-done")
    ledger.tool_finished(
        tool_name="read_file",
        tool_call_id="call-done",
        status="ok",
        artifact_refs=[{"type": "blob", "path": "objects/sha256/aa/digest", "safe_to_publish": False}],
        output={"content": "done"},
    )
    ledger.tool_started(tool_name="search", tool_call_id="call-failed")
    ledger.tool_failed(tool_name="search", tool_call_id="call-failed", status="error")
    ledger.tool_started(tool_name="skip", tool_call_id="call-skipped")
    ledger.tool_skipped(tool_name="skip", tool_call_id="call-skipped", status="skipped")
    _events_path(hermes_home, "run-recover").write_text(
        _events_path(hermes_home, "run-recover").read_text(encoding="utf-8")
        + '{"event_seq":99,"type":"partial"',
        encoding="utf-8",
    )

    recovery = recover_run("run-recover", hermes_home=hermes_home)

    assert list(recovery["recovery"]["in_flight"]) == ["call-live"]
    assert [item["tool_call_id"] for item in recovery["recovery"]["recent_completed_tools"]] == [
        "call-done",
        "call-failed",
        "call-skipped",
    ]
    assert recovery["recovery"]["artifact_refs"] == [
        {"type": "blob", "path": "objects/sha256/aa/digest", "safe_to_publish": False}
    ]
    assert recovery["corrupt_lines"][0]["line_number"] == 8


def test_recovery_max_completed_zero_returns_no_recent_completed_tools(hermes_home):
    from agent.run_ledger import RunLedger
    from agent.run_ledger_reader import recover_run

    ledger = RunLedger(run_id="run-zero-completed", session_id="session-a", hermes_home=hermes_home)
    ledger.tool_started(tool_name="terminal", tool_call_id="call-a")
    ledger.tool_finished(tool_name="terminal", tool_call_id="call-a", status="ok")

    recovery = recover_run("run-zero-completed", hermes_home=hermes_home, max_completed=0)

    assert recovery["recovery"]["recent_completed_tools"] == []


def test_fetch_rejects_symlinked_runs_root(hermes_home, tmp_path):
    from agent.run_ledger import RunLedger
    from agent.run_ledger_reader import RunLedgerReadError, fetch_run_events

    real_home = tmp_path / "real-home"
    RunLedger(run_id="run-behind-symlink", session_id="session-a", hermes_home=real_home).tool_started(
        tool_name="terminal",
        tool_call_id="call-a",
    )
    hermes_home.mkdir(parents=True)
    (hermes_home / "runs").symlink_to(real_home / "runs", target_is_directory=True)

    with pytest.raises(RunLedgerReadError, match="symlink|runs directory"):
        fetch_run_events("run-behind-symlink", hermes_home=hermes_home)
