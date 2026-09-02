from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "rsi-audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("rsi_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            end_reason TEXT,
            last_activity_description TEXT,
            last_activity_at REAL,
            started_at REAL,
            ended_at REAL,
            profile_name TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            effect_disposition TEXT,
            timestamp REAL
        );
        """
    )
    con.close()


def _message(
    db: Path,
    session_id: str,
    role: str,
    content: str = "",
    *,
    tool_name: str | None = None,
    tool_calls: list[dict] | None = None,
    timestamp: float = 101.0,
) -> None:
    con = sqlite3.connect(db)
    con.execute(
        """INSERT INTO messages
           (session_id, role, content, tool_calls, tool_name, timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (session_id, role, content, json.dumps(tool_calls) if tool_calls else None, tool_name, timestamp),
    )
    con.commit()
    con.close()


def _session(
    db: Path,
    session_id: str,
    *,
    source: str = "cli",
    title: str = "fixture",
    end_reason: str | None = "cli_close",
    ended_at: float | None = 110.0,
) -> None:
    con = sqlite3.connect(db)
    con.execute(
        """INSERT INTO sessions
           (id, source, title, end_reason, last_activity_description,
            last_activity_at, started_at, ended_at, profile_name)
           VALUES (?, ?, ?, ?, '', 110, 100, ?, 'qa')""",
        (session_id, source, title, end_reason, ended_at),
    )
    con.commit()
    con.close()


@pytest.fixture
def audit_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    home = tmp_path / "home"
    state = home / ".hermes" / "profiles" / "qa" / "state.db"
    state.parent.mkdir(parents=True)
    _state_db(state)
    monkeypatch.setattr(module, "HOME", home)
    monkeypatch.setattr(module, "KANBAN", home / ".hermes" / "kanban.db")
    monkeypatch.setattr(module, "EXEC_DB", home / ".hermes" / "cron" / "executions.db")
    return module, state, home


def _only(module) -> dict:
    rows = module.scan_sessions("qa", since=90)
    assert len(rows) == 1
    return rows[0]


def test_user_failure_words_and_successful_tool_payload_are_not_failures(audit_env):
    module, db, _ = audit_env
    sid = "lexical-success"
    _session(db, sid)
    _message(db, sid, "user", "Explain why the previous command failed with error: timeout")
    _message(
        db,
        sid,
        "tool",
        json.dumps({"output": "0 failed, 12 passed", "exit_code": 0, "error": None}),
        tool_name="terminal",
    )
    _message(db, sid, "assistant", '{"status":"completed","failed_checks":0}')

    row = _only(module)

    assert row["failed"] is False
    assert row["fail_hits"] == []


def test_actual_tool_error_is_a_failure(audit_env):
    module, db, _ = audit_env
    sid = "tool-error"
    _session(db, sid)
    _message(db, sid, "user", "Run the check")
    _message(
        db,
        sid,
        "tool",
        json.dumps({"output": "", "exit_code": 1, "error": "command failed"}),
        tool_name="terminal",
    )
    _message(db, sid, "assistant", "The command did not complete.")

    row = _only(module)

    assert row["failed"] is True
    assert "tool:terminal:exit_code=1" in row["fail_hits"]


def test_failure_shaped_domain_payload_is_not_a_tool_error(audit_env):
    module, db, _ = audit_env
    sid = "domain-status"
    _session(db, sid)
    _message(db, sid, "user", "List failed jobs")
    _message(
        db,
        sid,
        "tool",
        json.dumps({"status": "failed", "failed": True, "items": ["historical-job"]}),
        tool_name="search_files",
    )
    _message(db, sid, "assistant", "Found one historical failed job.")

    row = _only(module)

    assert row["failed"] is False
    assert row["fail_hits"] == []


def test_tool_error_after_more_than_two_hundred_messages_is_detected(audit_env):
    module, db, _ = audit_env
    sid = "long-session"
    _session(db, sid)
    _message(db, sid, "user", "Run a long workflow")
    for index in range(205):
        _message(db, sid, "tool", json.dumps({"item": index}), tool_name="search_files")
    _message(
        db,
        sid,
        "tool",
        json.dumps({"output": "", "exit_code": 2, "error": "late failure"}),
        tool_name="terminal",
    )
    _message(db, sid, "assistant", "The final command failed.")

    row = _only(module)

    assert "tool:terminal:exit_code=2" in row["fail_hits"]


def test_ended_session_with_only_pre_tool_assistant_text_has_no_final_response(audit_env):
    module, db, _ = audit_env
    sid = "preamble-only"
    _session(db, sid)
    _message(db, sid, "user", "Inspect the files")
    _message(db, sid, "assistant", "I am checking now.")
    _message(db, sid, "tool", json.dumps({"items": []}), tool_name="search_files")

    row = _only(module)

    assert "session:missing_final_response" in row["fail_hits"]


def test_new_user_turn_requires_a_new_terminal_assistant_response(audit_env):
    module, db, _ = audit_env
    sid = "unanswered-followup"
    _session(db, sid)
    _message(db, sid, "user", "First turn", timestamp=100)
    _message(db, sid, "assistant", "First answer", timestamp=101)
    _message(db, sid, "user", "Second turn", timestamp=102)

    row = _only(module)

    assert "session:missing_final_response" in row["fail_hits"]


def test_scan_does_not_truncate_failures_after_thirty_sessions(audit_env):
    module, db, _ = audit_env
    for index in range(45):
        sid = f"session-{index:02d}"
        _session(db, sid)
        _message(db, sid, "user", "Do the work")
        if index != 44:
            _message(db, sid, "assistant", "Done.")

    rows = module.scan_sessions("qa", since=90)

    assert len(rows) == 45
    assert next(row for row in rows if row["id"] == "session-44")["failed"] is True


def test_ended_session_without_terminal_assistant_response_is_a_failure(audit_env):
    module, db, _ = audit_env
    sid = "missing-final"
    _session(db, sid)
    _message(db, sid, "user", "Output JSON only")

    row = _only(module)

    assert row["failed"] is True
    assert "session:missing_final_response" in row["fail_hits"]


def test_explicit_needs_input_block_event_is_a_failure(audit_env):
    module, db, _ = audit_env
    sid = "needs-input"
    _session(db, sid, source="kanban")
    _message(db, sid, "user", "Work kanban task t_fixture")
    _message(
        db,
        sid,
        "assistant",
        tool_calls=[{
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "kanban_block",
                "arguments": json.dumps({"kind": "needs_input", "reason": "Choose a format"}),
            },
        }],
    )
    _message(
        db,
        sid,
        "tool",
        json.dumps({"ok": True, "status": "blocked", "block_kind": "needs_input"}),
        tool_name="kanban_block",
    )
    _message(db, sid, "assistant", "Blocked pending the required decision.")

    row = _only(module)

    assert row["failed"] is True
    assert "lifecycle:needs_input" in row["fail_hits"]


def test_timed_out_clarification_is_a_needs_input_failure(audit_env):
    module, db, _ = audit_env
    sid = "clarify-timeout"
    _session(db, sid)
    _message(db, sid, "user", "Finish autonomously")
    _message(
        db,
        sid,
        "tool",
        json.dumps({"responses": [], "timed_out": True}),
        tool_name="clarify",
    )
    _message(db, sid, "assistant", "I could not continue without an answer.")

    row = _only(module)

    assert row["failed"] is True
    assert "lifecycle:needs_input" in row["fail_hits"]


def test_compression_boundary_without_final_text_is_not_missing_response(audit_env):
    module, db, _ = audit_env
    sid = "compressed-parent"
    _session(db, sid, end_reason="compression")
    _message(db, sid, "user", "Continue the task")

    row = _only(module)

    assert row["failed"] is False
    assert row["fail_hits"] == []


@pytest.mark.parametrize("end_reason", ["timeout", "exception", "cron_incomplete_no_output"])
def test_crash_and_timeout_end_reasons_are_failures(audit_env, end_reason: str):
    module, db, _ = audit_env
    sid = f"ended-{end_reason}"
    _session(db, sid, end_reason=end_reason)
    _message(db, sid, "user", "Do the work")
    _message(db, sid, "assistant", "Partial progress")

    row = _only(module)

    assert row["failed"] is True
    assert f"session:end_reason={end_reason}" in row["fail_hits"]


def _kanban_db(
    path: Path,
    *,
    task_status: str,
    run_status: str,
    outcome: str,
    error: str | None,
    older_run: tuple[str, str, str | None] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            status TEXT,
            current_run_id INTEGER,
            session_id TEXT,
            block_kind TEXT
        );
        CREATE TABLE task_runs (
            id INTEGER PRIMARY KEY,
            task_id TEXT,
            status TEXT,
            outcome TEXT,
            error TEXT,
            started_at INTEGER,
            ended_at INTEGER
        );
        """
    )
    con.execute(
        "INSERT INTO tasks VALUES ('t_fixture', ?, 7, 'durable-kanban', NULL)",
        (task_status,),
    )
    if older_run is not None:
        con.execute(
            "INSERT INTO task_runs VALUES (6, 't_fixture', ?, ?, ?, 100, 110)",
            older_run,
        )
        run_started, run_ended = 200, 210
    else:
        run_started, run_ended = 100, 110
    con.execute(
        "INSERT INTO task_runs VALUES (7, 't_fixture', ?, ?, ?, ?, ?)",
        (run_status, outcome, error, run_started, run_ended),
    )
    con.commit()
    con.close()


def test_failed_durable_kanban_lifecycle_overrides_plausible_final_text(audit_env):
    module, db, home = audit_env
    _session(db, "durable-kanban", source="kanban")
    _message(db, "durable-kanban", "user", "Work kanban task t_fixture")
    _message(db, "durable-kanban", "assistant", "Done. All checks passed.")
    _kanban_db(
        home / ".hermes" / "kanban.db",
        task_status="ready",
        run_status="completed",
        outcome="failed",
        error="worker crashed",
    )

    row = _only(module)

    assert row["failed"] is True
    assert "kanban:outcome=failed" in row["fail_hits"]


def test_successful_durable_kanban_lifecycle_ignores_lexical_noise(audit_env):
    module, db, home = audit_env
    _session(db, "durable-kanban", source="kanban")
    _message(db, "durable-kanban", "user", "Investigate failed jobs")
    _message(
        db,
        "durable-kanban",
        "tool",
        json.dumps({"output": "found failed labels", "exit_code": 0, "error": None}),
        tool_name="terminal",
    )
    _message(db, "durable-kanban", "assistant", "Done.")
    _kanban_db(
        home / ".hermes" / "kanban.db",
        task_status="done",
        run_status="completed",
        outcome="success",
        error=None,
    )

    row = _only(module)

    assert row["failed"] is False
    assert row["fail_hits"] == []


def test_kanban_retry_does_not_hide_failure_for_a_prior_session(audit_env):
    module, db, home = audit_env
    _session(db, "durable-kanban", source="kanban")
    _message(db, "durable-kanban", "user", "Work kanban task t_fixture")
    _message(db, "durable-kanban", "assistant", "Stopping this attempt.")
    _kanban_db(
        home / ".hermes" / "kanban.db",
        task_status="done",
        run_status="completed",
        outcome="completed",
        error=None,
        older_run=("completed", "failed", "first attempt crashed"),
    )

    row = _only(module)

    assert "kanban:outcome=failed" in row["fail_hits"]


def test_multiple_task_ids_correlate_to_the_run_nearest_session_start(audit_env):
    module, db, home = audit_env
    _session(db, "durable-kanban", source="kanban")
    _message(db, "durable-kanban", "user", "Work t_other while reporting t_fixture")
    _message(db, "durable-kanban", "assistant", "Stopped.")
    path = home / ".hermes" / "kanban.db"
    _kanban_db(
        path,
        task_status="done",
        run_status="completed",
        outcome="failed",
        error="matched failure",
    )
    con = sqlite3.connect(path)
    con.execute("INSERT INTO tasks VALUES ('t_other', 'done', 8, NULL, NULL)")
    con.execute(
        "INSERT INTO task_runs VALUES (8, 't_other', 'completed', 'completed', NULL, 200, 210)"
    )
    con.commit()
    con.close()

    row = _only(module)

    assert "kanban:outcome=failed" in row["fail_hits"]


def _cron_db(
    path: Path,
    *,
    status: str,
    error: str | None,
    finished_at: str | None = "1970-01-01T00:01:50+00:00",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        """CREATE TABLE executions (
               id TEXT, job_id TEXT, status TEXT, error TEXT,
               started_at TEXT, finished_at TEXT
           )"""
    )
    con.execute(
        "INSERT INTO executions VALUES ('exec-1', 'job123', ?, ?, '1970-01-01T00:01:40+00:00', ?)",
        (status, error, finished_at),
    )
    con.commit()
    con.close()


def test_failed_durable_cron_execution_is_a_failure(audit_env):
    module, db, home = audit_env
    sid = "cron_job123_19700101_000140"
    _session(db, sid, source="cron", title="fixture-job", end_reason="cron_complete")
    _message(db, sid, "user", "Run scheduled work")
    _message(db, sid, "assistant", "Done.")
    _cron_db(home / ".hermes" / "cron" / "executions.db", status="failed", error="process crashed")

    row = _only(module)

    assert row["failed"] is True
    assert "cron:status=failed" in row["fail_hits"]


def test_running_cron_execution_is_not_reported_as_failed(audit_env):
    module, db, home = audit_env
    sid = "cron_job123_19700101_000140"
    _session(db, sid, source="cron", title="fixture-job", end_reason=None, ended_at=None)
    _message(db, sid, "user", "Run scheduled work")
    _cron_db(
        home / ".hermes" / "cron" / "executions.db",
        status="running",
        error=None,
        finished_at=None,
    )

    row = _only(module)

    assert row["failed"] is False
    assert module.cron_failures(since=90) == []


def test_cron_failure_scan_does_not_drop_older_failure_after_eighty_runs(audit_env):
    module, _, home = audit_env
    path = home / ".hermes" / "cron" / "executions.db"
    _cron_db(path, status="failed", error="old failure")
    con = sqlite3.connect(path)
    for index in range(1, 85):
        total_seconds = 40 + index
        timestamp = f"1970-01-01T00:{total_seconds // 60:02d}:{total_seconds % 60:02d}+00:00"
        con.execute(
            "INSERT INTO executions VALUES (?, 'job123', 'completed', NULL, ?, ?)",
            (f"exec-{index + 1}", timestamp, timestamp),
        )
    con.commit()
    con.close()

    failures = module.cron_failures(since=90)

    assert [failure["execution_id"] for failure in failures] == ["exec-1"]


def test_kanban_failure_scan_does_not_drop_older_failure_after_forty_runs(audit_env):
    module, _, home = audit_env
    path = home / ".hermes" / "kanban.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT, assignee TEXT, status TEXT,
            consecutive_failures INTEGER, last_failure_error TEXT,
            block_kind TEXT
        );
        CREATE TABLE task_runs (
            id INTEGER PRIMARY KEY, task_id TEXT, profile TEXT, status TEXT,
            outcome TEXT, error TEXT, summary TEXT,
            started_at INTEGER, ended_at INTEGER
        );
        """
    )
    for index in range(45):
        task_id = f"t_{index:02d}"
        con.execute(
            "INSERT INTO tasks VALUES (?, ?, 'qa', 'done', 0, NULL, NULL)",
            (task_id, f"task {index}"),
        )
        outcome = "failed" if index == 0 else "completed"
        error = "old failure" if index == 0 else None
        con.execute(
            "INSERT INTO task_runs VALUES (?, ?, 'qa', 'done', ?, ?, '', ?, ?)",
            (index + 1, task_id, outcome, error, 100 + index, 100 + index),
        )
    con.commit()
    con.close()

    failures = module.kanban_failures(since=90)

    assert [failure["run_id"] for failure in failures] == [1]


def test_fleet_includes_default_and_every_installed_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    home = tmp_path / "home"
    profiles = home / ".hermes" / "profiles"
    for name in ("buggy", "coder", "jade", "jade-ops", "product", "qa",
                 "research", "reviewer", "rsi", "x", "yuki", "yuki-ops"):
        (profiles / name).mkdir(parents=True)
    monkeypatch.setattr(module, "HOME", home)
    monkeypatch.setattr(module, "PROFILES_DIR", profiles)
    monkeypatch.setattr(module, "CONTRACT", tmp_path / "absent-contract.yaml")

    roster = module.fleet()

    # default + all 12 profile dirs = 13 installed; unlisted installs
    # still get a slice even with no contract present.
    assert len(roster) == 13
    assert roster[0] == "default"
    assert set(roster) >= {
        "default", "buggy", "coder", "jade", "jade-ops", "product", "qa",
        "research", "reviewer", "rsi", "x", "yuki", "yuki-ops",
    }


def test_fleet_contract_order_wins_but_covers_unlisted_installs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    home = tmp_path / "home"
    profiles = home / ".hermes" / "profiles"
    for name in ("qa", "reviewer", "x", "yuki", "yuki-ops"):
        (profiles / name).mkdir(parents=True)
    contract = tmp_path / "contract.yaml"
    contract.write_text("fleet: [coder, product, qa, reviewer, yuki, yuki-ops, x]\n")
    monkeypatch.setattr(module, "HOME", home)
    monkeypatch.setattr(module, "PROFILES_DIR", profiles)
    monkeypatch.setattr(module, "CONTRACT", contract)

    roster = module.fleet()

    # Only contracted names that are actually installed come first, in
    # contract order; installed-but-unlisted names (and default) still appear
    # so no profile lacks a slice.
    assert roster[:4] == ["qa", "reviewer", "yuki", "yuki-ops"]
    assert roster[-1] == "default"
    assert set(roster) == {"qa", "reviewer", "yuki", "yuki-ops", "x", "default"}


def test_audit_emits_empty_slice_for_every_installed_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    module = _load_module()
    home = tmp_path / "home"
    state = home / ".hermes" / "profiles" / "qa" / "state.db"
    state.parent.mkdir(parents=True)
    _state_db(state)
    store = tmp_path / "rsi"
    store.mkdir()
    contract = store / "contract.yaml"
    contract.write_text("fleet: [qa]\n")
    (store / "last_tick.json").write_text('{"unix": 50}', encoding="utf-8")
    monkeypatch.setattr(module, "HOME", home)
    monkeypatch.setattr(module, "KANBAN", home / ".hermes" / "kanban.db")
    monkeypatch.setattr(module, "EXEC_DB", home / ".hermes" / "cron" / "executions.db")
    monkeypatch.setattr(module, "PROFILES_DIR", home / ".hermes" / "profiles")
    monkeypatch.setattr(module, "CONTRACT", contract)
    monkeypatch.setattr(module, "STORE", store)
    monkeypatch.setattr(module, "LAST", store / "last_tick.json")

    module.main()
    payload = json.loads(capsys.readouterr().out)

    # Every installed profile (default + qa) gets an explicit slice, even the
    # ones with nothing to report and no contracted name.
    assert set(payload["profiles"]) == {"default", "qa"}
    for slice_ in payload["profiles"].values():
        assert isinstance(slice_, dict)
        assert isinstance(slice_["sessions"], list)
        assert isinstance(slice_["session_failures"], list)
        assert isinstance(slice_["cron_failures"], list)
        assert isinstance(slice_["kanban_failures"], list)
    on_disk = json.loads((store / "audit" / "latest.json").read_text(encoding="utf-8"))
    assert set(on_disk["profiles"]) == {"default", "qa"}
