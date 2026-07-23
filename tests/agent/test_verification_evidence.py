import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import agent.verification_evidence as verification_evidence
from agent.verification_evidence import (
    classify_verification_command,
    confirm_outcome_receipt,
    get_reusable_outcome_receipt,
    list_approval_decision_receipts,
    list_reusable_outcome_receipts,
    mark_workspace_edited,
    record_approval_decision_receipt,
    record_outcome_receipt,
    record_terminal_result,
    verification_status,
)


def _node_project(root: Path) -> None:
    (root / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest", "lint": "eslint .", "dev": "vite"}})
    )
    (root / "pnpm-lock.yaml").write_text("")
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / "run_tests.sh").write_text("#!/bin/sh\n")


def _python_project(root: Path) -> None:
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")


def _approval_record() -> dict:
    return {
        "id": "a1b2c3d4",
        "subsystem": "memory",
        "action": "add",
        "summary": "sensitive human-readable proposal",
        "origin": "background_review",
        "created_at": 1.0,
        "payload": {"action": "add", "target": "memory", "content": "secret payload"},
    }


def test_approval_decision_receipt_is_immutable_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    receipt = record_approval_decision_receipt(
        record=_approval_record(), decision="approved", terminal_outcome="applied"
    )

    assert receipt is not None
    assert receipt["decision"] == "approved"
    assert receipt["terminal_outcome"] == "applied"
    assert receipt["proposal_origin"] == "background_review"
    assert "payload" not in receipt
    assert "summary" not in receipt
    assert "secret payload" not in str(receipt)

    retry = record_approval_decision_receipt(
        record=_approval_record(), decision="approved", terminal_outcome="applied"
    )
    assert retry == receipt
    assert list_approval_decision_receipts(subsystem="memory") == [receipt]

    db_path = tmp_path / ".hermes" / "verification_evidence.db"
    with sqlite3.connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE approval_decision_receipts SET decision = 'rejected' WHERE id = ?",
                (receipt["id"],),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "DELETE FROM approval_decision_receipts WHERE id = ?", (receipt["id"],)
            )


def test_approval_decision_receipt_rejects_conflicting_terminalization(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    record = _approval_record()
    assert record_approval_decision_receipt(
        record=record, decision="approved", terminal_outcome="terminal_noop"
    ) is not None
    assert record_approval_decision_receipt(
        record=record, decision="rejected", terminal_outcome="rejected"
    ) is None
    assert len(list_approval_decision_receipts()) == 1


def test_approval_receipt_reader_does_not_create_or_migrate_database(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert list_approval_decision_receipts() == []
    assert not home.exists()

    home.mkdir()
    database_path = home / "verification_evidence.db"
    with sqlite3.connect(database_path) as conn:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '3')")
    before = database_path.read_bytes()

    assert list_approval_decision_receipts() == []
    assert database_path.read_bytes() == before
    with sqlite3.connect(database_path) as conn:
        assert conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0] == "3"
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'approval_decision_receipts'"
        ).fetchone() is None


def test_approval_receipt_writer_upgrades_v3_metadata_and_creates_guards(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    home.mkdir()
    database_path = home / "verification_evidence.db"
    with sqlite3.connect(database_path) as conn:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '3')")

    assert record_approval_decision_receipt(
        record=_approval_record(), decision="approved", terminal_outcome="applied"
    ) is not None

    with sqlite3.connect(database_path) as conn:
        version = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'approval_decision_receipts'"
        ).fetchone()
        triggers = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE 'approval_decision_receipts_no_%' ORDER BY name"
        ).fetchall()

    assert version == "6"
    assert table is not None
    assert [row[0] for row in triggers] == [
        "approval_decision_receipts_no_delete",
        "approval_decision_receipts_no_update",
    ]


def test_classifies_targeted_project_verify_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    evidence = classify_verification_command(
        "scripts/run_tests.sh tests/test_widget.py -q",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="1 passed",
    )

    assert evidence is not None
    assert evidence.canonical_command == "scripts/run_tests.sh"
    assert evidence.kind == "test"
    assert evidence.scope == "targeted"
    assert evidence.status == "passed"


def test_classifies_python_module_pytest_as_detected_pytest(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        "python -m pytest tests/test_calc.py::test_even -q",
        cwd=tmp_path,
        session_id="s1",
        exit_code=1,
        output="failed",
    )

    assert evidence is not None
    assert evidence.canonical_command == "pytest"
    assert evidence.kind == "test"
    assert evidence.scope == "targeted"
    assert evidence.status == "failed"


def test_records_passed_then_marks_stale_after_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    event = record_terminal_result(
        command="scripts/run_tests.sh",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="all green",
    )

    assert event is not None
    assert verification_status(session_id="s1", cwd=tmp_path)["status"] == "passed"

    mark_workspace_edited(
        session_id="s1",
        cwd=tmp_path,
        paths=[str(tmp_path / "src" / "app.ts")],
    )

    status = verification_status(session_id="s1", cwd=tmp_path)
    assert status["status"] == "stale"
    assert status["changed_paths"] == [str(tmp_path / "src" / "app.ts")]


def test_confirmed_passing_outcome_receipt_becomes_explicitly_reusable(tmp_path, monkeypatch):
    """Judge completion alone must never become automatic agent memory."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="all green",
    )

    receipt = record_outcome_receipt(
        session_id="s1",
        cwd=tmp_path,
        goal="ship the verified widget",
        terminal_kind="judge_done_unconfirmed",
    )

    assert receipt is not None
    assert receipt["goal_digest"] != "ship the verified widget"
    assert receipt["completion_contract_digest"]
    assert receipt["reusable"] is False
    assert list_reusable_outcome_receipts(cwd=tmp_path) == []

    confirmed = confirm_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    )
    assert confirmed is not None
    assert confirmed["terminal_kind"] == "achieved_confirmed"
    assert confirmed["user_confirmed_at"] is not None
    assert confirmed["verification_status"] == "passed"
    assert confirmed["reusable"] is True
    assert [row["id"] for row in list_reusable_outcome_receipts(cwd=tmp_path)] == [receipt["id"]]


def test_get_reusable_outcome_receipt_enforces_current_session_workspace_and_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    monkeypatch.setattr(
        "agent.coding_context.project_facts_for",
        lambda _cwd=None: {
            "root": str(tmp_path),
            "verifyCommands": ["pnpm test"],
            "manifests": ["package.json"],
            "packageManagers": ["pnpm"],
            "contextFiles": [],
        },
    )
    record_terminal_result(
        command="pnpm test", cwd=tmp_path, session_id="s1", exit_code=0, output="all green"
    )
    receipt = record_outcome_receipt(
        session_id="s1", cwd=tmp_path, goal="record a safe lesson", terminal_kind="judge_done_unconfirmed"
    )
    assert receipt is not None
    assert confirm_outcome_receipt(receipt["id"], expected_session_id="s1", cwd=tmp_path)

    eligible = get_reusable_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    )
    assert eligible is not None
    assert eligible["id"] == receipt["id"]
    assert eligible["verification_status"] == "passed"
    assert get_reusable_outcome_receipt(
        receipt["id"], expected_session_id="other-session", cwd=tmp_path
    ) is None

    mark_workspace_edited(session_id="s2", cwd=tmp_path, paths=["src/widget.ts"])
    assert get_reusable_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    ) is None


def test_evidence_expiry_window_normalizes_timezones_and_rejects_future_time():
    anchor = datetime(2030, 1, 31, 12, tzinfo=timezone.utc)

    assert verification_evidence._evidence_is_expired(
        "2030-01-01T21:00:00+09:00", now=anchor
    ) is False
    assert verification_evidence._evidence_is_expired(
        "2030-01-01T12:00:00+00:00", now=anchor
    ) is False
    assert verification_evidence._evidence_is_expired(
        "2030-01-01T11:59:59+00:00", now=anchor
    ) is True
    assert verification_evidence._evidence_is_expired(
        "2030-02-01T12:00:01+00:00", now=anchor
    ) is True


def test_idle_expired_evidence_is_not_reusable_without_later_ledger_write(
    tmp_path, monkeypatch
):
    """Retention is an eligibility rule, not only cleanup triggered by later work."""
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _node_project(tmp_path)
    monkeypatch.setattr(
        "agent.coding_context.project_facts_for",
        lambda _cwd=None: {
            "root": str(tmp_path),
            "verifyCommands": ["pnpm test"],
            "manifests": ["package.json"],
            "packageManagers": ["pnpm"],
            "contextFiles": [],
        },
    )
    record_terminal_result(
        command="pnpm test", cwd=tmp_path, session_id="s1", exit_code=0, output="all green"
    )
    receipt = record_outcome_receipt(
        session_id="s1", cwd=tmp_path, goal="ship a time-bounded lesson", terminal_kind="judge_done_unconfirmed"
    )
    assert receipt is not None
    assert confirm_outcome_receipt(receipt["id"], expected_session_id="s1", cwd=tmp_path)["reusable"]

    expired_at = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    with sqlite3.connect(home / "verification_evidence.db") as conn:
        conn.execute("UPDATE verification_events SET created_at = ?", (expired_at,))
        conn.commit()

    # No later record is written: this is the idle path that cleanup alone missed.
    assert verification_status(session_id="s1", cwd=tmp_path)["status"] == "expired"
    assert list_reusable_outcome_receipts(cwd=tmp_path) == []
    assert get_reusable_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    ) is None

    future_at = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    with sqlite3.connect(home / "verification_evidence.db") as conn:
        conn.execute("UPDATE verification_events SET created_at = ?", (future_at,))
        conn.commit()
    assert verification_status(session_id="s1", cwd=tmp_path)["status"] == "expired"
    assert list_reusable_outcome_receipts(cwd=tmp_path) == []
    assert get_reusable_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    ) is None


def test_outcome_receipt_binds_final_contract_and_ordered_subgoals(tmp_path, monkeypatch):
    """Learning candidates identify the exact criteria the judge evaluated."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    record_terminal_result(
        command="pnpm test", cwd=tmp_path, session_id="s1", exit_code=0, output="all green"
    )

    base = record_outcome_receipt(
        session_id="s1",
        cwd=tmp_path,
        goal="ship the verified widget",
        terminal_kind="judge_done_unconfirmed",
        completion_contract={"verification": "pnpm test", "boundaries": "src/widget"},
        subgoals=["add a regression test", "document the public API"],
    )
    same = record_outcome_receipt(
        session_id="s1",
        cwd=tmp_path,
        goal="ship the verified widget",
        terminal_kind="judge_done_unconfirmed",
        completion_contract={"boundaries": "src/widget", "verification": "pnpm test"},
        subgoals=["add a regression test", "document the public API"],
    )
    reordered = record_outcome_receipt(
        session_id="s1",
        cwd=tmp_path,
        goal="ship the verified widget",
        terminal_kind="judge_done_unconfirmed",
        completion_contract={"verification": "pnpm test", "boundaries": "src/widget"},
        subgoals=["document the public API", "add a regression test"],
    )

    assert base is not None and same is not None and reordered is not None
    assert base["completion_contract_digest"] == same["completion_contract_digest"]
    assert base["completion_contract_digest"] != reordered["completion_contract_digest"]
    assert "pnpm test" not in str(base)
    assert "regression test" not in str(base)


def test_outcome_receipt_writer_migrates_v4_contract_digest_column(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    home.mkdir()
    database_path = home / "verification_evidence.db"
    with sqlite3.connect(database_path) as conn:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '4')")
        conn.execute(
            """CREATE TABLE outcome_receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL,
                session_id TEXT NOT NULL, root TEXT NOT NULL, goal_digest TEXT NOT NULL,
                terminal_kind TEXT NOT NULL, verification_status TEXT NOT NULL,
                verification_event_id INTEGER, actor TEXT NOT NULL,
                user_confirmed_at TEXT, reusable INTEGER NOT NULL DEFAULT 0
            )"""
        )
    _node_project(tmp_path)
    receipt = record_outcome_receipt(
        session_id="s1", cwd=tmp_path, goal="migrate receipt", terminal_kind="blocked"
    )

    assert receipt is not None and receipt["completion_contract_digest"]
    with sqlite3.connect(database_path) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(outcome_receipts)")]
        version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
    assert "completion_contract_digest" in columns
    assert version == "6"


def test_approval_receipt_writer_migrates_v5_outcome_lineage_column(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    home.mkdir()
    database_path = home / "verification_evidence.db"
    with sqlite3.connect(database_path) as conn:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '5')")
        conn.execute(
            """CREATE TABLE approval_decision_receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL,
                subsystem TEXT NOT NULL, pending_id TEXT NOT NULL,
                proposal_digest TEXT NOT NULL, proposal_origin TEXT NOT NULL,
                decision TEXT NOT NULL, terminal_outcome TEXT NOT NULL,
                failure_code TEXT,
                UNIQUE (subsystem, pending_id, proposal_digest)
            )"""
        )

    record = _approval_record() | {
        "proposal_version": 3,
        "freshness": {"outcome_receipt_id": 73},
    }
    receipt = record_approval_decision_receipt(
        record=record, decision="approved", terminal_outcome="applied"
    )

    assert receipt is not None
    assert receipt["outcome_receipt_id"] == 73
    with sqlite3.connect(database_path) as conn:
        columns = [
            row[1] for row in conn.execute("PRAGMA table_info(approval_decision_receipts)")
        ]
        version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
    assert "outcome_receipt_id" in columns
    assert version == "6"


def test_reusable_outcome_listing_can_be_scoped_to_one_session(tmp_path, monkeypatch):
    """An interactive outcome view must not reveal another session's receipt."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    receipt_ids = []
    for session_id in ("s1", "s2"):
        record_terminal_result(
            command="pnpm test",
            cwd=tmp_path,
            session_id=session_id,
            exit_code=0,
            output="all green",
        )
        receipt = record_outcome_receipt(
            session_id=session_id,
            cwd=tmp_path,
            goal=f"ship the verified widget for {session_id}",
            terminal_kind="judge_done_unconfirmed",
        )
        assert receipt is not None
        assert confirm_outcome_receipt(
            receipt["id"], expected_session_id=session_id, cwd=tmp_path
        )["reusable"] is True
        receipt_ids.append(receipt["id"])

    assert [row["id"] for row in list_reusable_outcome_receipts(cwd=tmp_path, session_id="s1")] == [
        receipt_ids[0]
    ]
    assert [row["id"] for row in list_reusable_outcome_receipts(cwd=tmp_path, session_id="s2")] == [
        receipt_ids[1]
    ]


def test_session_scoped_outcome_listing_fails_closed_without_workspace_root(tmp_path, monkeypatch):
    """An interactive receipt view may never fall back to another workspace."""
    monkeypatch.setattr("agent.verification_evidence._outcome_root", lambda _cwd: None)

    assert list_reusable_outcome_receipts(cwd=tmp_path, session_id="s1") == []


def test_other_session_edit_stales_outcome_receipt_for_learning_candidates(
    tmp_path, monkeypatch
):
    """A receipt may not teach later work after its proof has gone stale."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="all green",
    )
    receipt = record_outcome_receipt(
        session_id="s1",
        cwd=tmp_path,
        goal="ship the verified widget",
        terminal_kind="judge_done_unconfirmed",
    )
    assert receipt is not None
    assert confirm_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    )["reusable"] is True

    mark_workspace_edited(
        session_id="s2",
        cwd=tmp_path,
        paths=[str(tmp_path / "src" / "app.ts")],
    )

    # Session-scoped status remains passed, but a reusable outcome receipt is
    # workspace-scoped and must not outlive an edit by another session.
    assert verification_status(session_id="s1", cwd=tmp_path)["status"] == "passed"
    assert list_reusable_outcome_receipts(cwd=tmp_path) == []


def test_reverification_after_workspace_edit_keeps_older_receipt_stale(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="stale-session",
        exit_code=0,
        output="all green",
    )
    stale = record_outcome_receipt(
        session_id="stale-session",
        cwd=tmp_path,
        goal="replace the verified widget",
        terminal_kind="judge_done_unconfirmed",
    )
    assert stale is not None
    assert confirm_outcome_receipt(
        stale["id"], expected_session_id="stale-session", cwd=tmp_path
    )["reusable"] is True
    mark_workspace_edited(
        session_id="edit-session",
        cwd=tmp_path,
        paths=[str(tmp_path / "src" / "app.ts")],
    )

    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="edit-session",
        exit_code=0,
        output="all green",
    )
    fresh = record_outcome_receipt(
        session_id="edit-session",
        cwd=tmp_path,
        goal="keep the verified widget",
        terminal_kind="judge_done_unconfirmed",
    )
    assert fresh is not None
    assert confirm_outcome_receipt(
        fresh["id"], expected_session_id="edit-session", cwd=tmp_path
    )["reusable"] is True
    assert [row["id"] for row in list_reusable_outcome_receipts(cwd=tmp_path)] == [
        fresh["id"]
    ]


def test_delayed_older_edit_cannot_restore_reusable_receipt(tmp_path, monkeypatch):
    """The root edit marker must stay monotonic across out-of-order writers."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    base = datetime.now(timezone.utc) - timedelta(seconds=10)
    clock_values = iter(
        (
            base.isoformat(),  # verification event
            base.isoformat(),  # receipt record
            base.isoformat(),  # confirmation
            (base + timedelta(seconds=2)).isoformat(),  # newer edit
            (base - timedelta(seconds=1)).isoformat(),  # delayed older edit
        )
    )
    monkeypatch.setattr(
        "agent.verification_evidence._utc_now", lambda: next(clock_values)
    )
    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="all green",
    )
    receipt = record_outcome_receipt(
        session_id="s1",
        cwd=tmp_path,
        goal="preserve the latest workspace edit",
        terminal_kind="judge_done_unconfirmed",
    )
    assert receipt is not None
    assert confirm_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    )["reusable"] is True

    # Simulate an edit that happened first but committed after a newer edit in
    # another process. The old timestamp must not overwrite the newer marker.
    mark_workspace_edited(
        session_id="newer-edit",
        cwd=tmp_path,
        paths=[str(tmp_path / "src" / "newer.ts")],
    )
    mark_workspace_edited(
        session_id="delayed-older-edit",
        cwd=tmp_path,
        paths=[str(tmp_path / "src" / "older.ts")],
    )

    assert list_reusable_outcome_receipts(cwd=tmp_path) == []


def test_confirmed_outcome_without_fresh_passing_evidence_stays_nonreusable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    receipt = record_outcome_receipt(
        session_id="s1",
        cwd=tmp_path,
        goal="attempt a risky migration",
        terminal_kind="judge_done_unconfirmed",
    )

    confirmed = confirm_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    )

    assert confirmed["terminal_kind"] == "achieved_confirmed"
    assert confirmed["verification_status"] == "unverified"
    assert confirmed["reusable"] is False
    assert list_reusable_outcome_receipts(cwd=tmp_path) == []


def test_outcome_receipt_rejects_unconfirmed_achievement_and_nonjudge_reconfirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    with pytest.raises(ValueError, match="user_confirmed"):
        record_outcome_receipt(
            session_id="s1",
            cwd=tmp_path,
            goal="claim success too early",
            terminal_kind="achieved_confirmed",
        )

    blocked = record_outcome_receipt(
        session_id="s1",
        cwd=tmp_path,
        goal="blocked work",
        terminal_kind="blocked",
    )
    with pytest.raises(ValueError, match="judge_done_unconfirmed"):
        confirm_outcome_receipt(
            blocked["id"], expected_session_id="s1", cwd=tmp_path
        )


def test_outcome_receipt_confirmation_requires_current_session_and_workspace(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="all green",
    )
    receipt = record_outcome_receipt(
        session_id="s1",
        cwd=tmp_path,
        goal="protect the explicit confirmation boundary",
        terminal_kind="judge_done_unconfirmed",
    )
    assert receipt is not None

    assert confirm_outcome_receipt(
        receipt["id"], expected_session_id="s2", cwd=tmp_path
    ) is None
    other_root = tmp_path / "other-workspace"
    other_root.mkdir()
    _node_project(other_root)
    assert confirm_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=other_root
    ) is None

    confirmed = confirm_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    )
    assert confirmed is not None
    assert confirmed["reusable"] is True


def test_confirmed_outcome_receipt_retry_is_idempotent_and_keeps_ownership_boundary(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="all green",
    )
    receipt = record_outcome_receipt(
        session_id="s1",
        cwd=tmp_path,
        goal="make confirmation retries safe",
        terminal_kind="judge_done_unconfirmed",
    )
    assert receipt is not None

    first = confirm_outcome_receipt(
        receipt["id"],
        expected_session_id="s1",
        cwd=tmp_path,
        actor="first-confirmation",
    )
    retry = confirm_outcome_receipt(
        receipt["id"],
        expected_session_id="s1",
        cwd=tmp_path,
        actor="retry-must-not-overwrite",
    )

    assert first is not None
    assert retry == first
    with sqlite3.connect(tmp_path / ".hermes" / "verification_evidence.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM outcome_receipts").fetchone()[0] == 1
    assert confirm_outcome_receipt(
        receipt["id"], expected_session_id="other-session", cwd=tmp_path
    ) is None


def test_confirmation_reserves_writer_before_binding_current_evidence(
    tmp_path, monkeypatch
):
    """An edit writer cannot land between confirmation's proof read and update."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    monkeypatch.setattr(
        "agent.coding_context.project_facts_for",
        lambda _cwd=None: {
            "root": str(tmp_path),
            "verifyCommands": ["pnpm test"],
            "manifests": ["package.json"],
            "packageManagers": ["pnpm"],
            "contextFiles": [],
        },
    )
    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="all green",
    )
    receipt = record_outcome_receipt(
        session_id="s1",
        cwd=tmp_path,
        goal="return the existing winner after a confirmation race",
        terminal_kind="judge_done_unconfirmed",
    )
    assert receipt is not None
    database_path = tmp_path / ".hermes" / "verification_evidence.db"
    original_status = verification_evidence._outcome_verification_status_in_conn
    observed = {"reservation": False, "competing_writer_blocked": False}

    def verify_under_reservation(conn, *, session_id, root):
        observed["reservation"] = conn.in_transaction
        competing = sqlite3.connect(database_path, timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                competing.execute(
                    "UPDATE outcome_receipts SET actor = actor WHERE id = ?",
                    (receipt["id"],),
                )
            observed["competing_writer_blocked"] = True
        finally:
            competing.close()
        return original_status(conn, session_id=session_id, root=root)

    monkeypatch.setattr(
        verification_evidence,
        "_outcome_verification_status_in_conn",
        verify_under_reservation,
    )
    confirmed = confirm_outcome_receipt(
        receipt["id"],
        expected_session_id="s1",
        cwd=tmp_path,
        actor="atomic-confirmation",
    )

    assert observed == {"reservation": True, "competing_writer_blocked": True}
    assert confirmed is not None
    assert confirmed["actor"] == "atomic-confirmation"
    assert confirmed["reusable"] is True
    assert confirmed["currently_reusable"] is True
    assert confirmed["current_verification_status"] == "passed"


def test_confirmation_retry_reports_current_eligibility_without_rewriting_snapshot(
    tmp_path, monkeypatch
):
    """Retry feedback must not report an edited receipt as currently reusable."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    monkeypatch.setattr(
        "agent.coding_context.project_facts_for",
        lambda _cwd=None: {
            "root": str(tmp_path),
            "verifyCommands": ["pnpm test"],
            "manifests": ["package.json"],
            "packageManagers": ["pnpm"],
            "contextFiles": [],
        },
    )
    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="all green",
    )
    receipt = record_outcome_receipt(
        session_id="s1",
        cwd=tmp_path,
        goal="keep the audit snapshot but show present eligibility",
        terminal_kind="judge_done_unconfirmed",
    )
    assert receipt is not None
    first = confirm_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    )
    assert first is not None and first["reusable"] is True
    mark_workspace_edited(session_id="s2", cwd=tmp_path, paths=["src/widget.ts"])

    retry = confirm_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    )

    assert retry is not None
    assert retry["reusable"] is True
    assert retry["verification_status"] == "passed"
    assert retry["currently_reusable"] is False
    assert retry["current_verification_status"] == "stale"


def test_confirmation_retry_never_promotes_a_previously_nonreusable_snapshot(
    tmp_path, monkeypatch
):
    """Fresh proof after confirmation cannot retroactively create learning consent."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    monkeypatch.setattr(
        "agent.coding_context.project_facts_for",
        lambda _cwd=None: {
            "root": str(tmp_path),
            "verifyCommands": ["pnpm test"],
            "manifests": ["package.json"],
            "packageManagers": ["pnpm"],
            "contextFiles": [],
        },
    )
    receipt = record_outcome_receipt(
        session_id="s1",
        cwd=tmp_path,
        goal="do not turn later proof into retroactive confirmation",
        terminal_kind="judge_done_unconfirmed",
    )
    assert receipt is not None
    first = confirm_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    )
    assert first is not None
    assert first["reusable"] is False
    assert first["currently_reusable"] is False

    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="later green evidence",
    )
    retry = confirm_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    )

    assert retry is not None
    assert retry["verification_status"] == "unverified"
    assert retry["reusable"] is False
    assert retry["current_verification_status"] == "passed"
    assert retry["currently_reusable"] is False
    assert get_reusable_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=tmp_path
    ) is None


def test_lint_and_typecheck_are_not_reported_as_full_tests(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    lint = classify_verification_command(
        "pnpm run lint",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )
    test = classify_verification_command(
        "pnpm run test -- tests/button.test.tsx",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )

    assert lint is not None
    assert lint.kind == "lint"
    assert lint.scope == "full"
    assert test is not None
    assert test.kind == "test"
    assert test.scope == "targeted"


def test_package_script_shorthand_matches_canonical_verify_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    evidence = classify_verification_command(
        "pnpm test -- tests/button.test.tsx",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )

    assert evidence is not None
    assert evidence.canonical_command == "pnpm run test"
    assert evidence.scope == "targeted"


def test_shell_wrappers_match_but_echo_does_not(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    wrapped = classify_verification_command(
        "env CI=1 bash scripts/run_tests.sh tests/test_widget.py",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )
    echoed = classify_verification_command(
        "echo scripts/run_tests.sh tests/test_widget.py",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )

    assert wrapped is not None
    assert wrapped.canonical_command == "scripts/run_tests.sh"
    assert wrapped.scope == "targeted"
    assert echoed is None


def test_uv_run_pytest_matches_detected_pytest(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _python_project(tmp_path)

    evidence = classify_verification_command(
        "uv run pytest tests/test_calc.py",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
    )

    assert evidence is not None
    assert evidence.canonical_command == "pytest"
    assert evidence.scope == "targeted"


def test_temp_script_records_ad_hoc_evidence_without_canonical_suite(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    script = Path(tempfile.gettempdir()) / f"hermes-ad-hoc-{tmp_path.name}.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    try:
        evidence = classify_verification_command(
            f"python {script}",
            cwd=tmp_path,
            session_id="s1",
            exit_code=0,
            output="ok",
        )
    finally:
        script.unlink(missing_ok=True)

    assert evidence is not None
    assert evidence.canonical_command == "ad-hoc verification script"
    assert evidence.kind == "ad_hoc"
    assert evidence.scope == "targeted"
    assert evidence.status == "passed"


def test_unprefixed_temp_script_is_not_ad_hoc_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    script = Path(tempfile.gettempdir()) / f"random-check-{tmp_path.name}.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    try:
        evidence = classify_verification_command(
            f"python {script}",
            cwd=tmp_path,
            session_id="s1",
            exit_code=0,
            output="ok",
        )
    finally:
        script.unlink(missing_ok=True)

    assert evidence is None


def test_temp_script_does_not_replace_detected_suite(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    script = Path(tempfile.gettempdir()) / f"hermes-ad-hoc-{tmp_path.name}.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    try:
        evidence = classify_verification_command(
            f"python {script}",
            cwd=tmp_path,
            session_id="s1",
            exit_code=0,
            output="ok",
        )
    finally:
        script.unlink(missing_ok=True)

    assert evidence is None


def test_non_temp_script_is_not_ad_hoc_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    script = tmp_path / "scripts" / "repro.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="utf-8")

    evidence = classify_verification_command(
        f"python {script}",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="ok",
    )

    assert evidence is None


def test_status_is_unverified_without_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    assert verification_status(session_id="s1", cwd=tmp_path)["status"] == "unverified"


def test_edit_without_prior_evidence_stays_unverified(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)

    mark_workspace_edited(
        session_id="s1",
        cwd=tmp_path,
        paths=[str(tmp_path / "src" / "app.ts")],
    )

    status = verification_status(session_id="s1", cwd=tmp_path)
    assert status["status"] == "unverified"
    assert status["changed_paths"] == [str(tmp_path / "src" / "app.ts")]


def test_file_tool_stales_evidence_by_session_id_for_absolute_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    target = tmp_path / "src" / "app.ts"
    target.parent.mkdir()

    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="conversation",
        exit_code=0,
        output="green",
    )

    from tools.file_tools import write_file_tool

    result = json.loads(
        write_file_tool(
            str(target),
            "export const ok = true\n",
            task_id="turn",
            session_id="conversation",
        )
    )

    assert result.get("files_modified") == [str(target.resolve())], result
    assert verification_status(session_id="conversation", cwd=tmp_path)["status"] == "stale"
    assert verification_status(session_id="turn", cwd=tmp_path)["status"] == "unverified"


def test_recording_prunes_old_events_but_keeps_latest_state(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _node_project(tmp_path)

    for index in range(120):
        record_terminal_result(
            command="pnpm test",
            cwd=tmp_path,
            session_id="s1",
            exit_code=0,
            output=f"green {index}",
        )

    with sqlite3.connect(home / "verification_evidence.db") as conn:
        event_count = conn.execute("SELECT COUNT(*) FROM verification_events").fetchone()[0]
        latest_summary = conn.execute(
            """
            SELECT output_summary
            FROM verification_events
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()[0]

    assert event_count == 100
    assert latest_summary == "green 119"
    assert verification_status(session_id="s1", cwd=tmp_path)["status"] == "passed"


def test_recording_expires_old_current_evidence(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _node_project(tmp_path)

    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="old-session",
        exit_code=0,
        output="old green",
    )
    cutoff = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    with sqlite3.connect(home / "verification_evidence.db") as conn:
        conn.execute("UPDATE verification_events SET created_at = ?", (cutoff,))
        conn.commit()

    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="new-session",
        exit_code=0,
        output="new green",
    )

    assert verification_status(session_id="old-session", cwd=tmp_path)["status"] == "unverified"
    assert verification_status(session_id="new-session", cwd=tmp_path)["status"] == "passed"
    with sqlite3.connect(home / "verification_evidence.db") as conn:
        old_rows = conn.execute(
            "SELECT COUNT(*) FROM verification_events WHERE session_id = 'old-session'"
        ).fetchone()[0]
    assert old_rows == 0


def test_recording_expires_old_edit_only_state(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _node_project(tmp_path)

    mark_workspace_edited(
        session_id="old-session",
        cwd=tmp_path,
        paths=[str(tmp_path / "src" / "app.ts")],
    )
    cutoff = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    with sqlite3.connect(home / "verification_evidence.db") as conn:
        conn.execute("UPDATE verification_state SET last_edit_at = ?", (cutoff,))
        conn.commit()

    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="new-session",
        exit_code=0,
        output="new green",
    )

    status = verification_status(session_id="old-session", cwd=tmp_path)
    assert status["status"] == "unverified"
    assert status["changed_paths"] == []


def test_windows_backslash_ad_hoc_script_path_is_matched(tmp_path, monkeypatch):
    """Ad-hoc verification scripts with Windows backslash paths must be
    matched by ``_find_ad_hoc_match`` trying ``posix=False`` in addition to
    the default ``posix=True``. (#53553 / #65919)

    On Linux, ``Path`` doesn't parse Windows backslash paths, so we mock
    ``_is_temp_script_path`` to simulate the Windows environment where the
    path resolves correctly. The test verifies the posix=False splitting
    fallback — the actual fix from #53553.
    """
    from agent.verification_evidence import _find_ad_hoc_match

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")

    # On Windows, shlex.split(posix=True) eats backslashes as escape chars;
    # posix=False preserves them. Mock _is_temp_script_path so the test
    # focuses on the splitting fallback without needing a real Windows FS.
    def mock_is_temp_script(token, root):
        return "hermes-ad-hoc" in token and ".py" in token

    monkeypatch.setattr(
        "agent.verification_evidence._is_temp_script_path",
        mock_is_temp_script,
    )

    win_script = r"C:\Users\test\AppData\Local\Temp\hermes-ad-hoc-check.py"
    result = _find_ad_hoc_match(f"python {win_script}", tmp_path)
    assert result is not None, (
        "Windows backslash path should be matched via posix=False fallback"
    )
