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
    list_reusable_outcome_receipts,
    mark_workspace_edited,
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
    clock_values = iter(
        (
            "2030-01-01T00:00:02+00:00",  # verification event
            "2030-01-01T00:00:02+00:00",  # receipt record
            "2030-01-01T00:00:02+00:00",  # confirmation
            "2030-01-01T00:00:04+00:00",  # newer edit
            "2030-01-01T00:00:01+00:00",  # delayed older edit
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


def test_confirmation_race_returns_first_winner_without_overwriting_it(
    tmp_path, monkeypatch
):
    """Force a winner between the guarded read and conditional UPDATE."""
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
        goal="return the existing winner after a confirmation race",
        terminal_kind="judge_done_unconfirmed",
    )
    assert receipt is not None
    database_path = tmp_path / ".hermes" / "verification_evidence.db"
    winner = {
        "actor": "concurrent-winner",
        "confirmed_at": "2026-07-21T00:00:00+00:00",
        "verification_status": "passed",
        "verification_event_id": receipt["verification_event_id"],
        "reusable": 1,
    }
    original_connect = verification_evidence._connect
    injected = False

    class RacingConnection:
        def __init__(self, connection):
            self.connection = connection

        def __enter__(self):
            self.connection.__enter__()
            return self

        def __exit__(self, *args):
            return self.connection.__exit__(*args)

        def execute(self, statement, parameters=()):
            nonlocal injected
            if not injected and "UPDATE outcome_receipts" in statement:
                injected = True
                with sqlite3.connect(database_path) as competing_connection:
                    competing_connection.execute(
                        """
                        UPDATE outcome_receipts
                        SET terminal_kind = 'achieved_confirmed',
                            verification_status = ?,
                            verification_event_id = ?,
                            actor = ?,
                            user_confirmed_at = ?,
                            reusable = ?
                        WHERE id = ?
                          AND terminal_kind = 'judge_done_unconfirmed'
                        """,
                        (
                            winner["verification_status"],
                            winner["verification_event_id"],
                            winner["actor"],
                            winner["confirmed_at"],
                            winner["reusable"],
                            receipt["id"],
                        ),
                    )
            return self.connection.execute(statement, parameters)

        def commit(self):
            return self.connection.commit()

    monkeypatch.setattr(
        verification_evidence,
        "_connect",
        lambda: RacingConnection(original_connect()),
    )

    raced = confirm_outcome_receipt(
        receipt["id"],
        expected_session_id="s1",
        cwd=tmp_path,
        actor="losing-retry-must-not-overwrite",
    )

    assert injected is True
    assert raced is not None
    assert raced["terminal_kind"] == "achieved_confirmed"
    assert raced["actor"] == winner["actor"]
    assert raced["user_confirmed_at"] == winner["confirmed_at"]
    assert raced["verification_status"] == winner["verification_status"]
    assert raced["verification_event_id"] == winner["verification_event_id"]
    assert raced["reusable"] is True
    assert confirm_outcome_receipt(
        receipt["id"], expected_session_id="other-session", cwd=tmp_path
    ) is None
    other_root = tmp_path / "other-workspace"
    other_root.mkdir()
    _node_project(other_root)
    assert confirm_outcome_receipt(
        receipt["id"], expected_session_id="s1", cwd=other_root
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

    assert result["files_modified"] == [str(target.resolve())]
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
