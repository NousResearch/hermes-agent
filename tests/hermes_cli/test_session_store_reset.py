"""Behavioral tests for the owner-only session-store reset command."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import session_store_reset
from hermes_cli.sessions_cmd import cmd_sessions
from hermes_state import SessionDB


def _seed_store(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("reset-fixture", "cli", cwd="/tmp/reset-fixture")
        db.append_message("reset-fixture", "user", "private session fixture")
        db.save_gateway_routing_entry("fixture-key", "{}", scope="fixture")
        db._conn.execute(
            "INSERT INTO compression_locks (session_id, holder, acquired_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            ("reset-fixture", "fixture", 1.0, 2.0),
        )
        db._conn.execute(
            "INSERT INTO session_turn_leases (conversation_id, holder, acquired_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            ("reset-fixture", "fixture", 1.0, 2.0),
        )
        db._conn.execute(
            "INSERT INTO async_delegations "
            "(delegation_id, origin_session, state, dispatched_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("fixture-delegation", "reset-fixture", "completed", 1.0, 2.0),
        )
    finally:
        db.close()
    sessions = home / "sessions" / "nested"
    sessions.mkdir(parents=True)
    (sessions / "reset-fixture.jsonl").write_text("session artifact\n", encoding="utf-8")


def _reset_home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _seed_store(home)
    return home


def _quarantine(home: Path, attempt: str) -> Path:
    return home / "session-reset-quarantine" / attempt


def test_reset_quarantines_entire_owned_store_and_keeps_unrelated_content(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)
    unrelated = home / "config.yaml"
    unrelated.write_text("memory:\n  provider: local\n", encoding="utf-8")

    report = session_store_reset.reset_session_store("devops-001")

    assert report["status"] == "completed"
    assert report["schema"] == "hermes-session-store-reset/v1"
    assert str(home) not in json.dumps(report)
    quarantined = _quarantine(home, "devops-001")
    assert (quarantined / "sqlite" / "state.db").is_file()
    assert (quarantined / "sessions" / "nested" / "reset-fixture.jsonl").is_file()
    assert json.loads((quarantined / "phases" / "completed.json").read_text())["phase"] == "completed"
    assert unrelated.read_text(encoding="utf-8") == "memory:\n  provider: local\n"

    db = SessionDB(db_path=home / "state.db", read_only=True)
    try:
        conn = db._conn
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM gateway_routing").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM compression_locks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM session_turn_leases").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM async_delegations").fetchone()[0] == 0
    finally:
        db.close()


def test_reset_refuses_untrusted_sidecars_without_moving_source(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)
    (home / "state.db-unknown").write_text("unexpected", encoding="utf-8")

    report = session_store_reset.reset_session_store("devops-unknown")

    assert report["status"] == "failed"
    assert report["code"] == "unknown_sidecar"
    assert (home / "state.db").is_file()
    assert not _quarantine(home, "devops-unknown").exists()
    assert not (home / "session-reset-quarantine").exists()


def test_reset_refuses_group_writable_owned_source_before_creating_attempt(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)
    os.chmod(home / "state.db", 0o664)

    report = session_store_reset.reset_session_store("devops-group-writable")

    assert report["status"] == "failed"
    assert report["code"] == "source_untrusted"
    assert (home / "state.db").is_file()
    assert not _quarantine(home, "devops-group-writable").exists()
    assert not (home / "session-reset-quarantine").exists()

    os.chmod(home / "state.db", 0o644)
    assert session_store_reset.reset_session_store("devops-group-writable")["status"] == "completed"


def test_reset_refuses_symlinked_session_artifact_without_moving_source(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)
    target = home / "outside.txt"
    target.write_text("not a session artifact", encoding="utf-8")
    (home / "sessions" / "bad-link").symlink_to(target)

    report = session_store_reset.reset_session_store("devops-symlink")

    assert report["status"] == "failed"
    assert report["code"] == "source_untrusted"
    assert (home / "state.db").is_file()
    assert not _quarantine(home, "devops-symlink").exists()
    assert not (home / "session-reset-quarantine").exists()


def test_reset_refuses_a_proven_foreign_holder_without_moving_source(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)
    monkeypatch.setattr(session_store_reset, "_foreign_holders_present", lambda *_args: True)

    report = session_store_reset.reset_session_store("devops-holder")

    assert report["status"] == "failed"
    assert report["code"] == "live_holder"
    assert (home / "state.db").is_file()
    assert not _quarantine(home, "devops-holder").exists()
    assert not (home / "session-reset-quarantine").exists()


def test_reset_revalidates_planned_source_after_sqlite_locks_before_rename(
    tmp_path, monkeypatch
):
    home = _reset_home(tmp_path, monkeypatch)

    def replace_source_at_lock_boundary(boundary: str) -> None:
        if boundary == "locks-held":
            with (home / "state.db").open("ab") as state:
                state.write(b"race")

    report = session_store_reset.reset_session_store(
        "devops-plan-race", failure_hook=replace_source_at_lock_boundary
    )

    assert report["status"] == "failed"
    assert report["code"] == "attempt_inconsistent"
    assert (home / "state.db").is_file()
    assert not (_quarantine(home, "devops-plan-race") / "sqlite" / "state.db").exists()


def test_reset_revalidates_session_tree_digest_after_sqlite_locks(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)

    def mutate_sessions_at_lock_boundary(boundary: str) -> None:
        if boundary == "locks-held":
            (home / "sessions" / "nested" / "late.jsonl").write_text(
                "late mutation\n", encoding="utf-8"
            )

    report = session_store_reset.reset_session_store(
        "devops-session-tree-race", failure_hook=mutate_sessions_at_lock_boundary
    )

    assert report["status"] == "failed"
    assert report["code"] == "attempt_inconsistent"
    phases = _quarantine(home, "devops-session-tree-race") / "phases"
    assert not (phases / "completed.json").exists()
    assert (home / "sessions" / "nested" / "late.jsonl").is_file()


def test_same_attempt_resumes_after_a_recorded_partial_move(
    tmp_path, monkeypatch
):
    home = _reset_home(tmp_path, monkeypatch)

    def fail_after_main(boundary: str) -> None:
        if boundary == "moved:state.db":
            raise RuntimeError("injected crash boundary")

    report = session_store_reset.reset_session_store("devops-crash", failure_hook=fail_after_main)

    assert report["status"] == "failed"
    assert report["code"] == "operational_failure"
    assert report["quarantined"] is True
    quarantined = _quarantine(home, "devops-crash")
    assert (quarantined / "sqlite" / "state.db").is_file()
    assert not (home / "state.db").exists()
    assert json.loads((quarantined / "phases" / "failed.json").read_text())["phase"] == "failed"
    assert session_store_reset.reset_session_store("devops-crash")["status"] == "completed"
    assert (home / "state.db").is_file()


def test_reset_refuses_live_sessions_created_at_quarantined_boundary(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)

    def create_live_sessions(boundary: str) -> None:
        if boundary == "quarantined":
            (home / "sessions").mkdir()

    report = session_store_reset.reset_session_store(
        "devops-boundary-sessions", failure_hook=create_live_sessions
    )

    assert report["status"] == "failed"
    assert report["code"] == "verification_failed"
    phases = _quarantine(home, "devops-boundary-sessions") / "phases"
    assert (phases / "quarantined.json").is_file()
    assert not (phases / "completed.json").exists()


def test_same_attempt_adopts_only_the_provable_rename_before_part_marker(
    tmp_path, monkeypatch
):
    home = _reset_home(tmp_path, monkeypatch)
    original = session_store_reset._write_part_record

    def lose_first_marker(*args, **kwargs):
        if args[2] == "state.db":
            raise RuntimeError("crash after rename")
        return original(*args, **kwargs)

    monkeypatch.setattr(session_store_reset, "_write_part_record", lose_first_marker)
    assert session_store_reset.reset_session_store("devops-adopt")["status"] == "failed"
    monkeypatch.setattr(session_store_reset, "_write_part_record", original)

    report = session_store_reset.reset_session_store("devops-adopt")

    assert report["status"] == "completed"
    assert (_quarantine(home, "devops-adopt") / "parts" / "state.db.json").is_file()


def test_same_attempt_refuses_a_destination_that_does_not_match_manifest(
    tmp_path, monkeypatch
):
    home = _reset_home(tmp_path, monkeypatch)
    original = session_store_reset._write_part_record

    def lose_first_marker(*args, **kwargs):
        if args[2] == "state.db":
            raise RuntimeError("crash after rename")
        return original(*args, **kwargs)

    monkeypatch.setattr(session_store_reset, "_write_part_record", lose_first_marker)
    assert session_store_reset.reset_session_store("devops-inconsistent")["status"] == "failed"
    moved = _quarantine(home, "devops-inconsistent") / "sqlite" / "state.db"
    moved.unlink()
    moved.write_bytes(b"not the planned inode")
    monkeypatch.setattr(session_store_reset, "_write_part_record", original)

    report = session_store_reset.reset_session_store("devops-inconsistent")

    assert report["status"] == "failed"
    assert report["code"] == "attempt_inconsistent"
    assert not (home / "state.db").exists()


def test_reset_reconciles_the_same_completed_attempt_after_revalidation(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)

    assert session_store_reset.reset_session_store("devops-once")["status"] == "completed"
    phases = set((_quarantine(home, "devops-once") / "phases").iterdir())
    report = session_store_reset.reset_session_store("devops-once")

    assert report["status"] == "completed"
    assert report["already_completed"] is True
    assert (home / "state.db").is_file()
    assert set((_quarantine(home, "devops-once") / "phases").iterdir()) == phases


def test_completed_reconciliation_defers_term_and_does_not_add_a_phase(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)
    assert session_store_reset.reset_session_store("devops-repeat-term")["status"] == "completed"
    phases = set((_quarantine(home, "devops-repeat-term") / "phases").iterdir())
    original = session_store_reset._validate_empty_store

    def signal_during_revalidation(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        return original(*args, **kwargs)

    monkeypatch.setattr(session_store_reset, "_validate_empty_store", signal_during_revalidation)
    report = session_store_reset.reset_session_store("devops-repeat-term")

    assert report["status"] == "completed"
    assert report["already_completed"] is True
    assert report["deferred_signal"] == signal.SIGTERM
    assert session_store_reset.report_exit_code(report) == 143
    assert set((_quarantine(home, "devops-repeat-term") / "phases").iterdir()) == phases


def test_completed_attempt_refuses_tampered_active_store_without_failed_marker(
    tmp_path, monkeypatch
):
    home = _reset_home(tmp_path, monkeypatch)
    assert session_store_reset.reset_session_store("devops-tampered")["status"] == "completed"
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("unexpected", "cli")
    finally:
        db.close()

    report = session_store_reset.reset_session_store("devops-tampered")

    assert report["status"] == "failed"
    assert report["code"] == "verification_failed"
    assert not (_quarantine(home, "devops-tampered") / "phases" / "failed.json").exists()


def test_completed_attempt_refuses_a_new_live_sessions_directory(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)
    assert session_store_reset.reset_session_store("devops-live-sessions")["status"] == "completed"
    (home / "sessions").mkdir()

    report = session_store_reset.reset_session_store("devops-live-sessions")

    assert report["status"] == "failed"
    assert report["code"] == "verification_failed"
    assert not (_quarantine(home, "devops-live-sessions") / "phases" / "failed.json").exists()


def test_completed_attempt_refuses_unexpected_terminal_part_record(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)
    assert session_store_reset.reset_session_store("devops-extra-part")["status"] == "completed"
    parts = _quarantine(home, "devops-extra-part") / "parts"
    (parts / "unexpected.json").write_text("{}", encoding="utf-8")

    report = session_store_reset.reset_session_store("devops-extra-part")

    assert report["status"] == "failed"
    assert report["code"] == "attempt_inconsistent"
    assert not (_quarantine(home, "devops-extra-part") / "phases" / "failed.json").exists()


def test_completed_attempt_refuses_unexpected_attempt_root_or_sqlite_entry(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)
    first = session_store_reset.reset_session_store("devops-extra-root")
    assert first["status"] == "completed"
    attempt = _quarantine(home, "devops-extra-root")
    (attempt / "unexpected").write_text("x", encoding="utf-8")

    root_report = session_store_reset.reset_session_store("devops-extra-root")

    assert root_report["code"] == "attempt_inconsistent"
    (attempt / "unexpected").unlink()
    (attempt / "sqlite" / "unexpected").write_text("x", encoding="utf-8")
    sqlite_report = session_store_reset.reset_session_store("devops-extra-root")

    assert sqlite_report["code"] == "attempt_inconsistent"


def test_completed_attempt_refuses_tampered_quarantined_sessions_tree(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)
    assert session_store_reset.reset_session_store("devops-tampered-sessions")["status"] == "completed"
    sessions = _quarantine(home, "devops-tampered-sessions") / "sessions"
    (sessions / "outside").symlink_to(home / "state.db")

    report = session_store_reset.reset_session_store("devops-tampered-sessions")

    assert report["status"] == "failed"
    assert report["code"] == "source_untrusted"
    assert not (
        _quarantine(home, "devops-tampered-sessions") / "phases" / "failed.json"
    ).exists()


def test_finalization_purges_only_a_completed_revalidated_attempt(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)
    reset = session_store_reset.reset_session_store("devops-finalize")
    assert reset["status"] == "completed"

    report = session_store_reset.finalize_session_store_reset(
        "devops-finalize", reset["attempt_manifest_sha256"]
    )

    assert report == {
        "schema": "hermes-session-store-reset/v1",
        "attempt": "devops-finalize",
        "status": "completed",
        "finalized": True,
        "attempt_manifest_sha256": reset["attempt_manifest_sha256"],
        "deferred_signal": None,
    }
    assert not _quarantine(home, "devops-finalize").exists()
    assert (home / "state.db").is_file()


def test_finalization_requires_the_exact_reset_manifest_digest(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)
    reset = session_store_reset.reset_session_store("devops-finalize-digest")
    assert reset["status"] == "completed"

    missing = session_store_reset.finalize_session_store_reset("devops-finalize-digest")
    wrong = session_store_reset.finalize_session_store_reset(
        "devops-finalize-digest", "0" * 64
    )

    assert missing["code"] == "manifest_required"
    assert wrong["code"] == "manifest_mismatch"
    assert _quarantine(home, "devops-finalize-digest").exists()


def test_finalization_refuses_tampered_quarantine_even_with_manifest(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)
    reset = session_store_reset.reset_session_store("devops-finalize-tamper")
    assert reset["status"] == "completed"
    sessions = _quarantine(home, "devops-finalize-tamper") / "sessions"
    (sessions / "altered.jsonl").write_text("not part of reset evidence\n", encoding="utf-8")

    report = session_store_reset.finalize_session_store_reset(
        "devops-finalize-tamper", reset["attempt_manifest_sha256"]
    )

    assert report["status"] == "failed"
    assert report["code"] == "attempt_inconsistent"
    assert _quarantine(home, "devops-finalize-tamper").exists()


def test_reset_defers_term_until_the_critical_section_has_completed(tmp_path, monkeypatch):
    home = _reset_home(tmp_path, monkeypatch)

    def send_term(boundary: str) -> None:
        if boundary == "prepared":
            os.kill(os.getpid(), signal.SIGTERM)

    report = session_store_reset.reset_session_store("devops-term", failure_hook=send_term)

    assert report["status"] == "completed"
    assert report["deferred_signal"] == signal.SIGTERM
    assert session_store_reset.report_exit_code(report) == 143
    assert (home / "state.db").is_file()
    phase = json.loads(
        (_quarantine(home, "devops-term") / "phases" / "completed.json").read_text()
    )
    assert phase["phase"] == "completed"


def test_cli_reset_store_emits_one_sanitized_json_document(tmp_path, monkeypatch, capsys):
    _reset_home(tmp_path, monkeypatch)

    rc = cmd_sessions(
        SimpleNamespace(sessions_action="reset-store", attempt="devops-cli", yes=True, json=True)
    )

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "completed"


def test_module_requires_explicit_confirmation_and_keeps_store(tmp_path, monkeypatch, capsys):
    home = _reset_home(tmp_path, monkeypatch)

    rc = session_store_reset.main(["--attempt", "devops-confirm", "--json"])

    assert rc == 1
    assert json.loads(capsys.readouterr().out)["code"] == "confirmation_required"
    assert (home / "state.db").is_file()
