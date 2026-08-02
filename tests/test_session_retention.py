import hashlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_state import SessionDB
from hermes_state_retention import (
    RETENTION_STAGE_METADATA,
    TOOL_RESULT_RETENTION_PLACEHOLDER,
    RetentionPolicy,
    SessionHistoryUnavailableError,
    VacuumDecision,
)


@pytest.fixture()
def db(tmp_path):
    value = SessionDB(db_path=tmp_path / "state.db")
    yield value
    value.close()


def _ended_session(
    db,
    sid,
    *,
    source="cli",
    days_old=8,
    parent_session_id=None,
    system_prompt=None,
):
    now = time.time()
    db.create_session(
        sid,
        source,
        parent_session_id=parent_session_id,
        system_prompt=system_prompt,
    )
    db.append_message(sid, "user", "keep me", timestamp=now - days_old * 86400)
    db.append_message(
        sid,
        "assistant",
        None,
        tool_calls=[{"id": "call-1", "type": "function"}],
        timestamp=now - days_old * 86400,
    )
    db.append_message(
        sid,
        "tool",
        "large secret payload",
        tool_name="read_file",
        tool_call_id="call-1",
        api_content="wire copy of secret",
        timestamp=now - days_old * 86400,
    )
    db.end_session(sid, "done")
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (now - days_old * 86400, sid),
    )
    db._conn.commit()
    return now


def _layered(**overrides):
    return RetentionPolicy.from_config({"retention_mode": "layered", **overrides})


def test_policy_defaults_to_compatible_delete_mode():
    policy = RetentionPolicy.from_config({})
    assert policy.mode == "delete"
    assert policy.retention_days == 90
    with pytest.raises(ValueError, match="must be a mapping"):
        RetentionPolicy.from_config([])
    policy = RetentionPolicy.from_config(
        {"retention_mode": "delete", "retention_by_source": "ignored"}
    )
    assert policy.mode == "delete"


def test_schema_v24_columns_are_reconciled(db):
    columns = {
        row["name"] for row in db._conn.execute("PRAGMA table_info(sessions)")
    }
    assert {"retention_stage", "retention_last_active", "archive_origin"} <= columns
    assert db._conn.execute("SELECT version FROM schema_version").fetchone()[0] == 24


def test_read_only_pre_v24_database_treats_history_as_available(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO sessions(id) VALUES ('legacy')")
    conn.commit()
    conn.close()
    legacy = SessionDB(db_path=path, read_only=True)
    try:
        assert legacy.retention_resume_status("legacy") == "available"
    finally:
        legacy.close()


def test_source_override_is_merged_and_validated():
    policy = _layered(
        retention_by_source={
            "Cron": {"compact_tool_results_after_days": 2, "retention_days": 60}
        }
    )
    cron = policy.thresholds_for("cron")
    assert (cron.compact_days, cron.metadata_days, cron.delete_days) == (2, 30, 60)
    with pytest.raises(ValueError, match="must satisfy"):
        _layered(metadata_only_after_days=7)
    with pytest.raises(ValueError, match="true or false"):
        _layered(vacuum_after_prune="false")
    with pytest.raises(ValueError, match="non-negative"):
        _layered(retention_days=float("nan"))
    with pytest.raises(ValueError, match="duplicate source"):
        _layered(retention_by_source={"Cron": {}, "cron": {}})


def test_compaction_preserves_protocol_fields_and_updates_fts(db):
    now = _ended_session(db, "old-tool", days_old=8)
    statements = []
    db._conn.set_trace_callback(statements.append)
    try:
        report = db.apply_retention_policy(_layered(), now=now, vacuum=False)
    finally:
        db._conn.set_trace_callback(None)

    assert report.totals.compacted_tool_results == 1
    row = db._conn.execute(
        "SELECT * FROM messages WHERE session_id = ? AND role = 'tool'",
        ("old-tool",),
    ).fetchone()
    assert row["content"] == TOOL_RESULT_RETENTION_PLACEHOLDER
    assert row["api_content"] is None
    assert row["tool_name"] == "read_file"
    assert row["tool_call_id"] == "call-1"
    assistant = db._conn.execute(
        "SELECT tool_calls FROM messages WHERE session_id = ? AND role = 'assistant'",
        ("old-tool",),
    ).fetchone()
    assert "call-1" in assistant["tool_calls"]
    assert db.search_messages("secret") == []
    assert not any("rebuild" in sql.lower() for sql in statements)


def test_metadata_stage_clears_content_and_cannot_resume_or_unarchive(db):
    now = _ended_session(
        db, "metadata", days_old=31, system_prompt="private system prompt"
    )
    report = db.apply_retention_policy(_layered(), now=now, vacuum=False)

    assert report.totals.metadata_lineages == 1
    row = db.get_session("metadata")
    assert row["retention_stage"] == RETENTION_STAGE_METADATA
    assert row["retention_last_active"] == pytest.approx(now - 31 * 86400)
    assert row["archived"] == 1
    assert row["message_count"] == 0
    assert row["tool_call_count"] == 0
    assert row["system_prompt"] is None
    assert db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'metadata'"
    ).fetchone()[0] == 0
    assert db.search_messages("keep me") == []
    with pytest.raises(SessionHistoryUnavailableError, match="history expired"):
        db.get_resume_conversations("metadata")
    assert db.get_messages_as_conversation(
        "metadata", allow_metadata_only=True
    ) == []
    with pytest.raises(SessionHistoryUnavailableError, match="history expired"):
        db.reopen_session("metadata")
    assert db.set_session_archived("metadata", False) is False


def test_old_full_session_requires_a_later_run_before_delete(db):
    now = _ended_session(db, "very-old", days_old=100)
    policy = _layered()

    first = db.apply_retention_policy(policy, now=now, vacuum=False)
    assert first.totals.metadata_lineages == 1
    assert first.totals.deleted_session_rows == 0
    assert db.get_session("very-old") is not None

    second = db.apply_retention_policy(policy, now=now + 1, vacuum=False)
    assert second.totals.deleted_session_rows == 1
    assert db.get_session("very-old") is None


def test_recent_activity_spares_a_long_lived_session(db):
    now = _ended_session(db, "long-lived", days_old=1)
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (now - 100 * 86400, "long-lived"),
    )
    db._conn.commit()

    report = db.apply_retention_policy(_layered(), now=now, vacuum=False)
    assert report.totals == type(report.totals)()
    assert db.get_session("long-lived")["retention_stage"] is None


@pytest.mark.parametrize(
    ("days_old", "field"),
    [(7, "compacted_lineages"), (30, "metadata_lineages")],
)
def test_layer_boundaries_are_inclusive(db, days_old, field):
    now = _ended_session(db, f"boundary-{days_old}", days_old=days_old)
    report = db.apply_retention_policy(_layered(), now=now, vacuum=False)
    assert getattr(report.totals, field) == 1


@pytest.mark.parametrize("protection", ["pinned", "archived"])
def test_user_protection_exempts_layered_retention(db, protection):
    now = _ended_session(db, protection, days_old=100)
    if protection == "pinned":
        db.set_session_pinned(protection, True)
    else:
        db.set_session_archived(protection, True)

    report = db.apply_retention_policy(_layered(), now=now, vacuum=False)
    assert report.totals == type(report.totals)()
    assert db.get_session(protection)[protection] == 1


def test_compression_lineage_uses_tip_source_and_latest_activity(db):
    now = _ended_session(db, "root", source="cli", days_old=40)
    db._conn.execute(
        "UPDATE sessions SET end_reason = 'compression' WHERE id = 'root'"
    )
    db._conn.commit()
    _ended_session(
        db,
        "tip",
        source="cron",
        days_old=8,
        parent_session_id="root",
    )
    policy = _layered(
        retention_by_source={
            "cron": {
                "compact_tool_results_after_days": 7,
                "metadata_only_after_days": 10,
                "retention_days": 20,
            }
        }
    )

    report = db.apply_retention_policy(policy, now=now, vacuum=False)
    assert report.by_source["cron"].compacted_lineages == 1
    assert report.totals.compacted_tool_results == 2
    assert db.get_session("root")["retention_stage"] == "tool_results_compacted"
    assert db.get_session("tip")["retention_stage"] == "tool_results_compacted"


def test_active_sibling_protects_an_imported_multi_tip_lineage(db):
    now = _ended_session(db, "multi-root", days_old=100)
    db._conn.execute(
        "UPDATE sessions SET end_reason = 'compression' WHERE id = 'multi-root'"
    )
    db._conn.commit()

    db.create_session(
        "active-child",
        "cli",
        parent_session_id="multi-root",
    )
    db.append_message(
        "active-child",
        "user",
        "still active",
        timestamp=now - 100 * 86400,
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = 'active-child'",
        (now - 101 * 86400,),
    )
    db._conn.commit()
    _ended_session(
        db,
        "newer-ended-child",
        days_old=100,
        parent_session_id="multi-root",
    )

    report = db.apply_retention_policy(_layered(), now=now, vacuum=False)

    assert report.totals == type(report.totals)()
    assert db.get_session("active-child")["retention_stage"] is None
    assert db.get_messages("multi-root")
    assert db.get_messages("active-child")
    assert db.get_messages("newer-ended-child")


def test_dry_run_is_read_only(db):
    now = _ended_session(db, "preview", days_old=31)
    report = db.apply_retention_policy(_layered(), now=now, dry_run=True)
    assert report.totals.metadata_lineages == 1
    assert db.get_session("preview")["retention_stage"] is None
    assert len(db.get_messages("preview")) == 3


def test_new_tool_results_are_compacted_on_a_later_run(db):
    now = _ended_session(db, "reused", days_old=8)
    policy = _layered()
    db.apply_retention_policy(policy, now=now, vacuum=False)
    db.reopen_session("reused")
    db.append_message(
        "reused",
        "tool",
        "new large payload",
        tool_name="shell",
        timestamp=now - 8 * 86400,
    )
    db.end_session("reused", "done")

    second = db.apply_retention_policy(policy, now=now, vacuum=False)
    assert second.totals.compacted_tool_results == 1
    assert db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'tool' AND content = ?",
        ("reused", TOOL_RESULT_RETENTION_PLACEHOLDER),
    ).fetchone()[0] == 2


def test_compaction_also_clears_a_stale_api_sidecar(db):
    now = _ended_session(db, "sidecar", days_old=8)
    db._conn.execute(
        "UPDATE messages SET content = ? WHERE session_id = ? AND role = 'tool'",
        (TOOL_RESULT_RETENTION_PLACEHOLDER, "sidecar"),
    )
    db._conn.commit()

    report = db.apply_retention_policy(_layered(), now=now, vacuum=False)
    assert report.totals.compacted_tool_results == 1
    assert db._conn.execute(
        "SELECT api_content FROM messages WHERE session_id = ? AND role = 'tool'",
        ("sidecar",),
    ).fetchone()[0] is None


def test_failed_transaction_does_not_remove_artifacts(db, tmp_path, monkeypatch):
    now = _ended_session(db, "rollback", days_old=31)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    artifact = sessions_dir / "session_rollback.json"
    artifact.write_text("full transcript", encoding="utf-8")
    original = db._apply_layered_action

    def fail_after_mutation(conn, logical, action):
        original(conn, logical, action)
        raise RuntimeError("forced rollback")

    monkeypatch.setattr(db, "_apply_layered_action", fail_after_mutation)
    with pytest.raises(RuntimeError, match="forced rollback"):
        db.apply_retention_policy(
            _layered(), now=now, sessions_dir=sessions_dir, vacuum=False
        )

    assert artifact.exists()
    assert len(db.get_messages("rollback")) == 3
    assert db.get_session("rollback")["retention_stage"] is None


def test_auto_maintenance_claim_is_atomic_across_connections(tmp_path):
    path = tmp_path / "state.db"
    first = SessionDB(db_path=path)
    second = SessionDB(db_path=path)
    now = time.time()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda database: database._claim_auto_retention(now, 24),
                    (first, second),
                )
            )
        assert sorted(results) == [False, True]
    finally:
        first.close()
        second.close()


def test_candidate_is_revalidated_after_a_concurrent_new_message(
    db, tmp_path, monkeypatch
):
    now = _ended_session(db, "race", days_old=31)
    sibling = SessionDB(db_path=db.db_path)
    original_execute = db._execute_write
    injected = False

    def interleaved_execute(fn, *args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            sibling.append_message("race", "user", "recent", timestamp=now)
        return original_execute(fn, *args, **kwargs)

    monkeypatch.setattr(db, "_execute_write", interleaved_execute)
    try:
        report = db.apply_retention_policy(_layered(), now=now, vacuum=False)
    finally:
        sibling.close()

    assert report.totals.metadata_lineages == 0
    assert db.get_session("race")["retention_stage"] is None
    assert len(db.get_messages("race")) == 4


def test_auto_archive_is_not_mistaken_for_user_protection(db):
    now = _ended_session(db, "auto-archived", days_old=31)
    assert db.archive_stale_sessions(30) == 1
    assert db.get_session("auto-archived")["archive_origin"] == "auto_archive"

    report = db.apply_retention_policy(_layered(), now=now, vacuum=False)
    assert report.totals.metadata_lineages == 1


def test_auto_archive_preserves_legacy_archive_protection_across_lineage(db):
    now = _ended_session(db, "protected-root", days_old=31)
    db._conn.execute(
        "UPDATE sessions SET end_reason = 'compression', archived = 1, "
        "archive_origin = NULL WHERE id = 'protected-root'"
    )
    db._conn.commit()
    _ended_session(
        db,
        "unarchived-tip",
        days_old=31,
        parent_session_id="protected-root",
    )

    assert db.archive_stale_sessions(30) == 1
    assert db.get_session("protected-root")["archive_origin"] is None
    assert db.get_session("unarchived-tip")["archive_origin"] == "auto_archive"

    report = db.apply_retention_policy(_layered(), now=now, vacuum=False)
    assert report.totals == type(report.totals)()
    assert db.get_session("protected-root")["retention_stage"] is None
    assert db.get_session("unarchived-tip")["retention_stage"] is None


def test_metadata_only_marker_survives_export_import(db, tmp_path):
    now = _ended_session(db, "portable", days_old=31)
    db.apply_retention_policy(_layered(), now=now, vacuum=False)
    exported = db.export_session("portable")
    target = SessionDB(db_path=tmp_path / "imported.db")
    try:
        result = target.import_sessions([exported])
        assert result["ok"] is True
        assert target.retention_resume_status("portable") == RETENTION_STAGE_METADATA
        with pytest.raises(SessionHistoryUnavailableError):
            target.get_resume_conversations("portable")
    finally:
        target.close()


def test_metadata_only_ancestor_blocks_partial_lineage_resume_and_unarchive(db):
    db.create_session("metadata-root", "cli")
    db.end_session("metadata-root", "compression")
    db._conn.execute(
        "UPDATE sessions SET archived = 1, retention_stage = 'metadata_only', "
        "archive_origin = 'layered_retention' WHERE id = 'metadata-root'"
    )
    db._conn.commit()
    db.create_session(
        "full-tip",
        "cli",
        parent_session_id="metadata-root",
    )
    db.append_message("full-tip", "user", "remaining partial history")
    db.end_session("full-tip", "done")

    assert db.retention_resume_status("full-tip") == RETENTION_STAGE_METADATA
    with pytest.raises(SessionHistoryUnavailableError):
        db.get_resume_conversations("full-tip")
    with pytest.raises(SessionHistoryUnavailableError):
        db.reopen_session("full-tip")
    assert db.set_session_archived("full-tip", False) is False
    assert db.get_session("metadata-root")["archived"] == 1
    assert db.get_session("full-tip")["archived"] == 0


def test_metadata_only_parent_does_not_block_independent_branch(db):
    db.create_session("expired-parent", "cli")
    db.end_session("expired-parent", "compression")
    db._conn.execute(
        "UPDATE sessions SET archived = 1, retention_stage = 'metadata_only', "
        "archive_origin = 'layered_retention' WHERE id = 'expired-parent'"
    )
    db._conn.commit()
    db.create_session(
        "independent-branch",
        "cli",
        parent_session_id="expired-parent",
        model_config={"_branched_from": "expired-parent"},
    )
    db.append_message("independent-branch", "user", "complete branch history")
    db.end_session("independent-branch", "done")

    assert db.retention_resume_status("independent-branch") == "available"
    db.reopen_session("independent-branch")
    assert db.get_session("independent-branch")["ended_at"] is None
    assert db.set_session_archived("independent-branch", False) is True
    assert db.get_session("expired-parent")["archived"] == 1
    assert db.set_session_pinned("independent-branch", True) is True
    assert db.get_session("independent-branch")["pinned"] == 1
    assert db.get_session("expired-parent")["pinned"] == 0


def test_vacuum_runs_only_after_an_eligible_space_decision(db, monkeypatch):
    now = _ended_session(db, "vacuum", days_old=31)
    calls = []
    monkeypatch.setattr(
        db,
        "_vacuum_decision",
        lambda _policy: VacuumDecision(reason="eligible"),
    )
    monkeypatch.setattr(db, "vacuum", lambda: calls.append(True))

    report = db.apply_retention_policy(_layered(), now=now)
    assert calls == [True]
    assert report.vacuum.ran is True


def test_vacuum_failure_does_not_undo_committed_retention(db, monkeypatch):
    now = _ended_session(db, "vacuum-failure", days_old=31)
    monkeypatch.setattr(
        db,
        "_vacuum_decision",
        lambda _policy: VacuumDecision(reason="eligible"),
    )

    def fail_vacuum():
        raise OSError("disk busy")

    monkeypatch.setattr(db, "vacuum", fail_vacuum)
    report = db.apply_retention_policy(_layered(), now=now)
    assert db.get_session("vacuum-failure")["retention_stage"] == "metadata_only"
    assert report.vacuum.ran is False
    assert "VACUUM failed" in report.warnings[0]


def test_config_driven_auto_maintenance_uses_layered_mode(db):
    _ended_session(db, "automatic", days_old=8)

    result = db.maybe_auto_maintain_sessions(
        {
            "retention_mode": "layered",
            "vacuum_after_prune": False,
        },
        min_interval_hours=0,
    )

    assert result["mode"] == "layered"
    assert result["totals"]["compacted_tool_results"] == 1
    assert result["pruned"] == 0


def test_invalid_automatic_config_never_raises_or_mutates(db):
    _ended_session(db, "invalid-auto", days_old=100)
    result = db.maybe_auto_maintain_sessions(
        {"retention_mode": "layered", "retention_days": 5},
        min_interval_hours=0,
    )
    assert "must satisfy" in result["error"]
    assert db.get_session("invalid-auto")["retention_stage"] is None


def test_artifact_cleanup_rejects_traversal_and_escapes_globs(db, tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    outside = tmp_path / "escape.json"
    outside.write_text("keep", encoding="utf-8")
    broad = sessions_dir / "request_dump_safeX_1.json"
    broad.write_text("keep", encoding="utf-8")
    exact = sessions_dir / "safe[1].jsonl"
    exact.write_text("remove", encoding="utf-8")
    digest = hashlib.sha256(b"safe[1]").hexdigest()[:12]
    snapshot = sessions_dir / f"session_safe_1_{digest}.json"
    snapshot.write_text("remove", encoding="utf-8")
    matched = sessions_dir / "request_dump_safe[1]_1.json"
    matched.write_text("remove", encoding="utf-8")

    db._remove_session_files(sessions_dir, "../escape")
    db._remove_session_files(sessions_dir, "safe[1]")

    assert outside.exists()
    assert broad.exists()
    assert not exact.exists()
    assert not snapshot.exists()
    assert not matched.exists()


def test_artifact_cleanup_matches_runtime_sanitizer_for_whitespace(db, tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    snapshot = sessions_dir / "session_spaced.json"
    request_dump = sessions_dir / "request_dump_spaced_1.json"
    snapshot.write_text("remove", encoding="utf-8")
    request_dump.write_text("remove", encoding="utf-8")

    db._remove_session_files(sessions_dir, " spaced ")

    assert not snapshot.exists()
    assert not request_dump.exists()
