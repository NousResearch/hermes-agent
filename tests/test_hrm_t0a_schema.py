"""HRM-T0a SessionDB schema + accessors + move primitive.

Covers steps 1-2 of the implementation plan
(work/hrm-t0a-hermes-topic-implementation-plan.md):

- ``active_topic_pointer`` and ``move_log`` tables apply via the opt-in
  ``apply_hrm_t0a_migration`` and are idempotent on re-apply.
- ``read_active_topic`` / ``set_active_topic`` / ``clear_active_topic``
  enforce PK uniqueness, capture provenance, and return prior state
  so the slash handler can compensate on confirmation-banner failure.
- ``move_turns`` is atomic (BEGIN IMMEDIATE), idempotent (replay
  returns byte-equal response body), dry-run rolls back, and the
  soft-delete columns on ``messages`` are populated only on commit.
"""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from hermes_state import SessionDB


# ── Schema / migration ────────────────────────────────────────────────


def test_messages_soft_delete_columns_reconciled(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    cols = {
        row[1]
        for row in db._conn.execute("PRAGMA table_info('messages')").fetchall()
    }
    for expected in ("moved_to_session_id", "moved_to_message_id", "moved_at"):
        assert expected in cols, f"missing reconciled column: {expected}"
    db.close()


def test_apply_hrm_t0a_migration_creates_tables_and_is_idempotent(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_hrm_t0a_migration()
    db.apply_hrm_t0a_migration()  # re-apply must be a no-op

    tables = {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "active_topic_pointer" in tables
    assert "move_log" in tables

    indexes = {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    }
    assert "idx_atp_app_topic" in indexes
    assert "idx_move_log_src" in indexes
    assert "idx_move_log_dst" in indexes

    assert db.get_meta("hrm_t0a_schema_version") == "1"

    applied_at = db.get_meta("hrm_t0a_applied_at")
    assert applied_at is not None
    assert float(applied_at) > 0.0
    db.close()


def test_apply_hrm_t0a_migration_stamps_high_water_mark(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    # Synthetic clock-skewed session: started_at in the future.
    future_ts = 9_999_999_999.0
    db.create_session(session_id="legacy", source="telegram", user_id="u")
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (future_ts, "legacy"),
    )
    db._conn.commit()
    db.apply_hrm_t0a_migration()
    stamped = float(db.get_meta("hrm_t0a_applied_at"))
    assert stamped > future_ts, "stamp must exceed max(sessions.started_at)+1"
    db.close()


# ── active_topic_pointer accessors ────────────────────────────────────


_PRINCIPAL = dict(
    platform="telegram",
    user_id="208214988",
    chat_id="208214988",
    app_id="hermes-agent",
)


def test_read_active_topic_returns_none_before_migration(tmp_path):
    """Read path stays cold — does not auto-create the side tables."""
    db = SessionDB(db_path=tmp_path / "state.db")
    assert db.read_active_topic(**_PRINCIPAL) is None
    tables = {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "active_topic_pointer" not in tables
    db.close()


def test_set_active_topic_roundtrip_and_returns_prior(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    first = db.set_active_topic(
        **_PRINCIPAL,
        topic_id="research",
        updated_by="slash:/topic switch",
    )
    assert first["prior"] is None
    assert first["current"]["topic_id"] == "research"

    snap = db.read_active_topic(**_PRINCIPAL)
    assert snap is not None
    assert snap["topic_id"] == "research"

    second = db.set_active_topic(
        **_PRINCIPAL,
        topic_id="implementation",
        updated_by="slash:/topic switch",
    )
    assert second["prior"]["topic_id"] == "research"
    assert second["current"]["topic_id"] == "implementation"
    db.close()


def test_set_active_topic_requires_provenance(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    with pytest.raises(ValueError, match="topic_id"):
        db.set_active_topic(**_PRINCIPAL, topic_id="", updated_by="x")
    with pytest.raises(ValueError, match="updated_by"):
        db.set_active_topic(**_PRINCIPAL, topic_id="t", updated_by="")
    db.close()


def test_pointer_pk_isolates_principals_and_apps(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_active_topic(**_PRINCIPAL, topic_id="t-A", updated_by="x")
    # Different app_id under the same principal
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="other-repo",
        topic_id="t-B",
        updated_by="x",
    )
    # Different user_id under the same chat
    db.set_active_topic(
        platform="telegram",
        user_id="other-user",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="t-C",
        updated_by="x",
    )
    assert db.read_active_topic(**_PRINCIPAL)["topic_id"] == "t-A"
    assert (
        db.read_active_topic(
            platform="telegram",
            user_id="208214988",
            chat_id="208214988",
            app_id="other-repo",
        )["topic_id"]
        == "t-B"
    )
    assert (
        db.read_active_topic(
            platform="telegram",
            user_id="other-user",
            chat_id="208214988",
            app_id="hermes-agent",
        )["topic_id"]
        == "t-C"
    )
    db.close()


def test_clear_active_topic_returns_prior_and_deletes(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_active_topic(**_PRINCIPAL, topic_id="research", updated_by="x")
    prior = db.clear_active_topic(**_PRINCIPAL, updated_by="slash:/topic clear")
    assert prior is not None
    assert prior["topic_id"] == "research"
    assert db.read_active_topic(**_PRINCIPAL) is None

    # Idempotent: clearing again returns None.
    again = db.clear_active_topic(**_PRINCIPAL, updated_by="slash:/topic clear")
    assert again is None
    db.close()


def test_bound_thread_id_is_optional_metadata_not_routing_key(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_active_topic(
        **_PRINCIPAL,
        topic_id="research",
        bound_thread_id="9001",
        updated_by="slash:/topic bind-thread",
    )
    row = db.read_active_topic(**_PRINCIPAL)
    assert row["topic_id"] == "research"
    assert row["bound_thread_id"] == "9001"
    db.close()


# ── move_turns primitive ───────────────────────────────────────────────


def _seed_two_sessions_with_messages(db: SessionDB, *, src_n: int, dst_n: int):
    db.create_session(session_id="src", source="telegram", user_id="u")
    db.create_session(session_id="dst", source="telegram", user_id="u")
    for i in range(src_n):
        db.append_message("src", "user", f"src-message-{i}")
    for i in range(dst_n):
        db.append_message("dst", "user", f"dst-message-{i}")


def test_move_turns_dry_run_does_not_mutate(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_two_sessions_with_messages(db, src_n=5, dst_n=0)

    plan = db.move_turns(
        src_session_id="src",
        dst_session_id="dst",
        range_spec="last:3",
        dry_run=True,
    )
    assert plan["dry_run"] is True
    assert len(plan["src_message_ids"]) == 3
    assert plan["dst_message_ids"] == []

    # State unchanged.
    src_active = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'src' AND active = 1"
    ).fetchone()[0]
    assert src_active == 5
    dst_count = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'dst'"
    ).fetchone()[0]
    assert dst_count == 0
    log_count = db._conn.execute("SELECT COUNT(*) FROM move_log").fetchone()[0]
    assert log_count == 0
    db.close()


def test_move_turns_commit_happy_path_last_n(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_two_sessions_with_messages(db, src_n=5, dst_n=2)

    resp = db.move_turns(
        src_session_id="src",
        dst_session_id="dst",
        range_spec="last:3",
        idempotency_key="key-A",
    )
    assert resp["dry_run"] is False
    assert resp["replay"] is False
    assert len(resp["src_message_ids"]) == 3
    assert len(resp["dst_message_ids"]) == 3

    # src: 3 tombstoned, 2 active remaining.
    src_active = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'src' AND active = 1"
    ).fetchone()[0]
    assert src_active == 2
    src_tombstoned = db._conn.execute(
        "SELECT COUNT(*) FROM messages "
        "WHERE session_id = 'src' AND moved_to_session_id = 'dst'"
    ).fetchone()[0]
    assert src_tombstoned == 3

    # dst: 2 original + 3 moved = 5 total.
    dst_total = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'dst'"
    ).fetchone()[0]
    assert dst_total == 5

    # move_log row exists and the response is reconstructible.
    log = db._conn.execute(
        "SELECT response_body FROM move_log WHERE idempotency_key = 'key-A'"
    ).fetchone()[0]
    cached = json.loads(log)
    assert cached["src_message_ids"] == resp["src_message_ids"]
    assert cached["dst_message_ids"] == resp["dst_message_ids"]
    db.close()


def test_move_turns_replay_returns_cached_body(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_two_sessions_with_messages(db, src_n=5, dst_n=0)
    first = db.move_turns(
        src_session_id="src",
        dst_session_id="dst",
        range_spec="last:2",
        idempotency_key="rk",
    )
    second = db.move_turns(
        src_session_id="src",
        dst_session_id="dst",
        range_spec="last:2",
        idempotency_key="rk",
    )
    assert second["replay"] is True
    # All fields except ``replay`` are byte-equal.
    f = dict(first)
    s = dict(second)
    f.pop("replay")
    s.pop("replay")
    assert f == s

    # No second batch of dst rows.
    dst_total = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'dst'"
    ).fetchone()[0]
    assert dst_total == 2
    db.close()


def test_move_turns_requires_idempotency_key_for_commit(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_two_sessions_with_messages(db, src_n=1, dst_n=0)
    with pytest.raises(ValueError, match="idempotency_key"):
        db.move_turns(
            src_session_id="src",
            dst_session_id="dst",
            range_spec="last:1",
        )
    db.close()


def test_move_turns_rejects_unknown_sessions(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="src", source="telegram", user_id="u")
    with pytest.raises(KeyError, match="dst session not found"):
        db.move_turns(
            src_session_id="src",
            dst_session_id="dst",
            range_spec="last:1",
            idempotency_key="k",
        )
    db.close()


def test_move_turns_empty_range_is_logged_idempotently(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_two_sessions_with_messages(db, src_n=0, dst_n=0)
    resp = db.move_turns(
        src_session_id="src",
        dst_session_id="dst",
        range_spec="last:5",
        idempotency_key="empty-k",
    )
    assert resp["src_message_ids"] == []
    assert resp["dst_message_ids"] == []
    # Replay returns byte-equal.
    again = db.move_turns(
        src_session_id="src",
        dst_session_id="dst",
        range_spec="last:5",
        idempotency_key="empty-k",
    )
    assert again["replay"] is True
    db.close()


def test_move_turns_atomic_on_crash_mid_transaction(tmp_path, monkeypatch):
    """A raise inside the write txn leaves src and dst untouched."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_two_sessions_with_messages(db, src_n=3, dst_n=0)

    # Capture pre-state.
    pre_src_active = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'src' AND active = 1"
    ).fetchone()[0]
    pre_dst_total = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'dst'"
    ).fetchone()[0]

    # Inject a fault during the move by wrapping resolve_move_range to
    # raise AFTER one INSERT into dst — so dst would otherwise see a
    # partial state if the transaction wasn't atomic.
    orig_resolve = db._resolve_move_range

    def faulty_resolve(conn, src_session_id, range_spec):
        ids = orig_resolve(conn, src_session_id, range_spec)
        # Sneak in a fault after resolution: mark the wrapper so the
        # next INSERT into messages raises.
        conn.execute(
            "CREATE TEMP TRIGGER hrm_t0a_fault BEFORE INSERT ON messages "
            "WHEN new.session_id = 'dst' "
            "BEGIN SELECT RAISE(ABORT, 'injected-fault'); END"
        )
        return ids

    monkeypatch.setattr(db, "_resolve_move_range", faulty_resolve)
    with pytest.raises(sqlite3.IntegrityError):
        db.move_turns(
            src_session_id="src",
            dst_session_id="dst",
            range_spec="last:2",
            idempotency_key="fault-k",
        )

    # Clean up the trigger so subsequent checks don't fire it.
    try:
        db._conn.execute("DROP TRIGGER IF EXISTS hrm_t0a_fault")
        db._conn.commit()
    except sqlite3.OperationalError:
        pass

    post_src_active = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'src' AND active = 1"
    ).fetchone()[0]
    post_dst_total = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'dst'"
    ).fetchone()[0]
    log_count = db._conn.execute(
        "SELECT COUNT(*) FROM move_log WHERE idempotency_key = 'fault-k'"
    ).fetchone()[0]
    assert post_src_active == pre_src_active, "src must be byte-equal after rollback"
    assert post_dst_total == pre_dst_total, "dst must be byte-equal after rollback"
    assert log_count == 0, "no move_log row may exist for a rolled-back move"
    db.close()


def test_move_turns_range_spec_validation(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_two_sessions_with_messages(db, src_n=1, dst_n=0)
    for bad in ["", "lastN", "last:0", "last:-1", "range:", "range:5..1"]:
        with pytest.raises(ValueError):
            db.move_turns(
                src_session_id="src",
                dst_session_id="dst",
                range_spec=bad,
                idempotency_key=f"k-{bad}",
            )
    db.close()


def test_read_move_log_filters_by_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_two_sessions_with_messages(db, src_n=3, dst_n=0)
    db.move_turns(
        src_session_id="src",
        dst_session_id="dst",
        range_spec="last:2",
        idempotency_key="audit-k",
    )
    log = db.read_move_log(session_id="src")
    assert len(log) == 1
    assert log[0]["idempotency_key"] == "audit-k"
    assert log[0]["range_spec"] == "last:2"
    log_dst = db.read_move_log(session_id="dst")
    assert len(log_dst) == 1
    log_other = db.read_move_log(session_id="unknown")
    assert log_other == []
    db.close()


def test_read_move_log_empty_before_migration(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    # No migration applied.
    assert db.read_move_log(session_id="anything") == []
    db.close()
