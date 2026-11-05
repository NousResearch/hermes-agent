"""Adversarial recoverable-unarchive contracts for PR #149132 (review 6662897210)."""

from contextlib import contextmanager
import sqlite3
import threading

import pytest

import hermes_state_sessions as sessions
from hermes_state import SessionDB
from hermes_state_lineage import (
    SessionIdOwnedElsewhereError,
    SessionLineageInventoryError,
)
from tests.hermes_state.test_recoverable_unarchive_fencing import _rows, _stores


def _foreign_live(db, sid):
    db._conn.execute(
        "INSERT INTO sessions (id,source,started_at,profile_name,session_key,message_count) "
        "VALUES (?,'cli',30,'science',?,0)",
        (sid, f"agent:science:local:dm:{sid}"),
    )
    db._conn.commit()


def _after_recovery_guard(monkeypatch, callback):
    original = sessions.session_lineage_guard
    acquisitions = []

    @contextmanager
    def scheduled(path, ids, **kwargs):
        # Other public writers receive a string; only recovery passes a set.
        if isinstance(ids, str):
            with original(path, ids, **kwargs) as snapshot:
                yield snapshot
            return
        acquisitions.append(frozenset(ids))
        with original(path, ids, **kwargs) as snapshot:
            yield snapshot
        callback(len(acquisitions))

    monkeypatch.setattr(sessions, "session_lineage_guard", scheduled)
    return acquisitions


def test_foreign_tip_revealed_between_attempts_is_rejected(tmp_path, monkeypatch):
    with _stores(tmp_path, monkeypatch) as (local, sibling):
        # A dormant legacy continuation is not reachable until tip is compressed.
        local.create_session("foreign", "cli", parent_session_id="tip")
        local.end_session("foreign", "ws_orphan_reap")
        local.set_session_archived("foreign", True)
        _foreign_live(sibling, "foreign")
        writer_snapshot = []

        def writer(attempt):
            if attempt == 1:
                # Both are real, fenced public lifecycle methods.
                local.reopen_session("tip")
                local.end_session("tip", "compression")
                assert local.get_compression_tip("root") == "foreign"
                writer_snapshot.append((_rows(local), _rows(sibling)))

        acquisitions = _after_recovery_guard(monkeypatch, writer)
        with pytest.raises(SessionIdOwnedElsewhereError):
            local.unarchive_recoverable_session("root")
        assert (_rows(local), _rows(sibling)) == writer_snapshot[0]
        assert acquisitions == [
            {"root"}, {"root", "tip"}, {"root", "tip", "foreign"}
        ]


def test_live_foreign_owner_appearing_after_discovery_is_rejected(tmp_path, monkeypatch):
    with _stores(tmp_path, monkeypatch) as (local, sibling):
        before = _rows(local)

        def writer(attempt):
            if attempt == 1:
                _foreign_live(sibling, "tip")

        _after_recovery_guard(monkeypatch, writer)
        with pytest.raises(SessionIdOwnedElsewhereError):
            local.unarchive_recoverable_session("root")
        assert _rows(local) == before
        assert sibling.get_session("tip")["ended_at"] is None


@pytest.mark.parametrize("rotations", [1, 2])
def test_real_rotations_between_attempts_stabilize_or_hit_cap(tmp_path, monkeypatch, rotations):
    with _stores(tmp_path, monkeypatch) as (local, sibling):
        latest = ["tip"]

        def writer(attempt):
            if attempt <= rotations:
                parent = latest[0]
                child = f"rotation-{attempt}"
                local.publish_compression_child(
                    parent_session_id=parent, child_session_id=child, source="cli",
                    messages=[{"role": "user", "content": "durable handoff"}],
                    require_compression_lease=False,
                )
                local.end_session(child, "ws_orphan_reap")
                latest[0] = child

        acquisitions = _after_recovery_guard(monkeypatch, writer)
        if rotations == 2:
            with pytest.raises(SessionLineageInventoryError, match="tip changed"):
                local.unarchive_recoverable_session("root")
            assert all(row["archived"] and row["ended_at"] is not None
                       for row in (local.get_session(sid) for sid in
                                   ["root", "tip", "rotation-1", "rotation-2"]))
        else:
            assert local.unarchive_recoverable_session("root") is True
            assert local.get_session(latest[0])["ended_at"] is None
            local.append_message(latest[0], "user", "recovery remains writable")
        assert len(acquisitions) == 3
        assert _rows(sibling) == ()


@pytest.mark.parametrize("failure", ["root_read", "tip_discovery", "tip_read"])
@pytest.mark.parametrize("error", [sqlite3.OperationalError, SessionLineageInventoryError])
def test_discovery_errors_do_not_mutate(tmp_path, monkeypatch, failure, error):
    with _stores(tmp_path, monkeypatch) as (local, sibling):
        original = local.get_session

        def get_session(sid):
            if (failure == "root_read" and sid == "root") or (failure == "tip_read" and sid == "tip"):
                raise error("injected read failure")
            return original(sid)

        def discovery(sid):
            raise error("injected discovery failure")

        monkeypatch.setattr(local, "get_session", get_session)
        if failure == "tip_discovery":
            # A recoverable root would have reopened under the old except: pass.
            local._write_sql("UPDATE sessions SET end_reason='ws_orphan_reap' WHERE id='root'")
            monkeypatch.setattr(local, "get_compression_tip", discovery)
        before = _rows(local), _rows(sibling)
        assert local.unarchive_recoverable_session("root") is False
        assert (_rows(local), _rows(sibling)) == before


def test_guard_inventory_error_propagates_before_mutation(tmp_path, monkeypatch):
    with _stores(tmp_path, monkeypatch) as (local, sibling):
        before = _rows(local), _rows(sibling)

        @contextmanager
        def unreadable(*args, **kwargs):
            raise SessionLineageInventoryError("inventory unreadable")
            yield

        monkeypatch.setattr(sessions, "session_lineage_guard", unreadable)
        with pytest.raises(SessionLineageInventoryError):
            local.unarchive_recoverable_session("root")
        assert (_rows(local), _rows(sibling)) == before


def test_tip_lock_is_held_through_both_recovery_writes(tmp_path, monkeypatch):
    with _stores(tmp_path, monkeypatch) as (local, _sibling):
        original_guard = sessions.session_lineage_guard
        attempts = []
        unfinished = []

        def competing_writer():
            try:
                with original_guard(local.db_path, "tip", timeout_s=0):
                    attempts.append("entered")
            except SessionLineageInventoryError:
                attempts.append("locked")

        def observe_write(sql):
            statement = " ".join(sql.split()).upper()
            if not any(marker in statement for marker in (
                "UPDATE SESSIONS SET ARCHIVED =", "UPDATE SESSIONS SET ENDED_AT =",
            )):
                return
            worker = threading.Thread(target=competing_writer)
            worker.start()
            worker.join(timeout=5)
            unfinished.append(worker.is_alive())

        # Observe the actual writes, independent of which transaction helper calls them.
        local._conn.set_trace_callback(observe_write)
        try:
            assert local.unarchive_recoverable_session("root") is True
        finally:
            local._conn.set_trace_callback(None)
        assert attempts == ["locked", "locked"]
        assert not any(unfinished)


def test_recovery_persists_across_reopen_and_deliberate_archive(tmp_path, monkeypatch):
    with _stores(tmp_path, monkeypatch) as (local, _sibling):
        local.append_message("tip", "user", "retained content")
        assert local.unarchive_recoverable_session("root") is True
        reopened = SessionDB(local.db_path)
        try:
            assert reopened.get_compression_tip("root") == "tip"
            assert reopened.get_session("tip")["ended_at"] is None
            assert any(m["content"] == "retained content" for m in reopened.get_messages("tip"))
            reopened.append_message("tip", "user", "continued after reopening")
            assert reopened.unarchive_recoverable_session("root") is False
            reopened.set_session_archived("root", True)
            assert reopened.unarchive_recoverable_session("root") is False
            assert reopened.get_session("tip")["archived"]
        finally:
            reopened.close()


def test_failed_end_stamp_clear_leaves_every_row_unchanged(tmp_path, monkeypatch):
    """An end-stamp failure rolls back every lineage row and leaves the sibling intact."""
    with _stores(tmp_path, monkeypatch) as (local, sibling):
        local._conn.execute(
            "CREATE TRIGGER reject_reopen BEFORE UPDATE OF ended_at ON sessions "
            "WHEN OLD.id='tip' AND NEW.ended_at IS NULL "
            "BEGIN SELECT RAISE(ABORT, 'injected end-stamp failure'); END"
        )
        local._conn.commit()
        before = _rows(local), _rows(sibling)
        with pytest.raises(sqlite3.IntegrityError, match="injected end-stamp failure"):
            local.unarchive_recoverable_session("root")
        assert (_rows(local), _rows(sibling)) == before
