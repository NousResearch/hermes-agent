"""Managed turn admission is one durable transcript write, not a second reservation."""
import sqlite3

import pytest

from hermes_state import SessionDB


def test_managed_lookup_follows_only_verified_compression_lineage(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("parent", source="desktop")
        row_id = db.append_message("parent", "user", content="original", managed_turn_key="e" * 32)
        db.end_session("parent", "compression")
        db.create_session("compressed-child", source="desktop", parent_session_id="parent")
        assert db.get_managed_turn("compressed-child", "e" * 32) == {"user_row_id": row_id}
        db.create_session("fork-parent", source="desktop")
        db.append_message("fork-parent", "user", content="private fork", managed_turn_key="f" * 32)
        db.end_session("fork-parent", "branched")
        db.create_session("fork-child", source="desktop", parent_session_id="fork-parent")
        assert db.get_managed_turn("fork-child", "f" * 32) is None
        db.create_session("reset-parent", source="desktop")
        db.append_message("reset-parent", "user", content="before reset", managed_turn_key="a" * 32)
        db.end_session("reset-parent", "reset")
        db.create_session("reset-child", source="desktop", parent_session_id="reset-parent")
        assert db.get_managed_turn("reset-child", "a" * 32) is None
    finally:
        db.close()


def test_managed_turn_receipt_failure_rolls_back_user_row(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        db.create_session("owned", source="desktop")
        raw = sqlite3.connect(path)
        try:
            raw.execute("""CREATE TRIGGER deny_managed_receipt BEFORE INSERT ON managed_turn_submissions
                           BEGIN SELECT RAISE(ABORT, 'fixture write failure'); END""")
            raw.commit()
        finally:
            raw.close()
        with pytest.raises(sqlite3.IntegrityError):
            db.append_message("owned", "user", content="must stay unadmitted", managed_turn_key="c" * 32)
        assert db.get_managed_turn("owned", "c" * 32) is None
        assert db.get_messages("owned") == []
    finally:
        db.close()


def test_two_database_handles_cannot_admit_same_key(tmp_path):
    path = tmp_path / "state.db"
    first = SessionDB(path)
    second = SessionDB(path)
    try:
        first.create_session("owned", source="desktop")
        row_id = first.append_message("owned", "user", content="first", managed_turn_key="d" * 32)
        with pytest.raises(sqlite3.IntegrityError):
            second.append_message("owned", "user", content="second", managed_turn_key="d" * 32)
        assert second.get_managed_turn("owned", "d" * 32) == {"user_row_id": row_id}
        assert len(second.get_messages("owned")) == 1
    finally:
        first.close()
        second.close()


def test_managed_turn_key_and_user_row_commit_atomically_across_reopen(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        db.create_session("session-a", source="desktop")
        db.create_session("session-b", source="desktop")
        key = "a" * 32
        row_id = db.append_message("session-a", "user", content="Plan a report", managed_turn_key=key)
        assert db.get_managed_turn("session-a", key) == {"user_row_id": row_id}
        assert db.get_managed_turn("session-b", key) is None
        with pytest.raises(sqlite3.IntegrityError):
            db.append_message("session-a", "user", content="Plan a report", managed_turn_key=key)
        with pytest.raises(sqlite3.IntegrityError):
            db.append_message("session-b", "user", content="Different work", managed_turn_key=key)
        assert len(db.get_messages("session-a")) == 1
        assert db.get_messages("session-b") == []
    finally:
        db.close()
    reopened = SessionDB(path)
    try:
        assert reopened.get_managed_turn("session-a", key) == {"user_row_id": row_id}
        assert reopened.get_managed_turn("session-b", key) is None
    finally:
        reopened.close()
