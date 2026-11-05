"""Adoption preserves deletion, store identity and complete transfer authority."""

import sqlite3
import time
from contextlib import contextmanager

import pytest

from hermes_state import SessionDB
from hermes_state_lineage import SessionLineageError, SessionLineageInventoryError
from tests.hermes_state.test_adoption_transfer_records import _open, _retire_fails_for, _seed_compression_lineage


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def _refused(call):
    try:
        return call().get("ok") is False
    except SessionLineageError:
        return True


@pytest.mark.parametrize("pending", [False, True])
def test_deleted_target_member_is_never_restored(home, pending):
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    donor, target = _open(home, default=True), _open(home)
    sid = "deleted-adoption-member"
    try:
        donor.create_session(sid, "tui")
        donor.append_message(sid, "user", "preserve donor")
        if pending:
            assert target.adopt_session_lineage_from(donor, sid, retire_donor=False)["adopted"]
        target._conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
        target._conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
        target._conn.execute("INSERT INTO state_meta(key,value) VALUES (?,?)", (RETIRED_PREFIX + sid, "{}"))
        target._conn.commit()
        assert _refused(lambda: target.adopt_session_lineage_from(donor, sid))
        assert target.get_session(sid) is None
        assert donor.get_session(sid)["end_reason"] is None
        assert [m["content"] for m in donor.get_messages(sid)] == ["preserve donor"]
    finally:
        donor.close()
        target.close()


def test_unrelated_ended_target_does_not_authorize_a_new_record(home):
    donor, target = _open(home, default=True), _open(home)
    sid = "unrelated-closed-target"
    try:
        donor.create_session(sid, "tui")
        donor.append_message(sid, "user", "donor history")
        target._conn.execute(
            "INSERT INTO sessions(id,source,started_at,ended_at,end_reason,message_count) VALUES (?,'tui',?,?,'user_close',1)",
            (sid, time.time(), time.time()),
        )
        target._conn.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES (?,'user','unrelated history',?)",
                             (sid, time.time()))
        target._conn.commit()
        assert _refused(lambda: target.adopt_session_lineage_from(donor, sid))
        assert donor.get_session(sid)["end_reason"] is None
        assert [m["content"] for m in target.get_messages(sid, include_inactive=True)] == ["unrelated history"]
    finally:
        donor.close()
        target.close()


def test_saved_record_cannot_be_rebound_to_another_standalone_donor(home, tmp_path):
    donor = SessionDB(tmp_path / "first" / "state.db")
    target = SessionDB(tmp_path / "target" / "state.db")
    other = SessionDB(tmp_path / "other" / "state.db")
    sid = "same-id-different-donor"
    try:
        donor.create_session(sid, "tui")
        donor.append_message(sid, "user", "original")
        assert target.adopt_session_lineage_from(donor, sid)["donor_retired"]
        other.create_session(sid, "tui")
        other.append_message(sid, "user", "unrelated")
        assert _refused(lambda: target.adopt_session_lineage_from(other, sid))
        assert other.get_session(sid)["end_reason"] is None
        assert [m["content"] for m in target.get_messages(sid)] == ["original"]
        rows = target._conn.execute("SELECT DISTINCT donor_store FROM adoption_transfers").fetchall()
        assert [row[0] for row in rows] == [str(donor.db_path.resolve())]
    finally:
        donor.close()
        target.close()
        other.close()


def test_record_lookup_failure_cannot_fall_back_to_a_truncated_walk(home, monkeypatch):
    donor, target = _open(home, default=True), _open(home)
    try:
        _seed_compression_lineage(donor, "lookup-root", "lookup-tip")
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for("lookup-tip"))
            assert target.adopt_session_lineage_from(donor, "lookup-tip")["donor_retired"] is False
        original = target._read_ctx

        class FailedLookup:
            def __init__(self, conn):
                self.conn = conn

            def execute(self, sql, *args):
                if "FROM adoption_transfers" in sql:
                    raise sqlite3.OperationalError("database is locked")
                return self.conn.execute(sql, *args)

        @contextmanager
        def failed_read():
            with original() as conn:
                yield FailedLookup(conn)

        with monkeypatch.context() as patch:
            patch.setattr(target, "_read_ctx", failed_read)
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                target.adopt_session_lineage_from(donor, "lookup-root")
        rows = target._conn.execute("SELECT member_id FROM adoption_transfers ORDER BY member_order").fetchall()
        assert [row[0] for row in rows] == ["lookup-root", "lookup-tip"]
        assert donor.get_session("lookup-tip")["end_reason"] is None
    finally:
        donor.close()
        target.close()


def test_unsupported_record_version_is_not_adoption_authority(home):
    donor, target = _open(home, default=True), _open(home)
    sid = "unsupported-transfer"
    try:
        donor.create_session(sid, "tui")
        donor.append_message(sid, "user", "q")
        assert target.adopt_session_lineage_from(donor, sid, retire_donor=False)["adopted"]
        target._conn.execute("UPDATE adoption_transfers SET format_version=999")
        target._conn.commit()
        assert _refused(lambda: target.adopt_session_lineage_from(donor, sid))
        assert donor.get_session(sid)["end_reason"] is None
        assert target._conn.execute("SELECT format_version FROM adoption_transfers").fetchone()[0] == 999
    finally:
        donor.close()
        target.close()


def test_resolution_error_is_not_a_standalone_classification(home, tmp_path, monkeypatch):
    import hermes_state_lineage as lineage
    donor = _open(home, default=True)
    target = SessionDB(tmp_path / "outside" / "state.db")
    sid = "unverifiable-donor"
    try:
        donor.create_session(sid, "tui")
        original = lineage._managed_root_and_current

        def resolve(path):
            if str(path) == str(donor.db_path):
                raise SessionLineageInventoryError("Cannot inspect donor")
            return original(path)

        monkeypatch.setattr(lineage, "_managed_root_and_current", resolve)
        assert _refused(lambda: target.adopt_session_lineage_from(donor, sid))
        assert target.get_session(sid) is None
        assert donor.get_session(sid)["end_reason"] is None
    finally:
        donor.close()
        target.close()
