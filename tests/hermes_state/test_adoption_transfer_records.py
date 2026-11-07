"""Durable adoption-transfer records (second-round review findings 1 and 2).

Pins the two HIGH findings from the second independent review:
1. Multi-segment retries: a failed donor retirement must NOT become unretryable (two
   ended predecessor copies), and a retry after PARTIAL retirement must resolve the
   COMPLETE original membership from the durable record — never a truncated compression
   walk that reports false completion while a segment stays live in both stores.
2. Mixed managed/standalone store pairs are refused before any mutation (donor-side
   serialization cannot be guaranteed); same-managed-home pairs run fully fenced, and
   the pre-existing standalone/standalone contract is unchanged.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_lineage import (
    SessionIdOwnedElsewhereError,
    SessionLineageAmbiguousError,
    session_lineage_guard,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def _open(root: Path, *, default: bool = False, name: str = "science") -> SessionDB:
    db_path = root / "state.db" if default else root / "profiles" / name / "state.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SessionDB(db_path)


def _seed_compression_lineage(donor: SessionDB, root_id: str, tip_id: str) -> None:
    """Root segment ended-by-compression with a live child tip (the stranded shape).
    Messages are appended while both rows are live — appending to a session already
    ended-by-compression is correctly refused — then the root is stamped."""
    donor._conn.execute(
        "INSERT INTO sessions (id, source, started_at, profile_name, session_key, message_count)"
        " VALUES (?, 'tui', ?, 'default', ?, 0)",
        (root_id, time.time(), f"agent:main:local:dm:{root_id}"),
    )
    donor._conn.execute(
        "INSERT INTO sessions (id, source, started_at, profile_name, session_key, parent_session_id,"
        " message_count) VALUES (?, 'tui', ?, 'default', ?, ?, 0)",
        (tip_id, time.time(), f"agent:main:local:dm:{tip_id}", root_id),
    )
    donor._conn.commit()
    donor.append_message(root_id, "user", "root q")
    donor.append_message(root_id, "assistant", "root a")
    donor.append_message(tip_id, "user", "tip q")
    donor.append_message(tip_id, "assistant", "tip a")
    donor._conn.execute(
        "UPDATE sessions SET ended_at = ?, end_reason = 'compression' WHERE id = ?",
        (time.time(), root_id),
    )
    donor._conn.commit()


def _retire_fails_for(*seg_ids):
    """Class-level override: retirement returns False exactly for the named segments and
    DELEGATES to the real retirement for every other segment (a bare True would claim
    retirement without stamping the donor)."""
    original = SessionDB._retire_donor_segment

    def _override(self, donor_db, seg_id):
        if seg_id in seg_ids:
            return False
        return original(self, donor_db, seg_id)

    return _override


def test_two_segment_retry_after_first_retirement_failure_completes(home, monkeypatch):
    """Case 1a: failing the ROOT's retirement leaves two ended predecessor copies (donor
    'compression' + imported 'compression') that ordinary selection rejects — the durable
    record must carry the retry through to complete membership retirement."""
    donor, target = _open(home, default=True), _open(home)
    root_id, tip_id = "20261105_segA", "20261105_segB"
    try:
        _seed_compression_lineage(donor, root_id, tip_id)
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(root_id))
            first = target.adopt_session_lineage_from(donor, tip_id)
        assert first["adopted"] is True and first["donor_retired"] is False

        retried = target.adopt_session_lineage_from(donor, tip_id)
        assert retried["adopted"] is True and retried["donor_retired"] is True
        assert retried["pending_ids"] == []
        assert sorted(first["member_ids"]) == sorted([root_id, tip_id])
        assert root_id in first["pending_ids"]
        for mid in (root_id, tip_id):
            row = donor.get_session(mid)
            assert row is not None and row["end_reason"] == "adopted_by_profile"
        assert len(target.get_messages(root_id, include_inactive=True)) == 2
        assert len(target.get_messages(tip_id)) == 2
        target.append_message(tip_id, "user", "post-adoption write")
        assert len(target.get_messages(tip_id)) == 3
    finally:
        donor.close()
        target.close()


def test_retry_by_root_after_partial_retirement_keeps_full_membership(home, monkeypatch):
    """Case 1b (the false-completion defect): root retired, tip retirement fails; a retry
    requested BY THE ROOT must still retire the tip — the truncated compression walk
    (root now 'adopted_by_profile') must not shrink the denominator."""
    donor, target = _open(home, default=True), _open(home)
    root_id, tip_id = "20261105_segC", "20261105_segD"
    try:
        _seed_compression_lineage(donor, root_id, tip_id)
        donor._conn.execute(
            "UPDATE messages SET display_metadata=? WHERE session_id=? AND role='user' AND active=1",
            ('{"_queued_prompt":true}', root_id),
        )
        donor._conn.commit()
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(tip_id))
            first = target.adopt_session_lineage_from(donor, tip_id)
        assert first["adopted"] is True and first["donor_retired"] is False
        assert donor.get_session(root_id)["end_reason"] == "adopted_by_profile"

        assert first["pending_ids"] == [tip_id]
        retried = target.adopt_session_lineage_from(donor, root_id)  # BY THE ROOT
        assert retried["donor_retired"] is True, "retry must complete the FULL recorded membership"
        assert sorted(retried["member_ids"]) == sorted([root_id, tip_id])
        tip_row = donor.get_session(tip_id)
        assert tip_row is not None and tip_row["end_reason"] == "adopted_by_profile"
        assert len(target.get_messages(tip_id)) == 2  # no transcript loss
        target_root = target.get_messages(root_id, include_inactive=True)
        donor_root = donor.get_messages(root_id, include_inactive=True)
        assert [row["content"] for row in target_root] == [row["content"] for row in donor_root]
        assert target_root[0]["active"] == 1 and donor_root[0]["active"] == 0
        assert target_root[0]["display_metadata"] == {"_queued_prompt": True}
        model_history, display_history = target.get_resume_conversations(tip_id)
        assert all("root q" not in str(row.get("content", "")) for row in model_history)
        assert any("root q" in str(row.get("content", "")) for row in display_history)
        target.append_message(tip_id, "user", "still writable")
        assert len(target.get_messages(tip_id)) == 3
    finally:
        donor.close()
        target.close()


def test_record_member_missing_from_donor_is_a_conflict_not_a_truncation(home):
    """A record naming a donor segment that no longer exists fails closed with an explicit
    conflict — never a shortened-list success."""
    donor, target = _open(home, default=True), _open(home)
    root_id, tip_id = "20261105_segE", "20261105_segF"
    try:
        _seed_compression_lineage(donor, root_id, tip_id)
        first = target.adopt_session_lineage_from(donor, tip_id, retire_donor=False)
        assert first["adopted"] is True
        donor._conn.execute("DELETE FROM messages WHERE session_id = ?", (tip_id,))
        donor._conn.execute("DELETE FROM sessions WHERE id = ?", (tip_id,))
        donor._conn.commit()
        result = target.adopt_session_lineage_from(donor, root_id)
        assert result.get("conflict") is True or result.get("ok") is False
        assert result.get("donor_retired") is not True
    finally:
        donor.close()
        target.close()


def test_mixed_managed_standalone_pair_is_refused_before_mutation(home, tmp_path):
    """Finding 2: a managed donor with a standalone target (and the reciprocal) is
    refused with NO import and NO retirement on either side."""
    donor = _open(home, default=True)
    outside = SessionDB(tmp_path / "elsewhere" / "state.db")  # outside the managed home
    sid = "20261105_mixed"
    try:
        donor.create_session(sid, source="tui", session_key="agent:main:local:dm:" + sid)
        donor.append_message(sid, "user", "q")
        result = outside.adopt_session_lineage_from(donor, sid)
        assert result.get("ok") is False and "mixed" in result.get("error", "")
        assert outside.get_session(sid) is None, "no target row may appear"
        row = donor.get_session(sid)
        assert row is not None and row.get("end_reason") is None, "donor untouched"
        assert len(donor.get_messages(sid)) == 1, "donor transcript untouched"

        # Reciprocal: standalone donor, managed target.
        sid2 = "20261105_mixed2"
        outside.create_session(sid2, source="tui", session_key="agent:main:local:dm:" + sid2)
        reciprocal = donor.adopt_session_lineage_from(outside, sid2)
        assert reciprocal.get("ok") is False and "mixed" in reciprocal.get("error", "")
        assert donor.get_session(sid2) is None
    finally:
        donor.close()
        outside.close()


def test_third_store_copy_blocks_pending_adoption_retry(home, monkeypatch):
    """A third live copy surfacing during a pending adoption refuses the retry with no
    mutation anywhere: the membership tolerance never absorbs extra owners."""
    donor, target, third = _open(home, default=True), _open(home), _open(home, name="other")
    root_id, tip_id = "20261105_segG", "20261105_segH"
    try:
        _seed_compression_lineage(donor, root_id, tip_id)
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(tip_id))
            first = target.adopt_session_lineage_from(donor, tip_id)
        assert first["adopted"] is True and first["donor_retired"] is False

        donor_before = len(donor.get_messages(tip_id, include_inactive=True))
        target_before = len(target.get_messages(tip_id, include_inactive=True))
        third._conn.execute(
            "INSERT INTO sessions (id, source, started_at, profile_name, session_key, message_count)"
            " VALUES (?, 'tui', ?, 'other', ?, 0)",
            (tip_id, time.time(), f"agent:main:local:dm:{tip_id}"),
        )
        third._conn.commit()
        refused = False
        try:
            target.adopt_session_lineage_from(donor, tip_id)
        except (SessionLineageAmbiguousError, SessionIdOwnedElsewhereError):
            refused = True
        assert refused, "a third live copy must block the pending retry"
        assert len(donor.get_messages(tip_id, include_inactive=True)) == donor_before
        assert len(target.get_messages(tip_id, include_inactive=True)) == target_before
    finally:
        donor.close()
        target.close()
        third.close()


def test_standalone_pair_adoption_contract_is_unchanged(tmp_path):
    """Both stores standalone: the pre-existing unfenced compatibility path still adopts."""
    donor = SessionDB(tmp_path / "a" / "state.db")
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    target = SessionDB(tmp_path / "b" / "state.db")
    sid = "20261105_standalone"
    try:
        donor.create_session(sid, source="tui", session_key="agent:main:local:dm:" + sid)
        donor.append_message(sid, "user", "q")
        result = target.adopt_session_lineage_from(donor, sid)
        assert result["adopted"] is True and result["donor_retired"] is True
    finally:
        donor.close()
        target.close()


@pytest.mark.parametrize("corruption", ["target_drift", "legacy", "malformed"])
def test_partial_retry_rejects_missing_or_changed_activity_witness(home, monkeypatch, corruption):
    import json

    donor, target = _open(home, default=True), _open(home)
    root, tip = "witness-root", "witness-tip"
    try:
        _seed_compression_lineage(donor, root, tip)
        donor._conn.execute(
            "UPDATE messages SET display_metadata=? WHERE session_id=? AND role='user'",
            ('{"_queued_prompt":true}', root))
        donor._conn.commit()
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(tip))
            assert target.adopt_session_lineage_from(donor, tip)["pending_ids"] == [tip]
        if corruption == "target_drift":
            target._conn.execute("UPDATE messages SET active=0 WHERE session_id=? AND role='user'", (root,))
        else:
            row = target._conn.execute("SELECT transfer_id,manifest_json FROM adoption_transfers WHERE manifest_json IS NOT NULL").fetchone()
            manifest = json.loads(row[1])
            member = next(m for m in manifest["members"] if m["id"] == root)
            if corruption == "legacy":
                del member["source_count"]
                del member["target_active_digest"]
            else:
                member["target_active_digest"] = None
            target._conn.execute("UPDATE adoption_transfers SET manifest_json=? WHERE transfer_id=? AND manifest_json IS NOT NULL",
                                 (json.dumps(manifest), row[0]))
        target._conn.commit()
        outcome = target.adopt_session_lineage_from(donor, root)
        assert outcome["conflict"] and "activity witness" in outcome["error"]
        assert donor.get_session(tip)["end_reason"] != "adopted_by_profile"
        assert target._conn.execute("SELECT count(*) FROM adoption_transfers").fetchone()[0] == 2
    finally:
        donor.close()
        target.close()


@pytest.mark.parametrize("metadata,role,should_retire", [
    ('{"_queued_prompt":true}', "user", True),
    ('{"_queued_prompt":1}', "user", True),
    ('{"_queued_prompt":1.0}', "user", True),
    ('{"_queued_prompt":"1"}', "user", False),
    ('{"_queued_prompt":0}', "user", False),
    ('{broken', "user", False),
    ('{"_queued_prompt":{}}', "user", False),
    ('{"_queued_prompt":true}', "assistant", False),
])
def test_retired_activity_exception_matches_sql_marker(home, monkeypatch, metadata, role, should_retire):
    donor, target = _open(home, default=True), _open(home)
    root, tip = "marker-root", "marker-tip"
    try:
        _seed_compression_lineage(donor, root, tip)
        donor._conn.execute("UPDATE messages SET display_metadata=? WHERE session_id=? AND role=?", (metadata, root, role))
        donor._conn.commit()
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(tip))
            first = target.adopt_session_lineage_from(donor, tip)
        if metadata == "{broken":
            # Import normalizes invalid metadata, so the strict copy check refuses it
            # before a recorded retry exists; retirement must not begin.
            assert not first["adopted"] and not first["donor_retired"]
            assert donor.get_session(root)["end_reason"] == "compression"
            return
        assert first["pending_ids"] == [tip]
        source = donor._conn.execute("SELECT active FROM messages WHERE session_id=? AND role=?", (root, role)).fetchone()[0]
        assert source == (0 if should_retire else 1)
        if not should_retire:
            donor._conn.execute("UPDATE messages SET active=0 WHERE session_id=? AND role=?", (root, role))
            donor._conn.commit()
        result = target.adopt_session_lineage_from(donor, root)
        if should_retire:
            assert result["donor_retired"]
        else:
            assert result["conflict"] and not result["donor_retired"]
            assert donor.get_session(tip)["end_reason"] is None
    finally:
        donor.close()
        target.close()


def test_three_member_retirement_retry_survives_reopening(home, monkeypatch):
    donor, target = _open(home, default=True), _open(home)
    root, middle, tip = "reopen-root", "reopen-middle", "reopen-tip"
    try:
        _seed_compression_lineage(donor, root, middle)
        donor._conn.execute("INSERT INTO sessions (id, source, started_at, parent_session_id) VALUES (?, 'tui', ?, ?)", (tip, time.time(), middle))
        donor._conn.commit()
        donor.append_message(tip, "user", "tip input")
        donor._conn.execute("UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (time.time(), middle))
        donor._conn.execute("UPDATE messages SET display_metadata=? WHERE session_id IN (?, ?) AND role='user'", ('{"_queued_prompt":true}', root, middle))
        donor._conn.commit()
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(middle))
            first = target.adopt_session_lineage_from(donor, tip)
        assert first["pending_ids"] == [middle, tip]
        donor.close(); target.close()
        donor, target = _open(home, default=True), _open(home)
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(tip))
            second = target.adopt_session_lineage_from(donor, root)
        assert second["pending_ids"] == [tip]
        donor.close(); target.close()
        donor, target = _open(home, default=True), _open(home)
        final = target.adopt_session_lineage_from(donor, middle)
        assert final["donor_retired"] and final["member_ids"] == [root, middle, tip]
        assert all(donor.get_session(mid)["end_reason"] == "adopted_by_profile" for mid in (root, middle, tip))
        assert [len(target.get_messages(mid, include_inactive=True)) for mid in (root, middle, tip)] == [2, 2, 1]
    finally:
        donor.close()
        target.close()


def test_legacy_saved_transfer_with_exact_history_can_retry(home, monkeypatch):
    import json

    donor, target = _open(home, default=True), _open(home)
    root, tip = "legacy-exact-root", "legacy-exact-tip"
    try:
        _seed_compression_lineage(donor, root, tip)
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(tip))
            assert target.adopt_session_lineage_from(donor, tip)["pending_ids"] == [tip]
        row = target._conn.execute("SELECT transfer_id, manifest_json FROM adoption_transfers WHERE manifest_json IS NOT NULL").fetchone()
        manifest = json.loads(row[1])
        for member in manifest["members"]:
            del member["source_count"]
            del member["target_active_digest"]
        target._conn.execute("UPDATE adoption_transfers SET manifest_json=? WHERE transfer_id=? AND manifest_json IS NOT NULL", (json.dumps(manifest), row[0]))
        target._conn.commit()
        result = target.adopt_session_lineage_from(donor, root)
        assert result["donor_retired"] and result["member_ids"] == [root, tip]
    finally:
        donor.close()
        target.close()


def test_recorded_malformed_queue_marker_cannot_authorize_activity_mismatch(home, monkeypatch):
    donor, target = _open(home, default=True), _open(home)
    root, tip = "malformed-root", "malformed-tip"
    try:
        _seed_compression_lineage(donor, root, tip)
        donor._conn.execute("UPDATE messages SET display_metadata=? WHERE session_id=? AND role='user'", ('{"_queued_prompt":true}', root))
        donor._conn.commit()
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(tip))
            assert target.adopt_session_lineage_from(donor, tip)["pending_ids"] == [tip]
        for db in (donor, target):
            db._conn.execute("UPDATE messages SET display_metadata=? WHERE session_id=? AND role='user'", ('{broken', root))
            db._conn.commit()
        result = target.adopt_session_lineage_from(donor, root)
        assert result["conflict"] and not result["donor_retired"]
        assert donor.get_session(tip)["end_reason"] is None
    finally:
        donor.close()
        target.close()


def test_archive_failure_rolls_back_entire_retirement_and_retry(home):
    donor, target = _open(home, default=True), _open(home)
    root, tip = "atomic-retire-root", "atomic-retire-tip"
    try:
        _seed_compression_lineage(donor, root, tip)
        donor._conn.execute("UPDATE messages SET display_metadata=? WHERE session_id=? AND role='user'", ('{"_queued_prompt":true}', root))
        donor._conn.execute("CREATE TRIGGER fail_retire_archive BEFORE UPDATE OF archived ON sessions "
                            "WHEN NEW.archived=1 BEGIN SELECT RAISE(ABORT, 'injected archive fault'); END")
        donor._conn.commit()
        old = donor.get_session(root)
        first = target.adopt_session_lineage_from(donor, tip)
        assert first["adopted"] and not first["donor_retired"]
        after = donor.get_session(root)
        assert (after["ended_at"], after["end_reason"], after["archived"]) == (old["ended_at"], old["end_reason"], old["archived"])
        assert donor.get_messages(root, include_inactive=True)[0]["active"] == 1
        donor._conn.execute("DROP TRIGGER fail_retire_archive")
        donor._conn.commit()
        done = target.adopt_session_lineage_from(donor, root)
        assert done["donor_retired"] and done["member_ids"] == [root, tip]
        assert all(donor.get_session(mid)["archived"] for mid in (root, tip))
        assert [len(target.get_messages(mid, include_inactive=True)) for mid in (root, tip)] == [2, 2]
    finally:
        donor.close()
        target.close()


def test_half_retired_legacy_mismatch_without_witness_stays_pending(home, monkeypatch):
    import json

    donor, target = _open(home, default=True), _open(home)
    root, tip = "legacy-half-root", "legacy-half-tip"
    try:
        _seed_compression_lineage(donor, root, tip)
        donor._conn.execute("UPDATE messages SET display_metadata=? WHERE session_id=? AND role='user'", ('{"_queued_prompt":true}', root))
        donor._conn.commit()
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(root))
            assert target.adopt_session_lineage_from(donor, tip)["pending_ids"] == [root, tip]
        row = target._conn.execute("SELECT transfer_id, manifest_json FROM adoption_transfers WHERE manifest_json IS NOT NULL").fetchone()
        manifest = json.loads(row[1])
        for member in manifest["members"]:
            del member["source_count"]
            del member["target_active_digest"]
        target._conn.execute("UPDATE adoption_transfers SET manifest_json=? WHERE transfer_id=? AND manifest_json IS NOT NULL", (json.dumps(manifest), row[0]))
        target._conn.commit()
        assert donor.end_session_as_adopted(root)
        result = target.adopt_session_lineage_from(donor, root)
        assert result["conflict"] and "activity witness" in result["error"]
        assert not donor.get_session(root)["archived"] and donor.get_session(tip)["end_reason"] is None
        assert target._conn.execute("SELECT count(*) FROM adoption_transfers").fetchone()[0] == 2
    finally:
        donor.close()
        target.close()


def test_disappearing_pending_transfer_fails_closed(home, monkeypatch):
    from contextlib import contextmanager
    from hermes_state_adoption_records import AdoptionConflict, retirement_pending_source

    donor, target = _open(home, default=True), _open(home)
    root, tip = "disappear-root", "disappear-tip"
    try:
        _seed_compression_lineage(donor, root, tip)
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(root))
            assert target.adopt_session_lineage_from(donor, tip)["pending_ids"] == [root, tip]
        original = target._read_ctx

        class Result:
            def __init__(self, cursor):
                self.cursor = cursor

            def fetchall(self):
                rows = self.cursor.fetchall()
                target._conn.execute("DELETE FROM adoption_transfers")
                target._conn.commit()
                return rows

        class Connection:
            def __init__(self, conn):
                self.conn = conn

            def execute(self, sql, params=()):
                result = self.conn.execute(sql, params)
                if sql.startswith("SELECT DISTINCT transfer_id FROM adoption_transfers"):
                    return Result(result)
                return result

        @contextmanager
        def intercepted():
            with original() as conn:
                yield Connection(conn)

        monkeypatch.setattr(target, "_read_ctx", intercepted)
        with pytest.raises(AdoptionConflict, match="disappeared"):
            retirement_pending_source(target, root)
        assert donor.get_session(root)["end_reason"] == "compression"
        assert target._conn.execute("SELECT COUNT(*) FROM sessions WHERE id IN (?, ?)", (root, tip)).fetchone()[0] == 2
    finally:
        donor.close()
        target.close()
