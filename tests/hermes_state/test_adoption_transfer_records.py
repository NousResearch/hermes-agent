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
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for(tip_id))
            first = target.adopt_session_lineage_from(donor, tip_id)
        assert first["adopted"] is True and first["donor_retired"] is False
        assert donor.get_session(root_id)["end_reason"] == "adopted_by_profile"

        retried = target.adopt_session_lineage_from(donor, root_id)  # BY THE ROOT
        assert retried["donor_retired"] is True, "retry must complete the FULL recorded membership"
        assert first["pending_ids"] == [tip_id]
        assert sorted(retried["member_ids"]) == sorted([root_id, tip_id])
        tip_row = donor.get_session(tip_id)
        assert tip_row is not None and tip_row["end_reason"] == "adopted_by_profile"
        assert len(target.get_messages(tip_id)) == 2  # no transcript loss
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
