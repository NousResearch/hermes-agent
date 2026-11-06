"""Import, physical-inventory and managed adoption-retry regressions for #149132.

Real worker compression receipt coverage lives in test_worker_receipt_lineage.py.
"""

from __future__ import annotations

import os
import time

import pytest

from hermes_state import SessionDB
from hermes_state_lineage import (
    SessionIdOwnedElsewhereError,
    SessionLineageAmbiguousError,
    session_lineage_guard,
)


@pytest.fixture
def profile_stores(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    profile = root / "profiles" / "science"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root, profile


def _seed_row(db: SessionDB, session_id: str, *, key: str, profile: str, ended_at=None, reason=None):
    db._conn.execute(
        "INSERT INTO sessions (id, source, started_at, profile_name, session_key, ended_at, end_reason, message_count) "
        "VALUES (?, 'tui', ?, ?, ?, ?, ?, 0)",
        (session_id, time.time(), profile, key, ended_at, reason),
    )
    db._conn.commit()


# ── finding 2: import fences every payload id ─────────────────────────────────

def test_import_rejects_an_id_moved_to_a_sibling_before_the_guard(profile_stores, monkeypatch):
    root, profile = profile_stores
    owner = SessionDB(root / "state.db")
    sibling = SessionDB(profile / "state.db")
    try:
        _seed_row(owner, "victim", key="agent:main:local:dm:victim", profile="main")
        payload = [{"id": "victim", "source": "tui", "started_at": 1, "messages": []}]

        import hermes_state_portability as portability
        real_guard = portability.session_lineage_guard

        def interposed_move(db_path, ids, **kwargs):
            # The deterministic race: between the presence read and the guard, the id
            # moves to the sibling store (insert there, vanish here).
            _seed_row(sibling, "victim", key="agent:science:local:dm:victim", profile="science")
            owner._conn.execute("DELETE FROM sessions WHERE id='victim'")
            owner._conn.commit()
            return real_guard(db_path, ids, **kwargs)

        monkeypatch.setattr(portability, "session_lineage_guard", interposed_move)
        with pytest.raises(SessionIdOwnedElsewhereError):
            owner.import_sessions(payload)
        assert owner.get_session("victim") is None  # source stays absent, not recreated
    finally:
        owner.close()
        sibling.close()


# ── finding 3: repair dual-home sees tombstoned physical stores ────────────────

def test_repair_dual_home_names_a_tombstoned_blocking_store(profile_stores):
    root, _live = profile_stores
    ghost_dir = root / "profiles" / "ghost"
    ghost_dir.mkdir(parents=True)
    owner = SessionDB(root / "state.db")
    ghost = SessionDB(ghost_dir / "state.db")
    try:
        _seed_row(owner, "dup", key="agent:main:local:dm:dup", profile="main")
        _seed_row(ghost, "dup", key="agent:ghost:local:dm:dup", profile="ghost")

        # Tombstone the profile: physical state.db remains (the fence still sees it),
        # but live enumeration must skip it.
        from hermes_constants import mark_named_profile_deleted
        mark_named_profile_deleted(ghost_dir)

        from hermes_cli.sessions_repair_profiles import RepairPlan, _Session, enumerate_stores

        stores = enumerate_stores()
        assert all(s.profile != "ghost" for s in stores)  # tombstoned is not enumerated live
        session = _Session(read_only=True)
        try:
            plan = RepairPlan(stores).scan(session)
        finally:
            session.close()  # read-only handle must not leak past the scan
        dual = [f for f in plan.findings if f.kind == "dual_home" and f.subject == "dup"]
        assert dual, "dual_home finding must exist for the tombstoned duplicate"
        assert "ghost" in dual[0].store  # …and it must NAME the blocking physical store
    finally:
        owner.close()
        ghost.close()


# ── finding 4: failed donor retirement must not block the adoption retry ──────

def test_failed_donor_retirement_retry_completes_adoption(profile_stores, monkeypatch):
    root, profile = profile_stores
    donor = SessionDB(root / "state.db")
    target = SessionDB(profile / "state.db")
    sid = "20261104_stranded_1"
    try:
        donor.create_session(sid, source="tui", session_key="agent:main:local:dm:" + sid)
        for i in range(1, 3):
            donor.append_message(sid, "user", f"q{i}")
            donor.append_message(sid, "assistant", f"a{i}")

        # Inject exactly one retirement failure — scoped so only the override is undone;
        # the fixture's managed HERMES_HOME must stay active through the whole retry.
        with monkeypatch.context() as retirement_patch:
            retirement_patch.setattr(SessionDB, "_retire_donor_segment",
                                     lambda self, donor_db, seg_id: False)
            first = target.adopt_session_lineage_from(donor, sid)
            assert first["adopted"] is True and first["donor_retired"] is False

        assert os.environ["HERMES_HOME"] == str(root)  # fence namespace never reset

        # Pending-transfer duplicate stays fenced for ORDINARY writes: two live owners,
        # no transfer_from → ambiguity, nothing mutated.
        with pytest.raises(SessionLineageAmbiguousError):
            with session_lineage_guard(target.db_path, (sid,)):
                pass  # guard entry itself must refuse the ambiguous pair

        retried = target.adopt_session_lineage_from(donor, sid)  # documented retry
        assert retried["adopted"] is True
        assert retried["donor_retired"] is True
        donor_row = donor.get_session(sid)
        assert donor_row is not None and donor_row["end_reason"] == "adopted_by_profile"
        assert len(target.get_messages(sid, include_inactive=True)) == 4
        with session_lineage_guard(target.db_path, (sid,)) as snapshot:
            assert snapshot is not None  # retry finished inside a managed home
        target.append_message(sid, "user", "post-adoption write")  # target stays writable
        assert len(target.get_messages(sid)) == 5
    finally:
        donor.close()
        target.close()
