"""Generic transfer boundaries for #149132.

Explicit generic transfers permit a donor-only row or the exact live donor/local pair.
Saved adoption authorizations additionally verify recorded lifecycle evidence in the
adoption tests. Unrelated owners, third live owners and source/current aliases never
receive a generic transfer exception.
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


def _seed_row(db: SessionDB, session_id: str, *, key: str, profile: str):
    """Insert a session row WITHOUT the fence: the shape tests need pre-existing multi-owner
    rows that guarded ``create_session`` correctly refuses to produce."""
    db._conn.execute(
        "INSERT INTO sessions (id, source, started_at, profile_name, session_key, message_count) "
        "VALUES (?, 'tui', ?, ?, ?, 0)",
        (session_id, time.time(), profile, key),
    )
    db._conn.commit()


def test_ordinary_write_with_two_live_owners_is_ambiguous(home):
    default, science = _open(home, default=True), _open(home)
    sid = "20261105_two_live"
    try:
        default.create_session(sid, source="tui", session_key="agent:main:local:dm:" + sid)
        _seed_row(science, sid, key="agent:main:local:dm:" + sid, profile="science")
        # No transfer_from anywhere: ordinary owner selection must refuse two live owners.
        with pytest.raises(SessionLineageAmbiguousError):
            with session_lineage_guard(default.db_path, (sid,)):
                pass
    finally:
        default.close()
        science.close()


def test_transfer_with_sole_unrelated_owner_is_refused(home):
    default, science, other = _open(home, default=True), _open(home), _open(home, name="other")
    sid = "20261105_unrelated"
    try:
        other.create_session(sid, source="tui", session_key="agent:main:local:dm:" + sid)
        # Declared donor (science) has NO row; the only row lives in `other`. The single-row
        # tolerance must prove the row belongs to the declared source, not merely exist.
        with pytest.raises((SessionIdOwnedElsewhereError, SessionLineageAmbiguousError)):
            with session_lineage_guard(default.db_path, (sid,), transfer_from=science.db_path):
                pass
    finally:
        default.close()
        science.close()
        other.close()


def test_adoption_with_third_live_owner_is_refused(home):
    default, science, other = _open(home, default=True), _open(home), _open(home, name="other")
    sid = "20261105_third_owner"
    try:
        science.create_session(sid, source="tui", session_key="agent:main:local:dm:" + sid)
        science.append_message(sid, "user", "donor message")
        # Fence-correct seeding: the third-party duplicates cannot be created through the
        # guarded API (that is the fix working) — seed the rows directly.
        _seed_row(default, sid, key="agent:main:local:dm:" + sid, profile="default")
        _seed_row(other, sid, key="agent:main:local:dm:" + sid, profile="other")
        result_of_refusal = None
        try:
            result_of_refusal = default.adopt_session_lineage_from(science, sid)
        except SessionLineageAmbiguousError:
            pass  # fail-closed at guard entry: the required outcome for a third live owner
        assert result_of_refusal is None or result_of_refusal.get("adopted") is False, \
            "adoption must refuse a third live owner, never complete"
        donor_row = science.get_session(sid)
        assert donor_row is not None and donor_row.get("end_reason") is None, \
            "refused adoption must not retire the donor"
        assert science.get_session(sid) is not None and \
            len(default.get_messages(sid)) == 0, "no target mutation, no donor transcript loss"
    finally:
        default.close()
        science.close()
        other.close()


def test_aliased_source_and_current_is_not_a_transfer_pair(home):
    default, science = _open(home, default=True), _open(home)
    sid = "20261105_alias"
    try:
        science.create_session(sid, source="tui", session_key="agent:main:local:dm:" + sid)
        # current == transfer_from (same file): the alias must fall through to ordinary
        # selection, which sees one live row owned elsewhere → refused, never a "pair".
        with pytest.raises(SessionIdOwnedElsewhereError):
            with session_lineage_guard(default.db_path, (sid,), transfer_from=default.db_path):
                pass
    finally:
        default.close()
        science.close()


def test_idempotent_all_local_re_import_is_skipped(home):
    default = _open(home, default=True)
    sid = "20261105_reimport"
    try:
        default.create_session(sid, source="tui", session_key="agent:main:local:dm:" + sid)
        default.append_message(sid, "user", "q1")
        payload = default.export_session(sid)
        result = default.import_sessions([payload])
        assert result.get("imported", 0) == 0, "sole-owner re-import must skip, not duplicate"
        assert len(default.get_messages(sid)) == 1
        assert default.get_session(sid) is not None
    finally:
        default.close()


# ── physical-store inventory boundaries (finding 3 follow-up) ──────────────────

def _repair_plan(root, stores):
    """Cross-version interface: the fixture's HERMES_HOME (ambient managed root) drives
    inventory, so the SAME call runs on the base tree and the candidate (a `root=`
    keyword would TypeError on the base and fake a RED)."""
    from hermes_cli.sessions_repair_profiles import RepairPlan
    return RepairPlan(stores)


def test_explicit_root_api_scans_that_root(home, tmp_path):
    """Candidate-only API surface: an explicit root is honored when given (no TypeError,
    same discovered stores as the ambient home for the same tree)."""
    from hermes_cli.sessions_repair_profiles import RepairPlan
    profile_db = home / "profiles" / "science" / "state.db"
    profile_db.parent.mkdir(parents=True)
    SessionDB(profile_db).close()
    from hermes_cli.sessions_repair_profiles import Store
    plan = RepairPlan([Store("science", profile_db.parent, routing=True)], root=home)
    assert "science" in {s.profile for s in plan.physical_stores}


def test_empty_home_inventories_without_creating_a_database(home):
    plan = _repair_plan(home, [])
    assert plan.physical_stores == []
    assert not (home / "state.db").exists(), "read-only inventory must never create a database"


def test_named_profile_only_home_inventories_profile(home):
    profile_db = home / "profiles" / "science" / "state.db"
    profile_db.parent.mkdir(parents=True)
    SessionDB(profile_db).close()
    from hermes_cli.sessions_repair_profiles import Store
    plan = _repair_plan(home, [Store("science", profile_db.parent, routing=True)])
    names = {store.profile for store in plan.physical_stores}
    assert "science" in names
    assert not (home / "state.db").exists(), "a missing default db stays missing after inventory"


def test_unsafe_existing_store_is_a_fatal_inventory_error(home):
    import os as _os
    from hermes_state_lineage import SessionLineageInventoryError
    real = home / "profiles" / "science" / "state.db"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"x")
    link = home / "profiles" / "evil" / "state.db"
    link.parent.mkdir(parents=True)
    _os.symlink(real, link)  # symlinked store file = unsafe, deterministic on every umask
    with pytest.raises(SessionLineageInventoryError):
        _repair_plan(home, [])
