"""Tests for MemoryStore.backup_before — the reversibility safety net.

Destructive ops (rename_entity, merge_entities) snapshot the live DB to
``<db_parent>/backups/{op}-{timestamp}.db`` before mutating. Cleanup keeps
the 10 most recent files.

Invariants:
  - backup_before writes a usable SQLite copy of the source DB
  - rotation prunes to the configured keep count, oldest first
  - rename_entity / merge_entities trigger a backup automatically
  - the auto-backup reflects pre-op state (snapshot is taken BEFORE mutation)
  - tests are hermetic: backups land next to the temp DB, not in ~/.hermes/
"""

from __future__ import annotations

import sqlite3

import pytest

from plugins.memory.holographic.store import MemoryStore


DIM = 2048


@pytest.fixture
def store_in_tmpdir(tmp_path):
    """A MemoryStore whose backups land in tmp_path/backups (hermetic)."""
    db_path = tmp_path / "memory_store.db"
    store = MemoryStore(db_path=str(db_path), hrr_dim=DIM, default_trust=0.85)
    yield store, tmp_path
    store.close()


def _entity_id_for(store: MemoryStore, name: str) -> int:
    row = store._conn.execute(
        "SELECT entity_id FROM entities WHERE LOWER(name) = LOWER(?)", (name,)
    ).fetchone()
    assert row is not None, f"entity {name!r} not found"
    return int(row["entity_id"])


def test_backup_before_creates_file(store_in_tmpdir):
    store, tmp_path = store_in_tmpdir
    store.add_fact("Apollo Energy Resources owns trucks.", category="general")

    backup_path = store.backup_before("manual_test")

    assert backup_path.exists()
    assert backup_path.parent == tmp_path / "backups"
    assert backup_path.name.startswith("manual_test-")
    assert backup_path.name.endswith(".db")


def test_backup_before_copy_is_readable_and_complete(store_in_tmpdir):
    """The backup must be a valid SQLite DB containing the source's facts."""
    store, _ = store_in_tmpdir
    store.add_fact("Apollo runs LNG logistics.", category="general")
    store.add_fact("Walker Anderson is Co-Founder.", category="identity")

    backup_path = store.backup_before("readable_test")

    conn = sqlite3.connect(str(backup_path))
    try:
        rows = conn.execute("SELECT content FROM facts ORDER BY fact_id").fetchall()
    finally:
        conn.close()
    contents = {r[0] for r in rows}
    assert "Apollo runs LNG logistics." in contents
    assert "Walker Anderson is Co-Founder." in contents


def test_backup_rotation_keeps_only_n_most_recent(store_in_tmpdir):
    store, tmp_path = store_in_tmpdir
    store.add_fact("seed fact for rotation test.", category="general")

    # Create 12 backups; rotation should leave only the last 10.
    for _ in range(12):
        store.backup_before("rotate_test", keep=10)

    backups_dir = tmp_path / "backups"
    files = sorted(backups_dir.glob("rotate_test-*.db"))
    assert len(files) == 10


def test_backup_rotation_drops_oldest_first(store_in_tmpdir):
    store, tmp_path = store_in_tmpdir
    store.add_fact("seed fact.", category="general")

    paths = [store.backup_before("order_test", keep=3) for _ in range(5)]

    backups_dir = tmp_path / "backups"
    surviving = {p.name for p in backups_dir.glob("order_test-*.db")}
    # Last 3 survive, first 2 are pruned.
    assert {p.name for p in paths[-3:]} == surviving
    for p in paths[:2]:
        assert not p.exists()


def test_rename_entity_triggers_backup(store_in_tmpdir):
    store, tmp_path = store_in_tmpdir
    store.add_fact("Apollo Energy Resources is a multi-division business.",
                   category="identity")
    eid = _entity_id_for(store, "Apollo Energy Resources")

    store.rename_entity(eid, "Apollo Energy Group")

    backups_dir = tmp_path / "backups"
    backups = list(backups_dir.glob("rename_entity-*.db"))
    assert len(backups) == 1


def test_rename_backup_captures_pre_rename_state(store_in_tmpdir):
    """The auto-backup must reflect the entity's *original* name, not the new one."""
    store, tmp_path = store_in_tmpdir
    store.add_fact("Apollo Energy Resources owns trucks.", category="general")
    eid = _entity_id_for(store, "Apollo Energy Resources")

    store.rename_entity(eid, "Apollo Energy Group")

    backup = next((tmp_path / "backups").glob("rename_entity-*.db"))
    conn = sqlite3.connect(str(backup))
    try:
        row = conn.execute(
            "SELECT name FROM entities WHERE entity_id = ?", (eid,)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == "Apollo Energy Resources"


def test_merge_entities_triggers_backup(store_in_tmpdir):
    store, tmp_path = store_in_tmpdir
    store.add_fact("Apollo Energy Resources owns trucks.", category="general")
    store.add_fact("Apollo Energy Group runs LNG logistics.", category="general")
    src = _entity_id_for(store, "Apollo Energy Resources")
    tgt = _entity_id_for(store, "Apollo Energy Group")

    store.merge_entities(src, tgt)

    backups_dir = tmp_path / "backups"
    backups = list(backups_dir.glob("merge_entities-*.db"))
    assert len(backups) == 1


def test_rename_failure_leaves_backup_in_place(store_in_tmpdir, monkeypatch):
    """If the rename rolls back, the backup still survives — that's the whole point.

    A failed destructive op shouldn't also cost the user their snapshot;
    the backup is what makes the failure recoverable.
    """
    from plugins.memory.holographic import holographic as hrr

    store, tmp_path = store_in_tmpdir
    store.add_fact("Apollo Energy Resources is a multi-division business.",
                   category="identity")
    store.add_fact("Apollo Energy Resources runs LNG logistics.", category="general")
    eid = _entity_id_for(store, "Apollo Energy Resources")

    call_count = {"n": 0}
    real_encode = hrr.encode_fact

    def flaky_encode(content, entities, dim):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("simulated encode failure")
        return real_encode(content, entities, dim)

    monkeypatch.setattr(
        "plugins.memory.holographic.store.hrr.encode_fact",
        flaky_encode,
    )

    with pytest.raises(RuntimeError, match="simulated encode failure"):
        store.rename_entity(eid, "Apollo Energy Group")

    backups = list((tmp_path / "backups").glob("rename_entity-*.db"))
    assert len(backups) == 1


def test_invalid_inputs_skip_backup(store_in_tmpdir):
    """Validation errors fire before any DB work — no backup should be written."""
    store, tmp_path = store_in_tmpdir
    store.add_fact("Apollo Energy Resources owns trucks.", category="general")
    eid = _entity_id_for(store, "Apollo Energy Resources")

    with pytest.raises(ValueError):
        store.rename_entity(eid, "   ")
    with pytest.raises(KeyError):
        store.rename_entity(99999, "Anything")
    with pytest.raises(KeyError):
        store.merge_entities(99999, eid)

    backups_dir = tmp_path / "backups"
    if backups_dir.exists():
        assert list(backups_dir.glob("*.db")) == []


def test_backup_before_rejects_empty_operation_name(store_in_tmpdir):
    """Empty op name is a caller bug — fail loud, don't silently bucket as 'unknown'."""
    store, _ = store_in_tmpdir
    store.add_fact("seed fact.", category="general")
    with pytest.raises(ValueError):
        store.backup_before("")
    with pytest.raises(ValueError):
        store.backup_before("   ")


def test_list_backups_empty_when_none_exist(store_in_tmpdir):
    store, _ = store_in_tmpdir
    assert store.list_backups() == []
    assert store.list_backups("rename_entity") == []


def test_list_backups_returns_newest_first(store_in_tmpdir):
    store, _ = store_in_tmpdir
    store.add_fact("seed fact.", category="general")

    paths = [store.backup_before("list_test") for _ in range(3)]

    listed = store.list_backups("list_test")
    assert listed == list(reversed(paths))


def test_list_backups_filters_by_operation(store_in_tmpdir):
    store, _ = store_in_tmpdir
    store.add_fact("Apollo Energy Resources owns trucks.", category="general")
    store.add_fact("Apollo Energy Group runs LNG.", category="general")
    eid = _entity_id_for(store, "Apollo Energy Resources")
    tgt = _entity_id_for(store, "Apollo Energy Group")

    store.rename_entity(eid, "Apollo Energy Group Inc.")
    # rename above merged the source into target via aliases, but eid is still
    # a distinct row; trigger merge with two real entities for the second snapshot.
    store.add_fact("Walker Anderson manages logistics.", category="general")
    walker = _entity_id_for(store, "Walker Anderson")
    store.merge_entities(walker, tgt)

    all_backups = store.list_backups()
    rename_only = store.list_backups("rename_entity")
    merge_only = store.list_backups("merge_entities")

    assert len(rename_only) == 1
    assert len(merge_only) == 1
    assert len(all_backups) == 2
    assert all(p.name.startswith("rename_entity-") for p in rename_only)
    assert all(p.name.startswith("merge_entities-") for p in merge_only)


def test_backup_path_traversal_is_neutralized(store_in_tmpdir):
    """Operation names with path separators must not escape the backups dir."""
    store, tmp_path = store_in_tmpdir
    store.add_fact("seed fact.", category="general")

    backup = store.backup_before("../evil")

    assert backup.parent == tmp_path / "backups"
    assert ".." not in backup.name or backup.name.startswith("..")  # name is escaped
    # Concretely: no file landed outside the backups dir.
    assert backup.exists()
    assert (tmp_path / "backups").exists()
    # No sibling 'evil-*.db' file landed in tmp_path itself.
    assert not list(tmp_path.glob("evil-*.db"))
