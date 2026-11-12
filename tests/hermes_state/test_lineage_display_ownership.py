"""Resolved reaction writes and retired-copy display reads for #149132."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_lineage import SessionIdOwnedElsewhereError, SessionLineageInventoryError


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile = tmp_path / "profiles" / "science"
    profile.mkdir(parents=True)
    local, sibling = SessionDB(tmp_path / "state.db"), SessionDB(profile / "state.db")
    try:
        yield local, sibling
    finally:
        local.close()
        sibling.close()


def _rows(db):
    return {table: [tuple(row) for row in db._conn.execute(f"SELECT * FROM {table} ORDER BY id")]
            for table in ("sessions", "messages")}


def _chain(db):
    db.create_session("ancestor", "cli")
    target = db.append_message("ancestor", "assistant", "Visible ancestor answer")
    db.create_session("tip", "cli", parent_session_id="ancestor")
    db._conn.execute("UPDATE sessions SET ended_at=1, end_reason='compression' WHERE id='ancestor'")
    db._conn.commit()
    assert db._resume_lineage_ids("tip") == ["ancestor", "tip"]
    return target


@pytest.mark.parametrize("operation", ["set", "clear", "take"])
def test_ancestor_reaction_write_requires_its_actual_owner(stores, operation):
    local, sibling = stores
    target = _chain(local)
    local.set_message_reaction("tip", target, "yes")
    sibling._conn.execute("INSERT INTO sessions (id,source,started_at,profile_name) VALUES ('ancestor','cli',1,'science')")
    sibling._conn.commit()
    before = _rows(local), _rows(sibling)
    with pytest.raises(SessionIdOwnedElsewhereError):
        if operation == "take":
            local.take_unseen_reactions("tip")
        else:
            local.set_message_reaction("tip", target, None if operation == "clear" else "new")
    assert (_rows(local), _rows(sibling)) == before


def test_uncontested_ancestor_reaction_preserves_once_only_delivery(stores):
    local, _ = stores
    target = _chain(local)
    assert local.set_message_reaction("tip", target, "yes")[0]["emoji"] == "yes"
    assert [row["row_id"] for row in local.take_unseen_reactions("tip")] == [target]
    assert local.take_unseen_reactions("tip") == []
    assert local.set_message_reaction("tip", target, None) == []


@pytest.mark.parametrize("operation", ["set", "take"])
def test_reaction_discovery_expands_before_any_write(stores, monkeypatch, operation):
    local, sibling = stores
    target = _chain(local)
    local.set_message_reaction("tip", target, "yes")
    sibling._conn.execute("INSERT INTO sessions (id,source,started_at,profile_name) VALUES ('ancestor','cli',1,'science')")
    sibling._conn.commit()
    real_discover = local._resume_lineage_ids
    calls = []
    def discover(sid):
        calls.append(sid)
        return ["tip"] if len(calls) == 1 else real_discover(sid)
    monkeypatch.setattr(local, "_resume_lineage_ids", discover)
    before = _rows(local), _rows(sibling)
    with pytest.raises(SessionIdOwnedElsewhereError):
        local.set_message_reaction("tip", target, "new") if operation == "set" else local.take_unseen_reactions("tip")
    assert len(calls) > 1
    assert (_rows(local), _rows(sibling)) == before


def test_retired_legacy_donor_reads_without_backfill_writes(stores):
    donor, target = stores
    donor.create_session("legacy", "cli")
    donor.append_message("legacy", "user", "Preserve legacy history")
    target.adopt_session_lineage_from(donor, "legacy")
    donor._conn.execute("UPDATE messages SET display_order=NULL WHERE session_id='legacy'")
    donor._conn.commit()
    before = _rows(donor), _rows(target)
    assert [row["content"] for row in donor.get_messages("legacy", include_compacted=True)] == ["Preserve legacy history"]
    assert (_rows(donor), _rows(target)) == before
    assert donor._conn.execute("SELECT display_order FROM messages WHERE session_id='legacy'").fetchone()[0] is None


def test_live_owner_backfills_legacy_display_order(stores):
    local, _ = stores
    local.create_session("legacy", "cli")
    local.append_message("legacy", "user", "Legacy content")
    local._conn.execute("UPDATE messages SET display_order=NULL WHERE session_id='legacy'")
    local._conn.commit()
    assert [row["content"] for row in local.get_messages("legacy", include_compacted=True)] == ["Legacy content"]
    assert local._conn.execute("SELECT display_order FROM messages WHERE session_id='legacy'").fetchone()[0] is not None


@pytest.mark.parametrize("error", [sqlite3.OperationalError("injected IO"), SessionLineageInventoryError("injected inventory")])
def test_display_backfill_does_not_hide_unrelated_failures(stores, monkeypatch, error):
    import hermes_state_messages as messages
    import hermes_state_compaction_display as display
    local, _ = stores
    local.create_session("legacy", "cli")
    local.append_message("legacy", "user", "Legacy content")
    local._conn.execute("UPDATE messages SET display_order=NULL WHERE session_id='legacy'")
    local._conn.commit()
    @contextmanager
    def broken(*args, **kwargs):
        raise error
        yield
    monkeypatch.setattr(messages, "session_lineage_guard", broken)
    monkeypatch.setattr(display, "session_lineage_guard", broken, raising=False)
    before = _rows(local)
    with pytest.raises(type(error), match="injected"):
        local.get_messages("legacy", include_compacted=True)
    assert _rows(local) == before


@pytest.mark.parametrize("marker", ["_branched_from", "_reset_from"])
def test_reaction_never_crosses_an_explicit_fork_boundary(stores, marker):
    import json
    local, sibling = stores
    target = _chain(local)
    local._conn.execute("UPDATE sessions SET model_config=? WHERE id='tip'", (json.dumps({marker: "ancestor"}),))
    local._conn.commit()
    sibling._conn.execute("INSERT INTO sessions (id,source,started_at,profile_name) VALUES ('ancestor','cli',1,'science')")
    sibling._conn.commit()
    before = _rows(local), _rows(sibling)
    assert local.set_message_reaction("tip", target, "yes") is None
    assert local.take_unseen_reactions("tip") == []
    assert (_rows(local), _rows(sibling)) == before


def test_real_display_backfill_abort_remains_visible_and_atomic(stores):
    local, _ = stores
    local.create_session("legacy", "cli")
    local.append_message("legacy", "user", "Legacy content")
    local._conn.execute("UPDATE messages SET display_order=NULL WHERE session_id='legacy'")
    local._conn.execute("CREATE TRIGGER reject_display BEFORE UPDATE OF display_order ON messages "
                        "BEGIN SELECT RAISE(ABORT, 'injected display abort'); END")
    local._conn.commit()
    before = _rows(local)
    with pytest.raises(sqlite3.IntegrityError, match="injected display abort"):
        local.get_messages("legacy", include_compacted=True)
    assert _rows(local) == before


def test_unstable_reaction_discovery_fails_closed(stores, monkeypatch):
    local, _ = stores
    target = _chain(local)
    seen = []
    def growing(sid):
        seen.append(f"new-{len(seen)}")
        return ["ancestor", "tip", *seen]
    monkeypatch.setattr(local, "_resume_lineage_ids", growing)
    before = _rows(local)
    with pytest.raises(SessionLineageInventoryError, match="changed"):
        local.set_message_reaction("tip", target, "yes")
    assert len(seen) < 20
    assert _rows(local) == before
