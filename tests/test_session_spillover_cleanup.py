"""Session-tied spillover unlinking (#126351): an archive referenced by a retained session
stays available exactly as long as that session does.

Two halves under test:
- the age sweep (``cleanup_spillover_cache``) never deletes a referenced archive, and fails
  closed (keeps files) when the reference set is unreadable;
- ``delete_session`` / ``delete_sessions`` / ``prune_sessions`` capture the doomed messages'
  pointers inside the delete transaction and unlink the archives afterwards — but only once no
  SURVIVING session references them (branch/compression siblings may share the pointer)."""

import os
import time

import pytest

from hermes_state import SessionDB
from tools.tool_result_storage import (
    _build_persisted_message,
    cleanup_spillover_cache,
    referenced_spillover_names,
)

THREE_DAYS_S = 3 * 86400


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    """One scratch home per test: state.db at the root, spillover under cache/ — the same
    layout ``referenced_spillover_names`` derives from the spill dir."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.tool_result_storage as trs
    monkeypatch.setattr(trs, "_spillover_pruned_homes", set())
    return tmp_path


def _spill(home, name, age_hours=0.0):
    spill_dir = home / "cache" / "spillover"
    spill_dir.mkdir(parents=True, exist_ok=True)
    path = spill_dir / name
    path.write_text("archived tool result", encoding="utf-8")
    if age_hours:
        stale = time.time() - age_hours * 3600
        os.utime(path, (stale, stale))
    return path


def _pointer(home, name, size=150_000):
    return _build_persisted_message(
        preview="preview", has_more=True, original_size=size,
        file_path=str(home / "cache" / "spillover" / name))


def _aged_session(db, sid, name, ended=True):
    db.create_session(sid, "cli")
    db.append_message(sid, "tool", _pointer(db.db_path.parent, name), timestamp=time.time() - THREE_DAYS_S)
    if ended:
        db.end_session(sid, "completed")
    db._write_sql(
        "UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = ?",
        (time.time() - THREE_DAYS_S, time.time() - THREE_DAYS_S, sid))


def test_retained_sessions_spill_survives_the_age_sweep(_home):
    """The blocking scenario from the review: ``retention_days: 2`` (48h window), the spill is
    49h old, yet the session that references it was active an hour ago and is retained — the
    sweep must not strand the transcript pointer."""
    db = SessionDB(db_path=_home / "state.db")
    try:
        db.create_session("s_recent", "cli")
        db.append_message("s_recent", "tool", _pointer(_home, "recent.txt"))
        db.end_session("s_recent", "completed")  # ended, unpinned, active 1h ago: not prunable
        db._write_sql("UPDATE sessions SET last_activity_at = ? WHERE id = ?",
                      (time.time() - 3600, "s_recent"))
        aged = _spill(_home, "recent.txt", age_hours=49)

        assert db.prune_sessions(older_than_days=2) == 0  # session retained...
        assert cleanup_spillover_cache(max_age_hours=48) == 0  # ...so the sweep keeps the archive
        assert aged.exists()
    finally:
        db.close()


def test_structured_and_backslashed_pointers_are_owned(_home):
    """A pointer inside multimodal content (stored behind the NUL sentinel, invisible to LIKE)
    and a backslashed Windows host path both count as references, and both leave with their
    session."""
    db = SessionDB(db_path=_home / "state.db")
    try:
        db.create_session("s", "cli")
        db.append_message("s", "tool", [{"type": "text", "text": _pointer(_home, "parts.txt")},
                                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}])
        db.append_message("s", "tool", "Full output saved to: C:\\h\\cache\\spillover\\win.txt")
        files = [_spill(_home, "parts.txt", age_hours=200 * 24), _spill(_home, "win.txt", age_hours=200 * 24)]

        assert cleanup_spillover_cache(max_age_hours=24) == 0
        assert all(f.exists() for f in files)
        db.delete_session("s")
        assert not any(f.exists() for f in files)
    finally:
        db.close()


def test_prune_sessions_unlinks_the_spill_with_its_session(_home):
    db = SessionDB(db_path=_home / "state.db")
    try:
        _aged_session(db, "s_old", "dead.txt")
        aged = _spill(_home, "dead.txt", age_hours=72)

        assert db.prune_sessions(older_than_days=2) == 1
        assert not aged.exists()
    finally:
        db.close()


def test_delete_session_unlinks_the_spill(_home):
    db = SessionDB(db_path=_home / "state.db")
    try:
        _aged_session(db, "s_del", "del.txt")
        spill = _spill(_home, "del.txt")

        assert db.delete_session("s_del") is True
        assert not spill.exists()
    finally:
        db.close()


def test_delete_sessions_bulk_unlinks_the_spill(_home):
    db = SessionDB(db_path=_home / "state.db")
    try:
        _aged_session(db, "s_a", "bulk_a.txt")
        _aged_session(db, "s_b", "bulk_b.txt")
        a = _spill(_home, "bulk_a.txt")
        b = _spill(_home, "bulk_b.txt")

        assert db.delete_sessions(["s_a", "s_b"]) == 2
        assert not a.exists() and not b.exists()
    finally:
        db.close()


def test_shared_pointer_survives_until_the_last_session_dies(_home):
    """Branch/compression siblings can share one archive: deleting one holder must not
    strand the other's pointer."""
    db = SessionDB(db_path=_home / "state.db")
    try:
        _aged_session(db, "s_first", "shared.txt")
        _aged_session(db, "s_second", "shared.txt")
        shared = _spill(_home, "shared.txt")

        db.delete_session("s_first")
        assert shared.exists()  # s_second still references it

        db.delete_session("s_second")
        assert not shared.exists()
    finally:
        db.close()


def test_canonical_delete_retires_the_spill_for_collection(_home):
    """The web/gateway DELETE goes through ``mutate_runtime_session(operation='delete')``, not
    ``delete_session``: the deletion fence records the pointer, the owner's post-delete
    collection unlinks it."""
    import hermes_state_runtime as rt
    from hermes_state_media import collect_retired_media
    db = SessionDB(db_path=_home / "state.db")
    try:
        _aged_session(db, "s_web", "web.txt")
        spill = _spill(_home, "web.txt")
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        rt.mutate_runtime_session(db, epoch=epoch, principal_id="human", session_id="s_web", request_id="del",
                                  expected_revision=0, expected_generation=0, operation="delete", payload={})
        assert spill.exists()  # the transaction only records the candidate
        collect_retired_media(db)  # gateway/session_mutations.py runs this after every delete
        assert not spill.exists()
    finally:
        db.close()


def test_unreferenced_spill_still_sweeps_by_age(_home):
    db = SessionDB(db_path=_home / "state.db")
    try:
        db.create_session("s_alive", "cli")  # a live db with no spillover references
        orphan = _spill(_home, "orphan.txt", age_hours=49)

        assert cleanup_spillover_cache(max_age_hours=48) == 1
        assert not orphan.exists()
    finally:
        db.close()


def test_unreadable_db_fails_closed_keeps_files(_home):
    """A damaged state.db must pause the sweep, not guess: every aged file stays."""
    (_home / "state.db").write_bytes(b"this is not a sqlite database" * 100)
    aged = _spill(_home, "unknowable.txt", age_hours=49)

    assert referenced_spillover_names(_home / "cache" / "spillover") is None
    assert cleanup_spillover_cache(max_age_hours=48) == 0
    assert aged.exists()


def test_nonfinite_retention_falls_back_and_still_sweeps(_home):
    """YAML ``.inf`` / ``.nan`` used to raise out of ``_spillover_retention_hours`` after the
    once-per-home flag was set — every later sweep silently skipped. Fallback cleanup must
    actually remove a stale file."""
    import hermes_yaml as yaml  # the repo's YAML seam; PyYAML is not a pinned test dep

    with (_home / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"sessions": {"retention_days": float("inf")}}, fh)
    aged = _spill(_home, "pre_inf.txt", age_hours=49)

    assert cleanup_spillover_cache() == 1  # .inf → 24h fallback → the 49h orphan goes
    assert not aged.exists()
