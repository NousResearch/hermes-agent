"""Chain-aware deletion invariants for the dashboard delete endpoints (#57543).

``list_sessions_rich`` surfaces one row per logical conversation: compression roots
are projected forward and the row carries the chain *tip's* id. Deleting only that
physical row leaves the root behind, and the next list re-projects it onto the
previous link — the user sees the "deleted" conversation reappear ("deletes some
of them, but not all"). Invariants this module locks in:

1. Deleting a tip removes the whole chain (backward walk to root).
2. Deleting a root removes every continuation — including stale sibling
   continuations, which orphaning would otherwise promote into visible
   top-level rows.
3. Branch / reset / delegate children of chain members keep the plain
   contract (orphaned, not deleted); delegate children are cascade-deleted.
4. The returned count reflects the *selected* rows, not the expanded chain
   links, so the dashboard toast matches what the user picked.
5. The default stays chain-blind — existing callers (CLI ``/exit --delete``,
   ACP, gateway API) are behaviour-identical.
6. Deletion is EXACT: only the selected roots' own compression chains die;
   every other lineage and standalone row survives untouched.
"""

from hermes_state import SessionDB


def _make_chain(db, root="root", mid="mid", tip="tip"):
    """root --compression--> mid --compression--> tip (live)."""
    db.create_session(root, source="tui")
    db.append_message(root, role="user", content="the start")
    db.end_session(root, "compression")
    db.create_session(mid, source="tui", parent_session_id=root)
    db.append_message(mid, role="user", content="continued once")
    db.end_session(mid, "compression")
    db.create_session(tip, source="tui", parent_session_id=mid)
    db.append_message(tip, role="user", content="continued twice")


def _seeded_db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def test_deleting_tip_removes_whole_chain(tmp_path):
    """The exact #57543 repro: the dashboard row carries the tip id; bulk-deleting
    it must not leave the root to resurface on the next list reload."""
    db = _seeded_db(tmp_path)
    _make_chain(db)
    assert [r["id"] for r in db.list_sessions_rich(limit=20)] == ["tip"]

    assert db.delete_sessions(["tip"], include_compression_chain=True) == 1
    assert db.get_session("root") is None
    assert db.get_session("mid") is None
    assert db.get_session("tip") is None
    assert db.list_sessions_rich(limit=20) == []
    db.close()


def test_bulk_delete_removes_whole_chain_of_each_selected_root_and_nothing_else(tmp_path):
    """EXACTNESS (#57543 sweeper risk): two selected roots with their own chains —
    both chains vanish whole, and every other lineage / standalone row survives."""
    db = _seeded_db(tmp_path)
    # chain A (selected)
    _make_chain(db, root="a_root", mid="a_mid", tip="a_tip")
    # chain B (selected)
    db.create_session("b_root", source="tui")
    db.append_message("b_root", role="user", content="b start")
    db.end_session("b_root", "compression")
    db.create_session("b_tip", source="tui", parent_session_id="b_root")
    # chain C — NOT selected, must survive whole
    db.create_session("c_root", source="tui")
    db.end_session("c_root", "compression")
    db.create_session("c_tip", source="tui", parent_session_id="c_root")
    # standalone rows — must survive
    db.create_session("solo1", source="tui")
    db.create_session("solo2", source="tui")

    deleted = db.delete_sessions(["a_tip", "b_tip"], include_compression_chain=True)

    assert deleted == 2  # counts selected rows, not the 5 physical links
    for gone in ("a_root", "a_mid", "a_tip", "b_root", "b_tip"):
        assert db.get_session(gone) is None, gone
    for kept in ("c_root", "c_tip", "solo1", "solo2"):
        assert db.get_session(kept) is not None, kept
    # chain C stays a single projected row; standalones still listed
    assert {r["id"] for r in db.list_sessions_rich(limit=20)} == {"c_tip", "solo2", "solo1"}
    db.close()


def test_deleting_root_removes_continuations_and_stale_siblings(tmp_path):
    """Forward direction: every compression child dies with the chain — including a
    stale sibling continuation, which orphaning would otherwise promote into a
    visible top-level row."""
    db = _seeded_db(tmp_path)
    db.create_session("root", source="tui")
    db.end_session("root", "compression")
    db.create_session("tip", source="tui", parent_session_id="root")
    db.create_session("stale", source="tui", parent_session_id="root")
    db.end_session("stale", "ws_orphan_reap")

    assert db.delete_sessions(["root"], include_compression_chain=True) == 1
    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    assert db.get_session("stale") is None
    db.close()


def test_chain_delete_orphans_branches_and_cascades_delegates(tmp_path):
    """Branch / reset / delegate children of chain members keep the plain per-row
    contract: a branch (or reset fork) is its own conversation and survives
    (orphaned); a delegate subagent run is cascade-deleted."""
    db = _seeded_db(tmp_path)
    _make_chain(db)
    db.create_session("branch", source="tui", parent_session_id="root",
                      model_config={"_branched_from": "root"})
    db.create_session("reset", source="tui", parent_session_id="mid",
                      model_config={"_reset_from": "mid"})
    db.create_session("delegate", source="tui", parent_session_id="tip",
                      model_config={"_delegate_from": "tip"})

    assert db.delete_sessions(["tip"], include_compression_chain=True) == 1
    branch = db.get_session("branch")
    assert branch is not None and branch["parent_session_id"] is None
    reset = db.get_session("reset")
    assert reset is not None and reset["parent_session_id"] is None
    assert db.get_session("delegate") is None
    db.close()


def test_count_reflects_selected_rows_not_chain_links(tmp_path):
    """Selecting 2 rows deletes 5 physical sessions but reports 2 — the toast
    matches the user's selection."""
    db = _seeded_db(tmp_path)
    _make_chain(db)
    db.create_session("plain", source="tui")

    assert db.delete_sessions(["tip", "plain"], include_compression_chain=True) == 2
    db.close()


def test_default_remains_chain_blind(tmp_path):
    """Without the flag, only the listed row is deleted — pins the historic
    contract for non-dashboard callers."""
    db = _seeded_db(tmp_path)
    _make_chain(db)

    assert db.delete_sessions(["tip"]) == 1
    assert db.get_session("tip") is None
    assert db.get_session("mid") is not None
    assert db.get_session("root") is not None
    db.close()


def test_single_delete_session_chain_variant(tmp_path):
    """``delete_session(..., include_compression_chain=True)`` — the single-row
    dashboard endpoint has the same reappearing-conversation bug, so it routes
    through the same expansion."""
    db = _seeded_db(tmp_path)
    _make_chain(db)
    assert db.delete_session("tip", include_compression_chain=True) is True
    assert db.get_session("root") is None
    assert db.get_session("mid") is None
    assert db.get_session("tip") is None
    # Missing sessions still report False.
    assert db.delete_session("ghost", include_compression_chain=True) is False
    db.close()


def test_chain_transcript_files_cleaned(tmp_path):
    """On-disk transcripts of every expanded chain member are swept, not just the
    selected row's."""
    db = _seeded_db(tmp_path)
    _make_chain(db)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "root.jsonl").write_text("")
    (sessions_dir / "tip.json").write_text("{}")
    # Another session's artifacts must survive.
    (sessions_dir / "session_other.json").write_text("{}")

    assert db.delete_sessions(["tip"], sessions_dir=sessions_dir,
                              include_compression_chain=True) == 1
    assert not (sessions_dir / "root.jsonl").exists()
    assert not (sessions_dir / "tip.json").exists()
    assert (sessions_dir / "session_other.json").exists()
    db.close()


def test_chain_delete_skips_guarded_lineage(tmp_path):
    """A live turn lease anywhere in the chain keeps the selected root: a torn
    delete would re-surface the conversation, worse than skipping it."""
    import os

    db = _seeded_db(tmp_path)
    _make_chain(db)
    holder = f"pid={os.getpid()}:turn=1"
    assert db.try_acquire_session_turn_lease("mid", holder, ttl_seconds=300.0) is True

    skipped: list[str] = []
    assert db.delete_sessions(["tip"], exclude_active_write_guards=True,
                              skipped_ids=skipped, include_compression_chain=True) == 0
    assert skipped == ["tip"]
    # whole chain survives
    for kept in ("root", "mid", "tip"):
        assert db.get_session(kept) is not None, kept
    db.close()
