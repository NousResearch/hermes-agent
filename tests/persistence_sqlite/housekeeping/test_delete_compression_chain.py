"""Chain-aware deletion invariants of the conversation store's ``delete_sessions`` / ``delete_session`` on a SQLite backend store, behind the dashboard delete endpoints (#57543).

``list_sessions_rich`` surfaces one row per logical conversation: compression roots
are projected forward and the row carries the chain *tip's* id. Deleting only that
physical row leaves the root behind, and the next list re-projects it onto the
previous link: the user sees the "deleted" conversation reappear ("deletes some
of them, but not all"). Invariants this module locks in:

1. Deleting a tip removes the whole chain (backward walk to root).
2. Deleting a root removes every continuation, including stale sibling
   continuations, which orphaning would otherwise promote into visible
   top-level rows.
3. Branch / reset / delegate children of chain members keep the plain
   contract (orphaned, not deleted); delegate children are cascade-deleted.
4. The returned count reflects the *selected* rows, not the expanded chain
   links, so the dashboard toast matches what the user picked.
5. The default stays chain-blind: existing callers (CLI ``/exit --delete``,
   ACP, gateway API) are behaviour-identical.
6. Deletion is EXACT: only the selected roots' own compression chains die;
   every other lineage and standalone row survives untouched.
"""

import pytest

from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    create_session, delete_session, delete_sessions, end_session, get_session, list_sessions_rich,
)
from session_retention import after_sessions_removed
from tests.contracts._scopes import MEDIA_ROOT


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def _make_chain(db, root="root", mid="mid", tip="tip"):
    """root --compression--> mid --compression--> tip (live)."""
    create_session(db, root, source="tui")
    append_message(db, root, role="user", content="the start")
    end_session(db, root, "compression")
    create_session(db, mid, source="tui", parent_session_id=root)
    append_message(db, mid, role="user", content="continued once")
    end_session(db, mid, "compression")
    create_session(db, tip, source="tui", parent_session_id=mid)
    append_message(db, tip, role="user", content="continued twice")


def test_deleting_tip_removes_whole_chain(db):
    """The exact #57543 repro: the dashboard row carries the tip id; bulk-deleting
    it must not leave the root to resurface on the next list reload."""
    _make_chain(db)
    assert [r["id"] for r in list_sessions_rich(db, limit=20)] == ["tip"]

    assert delete_sessions(db, ["tip"], include_compression_chain=True, media_root=MEDIA_ROOT)[0] == 1
    assert get_session(db, "root") is None
    assert get_session(db, "mid") is None
    assert get_session(db, "tip") is None
    assert list_sessions_rich(db, limit=20) == []


def test_bulk_delete_removes_whole_chain_of_each_selected_root_and_nothing_else(db):
    """EXACTNESS (#57543 sweeper risk): two selected roots with their own chains;
    both chains vanish whole, and every other lineage / standalone row survives."""
    # chain A (selected)
    _make_chain(db, root="a_root", mid="a_mid", tip="a_tip")
    # chain B (selected)
    create_session(db, "b_root", source="tui")
    append_message(db, "b_root", role="user", content="b start")
    end_session(db, "b_root", "compression")
    create_session(db, "b_tip", source="tui", parent_session_id="b_root")
    # chain C, NOT selected, must survive whole
    create_session(db, "c_root", source="tui")
    end_session(db, "c_root", "compression")
    create_session(db, "c_tip", source="tui", parent_session_id="c_root")
    # standalone rows must survive
    create_session(db, "solo1", source="tui")
    create_session(db, "solo2", source="tui")

    deleted = delete_sessions(db, ["a_tip", "b_tip"], include_compression_chain=True, media_root=MEDIA_ROOT)[0]

    assert deleted == 2  # counts selected rows, not the 5 physical links
    for gone in ("a_root", "a_mid", "a_tip", "b_root", "b_tip"):
        assert get_session(db, gone) is None, gone
    for kept in ("c_root", "c_tip", "solo1", "solo2"):
        assert get_session(db, kept) is not None, kept
    # chain C stays a single projected row; standalones still listed
    assert {r["id"] for r in list_sessions_rich(db, limit=20)} == {"c_tip", "solo2", "solo1"}


def test_deleting_root_removes_continuations_and_stale_siblings(db):
    """Forward direction: every compression child dies with the chain, including a
    stale sibling continuation, which orphaning would otherwise promote into a
    visible top-level row."""
    create_session(db, "root", source="tui")
    end_session(db, "root", "compression")
    create_session(db, "tip", source="tui", parent_session_id="root")
    create_session(db, "stale", source="tui", parent_session_id="root")
    end_session(db, "stale", "ws_orphan_reap")

    assert delete_sessions(db, ["root"], include_compression_chain=True, media_root=MEDIA_ROOT)[0] == 1
    assert get_session(db, "root") is None
    assert get_session(db, "tip") is None
    assert get_session(db, "stale") is None


def test_chain_delete_orphans_branches_and_cascades_delegates(db):
    """Branch / reset / delegate children of chain members keep the plain per-row
    contract: a branch (or reset fork) is its own conversation and survives
    (orphaned); a delegate subagent run is cascade-deleted."""
    _make_chain(db)
    create_session(db, "branch", source="tui", parent_session_id="root",
                   model_config={"_branched_from": "root"})
    create_session(db, "reset", source="tui", parent_session_id="mid",
                   model_config={"_reset_from": "mid"})
    create_session(db, "delegate", source="tui", parent_session_id="tip",
                   model_config={"_delegate_from": "tip"})

    assert delete_sessions(db, ["tip"], include_compression_chain=True, media_root=MEDIA_ROOT)[0] == 1
    branch = get_session(db, "branch")
    assert branch is not None and branch["parent_session_id"] is None
    reset = get_session(db, "reset")
    assert reset is not None and reset["parent_session_id"] is None
    assert get_session(db, "delegate") is None


def test_count_reflects_selected_rows_not_chain_links(db):
    """Selecting 2 rows deletes 5 physical sessions but reports 2: the toast
    matches the user's selection."""
    _make_chain(db)
    create_session(db, "plain", source="tui")

    assert delete_sessions(db, ["tip", "plain"], include_compression_chain=True, media_root=MEDIA_ROOT)[0] == 2


def test_default_remains_chain_blind(db):
    """Without the flag, only the listed row is deleted; pins the historic
    contract for non-dashboard callers."""
    _make_chain(db)

    assert delete_sessions(db, ["tip"], media_root=MEDIA_ROOT)[0] == 1
    assert get_session(db, "tip") is None
    assert get_session(db, "mid") is not None
    assert get_session(db, "root") is not None


def test_single_delete_session_chain_variant(db):
    """``delete_session(..., include_compression_chain=True)``: the single-row
    dashboard endpoint has the same reappearing-conversation bug, so it routes
    through the same expansion."""
    _make_chain(db)
    assert "tip" in delete_session(db, "tip", include_compression_chain=True, media_root=MEDIA_ROOT)
    assert get_session(db, "root") is None
    assert get_session(db, "mid") is None
    assert get_session(db, "tip") is None
    # Missing sessions still report nothing removed.
    assert delete_session(db, "ghost", include_compression_chain=True, media_root=MEDIA_ROOT) == ()


def test_chain_transcript_files_cleaned(db, store, tmp_path):
    """On-disk transcripts of every expanded chain member are swept, not just the
    selected row's: the removed ids go through ``session_retention.after_sessions_removed``."""
    _make_chain(db)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "root.jsonl").write_text("")
    (sessions_dir / "tip.json").write_text("{}")
    # Another session's artifacts must survive.
    (sessions_dir / "session_other.json").write_text("{}")

    deleted, removed = delete_sessions(db, ["tip"], sessions_dir=sessions_dir,
                                       include_compression_chain=True, media_root=MEDIA_ROOT)
    after_sessions_removed(store, db.db_path.parent, removed, reason="delete", sessions_dir=sessions_dir)
    assert deleted == 1
    assert not (sessions_dir / "root.jsonl").exists()
    assert not (sessions_dir / "tip.json").exists()
    assert (sessions_dir / "session_other.json").exists()


def test_chain_delete_skips_guarded_lineage(db, store):
    """A live turn lease anywhere in the chain keeps the selected root: a torn
    delete would re-surface the conversation, worse than skipping it."""
    import os

    _make_chain(db)
    holder = f"pid={os.getpid()}:turn=1"
    assert store.coordination.try_acquire_turn_lease(session_id="mid", holder=holder, ttl_seconds=300.0) is True

    skipped: list[str] = []
    assert delete_sessions(db, ["tip"], exclude_active_write_guards=True,
                           skipped_ids=skipped, include_compression_chain=True, media_root=MEDIA_ROOT)[0] == 0
    assert skipped == ["tip"]
    # whole chain survives
    for kept in ("root", "mid", "tip"):
        assert get_session(db, kept) is not None, kept
