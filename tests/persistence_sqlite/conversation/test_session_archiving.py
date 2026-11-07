import time

import pytest

import session_retention
from persistence.housekeeping.maintenance import PruneCriteria
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    create_session, get_session, list_sessions_rich, set_session_archived, set_session_hidden,
)


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def holder(store):
    return store.holder()


def _compression_pair(holder):
    base = time.time() - 100
    create_session(holder, "root", source="cli")
    create_session(holder, "tip", source="cli", parent_session_id="root")
    holder._write_sql(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    holder._write_sql(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )


def test_archiving_compression_tip_archives_projected_root(holder):
    _compression_pair(holder)

    assert set_session_archived(holder, "tip", True)[0] is True

    assert get_session(holder, "root")["archived"] == 1
    assert get_session(holder, "tip")["archived"] == 1
    assert [s["id"] for s in list_sessions_rich(holder, order_by_last_active=True)] == []
    assert [s["id"] for s in list_sessions_rich(holder, order_by_last_active=True, archived_only=True)] == ["tip"]


def test_unarchiving_compression_tip_unarchives_projected_root(holder):
    _compression_pair(holder)
    set_session_archived(holder, "tip", True)

    assert set_session_archived(holder, "tip", False)[0] is True

    assert get_session(holder, "root")["archived"] == 0
    assert get_session(holder, "tip")["archived"] == 0
    assert [s["id"] for s in list_sessions_rich(holder, order_by_last_active=True)] == ["tip"]


def test_archived_only_view_includes_hidden_archived_sessions(holder):
    """The archived-only view is the recovery surface: a session that is both
    archived and hidden (Bot Mode marks its sessions hidden) must appear
    there, otherwise it is unreachable from every UI list (#90946)."""
    create_session(holder, "plain", source="cli")
    create_session(holder, "both", source="cli")
    assert set_session_hidden(holder, "both", True) is True
    assert set_session_archived(holder, "both", True)[0] is True

    # Default list: hidden rows stay excluded (unchanged behaviour)...
    assert [s["id"] for s in list_sessions_rich(holder, order_by_last_active=True)] == ["plain"]
    # ...and the archived-only view must surface the archived+hidden row.
    assert [s["id"] for s in list_sessions_rich(holder, order_by_last_active=True, archived_only=True)] == ["both"]


def _stale_lineage(holder, prefix: str) -> tuple[str, str]:
    """root(compression, 40 days old) -> tip; the tip's state is the caller's."""
    root, tip = f"{prefix}-root", f"{prefix}-tip"
    create_session(holder, root, source="feishu")
    create_session(holder, tip, source="feishu", parent_session_id=root)
    base = time.time() - 40 * 86400
    holder._write_sql(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', last_activity_at = ? WHERE id = ?",
        (base, base + 10, base + 10, root))
    return root, tip


def test_bulk_archive_matches_a_lineage_through_its_tip_only(store, holder, scope_a):
    """#115489: `hermes sessions archive --older-than` must never hide an OPEN, active continuation
    because its compression ancestor is old: the lineage is archived through its tip, and an idle
    ended tip still takes its whole chain with it."""
    live_root, live_tip = _stale_lineage(holder, "live")
    append_message(holder, live_tip, "user", "still chatting")
    stale_root, stale_tip = _stale_lineage(holder, "stale")
    stale = time.time() - 35 * 86400
    holder._write_sql("UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'cli_close', last_activity_at = ? "
                      "WHERE id = ?", (stale, stale + 10, stale + 10, stale_tip))

    criteria = PruneCriteria(archived=False, lineage_tips_only=True)
    assert [r.session_id for r in store.maintenance.list_prune_candidates(30, criteria=criteria)] == [stale_tip]
    assert session_retention.archive_sessions(store, scope_a.profile_dir, older_than_days=30) == 1

    assert {s: get_session(holder, s)["archived"] for s in (live_root, live_tip)} == {live_root: 0, live_tip: 0}
    assert {s: get_session(holder, s)["archived"] for s in (stale_root, stale_tip)} == {stale_root: 1, stale_tip: 1}
    assert [s["id"] for s in list_sessions_rich(holder, order_by_last_active=True)] == [live_tip]
