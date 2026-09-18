"""Tests: automatic reaping of stale duplicate clones (kanban task t_a95da99a).

Problem: a respawn loop (e.g. a supervisor script) that violates the
one-clone-per-lineage protocol creates a FRESH card every time it wakes,
instead of finding and resuming its own prior card. Each generation blocks on
the same unmet precondition the last one did, and nothing ever retired the
old ones — left unchecked, 12 blocked clones of a single logical task
accumulated on one real board with no automatic recovery.

``reap_stale_sibling`` is the safety-checked primitive: archives one
``blocked`` task because a same-lineage sibling (same title/created_by/
assignee) already reached a status that proves it supersedes the stale one.
It re-derives and re-checks lineage from the DB itself — it does not trust
the caller's claim that two ids are siblings.

``_reap_superseded_blocked_siblings`` is the dispatcher-side finder wired
into every reclaim phase: groups ``blocked`` rows by lineage, finds each
lineage's furthest-along member, and reaps every other blocked member of
that lineage against it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _mk(conn, *, status, title="dup-lineage", created_by="supervisor", assignee="worker"):
    kwargs = {"initial_status": "blocked"} if status == "blocked" else {}
    tid = kb.create_task(conn, title=title, assignee=assignee, created_by=created_by, **kwargs)
    if status not in ("blocked", "ready"):
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, tid))
        conn.commit()
    return tid


class TestReapStaleSibling:
    """Unit tests for the safety-checked single-pair primitive."""

    def test_archives_stale_when_keeper_is_ready(self, conn):
        stale = _mk(conn, status="blocked")
        keeper = _mk(conn, status="ready")

        assert kb.reap_stale_sibling(conn, stale, keeper) is True
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (stale,)
        ).fetchone()["status"] == "archived"

    def test_keeper_status_unaffected(self, conn):
        stale = _mk(conn, status="blocked")
        keeper = _mk(conn, status="ready")

        kb.reap_stale_sibling(conn, stale, keeper)

        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (keeper,)
        ).fetchone()["status"] == "ready"

    def test_rejects_mismatched_title(self, conn):
        stale = _mk(conn, status="blocked", title="alpha")
        keeper = _mk(conn, status="ready", title="beta")

        with pytest.raises(ValueError, match="lineage"):
            kb.reap_stale_sibling(conn, stale, keeper)
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (stale,)
        ).fetchone()["status"] == "blocked"

    def test_rejects_mismatched_created_by(self, conn):
        stale = _mk(conn, status="blocked", created_by="supervisor-a")
        keeper = _mk(conn, status="ready", created_by="supervisor-b")

        with pytest.raises(ValueError, match="lineage"):
            kb.reap_stale_sibling(conn, stale, keeper)

    def test_rejects_mismatched_assignee(self, conn):
        stale = _mk(conn, status="blocked", assignee="worker-a")
        keeper = _mk(conn, status="ready", assignee="worker-b")

        with pytest.raises(ValueError, match="lineage"):
            kb.reap_stale_sibling(conn, stale, keeper)

    def test_rejects_stale_task_not_blocked(self, conn):
        """The target being reaped must itself be blocked — refuse to touch
        anything live (ready/running/review) even same-lineage."""
        not_stale = _mk(conn, status="ready")
        keeper = _mk(conn, status="done")

        with pytest.raises(ValueError, match="blocked"):
            kb.reap_stale_sibling(conn, not_stale, keeper)
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (not_stale,)
        ).fetchone()["status"] == "ready"

    def test_rejects_keeper_also_blocked(self, conn):
        """A 'keeper' that is itself blocked proves nothing supersedes the
        stale task yet — must not archive on that basis."""
        stale = _mk(conn, status="blocked")
        also_blocked = _mk(conn, status="blocked")

        with pytest.raises(ValueError, match="does not prove it supersedes"):
            kb.reap_stale_sibling(conn, stale, also_blocked)

    def test_rejects_keeper_archived(self, conn):
        stale = _mk(conn, status="blocked")
        archived_keeper = _mk(conn, status="archived")

        with pytest.raises(ValueError, match="does not prove it supersedes"):
            kb.reap_stale_sibling(conn, stale, archived_keeper)

    def test_rejects_self_reap(self, conn):
        tid = _mk(conn, status="blocked")
        with pytest.raises(ValueError):
            kb.reap_stale_sibling(conn, tid, tid)

    def test_rejects_unknown_stale_id(self, conn):
        keeper = _mk(conn, status="ready")
        with pytest.raises(ValueError):
            kb.reap_stale_sibling(conn, "t_doesnotexist", keeper)

    def test_rejects_unknown_keeper_id(self, conn):
        stale = _mk(conn, status="blocked")
        with pytest.raises(ValueError):
            kb.reap_stale_sibling(conn, stale, "t_doesnotexist")

    def test_accepts_keeper_in_each_alive_status(self, conn):
        for status in ("ready", "running", "review", "todo", "scheduled", "done"):
            stale = _mk(conn, status="blocked", title=f"lineage-{status}")
            keeper = _mk(conn, status=status, title=f"lineage-{status}")
            assert kb.reap_stale_sibling(conn, stale, keeper) is True, status

    def test_event_and_comment_logged(self, conn):
        stale = _mk(conn, status="blocked")
        keeper = _mk(conn, status="ready")

        kb.reap_stale_sibling(conn, stale, keeper)

        events = kb.list_events(conn, stale)
        reaped = [e for e in events if e.kind == "reaped"]
        assert len(reaped) == 1
        assert reaped[0].payload["superseded_by"] == keeper
        comments = kb.list_comments(conn, stale)
        assert any("reaped" in (c.body or "").lower() for c in comments)
        assert any(keeper in (c.body or "") for c in comments)


class TestReapSupersededBlockedSiblings:
    """Integration tests for the dispatcher-wired finder."""

    def test_three_stale_clones_reaped_against_one_ready_keeper(self, conn):
        clones = [_mk(conn, status="blocked") for _ in range(3)]
        keeper = _mk(conn, status="ready")

        reaped = kbd._reap_superseded_blocked_siblings(conn)

        assert sorted(reaped) == sorted(clones)
        statuses = {r["id"]: r["status"] for r in conn.execute(
            "SELECT id, status FROM tasks"
        ).fetchall()}
        for c in clones:
            assert statuses[c] == "archived"
        assert statuses[keeper] == "ready"

    def test_unrelated_blocked_task_untouched(self, conn):
        clone = _mk(conn, status="blocked", title="lineage-a")
        _mk(conn, status="ready", title="lineage-a")
        unrelated = _mk(conn, status="blocked", title="totally different card")

        reaped = kbd._reap_superseded_blocked_siblings(conn)

        assert clone in reaped
        assert unrelated not in reaped
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (unrelated,)
        ).fetchone()["status"] == "blocked"

    def test_lineage_with_no_alive_member_untouched(self, conn):
        """Every member of a lineage blocked -> nothing supersedes them yet;
        must not archive anything (there is no evidence any of them is stale
        rather than genuinely waiting)."""
        a = _mk(conn, status="blocked", title="all-blocked-lineage")
        b = _mk(conn, status="blocked", title="all-blocked-lineage")

        reaped = kbd._reap_superseded_blocked_siblings(conn)

        assert reaped == []
        statuses = {r["id"]: r["status"] for r in conn.execute(
            "SELECT id, status FROM tasks"
        ).fetchall()}
        assert statuses[a] == "blocked"
        assert statuses[b] == "blocked"

    def test_no_blocked_tasks_is_noop(self, conn):
        _mk(conn, status="ready")
        assert kbd._reap_superseded_blocked_siblings(conn) == []

    def test_keeper_is_newest_alive_member_not_oldest(self, conn):
        """When multiple alive siblings exist, reaping must not fail merely
        because more than one keeper candidate is present."""
        stale = _mk(conn, status="blocked", title="multi-keeper")
        older_done = _mk(conn, status="done", title="multi-keeper")
        newer_running = _mk(conn, status="running", title="multi-keeper")

        reaped = kbd._reap_superseded_blocked_siblings(conn)

        assert stale in reaped
        statuses = {r["id"]: r["status"] for r in conn.execute(
            "SELECT id, status FROM tasks"
        ).fetchall()}
        assert statuses[older_done] == "done"
        assert statuses[newer_running] == "running"

    def test_distinct_lineages_each_handled_independently(self, conn):
        stale_a = _mk(conn, status="blocked", title="lineage-a", assignee="w1")
        keeper_a = _mk(conn, status="ready", title="lineage-a", assignee="w1")
        stale_b = _mk(conn, status="blocked", title="lineage-b", assignee="w2")
        keeper_b = _mk(conn, status="running", title="lineage-b", assignee="w2")

        reaped = kbd._reap_superseded_blocked_siblings(conn)

        assert sorted(reaped) == sorted([stale_a, stale_b])
        statuses = {r["id"]: r["status"] for r in conn.execute(
            "SELECT id, status FROM tasks"
        ).fetchall()}
        assert statuses[keeper_a] == "ready"
        assert statuses[keeper_b] == "running"

    def test_twelve_clone_reproduction_of_real_incident_shape(self, conn):
        """Mirrors the real board shape that motivated this feature: many
        generations of a respawned coordinator, all blocked, plus one that
        finally got past the blocker."""
        clones = [_mk(conn, status="blocked", title="nsl-catalogue-coordinator",
                       created_by="nsl-catalogue-supervisor", assignee="coordinator")
                  for _ in range(11)]
        keeper = _mk(conn, status="ready", title="nsl-catalogue-coordinator",
                     created_by="nsl-catalogue-supervisor", assignee="coordinator")

        reaped = kbd._reap_superseded_blocked_siblings(conn)

        assert sorted(reaped) == sorted(clones)
        assert len(reaped) == 11
        remaining_blocked = conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE status='blocked'"
        ).fetchone()["c"]
        assert remaining_blocked == 0

    def test_wired_into_dispatch_once(self, conn):
        stale = _mk(conn, status="blocked")
        keeper = _mk(conn, status="ready")
        # keeper would otherwise get spawned by dispatch_once; that's fine,
        # this test only asserts the reap counter surfaces on the result.
        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: (True, ""),
                                    dry_run=True)

        assert stale in result.reaped_superseded_siblings
