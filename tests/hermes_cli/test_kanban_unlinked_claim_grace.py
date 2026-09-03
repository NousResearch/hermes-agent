"""Create→link claim race + dispatcher self-supervision.

A card created ``ready`` with no parent links used to be claimable before
the writer inserted ``task_links``. The claim gate now holds young
unlinked cards for ``CLAIM_UNLINKED_GRACE_SECONDS``. The dispatcher also
skips cards assigned to its own operator profile.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


def _fake_spawn(*args, **kwargs):
    return 12345


@pytest.mark.real_unlinked_claim_grace
def test_claim_task_skips_young_unlinked_root(kanban_home, monkeypatch):
    now = [1_700_000_000]
    monkeypatch.setattr(kb.time, "time", lambda: now[0])

    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="soon-to-be-linked", assignee="scout")
        assert kb.claim_task(conn, tid) is None
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        assert task.claim_lock is None


@pytest.mark.real_unlinked_claim_grace
def test_claim_task_after_grace_claims_unlinked_root(kanban_home, monkeypatch):
    now = [1_700_000_000]
    monkeypatch.setattr(kb.time, "time", lambda: now[0])

    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="genuine root", assignee="scout")
        now[0] += kb.CLAIM_UNLINKED_GRACE_SECONDS + 1
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert claimed.status == "running"
        assert claimed.claim_lock is not None


@pytest.mark.real_unlinked_claim_grace
def test_claim_task_young_linked_child_uses_parent_guard(kanban_home, monkeypatch):
    now = [1_700_000_000]
    monkeypatch.setattr(kb.time, "time", lambda: now[0])

    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="nexus")
        child = kb.create_task(
            conn, title="child", assignee="scout", parents=[parent],
        )
        assert kb.get_task(conn, child).status == "todo"
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))
        conn.commit()
        assert kb.claim_task(conn, child) is None
        after = kb.get_task(conn, child)
        assert after is not None
        assert after.status == "todo"
        assert after.claim_lock is None


def test_dispatch_skips_operator_profile_other_assignees_claim(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setenv("HERMES_PROFILE", "nexus")

    with kb.connect_closing() as conn:
        self_id = kb.create_task(conn, title="ops card", assignee="nexus")
        other_id = kb.create_task(conn, title="worker card", assignee="scout")
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)

    spawned_ids = [tid for tid, _who, _ws in res.spawned]
    assert self_id in res.skipped_self_dispatch
    assert self_id not in spawned_ids
    assert other_id in spawned_ids
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, self_id).status == "ready"
        assert kb.get_task(conn, other_id).status == "running"


def test_dispatch_skips_operator_profile_on_review_lane(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setenv("HERMES_PROFILE", "nexus")

    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="needs review", assignee="scout")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = kb.get_task(conn, tid).current_run_id
        assert kb.request_review(
            conn, tid, summary="done", reviewer="nexus", expected_run_id=run_id,
        )
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)

    spawned_ids = [sid for sid, _who, _ws in res.spawned]
    assert tid in res.skipped_self_dispatch
    assert tid not in spawned_ids
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "review"
