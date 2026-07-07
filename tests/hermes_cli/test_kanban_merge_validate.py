"""Tests for merge-card pre-dispatch validation (_validate_merge_card)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

APPROVED_SHA = "a" * 40  # 40-char hex


def _make_merge_body(sha: str = APPROVED_SHA) -> str:
    """Return a valid merge-intent card body."""
    return (
        f"Merge the approved PR.\n\n"
        f"gh pr merge 42 --squash --match-head-commit {sha}"
    )


def _make_preflight_body(sha: str = APPROVED_SHA) -> str:
    """Return a preflight evidence body with CODE_APPROVED and SHA."""
    return (
        f"CODE_APPROVED\n\n"
        f"Verified head SHA: {sha}\n"
        f"Tests passed, diff clean."
    )


# ---------------------------------------------------------------------------
# Test 1: Merge card with no parent → blocked
# ---------------------------------------------------------------------------

def test_merge_card_no_parent_blocked(kanban_home, all_assignees_spawnable):
    """A merge-intent card with no parent must be blocked, not spawned."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="merge PR",
            body=_make_merge_body(),
            assignee="builder",
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert not spawns, "merge card without parent must not be spawned"
    assert tid in res.auto_blocked
    with kb.connect() as conn:
        t = kb.get_task(conn, tid)
        assert t.status == "blocked"
        assert t.block_kind == "policy_violation"


# ---------------------------------------------------------------------------
# Test 2: Merge card with parent not assigned to preflight → blocked
# ---------------------------------------------------------------------------

def test_merge_card_no_preflight_parent_blocked(kanban_home, all_assignees_spawnable):
    """A merge card whose only parent is not preflight must be blocked."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        # Parent is builder, not preflight.
        parent_id = kb.create_task(
            conn,
            title="build something",
            assignee="builder",
        )
        kb.complete_task(conn, parent_id)
        merge_id = kb.create_task(
            conn,
            title="merge PR",
            body=_make_merge_body(),
            assignee="builder",
            parents=[parent_id],
        )
        # complete parent first for DAG promotion
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert not spawns, "merge card without preflight parent must not be spawned"
    assert merge_id in res.auto_blocked
    with kb.connect() as conn:
        t = kb.get_task(conn, merge_id)
        assert t.status == "blocked"
        assert t.block_kind == "policy_violation"


# ---------------------------------------------------------------------------
# Test 3: Preflight parent without CODE_APPROVED → blocked
# ---------------------------------------------------------------------------

def test_merge_card_preflight_no_approved_blocked(kanban_home, all_assignees_spawnable):
    """Preflight parent exists but has no CODE_APPROVED → blocked."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        preflight_id = kb.create_task(
            conn,
            title="preflight check",
            body="FIXUP_REQUIRED: tests fail",
            assignee="preflight",
        )
        kb.complete_task(conn, preflight_id)
        merge_id = kb.create_task(
            conn,
            title="merge PR",
            body=_make_merge_body(),
            assignee="builder",
            parents=[preflight_id],
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert not spawns
    assert merge_id in res.auto_blocked
    with kb.connect() as conn:
        t = kb.get_task(conn, merge_id)
        assert t.status == "blocked"
        assert t.block_kind == "policy_violation"


# ---------------------------------------------------------------------------
# Test 4: CODE_APPROVED present but no SHA → blocked
# ---------------------------------------------------------------------------

def test_merge_card_approved_no_sha_blocked(kanban_home, all_assignees_spawnable):
    """Preflight has CODE_APPROVED but no 40-char hex SHA → blocked."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        preflight_id = kb.create_task(
            conn,
            title="preflight check",
            body="CODE_APPROVED\n\nAll checks passed. (no SHA in body)",
            assignee="preflight",
        )
        kb.complete_task(conn, preflight_id)
        merge_id = kb.create_task(
            conn,
            title="merge PR",
            body=_make_merge_body(),
            assignee="builder",
            parents=[preflight_id],
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert not spawns
    assert merge_id in res.auto_blocked
    with kb.connect() as conn:
        t = kb.get_task(conn, merge_id)
        assert t.status == "blocked"


# ---------------------------------------------------------------------------
# Test 5: SHA mismatch between preflight and merge card → blocked
# ---------------------------------------------------------------------------

def test_merge_card_sha_mismatch_blocked(kanban_home, all_assignees_spawnable):
    """Preflight approved SHA differs from merge card SHA → blocked."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    preflight_sha = "b" * 40
    merge_sha = "c" * 40

    with kb.connect() as conn:
        preflight_id = kb.create_task(
            conn,
            title="preflight check",
            body=_make_preflight_body(sha=preflight_sha),
            assignee="preflight",
        )
        kb.complete_task(conn, preflight_id)
        merge_id = kb.create_task(
            conn,
            title="merge PR",
            body=_make_merge_body(sha=merge_sha),
            assignee="builder",
            parents=[preflight_id],
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert not spawns
    assert merge_id in res.auto_blocked
    with kb.connect() as conn:
        t = kb.get_task(conn, merge_id)
        assert t.status == "blocked"
        assert t.block_kind == "policy_violation"


# ---------------------------------------------------------------------------
# Test 6a: Merge card with --merge flag → blocked
# ---------------------------------------------------------------------------

def test_merge_card_forbidden_merge_flag_blocked(kanban_home, all_assignees_spawnable):
    """Merge card uses --merge instead of --squash → blocked."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        preflight_id = kb.create_task(
            conn,
            title="preflight check",
            body=_make_preflight_body(),
            assignee="preflight",
        )
        kb.complete_task(conn, preflight_id)
        merge_id = kb.create_task(
            conn,
            title="merge PR",
            body=f"gh pr merge 42 --merge --match-head-commit {APPROVED_SHA}",
            assignee="builder",
            parents=[preflight_id],
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert not spawns
    assert merge_id in res.auto_blocked
    with kb.connect() as conn:
        t = kb.get_task(conn, merge_id)
        assert t.status == "blocked"


# ---------------------------------------------------------------------------
# Test 6b: Merge card with --rebase flag → blocked
# ---------------------------------------------------------------------------

def test_merge_card_forbidden_rebase_flag_blocked(kanban_home, all_assignees_spawnable):
    """Merge card uses --rebase instead of --squash → blocked."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        preflight_id = kb.create_task(
            conn,
            title="preflight check",
            body=_make_preflight_body(),
            assignee="preflight",
        )
        kb.complete_task(conn, preflight_id)
        merge_id = kb.create_task(
            conn,
            title="merge PR",
            body=f"gh pr merge 42 --rebase --match-head-commit {APPROVED_SHA}",
            assignee="builder",
            parents=[preflight_id],
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert not spawns
    assert merge_id in res.auto_blocked


# ---------------------------------------------------------------------------
# Test 7: Valid merge card with all checks → eligible to spawn
# ---------------------------------------------------------------------------

def test_merge_card_valid_spawns(kanban_home, all_assignees_spawnable):
    """A merge card passing all validation checks must be eligible to spawn."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        preflight_id = kb.create_task(
            conn,
            title="preflight check",
            body=_make_preflight_body(),
            assignee="preflight",
        )
        kb.complete_task(conn, preflight_id)
        merge_id = kb.create_task(
            conn,
            title="merge PR",
            body=_make_merge_body(),
            assignee="builder",
            parents=[preflight_id],
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(spawns) == 1, f"valid merge card must spawn, got {len(spawns)}"
    assert spawns[0] == merge_id
    assert merge_id not in res.auto_blocked
    with kb.connect() as conn:
        t = kb.get_task(conn, merge_id)
        assert t.status == "running"


# ---------------------------------------------------------------------------
# Test 8: Non-merge cards pass through without validation
# ---------------------------------------------------------------------------

def test_non_merge_card_passes_through(kanban_home, all_assignees_spawnable):
    """Cards without 'gh pr merge' must not be affected by the validator."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="fix a bug",
            body="Fix the login button color.",
            assignee="builder",
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(spawns) == 1
    assert spawns[0] == tid
    assert tid not in res.auto_blocked


# ---------------------------------------------------------------------------
# Test 9: Valid merge card via comments (CODE_APPROVED + SHA in comment)
# ---------------------------------------------------------------------------

def test_merge_card_valid_via_comment(kanban_home, all_assignees_spawnable):
    """CODE_APPROVED and SHA found in preflight comments, not body."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        preflight_id = kb.create_task(
            conn,
            title="preflight check",
            body="Preflight inspection complete.",
            assignee="preflight",
        )
        # Add CODE_APPROVED + SHA via comment instead of body
        kb.add_comment(
            conn, preflight_id, "preflight",
            f"CODE_APPROVED\n\nVerified SHA: {APPROVED_SHA}\nAll checks passed.",
        )
        kb.complete_task(conn, preflight_id)
        merge_id = kb.create_task(
            conn,
            title="merge PR",
            body=_make_merge_body(),
            assignee="builder",
            parents=[preflight_id],
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(spawns) == 1
    assert spawns[0] == merge_id


# ---------------------------------------------------------------------------
# Test 10: Valid merge card with multiple parents, one preflight
# ---------------------------------------------------------------------------

def test_merge_card_multiple_parents_one_preflight(kanban_home, all_assignees_spawnable):
    """Merge card with builder + preflight parents; preflight approves → spawns."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        builder_parent = kb.create_task(
            conn,
            title="implement feature",
            assignee="builder",
        )
        kb.complete_task(conn, builder_parent)
        preflight_id = kb.create_task(
            conn,
            title="preflight check",
            body=_make_preflight_body(),
            assignee="preflight",
        )
        kb.complete_task(conn, preflight_id)
        merge_id = kb.create_task(
            conn,
            title="merge PR",
            body=_make_merge_body(),
            assignee="builder",
            parents=[builder_parent, preflight_id],
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(spawns) == 1
    assert spawns[0] == merge_id


# ---------------------------------------------------------------------------
# Test 11: Dry run does not block but reports auto_blocked
# ---------------------------------------------------------------------------

def test_merge_card_dry_run_reports_blocked(kanban_home, all_assignees_spawnable):
    """Dry run must not mutate status but must report the card as auto_blocked."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="merge PR",
            body=_make_merge_body(),
            assignee="builder",
        )
        res = kb.dispatch_once(conn, dry_run=True)

    assert tid in res.auto_blocked
    with kb.connect() as conn:
        t = kb.get_task(conn, tid)
        # Dry run must NOT mutate status
        assert t.status == "ready", f"dry run must not change status, got {t.status}"