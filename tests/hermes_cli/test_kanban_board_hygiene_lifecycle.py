"""Board-state-hygiene lifecycle transitions (BUI-942 item 2).

A parent card that already produced a PR — or that a replacement has
explicitly superseded — must NOT linger in ``ready`` forever behind the
respawn guard. It must transition into an appropriate non-ready lifecycle
state:

* a ready task with a recent PR comment  → ``review`` (auto handoff), so the
  review lane / a human picks up the PR instead of the dispatcher re-spawning
  the original worker every tick and only recording ``respawn_guarded``.
* a ready parent explicitly superseded by its replacement → ``archived`` with
  an auditable ``superseded`` event + comment, preserving child/dependency
  semantics (archived parents unblock dependents exactly like done ones).

Guard behavior for the OTHER reasons (blocker_auth / recent_success /
rate_limit_cooldown) is unchanged: those still defer in ``ready``.

Everything runs against a temp board — nothing here touches a live board.
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


PR_URL = "https://github.com/totemx-AI/subsidysmart/pull/77"


# ---------------------------------------------------------------------------
# (a) PR-producing ready card → review
# ---------------------------------------------------------------------------

def test_reconcile_moves_ready_pr_card_to_review(kanban_home):
    """A ready task with a recent PR comment is swept to ``review``."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        kb.add_comment(conn, t, "worker", f"Opened {PR_URL}")
        moved = kb.reconcile_pr_ready_to_review(conn)
        assert moved == [t]
        assert kb.get_task(conn, t).status == "review"


def test_reconcile_pr_handoff_emits_auditable_event(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        kb.add_comment(conn, t, "worker", f"PR up: {PR_URL}")
        kb.reconcile_pr_ready_to_review(conn)
        kinds = [
            r["kind"]
            for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
                (t,),
            ).fetchall()
        ]
    assert "auto_review_handoff" in kinds
    # A 'status' event drives the live feed / notifier so operators see the move.
    assert "status" in kinds


def test_reconcile_ignores_ready_card_with_old_pr_comment(kanban_home):
    """A PR comment older than the guard window is not a live PR — leave it."""
    import time

    with kb.connect() as conn:
        t = kb.create_task(conn, title="old-pr", assignee="alice")
        old_ts = int(time.time()) - kb._RESPAWN_GUARD_PR_WINDOW - 60
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, 'worker', ?, ?)",
            (t, f"stale {PR_URL}", old_ts),
        )
        moved = kb.reconcile_pr_ready_to_review(conn)
        assert moved == []
        assert kb.get_task(conn, t).status == "ready"


def test_reconcile_ignores_ready_card_without_pr(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="plain", assignee="alice")
        kb.add_comment(conn, t, "worker", "no pr here, just a note")
        moved = kb.reconcile_pr_ready_to_review(conn)
    assert moved == []


def test_dispatch_hands_off_ready_pr_card_to_review_instead_of_guard_sitting(
    kanban_home
):
    """End-to-end through dispatch_once: the PR card leaves ``ready`` (to
    ``review``) and is surfaced in ``pr_handoff_to_review`` rather than only
    being recorded in ``respawn_guarded`` and left sitting in ready.

    No ``all_assignees_spawnable`` fixture here, so the real profile check
    skips the (non-profile) 'alice' assignee in the review lane — leaving the
    card cleanly in ``review`` for a deterministic assertion."""
    spawned = []

    def fake_spawn(task, workspace):
        spawned.append(task.id)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        kb.add_comment(conn, t, "worker", f"Opened {PR_URL}")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        assert t in res.pr_handoff_to_review
        # It left ready — it is NOT sitting guard-deferred.
        assert (t, "active_pr") not in res.respawn_guarded
        assert kb.get_task(conn, t).status == "review"


def test_dispatch_preserves_non_pr_guard_reasons_in_ready(
    kanban_home, all_assignees_spawnable
):
    """blocker_auth (and the other guard reasons) still defer in ready — item 2
    only changes the active_pr lifecycle, not the rest of the guard."""
    spawned = []

    def fake_spawn(task, workspace):
        spawned.append(task.id)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="quota", assignee="alice")
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("quota exceeded", t),
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        assert (t, "blocker_auth") in res.respawn_guarded
        assert t not in spawned
        assert t not in res.pr_handoff_to_review
        assert kb.get_task(conn, t).status == "ready"


# ---------------------------------------------------------------------------
# (b) explicit supersession → archived
# ---------------------------------------------------------------------------

def test_supersede_task_moves_ready_parent_to_archived(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="old-plan", assignee="alice")
        replacement = kb.create_task(conn, title="new-plan", assignee="alice")
        ok = kb.supersede_task(
            conn, parent, replaced_by=replacement, actor="peewee"
        )
        assert ok is True
        assert kb.get_task(conn, parent).status == "archived"
        # Auditable superseded event carries the replacement pointer.
        ev = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'superseded'",
            (parent,),
        ).fetchone()
        assert ev is not None and replacement in (ev["payload"] or "")
        # Auditable comment for humans reading the card.
        bodies = [c.body for c in kb.list_comments(conn, parent)]
        assert any(replacement in b for b in bodies)


def _parents_of(conn, child):
    return {
        r["parent_id"]
        for r in conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id = ?", (child,)
        ).fetchall()
    }


def _children_of(conn, parent):
    return {
        r["child_id"]
        for r in conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ?", (parent,)
        ).fetchall()
    }


def test_supersede_transfers_dependencies_to_replacement_and_keeps_blocked(
    kanban_home
):
    """The old parent's dependents must be RE-PARENTED onto the replacement —
    not released — so they stay blocked until the replacement finishes. This is
    the core BUI-942 BLOCKER-2 fix: recompute_ready must not free a child just
    because the superseded parent got archived."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="old", assignee="alice")
        replacement = kb.create_task(conn, title="new", assignee="alice")
        other_parent = kb.create_task(conn, title="p2", assignee="alice")
        only_child = kb.create_task(
            conn, title="c1", assignee="alice", parents=[parent]
        )
        multi_child = kb.create_task(
            conn, title="c2", assignee="alice", parents=[parent, other_parent]
        )
        assert kb.get_task(conn, only_child).status == "todo"

        assert kb.supersede_task(conn, parent, replaced_by=replacement) is True

        # Edge retargeted: parent -> child removed, replacement -> child added.
        assert _children_of(conn, parent) == set()
        assert only_child in _children_of(conn, replacement)
        assert multi_child in _children_of(conn, replacement)
        assert _parents_of(conn, only_child) == {replacement}
        assert _parents_of(conn, multi_child) == {replacement, other_parent}

        # Dependents stay BLOCKED (replacement still active) — NOT released.
        assert kb.get_task(conn, only_child).status == "todo"
        assert kb.get_task(conn, multi_child).status == "todo"
        assert kb.get_task(conn, parent).status == "archived"


def test_supersede_transferred_child_promotes_when_replacement_completes(
    kanban_home
):
    """After transfer, the child is released only once the REPLACEMENT reaches a
    terminal state — the existing terminal-state gating semantics."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="old", assignee="alice")
        replacement = kb.create_task(conn, title="new", assignee="alice")
        child = kb.create_task(
            conn, title="c", assignee="alice", parents=[parent]
        )
        kb.supersede_task(conn, parent, replaced_by=replacement)
        assert kb.get_task(conn, child).status == "todo"  # still gated

        # Replacement finishes → child's sole (transferred) blocker clears.
        kb.complete_task(conn, replacement)
        assert kb.get_task(conn, child).status == "ready"


def test_supersede_requires_replacement_when_dependents_exist(kanban_home):
    """A parent with dependents cannot be superseded without a replacement — that
    would prematurely release the children (the bug). Reject it loudly."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="p", assignee="alice")
        kb.create_task(conn, title="c", assignee="alice", parents=[parent])
        with pytest.raises(ValueError):
            kb.supersede_task(conn, parent, replaced_by=None)
        # Nothing mutated — parent still active, child still gated.
        assert kb.get_task(conn, parent).status == "ready"


def test_supersede_rejects_self_replacement(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="t", assignee="alice")
        with pytest.raises(ValueError):
            kb.supersede_task(conn, t, replaced_by=t)
        assert kb.get_task(conn, t).status == "ready"  # unchanged


def test_supersede_rejects_missing_replacement(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="t", assignee="alice")
        with pytest.raises(ValueError):
            kb.supersede_task(conn, t, replaced_by="t_ghost")
        assert kb.get_task(conn, t).status == "ready"


def test_supersede_rejects_terminal_replacement(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="t", assignee="alice")
        dead = kb.create_task(conn, title="dead", assignee="alice")
        kb.complete_task(conn, dead)  # replacement is terminal (done)
        with pytest.raises(ValueError):
            kb.supersede_task(conn, t, replaced_by=dead)
        assert kb.get_task(conn, t).status == "ready"


@pytest.mark.parametrize("source_status", ["todo", "blocked", "running", "review", "done"])
def test_supersede_rejects_non_ready_source(kanban_home, source_status):
    """Supersession is the explicit ready-card hygiene transition, not a way to
    rewrite in-flight, blocked, review, or terminal lifecycle states."""
    with kb.connect() as conn:
        source = kb.create_task(conn, title="source", assignee="alice")
        repl = kb.create_task(conn, title="r", assignee="alice")
        conn.execute(
            "UPDATE tasks SET status = ? WHERE id = ?", (source_status, source)
        )
        with pytest.raises(ValueError):
            kb.supersede_task(conn, source, replaced_by=repl)
        unchanged = kb.get_task(conn, source)
        assert unchanged is not None
        assert unchanged.status == source_status


def test_supersede_transfer_avoids_duplicate_edge(kanban_home):
    """If a child already depends on BOTH the old parent and the replacement,
    the transfer must not create a duplicate edge."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="p", assignee="alice")
        replacement = kb.create_task(conn, title="r", assignee="alice")
        child = kb.create_task(
            conn, title="c", assignee="alice", parents=[parent, replacement]
        )
        kb.supersede_task(conn, parent, replaced_by=replacement)
        # Exactly one replacement->child edge; parent->child gone.
        edge_count = conn.execute(
            "SELECT COUNT(*) FROM task_links WHERE parent_id = ? AND child_id = ?",
            (replacement, child),
        ).fetchone()[0]
        assert edge_count == 1
        assert _parents_of(conn, child) == {replacement}


def test_supersede_transfer_rejects_cycle_without_mutation(kanban_home):
    """A cyclic retarget must reject the whole transaction. Dropping the old
    edge and skipping the new one would prematurely release the dependent."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="p", assignee="alice")
        child = kb.create_task(
            conn, title="c", assignee="alice", parents=[parent]
        )
        # replacement depends on child → adding child->... no; we want
        # replacement->child to cycle, which happens iff child is an ancestor
        # of replacement. Link child -> replacement to set that up.
        replacement = kb.create_task(
            conn, title="r", assignee="alice", parents=[child]
        )
        # replacement is now a descendant of child; transferring parent->child
        # onto replacement->child would cycle.
        before_edges = {
            (r["parent_id"], r["child_id"])
            for r in conn.execute("SELECT parent_id, child_id FROM task_links")
        }
        with pytest.raises(ValueError, match="would create a cycle"):
            kb.supersede_task(conn, parent, replaced_by=replacement)
        # The transaction is unchanged: old edge retained and source still ready.
        after_edges = {
            (r["parent_id"], r["child_id"])
            for r in conn.execute("SELECT parent_id, child_id FROM task_links")
        }
        assert after_edges == before_edges
        assert (replacement, child) not in {
            (r["parent_id"], r["child_id"])
            for r in conn.execute("SELECT parent_id, child_id FROM task_links").fetchall()
        }
        assert parent in _parents_of(conn, child)
        unchanged = kb.get_task(conn, parent)
        assert unchanged is not None
        assert unchanged.status == "ready"


def test_supersede_task_is_idempotent_on_archived(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="alice")
        assert kb.supersede_task(conn, t) is True
        # Already archived → no-op, returns False.
        assert kb.supersede_task(conn, t) is False


def test_supersede_task_unknown_task_returns_false(kanban_home):
    with kb.connect() as conn:
        assert kb.supersede_task(conn, "t_does_not_exist") is False


# ---------------------------------------------------------------------------
# (b) operational entry point — the CLI lifecycle verb that reaches
#     supersede_task for an explicit, t_32bfd9f4-style supersession.
# ---------------------------------------------------------------------------

def _build_kanban_parser():
    import argparse

    from hermes_cli import kanban as kc

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    return parser


def test_cli_supersede_verb_transitions_ready_parent(kanban_home):
    """`hermes kanban supersede <id> --replaced-by <id>` is the explicit
    operational entry point: an operator/integration that decides a ready
    parent is obsolete runs it, and it reaches kb.supersede_task."""
    from hermes_cli import kanban as kc

    with kb.connect() as conn:
        parent = kb.create_task(conn, title="obsolete", assignee="alice")
        replacement = kb.create_task(conn, title="fresh", assignee="alice")

    parser = _build_kanban_parser()
    args = parser.parse_args(
        ["kanban", "supersede", parent, "--replaced-by", replacement]
    )
    rc = kc.kanban_command(args)
    assert rc == 0

    with kb.connect() as conn:
        assert kb.get_task(conn, parent).status == "archived"
        ev = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'superseded'",
            (parent,),
        ).fetchone()
        assert ev is not None and replacement in (ev["payload"] or "")


def test_cli_supersede_verb_reports_failure_for_unknown(kanban_home):
    from hermes_cli import kanban as kc

    parser = _build_kanban_parser()
    args = parser.parse_args(["kanban", "supersede", "t_nope"])
    rc = kc.kanban_command(args)
    assert rc == 1


def test_cli_supersede_verb_reports_invalid_self(kanban_home, capsys):
    """An invalid supersession (self-replacement) exits non-zero with the exact
    reason surfaced from supersede_task's ValueError."""
    from hermes_cli import kanban as kc

    with kb.connect() as conn:
        t = kb.create_task(conn, title="t", assignee="alice")

    parser = _build_kanban_parser()
    args = parser.parse_args(["kanban", "supersede", t, "--replaced-by", t])
    rc = kc.kanban_command(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "cannot supersede" in err and "itself" in err
    with kb.connect() as conn:
        assert kb.get_task(conn, t).status == "ready"  # unchanged


# ---------------------------------------------------------------------------
# (a) Q2 — the PR→review sweep must not cause duplicate maker work, and must
#     be idempotent across ticks; review dispatch semantics are intended.
# ---------------------------------------------------------------------------

def test_pr_sweep_does_not_respawn_the_original_maker(kanban_home):
    """The swept card must not be re-spawned by the ready/maker lane — that
    would risk a duplicate PR. With a non-profile assignee neither lane spawns
    it, so it rests cleanly in ``review`` with zero maker spawns."""
    spawned = []

    def fake_spawn(task, workspace):
        spawned.append(task.id)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        kb.add_comment(conn, t, "worker", f"Opened {PR_URL}")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        assert t in res.pr_handoff_to_review
        assert spawned == []  # no maker (or review) spawn for a non-profile
        assert kb.get_task(conn, t).status == "review"


def test_pr_swept_card_is_review_work_not_maker_work(
    kanban_home, all_assignees_spawnable
):
    """When the swept card IS spawnable, the review lane claims it and loads the
    sdlc-review skill (verify/merge the PR) — it is NOT re-run as a fresh maker
    worker. This confirms the intended review-dispatch semantics: no duplicate
    PR-producing work."""
    spawned_skills = {}

    def fake_spawn(task, workspace):
        spawned_skills[task.id] = list(task.skills or [])

    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        kb.add_comment(conn, t, "worker", f"Opened {PR_URL}")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        assert t in res.pr_handoff_to_review
        # Spawned via the review lane → sdlc-review skill loaded.
        assert spawned_skills.get(t) == ["sdlc-review"]
        assert kb.get_task(conn, t).status == "running"  # claimed by review lane


def test_pr_sweep_is_idempotent_across_ticks(kanban_home):
    """A second dispatch tick must not re-move or duplicate the handoff event —
    once in review the card is no longer ready, so the sweep skips it."""
    def noop_spawn(task, workspace):
        return None

    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        kb.add_comment(conn, t, "worker", f"Opened {PR_URL}")
        res1 = kb.dispatch_once(conn, spawn_fn=noop_spawn)
        res2 = kb.dispatch_once(conn, spawn_fn=noop_spawn)
        assert res1.pr_handoff_to_review == [t]
        assert res2.pr_handoff_to_review == []
        handoff_events = conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'auto_review_handoff'",
            (t,),
        ).fetchone()[0]
        assert handoff_events == 1


def test_pr_sweep_skipped_under_dry_run(kanban_home):
    """dry_run must not mutate the board — the PR card stays ready and is not
    swept, matching the no-writes contract of a dry dispatch pass."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        kb.add_comment(conn, t, "worker", f"Opened {PR_URL}")
        res = kb.dispatch_once(conn, dry_run=True)
        assert res.pr_handoff_to_review == []
        assert kb.get_task(conn, t).status == "ready"
