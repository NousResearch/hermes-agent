"""Tests for decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None, tenant=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        triage=True,
    )


def _create_blocked(conn, title="stuck card", assignee="orchestrator"):
    return kb.create_task(
        conn,
        title=title,
        assignee=assignee,
        initial_status="blocked",
    )


def test_decompose_blocked_resume_fanout_children_claimable(kanban_home):
    """Regression for the 2026-09-01 resume fan-out deadlock.

    A blocked card (an operator split a stuck card into children) must fan
    out into children that run IMMEDIATELY — children as parents of the
    closing card, never children linked UNDER the blocked root. Split a
    blocked card into 3 independent children and assert at least one is
    immediately claimable (ready) and none waits on the root.
    """
    with kbc.connect() as conn:
        tid = _create_blocked(conn)
        assert kb.get_task(conn, tid).status == "blocked"

    children = [
        {"title": "child A", "assignee": "researcher", "parents": []},
        {"title": "child B", "assignee": "engineer", "parents": []},
        {"title": "child C", "assignee": "default", "parents": []},
    ]
    with kbc.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 3

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        kids = [kb.get_task(conn, cid) for cid in child_ids]
        # At least one child is immediately claimable.
        assert any(k.status == "ready" for k in kids), (
            "no child is immediately claimable — resume fan-out deadlocked"
        )
        # NO child may be parent-gated under the root (the 2026-09-01 bug).
        gated_under_root = [
            cid for cid in child_ids
            if root.id in {p for p in kb.parent_ids(conn, cid)}
        ]
        assert gated_under_root == [], (
            f"children {gated_under_root} gated under blocked root — deadlock"
        )
        # The root waits on the whole graph: it is a child of every child.
        for cid in child_ids:
            assert root.id in set(kb.child_ids(conn, cid))
        # Root flipped to todo (gated, promote on children completion).
        assert root.status == "todo"


def test_decompose_all_triage_parked_children_is_pm_recoverable(kanban_home):
    """An all-triage-parked fan-out is PM-recoverable, not a deadlock.

    A fan-out whose children all park in triage (decision-shaped / unknown
    assignee) is NOT refused: the root flips to ``todo`` and waits on the
    graph, and the PM can accept the parked children to make the graph
    runnable (``unblock_task``/``specify_triage_task``). The true cycle guard
    is the sibling Kahn check earlier — an all-parked fan-out has a legal link
    set. (Regression: e1d27f786d wrongly raised ValueError here and broke
    test_list_triage_ids_excludes_auto_decomposer_created.)
    """
    with kbc.connect() as conn:
        tid = _create_triage(conn)
    children = [
        {"title": "decision A", "assignee": "researcher", "triage": True, "parents": []},
        {"title": "decision B", "assignee": "engineer", "triage": True, "parents": []},
    ]
    with kbc.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        kids = [kb.get_task(conn, cid) for cid in child_ids]
    # Root flips to todo and waits on the (parked) graph.
    assert root.status == "todo"
    # The decision children stay parked in triage for PM acceptance.
    assert all(k.status == "triage" for k in kids)


def test_decompose_creates_children_and_promotes_root(kanban_home):
    # No ``all_assignees_spawnable`` needed: the autouse assignee neutralizer
    # (root conftest) already patches profile_exists->True for kanban tests.
    # (Verified in Rodge review t_788d2b96 follow-up.)
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kbc.connect() as conn:
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])

    # Root flipped to todo with orchestrator assignee, gated by children.
    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    # First child has no internal parents → ready on recompute_ready.
    assert c0.status == "ready"
    assert c0.assignee == "researcher"
    # Second child has parents=[0] → stays in todo until c0 completes.
    assert c1.status == "todo"
    assert c1.assignee == "engineer"


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert any("Decomposed into" in (c.body or "") for c in comments)
    assert any(ev.kind == "decomposed" for ev in events)


@pytest.mark.real_assignees
def test_create_known_assignee_not_parked(kanban_home):
    """A real assignee (``default`` is Agent Smith, always spawnable) does NOT
    get triage-parked. Marked ``real_assignees`` so the autouse assignee
    neutralizer is skipped and this exercises the REAL ``profile_exists``
    ``('default')->True`` path against on-disk profile dirs."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="good", assignee="default")
        task = kb.get_task(conn, tid)
    assert tid is not None
    assert task is not None
    assert task.status != "triage"


@pytest.mark.real_assignees
def test_create_unknown_assignee_parked_in_triage(kanban_home):
    """An assignee that is not a real profile parks the card in triage with a
    comment, and the bogus name is preserved on the row (not rejected)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="phantom", assignee="engineer")
        task = kb.get_task(conn, tid)
        comments = kb.list_comments(conn, tid)
    assert task is not None
    assert task.status == "triage"
    assert task.assignee == "engineer"
    assert any("unknown assignee" in (c.body or "") for c in comments)


@pytest.mark.real_assignees
def test_create_blocked_unknown_assignee_not_comment_parked(kanban_home):
    """A card with an explicit ``initial_status="blocked"`` and an unknown
    assignee stays BLOCKED (no triage clobber) and must NOT get the 'parked
    in triage' system comment — the parking comment is gated on actually
    being moved to triage (Rodge review t_788d2b96 finding 1)."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="human ops", assignee="phantom", initial_status="blocked"
        )
        task = kb.get_task(conn, tid)
        comments = kb.list_comments(conn, tid)
    assert task is not None
    assert task.status == "blocked"
    assert task.assignee == "phantom"
    assert not any(
        "parked in triage" in (c.body or "") for c in comments
    )
    assert not any(
        "unknown assignee" in (c.body or "") for c in comments
    )


@pytest.mark.real_assignees
def test_decompose_unknown_assignee_child_parked_in_triage(kanban_home):
    """A decomposer-spawned child naming a phantom profile parks in triage AND
    carries a comment, exactly like a top-level phantom card (the 2026-08-30
    incident produced 12 'engineer' + 12 'orchestrator' junk children)."""
    with kbc.connect() as conn:
        tid = _create_triage(conn)
    # A dispatchable sibling keeps the fan-out legal; the phantom child still
    # parks in triage with the routing comment. (2026-09-01: a decomposition
    # whose link set has NO dispatchable member is refused outright — see
    # test_decompose_all_triage_parked_children_refused.) ``default`` is a
    # real spawnable profile; ``engineer`` is phantom under real_assignees.
    children = [
        {"title": "build it", "assignee": "engineer", "parents": []},
        {"title": "parallel real", "assignee": "default", "parents": []},
    ]
    with kbc.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    kid = child_ids[0]
    with kbc.connect() as conn:
        child = kb.get_task(conn, kid)
        comments = kb.list_comments(conn, kid)
    assert child is not None
    assert child.status == "triage"
    assert child.assignee == "engineer"
    assert any("unknown assignee" in (c.body or "") for c in comments)


# ---------------------------------------------------------------------------
# Guards 2 & 3 (t_c52b9bc3): worktree inheritance + park parent unchanged
# ---------------------------------------------------------------------------

def _make_worktree(repo, task_id, branch=None):
    target = repo / ".worktrees" / task_id
    kb._ensure_git_worktree(repo, target, branch or f"wt/{task_id}")
    return target


def test_decompose_impl_child_inherits_dirty_parent_worktree(kanban_home, tmp_path):
    """Guard 2: a decomposition of a parent whose worktree holds a dirty diff
    must NOT mint a fresh worktree for the implementation child — the child
    reuses the parent's worktree/branch so the half-done diff is not stranded.
    """
    import subprocess

    def _git(*a, cwd=None):
        r = subprocess.run(["git", *a], cwd=cwd, capture_output=True,
                           text=True, encoding="utf-8", timeout=60)
        assert r.returncode == 0, (a, r.stderr)
        return r.stdout

    origin = tmp_path / "origin.git"
    _git("init", "--bare", str(origin))
    project = tmp_path / "project"
    _git("clone", str(origin), str(project))
    _git("-C", str(project), "config", "user.email", "t@ex.com")
    _git("-C", str(project), "config", "user.name", "t")
    (project / "README.md").write_text("hello\n", encoding="utf-8")
    _git("-C", str(project), "add", "README.md")
    _git("-C", str(project), "commit", "-m", "init")
    _git("-C", str(project), "push", "origin", "HEAD")

    # Parent owns a worktree with an uncommitted (dirty) diff.
    wt = _make_worktree(project, "t_parent1111")
    (wt / "wip.txt").write_text("half-done\n", encoding="utf-8")

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="code task", assignee="orchestrator",
            workspace_kind="worktree", workspace_path=str(wt),
        )
        # force to triage so decompose_triage_task will fan it out
        conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
        conn.commit()

    children = [
        {"title": "change scoped", "assignee": "default", "parents": [], "body": "wip"},
        {"title": "decision", "assignee": "researcher", "parents": [], "triage": True},
    ]
    with kbc.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator", children=children,
            author="decomposer", auto_promote=False,
        )
    assert child_ids is not None and len(child_ids) == 2
    with kbc.connect() as conn:
        # The implementation child (first dispatchable/todo) inherits wt.
        impl = kb.get_task(conn, child_ids[0])
        other = kb.get_task(conn, child_ids[1])
    assert impl is not None and other is not None
    assert impl.workspace_kind == "worktree"
    assert impl.workspace_path == str(wt), (
        "implementation child must inherit the parent's dirty worktree, not "
        "mint a fresh one"
    )
    # The decision-shaped (triage-parked) sibling still gets a fresh worktree.
    assert other.workspace_path is None


def test_decompose_blocked_resume_preserves_root_assignee(kanban_home):
    """Guard 3: a BLOCKED-resume fan-out leaves the parent's assignee unchanged
    (no reassignment to switch / any router profile) and parks it as todo.
    Fresh triage fan-outs keep the historical orchestrator-wake behavior.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="stuck impl", assignee="bob", initial_status="blocked",
        )
    children = [
        {"title": "child A", "assignee": "researcher", "parents": []},
        {"title": "child B", "assignee": "engineer", "parents": []},
    ]
    with kbc.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="switch", children=children,
            author="decomposer",
        )
    assert child_ids is not None and len(child_ids) == 2
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
    assert root is not None
    assert root.status == "todo"
    # assignee preserved — NOT reassigned to the router profile "switch"
    assert root.assignee == "bob"


def test_decompose_fresh_triage_still_sets_root_assignee(kanban_home):
    """Guard 3 regression: a FRESH triage fan-out keeps the historical
    orchestrator-wake assignment (root.assignee == root_assignee). Only the
    blocked-resume path is exempt.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="fresh idea", triage=True)
    children = [{"title": "research", "assignee": "researcher", "parents": []}]
    with kbc.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator", children=children,
            author="decomposer",
        )
    assert child_ids is not None and len(child_ids) == 1
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
    assert root is not None
    assert root.assignee == "orchestrator"


def _complete(conn, tid):
    kb.complete_task(conn, tid)


def test_decompose_hold_child_is_operator_hold_and_never_autopromotes(kanban_home):
    """Change A (2026-09-04 GlobalAside): a decomposed child marked ``hold``
    must be created as a REAL operator_hold block — never todo/ready — so it
    cannot auto-promote past an owner approval gate. A prose 'HELD' in the body
    is not a gate: recompute_ready auto-promotes a todo child the instant its
    parent completes. Assert:
      1) child with hold=True -> status 'blocked', block_kind 'operator_hold'
      2) a blocked 'operator_hold' child whose parent completes is NOT promoted
         by recompute_ready (sticky block), i.e. no bypass via parent-gating.
    """
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
    children = [
        # Non-held child: normal todo -> ready on parent completion.
        {"title": "implement", "assignee": "engineer", "parents": [], "hold": False},
        # Held terminal deploy child children of the impl child.
        {"title": "Deploy to production", "assignee": "researcher",
         "parents": [0], "hold": True},
    ]
    with kbc.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator", children=children,
            author="decomposer",
        )
    assert child_ids is not None and len(child_ids) == 2
    impl_id, deploy_id = child_ids

    with kbc.connect() as conn:
        impl = kb.get_task(conn, impl_id)
        deploy = kb.get_task(conn, deploy_id)
        assert impl.status == "ready"          # non-held: promoted normally
        assert deploy.status == "blocked"      # held: parked as a block
        assert deploy.block_kind == "operator_hold"

    # Complete the impl parent. recompute_ready must NOT promote the held child.
    with kbc.connect() as conn:
        kb.complete_task(conn, impl_id)
        kb.recompute_ready(conn)

    with kbc.connect() as conn:
        deploy = kb.get_task(conn, deploy_id)
        assert deploy.status == "blocked", (
            "operator_hold child auto-promoted past approval gate — Change A "
            "regression"
        )
        assert deploy.block_kind == "operator_hold"

    # A real unblock (owner approval) is the only exit.
    with kbc.connect() as conn:
        kb.unblock_task(conn, deploy_id)
    with kbc.connect() as conn:
        deploy = kb.get_task(conn, deploy_id)
        # After unblock, if its parent is done it resumes (ready) — the gate
        # is honoured, not firewalled forever.
        assert deploy.status in ("ready", "todo")


def test_decompose_hold_child_emits_blocked_event(kanban_home):
    """The held child writes a typed 'blocked' event so escalation-watch,
    fleet-preflight, stalled-card-watch and _has_sticky_block see the hold."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="rollout")
    children = [{"title": "Release build", "assignee": "researcher",
                 "parents": [], "hold": True}]
    with kbc.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator", children=children,
            author="decomposer",
        )
    assert child_ids is not None and len(child_ids) == 1
    with kbc.connect() as conn:
        events = kb.list_events(conn, child_ids[0])
    assert any(ev.kind == "blocked" for ev in events), (
        "held deploy child missing typed blocked event"
    )
    blocked = [ev for ev in events if ev.kind == "blocked"]
    assert blocked and blocked[-1].payload.get("kind") == "operator_hold"


def test_decompose_hold_false_child_not_blocked(kanban_home):
    """hold:false (or absent) children keep normal behavior — no over-hold."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="feature")
    for hold_flag in (False, None):
        with kbc.connect() as conn:
            _tid = kb.create_task(conn, title=f"feature {hold_flag}", triage=True)
        children = [{"title": "code", "assignee": "researcher",
                     "parents": [], "hold": hold_flag}]
        with kbc.connect() as conn:
            child_ids = kb.decompose_triage_task(
                conn, _tid, root_assignee="orch", children=children,
                author="decomposer",
            )
        assert child_ids is not None and len(child_ids) == 1
        with kbc.connect() as conn:
            child = kb.get_task(conn, child_ids[0])
        assert child.status == "ready"
        assert child.block_kind is None
