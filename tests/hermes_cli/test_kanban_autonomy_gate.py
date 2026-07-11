"""REQ-050 (decision 0007) — forced-failure safety-path tests for the deployment
autonomy enforcement floor. These GATE the first real deploy: the dispatch gate
must refuse to spawn work above a repo's autonomy ceiling, the kill-switch must
halt all spawning, and a Tier-C card must never execute regardless of config.

Covers REQ-044 (risk_tier plumbing), REQ-045 (autonomy_ceiling denormalisation),
and REQ-046 (dispatch gate + kill-switch).
"""
from __future__ import annotations

import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Fresh HERMES_HOME with a clean kanban DB (mirrors the default-assignee
    test fixture so the reimport picks up the isolated home)."""
    test_home = tempfile.mkdtemp(prefix="kanban_autonomy_gate_test_")
    monkeypatch.setenv("HERMES_HOME", test_home)
    # Ensure the kill-switch is off by default for every test.
    monkeypatch.delenv("HERMES_AUTONOMY_PAUSED", raising=False)
    monkeypatch.delenv("HERMES_AUTONOMY_PAUSE_FILE", raising=False)
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("hermes_cli")
            or mod.startswith("hermes_state")
            or mod == "hermes_constants"
        ):
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db, test_home


def _fake_spawn(*args, **kwargs):
    return 12345


def _mk_ready(kb, conn, **kw):
    """Create a ready, assigned card (defaults to a spawnable assignee)."""
    kw.setdefault("assignee", "default")
    return kb.create_task(conn, **kw)


# --------------------------------------------------------------------------
# Pure tier/ceiling logic (REQ-044)
# --------------------------------------------------------------------------

def test_tier_permitted_matrix(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    f = kb.tier_permitted_by_ceiling
    assert f("deploy-A", "deploy-A") is True
    assert f("deploy-A", "deploy-B") is True
    assert f("deploy-B", "deploy-A") is False
    # Tier-C never permitted, even under the highest ceiling.
    assert f("deploy-C", "deploy-B") is False
    # Unclassified card => repo-only floor (never a deploy).
    assert f(None, "deploy-B") is True
    assert f("deploy-A", None) is False          # no ceiling => repo-only default
    assert f("repo-only", None) is True
    # plan-only permits only read-only inspect.
    assert f("inspect", "plan-only") is True
    assert f("repo-only", "plan-only") is False
    # Garbage fails closed (treated as max-risk / default ceiling).
    assert f("bogus", "deploy-B") is True        # unknown tier => repo-only floor
    assert f("deploy-B", "bogus") is False       # unknown ceiling => repo-only


# --------------------------------------------------------------------------
# Dispatch gate (REQ-046)
# --------------------------------------------------------------------------

def test_deploy_card_over_ceiling_is_blocked(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = _mk_ready(
            kb, conn, title="deploy fleet",
            risk_tier="deploy-B", autonomy_ceiling="deploy-A",
        )
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    assert tid in res.skipped_ceiling
    assert tid not in [s[0] for s in res.spawned]
    with kb.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        ev = conn.execute(
            "SELECT COUNT(*) c FROM task_events "
            "WHERE task_id = ? AND kind = 'skipped_autonomy_ceiling'",
            (tid,),
        ).fetchone()
    assert row["status"] == "blocked"
    assert row["block_kind"] == "needs_input"
    assert ev["c"] == 1


def test_deploy_card_within_ceiling_passes_gate(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = _mk_ready(
            kb, conn, title="deploy bot",
            risk_tier="deploy-A", autonomy_ceiling="deploy-B",
        )
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    # The gate let it through — it is NOT ceiling-blocked (whether it then
    # spawned depends on profile resolution, which is not what we assert here).
    assert tid not in res.skipped_ceiling
    with kb.connect_closing() as conn:
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert row["status"] != "blocked"


def test_tier_c_card_never_executes(isolated_kanban_home):
    """Hard-exclude proof: a deploy-C card (wanctl/work-ansible surface) is
    blocked even under the most permissive ceiling."""
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = _mk_ready(
            kb, conn, title="touch live router",
            risk_tier="deploy-C", autonomy_ceiling="deploy-B",
        )
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    assert tid in res.skipped_ceiling


def test_unclassified_deploy_defaults_safe(isolated_kanban_home):
    """No ceiling (no project) => repo-only default: a deploy card is blocked,
    a repo-only card passes. No autonomous deploy without an explicit opt-in."""
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        deploy = _mk_ready(kb, conn, title="deploy", risk_tier="deploy-A")
        repo = _mk_ready(kb, conn, title="edit docs", risk_tier="repo-only")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    assert deploy in res.skipped_ceiling
    assert repo not in res.skipped_ceiling


def test_dry_run_does_not_block(isolated_kanban_home):
    """A dry-run tick reports the ceiling skip but must NOT mutate the row."""
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = _mk_ready(
            kb, conn, title="deploy fleet",
            risk_tier="deploy-B", autonomy_ceiling="deploy-A",
        )
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert tid in res.skipped_ceiling
    with kb.connect_closing() as conn:
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert row["status"] == "ready"  # untouched by dry-run


# --------------------------------------------------------------------------
# Kill-switch (REQ-046)
# --------------------------------------------------------------------------

def test_kill_switch_env_halts_spawning(isolated_kanban_home, monkeypatch):
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = _mk_ready(kb, conn, title="safe repo card", risk_tier="repo-only")
    monkeypatch.setenv("HERMES_AUTONOMY_PAUSED", "1")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    assert res.autonomy_paused is True
    assert not res.spawned
    with kb.connect_closing() as conn:
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    # Paused => not spawned, but NOT blocked either (it just waits).
    assert row["status"] == "ready"
    # Resume: same card now dispatches (gate lets a repo-only card through).
    monkeypatch.delenv("HERMES_AUTONOMY_PAUSED", raising=False)
    with kb.connect_closing() as conn:
        res2 = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    assert res2.autonomy_paused is False
    assert tid not in res2.skipped_ceiling


def test_kill_switch_sentinel_file(isolated_kanban_home, monkeypatch):
    kb, _ = isolated_kanban_home
    pause_file = tempfile.mktemp(prefix="autonomy_pause_")
    monkeypatch.setenv("HERMES_AUTONOMY_PAUSE_FILE", pause_file)
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        _mk_ready(kb, conn, title="card", risk_tier="repo-only")
    # No file yet => not paused.
    assert kb._autonomy_is_paused() is False
    import pathlib
    pathlib.Path(pause_file).write_text("paused by operator\n")
    assert kb._autonomy_is_paused() is True
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    assert res.autonomy_paused is True


# --------------------------------------------------------------------------
# Ceiling denormalisation from the project (REQ-045)
# --------------------------------------------------------------------------

def test_ceiling_denormalised_from_project(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    from hermes_cli import projects_db as pdb
    repo = tempfile.mkdtemp(prefix="proj_repo_")
    with pdb.connect_closing() as pconn:
        pid = pdb.create_project(pconn, name="wanctl", primary_path=repo)
        assert pdb.update_project(pconn, pid, autonomy_ceiling="plan-only")
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="wanctl work", assignee="default",
                             project_id=pid)
        t = kb.get_task(conn, tid)
    assert t.autonomy_ceiling == "plan-only"


def test_invalid_ceiling_rejected(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        with pytest.raises(ValueError):
            kb.create_task(conn, title="x", autonomy_ceiling="bogus")
        with pytest.raises(ValueError):
            kb.create_task(conn, title="x", risk_tier="bogus")


# --------------------------------------------------------------------------
# Risk-gated completion (REQ-047)
# --------------------------------------------------------------------------


def test_deploy_card_completion_held_for_review(isolated_kanban_home):
    """A deploy card with no attended approval parks in `review`, not `done`."""
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(
            conn, title="deploy", assignee="default", risk_tier="deploy-A"
        )
        ok = kb.complete_task(conn, tid, result="deployed")
    assert ok is True
    with kb.connect_closing() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
        ev = conn.execute(
            "SELECT COUNT(*) c FROM task_events "
            "WHERE task_id = ? AND kind = 'completion_held_for_review'",
            (tid,),
        ).fetchone()
    assert row["status"] == "review"
    assert ev["c"] == 1


def test_deploy_card_completes_with_approval(isolated_kanban_home):
    """A recorded `deploy_approved` event lets a deploy card auto-close."""
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(
            conn, title="deploy", assignee="default", risk_tier="deploy-A"
        )
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "deploy_approved", {"by": "operator"})
        ok = kb.complete_task(conn, tid, result="deployed")
    assert ok is True
    with kb.connect_closing() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "done"


def test_repo_card_auto_completes(isolated_kanban_home):
    """Non-deploy cards (repo-only/docs/tests/inspect) auto-close as before."""
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(
            conn, title="edit docs", assignee="default", risk_tier="repo-only"
        )
        ok = kb.complete_task(conn, tid, result="done")
    assert ok is True
    with kb.connect_closing() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "done"
