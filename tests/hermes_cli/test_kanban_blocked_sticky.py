"""Regression tests for #28712 — kanban dispatcher must not auto-promote
worker-initiated ``kanban_block`` (sticky blocks), but must keep
auto-recovering circuit-breaker blocks.

The bug: when a worker called ``kanban_block(reason="review-required:
...")`` to hand off to a human, the dispatcher's ``recompute_ready``
would promote the task back to ``ready`` on the next tick.  The fresh
worker found nothing to do (work already applied), exited cleanly, and
got recorded as a ``protocol_violation`` → ``gave_up`` → promote → loop
until manual intervention.

These tests pin down:

* Worker / operator-initiated blocks are sticky and survive
  ``recompute_ready``.
* Circuit-breaker blocks (``gave_up`` event, status flipped via
  ``_record_task_failure``) still auto-recover — the original intent
  of #40c1decb3 is preserved.
* An explicit ``kanban_unblock`` clears the sticky state.
* The full block → promote → crash → ``gave_up`` loop is broken after
  this fix: subsequent ticks leave the task blocked.

The tangentially related schema-init ordering bug originally reported
in #28712 (``init_db`` crashing on legacy DBs that pre-dated the
``session_id`` migration) is covered separately by
``test_kanban_db.py::test_connect_migrates_legacy_db_before_optional_column_indexes``,
landed via #28754 / #28781 ahead of this fix.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_dashboard_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_blocked_sticky_test",
        plugin_file,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.router


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def dashboard_client(kanban_home: Path) -> TestClient:
    app = FastAPI()
    app.include_router(_load_dashboard_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


# ---------------------------------------------------------------------------
# Worker-initiated kanban_block must be sticky
# ---------------------------------------------------------------------------


def test_worker_block_is_not_auto_promoted_by_recompute_ready(kanban_home: Path) -> None:
    """A standalone task that a worker explicitly blocks for review
    must stay blocked across an arbitrary number of dispatcher ticks.
    Before #28712's fix, ``recompute_ready`` would silently flip it
    back to ``ready`` on the very next tick."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="needs human review")
        kb.claim_task(conn, tid)
        assert kb.block_task(
            conn, tid,
            reason="review-required: please verify ACL change",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"

        # Hammer the promotion code — exactly the dispatcher loop's
        # behaviour, just compressed in time.
        for _ in range(5):
            promoted = kb.recompute_ready(conn)
            assert promoted == 0, "worker-blocked task must not auto-promote"
            assert kb.get_task(conn, tid).status == "blocked"


def test_initially_blocked_child_requires_explicit_unblock_after_parent_completion(
    kanban_home: Path,
) -> None:
    """A child created as blocked is a human gate, not a dependency wait."""
    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="implementation")
        blocked_child_id = kb.create_task(
            conn,
            title="human approval",
            parents=[parent_id],
            initial_status="blocked",
        )
        ready_child_id = kb.create_task(
            conn,
            title="ordinary dependent",
            parents=[parent_id],
        )

        blocked_child = kb.get_task(conn, blocked_child_id)
        ready_child = kb.get_task(conn, ready_child_id)
        assert blocked_child is not None and blocked_child.status == "blocked"
        assert ready_child is not None and ready_child.status == "todo"

        parent = kb.claim_task(conn, parent_id, claimer="test-parent")
        assert parent is not None
        assert kb.complete_task(
            conn,
            parent_id,
            summary="implementation complete",
            expected_run_id=parent.current_run_id,
        )

        blocked_child = kb.get_task(conn, blocked_child_id)
        ready_child = kb.get_task(conn, ready_child_id)
        assert ready_child is not None and ready_child.status == "ready"
        assert blocked_child is not None and blocked_child.status == "blocked"
        assert kb.claim_task(conn, blocked_child_id, claimer="must-not-claim") is None
        before_unblock_events = {
            row["kind"]
            for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?",
                (blocked_child_id,),
            ).fetchall()
        }
        assert before_unblock_events.isdisjoint({"promoted", "claimed", "spawned"})

        assert kb.unblock_task(conn, blocked_child_id)
        blocked_child = kb.get_task(conn, blocked_child_id)
        assert blocked_child is not None and blocked_child.status == "ready"
        assert kb.claim_task(conn, blocked_child_id, claimer="after-unblock") is not None


@pytest.mark.parametrize("transition", ["promote", "status"])
def test_operator_transition_clears_initial_block_stickiness(
    kanban_home: Path,
    dashboard_client: TestClient,
    transition: str,
) -> None:
    """Audited operator overrides must not poison later breaker recovery."""
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="human gate",
            initial_status="blocked",
        )

        if transition == "promote":
            ok, error = kb.promote_task(conn, task_id, actor="operator")
            assert ok and error is None
        else:
            response = dashboard_client.patch(
                f"/api/plugins/kanban/tasks/{task_id}",
                json={"status": "todo"},
            )
            assert response.status_code == 200, response.text
            assert kb.recompute_ready(conn) == 1

        claimed = kb.claim_task(conn, task_id, claimer="test-worker")
        assert claimed is not None
        assert kb._record_task_failure(
            conn,
            task_id,
            "spawn failed",
            outcome="spawn_failed",
            failure_limit=1,
            release_claim=True,
            end_run=True,
        )
        blocked = kb.get_task(conn, task_id)
        assert blocked is not None and blocked.status == "blocked"

        assert kb.recompute_ready(conn) == 1
        recovered = kb.get_task(conn, task_id)
        assert recovered is not None and recovered.status == "ready"




# ---------------------------------------------------------------------------
# Circuit-breaker blocks still auto-recover (preserve #40c1decb3 intent)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# unblock_task clears the sticky state
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full bug-shaped loop: block → promote → crash → gave_up → next tick
# ---------------------------------------------------------------------------


def test_protocol_violation_loop_is_broken(kanban_home: Path) -> None:
    """Reproduces the exact #28712 loop and asserts the dispatcher
    leaves the task blocked instead of cycling.

    Loop shape from the issue:

    1. Worker calls ``kanban_block`` → status='blocked',
       ``task_runs.outcome='blocked'``, ``blocked`` event.
    2. (Bug) Dispatcher promotes back to ``ready``.
    3. Fresh worker exits cleanly without terminal tool call →
       ``protocol_violation`` event.
    4. ``_record_task_failure(failure_limit=1)`` → ``gave_up`` event,
       status='blocked' again.
    5. (Bug) Dispatcher promotes again → infinite loop.

    With the fix in place, step 2 never happens — the test simulates
    one would-be loop cycle by faking the crash-then-gave_up entries
    that *would* have been written and asserts the *next* tick still
    leaves the task blocked.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="loop reproducer")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid,
            reason="review-required: human eyes please",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"

        # First dispatcher tick — must NOT promote.
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"

        # Simulate the (hypothetical) protocol_violation + gave_up
        # entries that the dispatcher would have written if the bug
        # were still present.  Even with those event rows in place,
        # the worker-initiated ``blocked`` event is the most recent
        # of the ``{blocked, unblocked}`` pair, so the sticky guard
        # still fires.
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'protocol_violation', NULL, ?)",
            (tid, now),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'gave_up', NULL, ?)",
            (tid, now + 1),
        )
        conn.commit()

        # Subsequent ticks must still leave it blocked.
        for _ in range(3):
            promoted = kb.recompute_ready(conn)
            assert promoted == 0
            assert kb.get_task(conn, tid).status == "blocked"


# ---------------------------------------------------------------------------
# Schema-init recovery on legacy DBs is covered by
# tests/hermes_cli/test_kanban_db.py::test_connect_migrates_legacy_db_before_optional_column_indexes
# (landed via #28754 / #28781).  The original PR shipped a duplicate test
# here; dropped during salvage to avoid two assertions of the same contract.
# ---------------------------------------------------------------------------
