"""Block-loop park: human-review lanes stay parked through auto-decomposer.

A valid ``needs_input`` blocker that hits ``BLOCK_RECURRENCE_LIMIT`` must
remain in triage with its original reason/kind/title/body. The auto-decomposer
must not rewrite or promote it. Operator ``admit_block_loop_task`` (CLI
``kanban unblock``) admits exactly one run. Active-PR and goal-mode behavior
is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_specify as spec
from hermes_cli.kanban_db_graph import decompose_triage_task


PR_URL = "https://github.com/Replay-Sales/Andy/pull/5071"
PARK_REASON = f"review-required: exact-head approval for {PR_URL}"
PARK_TITLE = "Land exact-head approval"
PARK_BODY = f"Do not redispatch. Human must approve {PR_URL}."


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claim_ready(conn, tid: str, claimer: str = "worker") -> None:
    task = kb.get_task(conn, tid)
    if task is not None and task.status != "ready":
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer=claimer)
    assert claimed is not None


def _park_needs_input(conn, *, tenant: str | None = None) -> str:
    """Drive a running card to the block-loop park (needs_input x limit)."""
    tid = kb.create_task(
        conn, title=PARK_TITLE, body=PARK_BODY, assignee="worker", tenant=tenant,
    )
    _claim_ready(conn, tid)
    assert kb.block_task(conn, tid, reason=PARK_REASON, kind="needs_input")
    assert kb.get_task(conn, tid).status == "blocked"
    assert kb.unblock_task(conn, tid)
    _claim_ready(conn, tid)
    assert kb.block_task(conn, tid, reason=PARK_REASON, kind="needs_input")
    parked = kb.get_task(conn, tid)
    assert parked.status == "triage"
    assert kb.is_block_loop_parked(parked)
    return tid


def _events(conn, tid: str, kind: str) -> list:
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


# ---------------------------------------------------------------------------
# Park destination
# ---------------------------------------------------------------------------


def test_needs_input_loop_parks_in_triage_with_structured_signal(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _park_needs_input(conn)
        task = kb.get_task(conn, tid)
        loop_events = _events(conn, tid, "block_loop_detected")

    assert task.status == "triage"
    assert task.block_kind == "needs_input"
    assert task.block_recurrences >= kb.BLOCK_RECURRENCE_LIMIT
    assert task.title == PARK_TITLE
    assert task.body == PARK_BODY
    assert loop_events, "expected block_loop_detected"
    payload = loop_events[-1].payload or {}
    assert payload.get("kind") == "needs_input"
    assert payload.get("reason") == PARK_REASON
    assert payload.get("parked") is True
    assert payload.get("recurrences") >= kb.BLOCK_RECURRENCE_LIMIT


def test_specify_and_decompose_refuse_parked_card_preserving_evidence(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _park_needs_input(conn)
        assert kb.specify_triage_task(
            conn, tid, title="rewritten title", body="rewritten body", author="auto-decomposer",
        ) is False
        assert decompose_triage_task(
            conn, tid, root_assignee="worker",
            children=[{"title": "child", "body": "nope", "assignee": "worker", "parents": []}],
            author="auto-decomposer",
        ) is None
        task = kb.get_task(conn, tid)

    assert task is not None
    assert task.status == "triage"
    assert task.title == PARK_TITLE
    assert task.body == PARK_BODY
    assert task.block_kind == "needs_input"

    with kbc.connect_closing() as conn:
        assert _events(conn, tid, "specified") == []
        assert _events(conn, tid, "decomposed") == []
        assert _events(conn, tid, "promoted") == []
        loop = _events(conn, tid, "block_loop_detected")[-1]
        assert (loop.payload or {}).get("reason") == PARK_REASON


def test_list_triage_ids_skips_parked_but_keeps_ordinary_ideas(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        parked = _park_needs_input(conn)
        ordinary = kb.create_task(conn, title="rough idea", body="please flesh out", triage=True)

    assert parked not in spec.list_triage_ids()
    assert parked not in decomp.list_triage_ids()
    assert ordinary in spec.list_triage_ids()
    assert ordinary in decomp.list_triage_ids()


def test_load_triage_task_refuses_parked_without_llm(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _park_needs_input(conn)

    task, reason = spec._load_triage_task(tid)
    assert task is None
    assert "parked after a block loop" in reason


def test_ordinary_triage_still_specifies_and_promotes(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="rough idea", body="one liner", triage=True)
        ok = kb.specify_triage_task(
            conn, tid, title="Do the thing", body="**Goal**\nShip it.", author="specifier",
        )
        task = kb.get_task(conn, tid)

    assert ok is True
    assert task.status == "ready"
    assert task.title == "Do the thing"
    assert "**Goal**" in (task.body or "")


def test_tenant_isolation_of_decomposable_triage_lists(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        parked_a = _park_needs_input(conn, tenant="replay")
        idea_a = kb.create_task(conn, title="replay idea", triage=True, tenant="replay")
        idea_b = kb.create_task(conn, title="other idea", triage=True, tenant="other")
        replay_ids = kb.list_decomposable_triage_ids(conn, tenant="replay")
        other_ids = kb.list_decomposable_triage_ids(conn, tenant="other")

    assert parked_a not in replay_ids
    assert idea_a in replay_ids
    assert idea_b not in replay_ids
    assert idea_b in other_ids
    assert parked_a not in other_ids


# ---------------------------------------------------------------------------
# Dispatcher / auto-decomposer interval
# ---------------------------------------------------------------------------


def test_parked_card_survives_auto_decompose_and_dispatch_interval(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live defect: auto-decomposer specified+promoted a parked card to
    ready, then the dispatcher emitted respawn_guarded(active_pr) forever."""
    import hermes_cli.profiles as profmod
    from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)

    with kbc.connect_closing() as conn:
        tid = _park_needs_input(conn)
        kb.add_comment(conn, tid, author="worker", body=f"Opened {PR_URL}")

    decompose_calls: list[str] = []

    def _explode_if_called(task_id, author=None, timeout=None):
        decompose_calls.append(task_id)
        raise AssertionError(f"auto-decomposer must not touch parked card {task_id}")

    monkeypatch.setattr(decomp, "decompose_task", _explode_if_called)

    dispatcher = _KanbanDispatcher(
        kb,
        _DispatcherSettings(
            interval=60.0, max_spawn=1, max_in_progress=4, failure_limit=5,
            stale_timeout_seconds=0, reconcile_orphans=True, default_assignee=None,
            max_in_progress_per_profile=None,
        ),
    )
    decomposed = dispatcher.auto_decompose_tick(auto_decompose_per_tick=3)
    assert decomposed == 0
    assert decompose_calls == []

    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 99)
        task = kb.get_task(conn, tid)
        guarded = dict(res.respawn_guarded)
        spawned_ids = [s[0] for s in res.spawned]
        kinds = [e.kind for e in kb.list_events(conn, tid)]

    assert task.status == "triage"
    assert task.title == PARK_TITLE
    assert task.body == PARK_BODY
    assert task.block_kind == "needs_input"
    assert tid not in spawned_ids
    assert tid not in guarded
    assert "specified" not in kinds
    assert "respawn_guarded" not in kinds
    assert "block_loop_detected" in kinds


def test_ordinary_triage_is_still_picked_up_by_auto_decompose_list(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        parked = _park_needs_input(conn)
        ordinary = kb.create_task(conn, title="underspecified", triage=True)
        decomposable = decomp.list_triage_ids()

    assert ordinary in decomposable
    assert parked not in decomposable


# ---------------------------------------------------------------------------
# Operator recovery: exactly one parked canary
# ---------------------------------------------------------------------------


def test_operator_admit_promotes_exactly_one_parked_canary(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _park_needs_input(conn)
        original = kb.get_task(conn, tid)
        assert kb.admit_block_loop_task(conn, tid) is True
        admitted = kb.get_task(conn, tid)
        events = _events(conn, tid, "block_loop_admitted")
        assert kb.admit_block_loop_task(conn, tid) is False
        still = kb.get_task(conn, tid)

    assert admitted.status == "ready"
    assert still.status == "ready"
    assert admitted.title == original.title == PARK_TITLE
    assert admitted.body == original.body == PARK_BODY
    assert admitted.block_kind == "needs_input"
    assert admitted.block_recurrences >= kb.BLOCK_RECURRENCE_LIMIT
    assert len(events) == 1
    payload = events[0].payload or {}
    assert payload.get("once") is True
    assert payload.get("kind") == "needs_input"
    assert payload.get("status") == "ready"


def test_cli_unblock_admits_parked_canary(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _park_needs_input(conn)

    out = kc.run_slash(f"unblock {tid}")
    assert "Admitted one run" in out

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.title == PARK_TITLE
        assert _events(conn, tid, "block_loop_admitted")


def test_admitted_canary_reparked_on_same_kind_block(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _park_needs_input(conn)
        assert kb.admit_block_loop_task(conn, tid)
        _claim_ready(conn, tid)
        assert kb.block_task(conn, tid, reason=PARK_REASON, kind="needs_input")
        reparked = kb.get_task(conn, tid)
        assert kb.is_block_loop_parked(reparked)
        assert kb.admit_block_loop_task(conn, tid) is True
        # Second admit in the ready state fails — one run per park.
        assert kb.admit_block_loop_task(conn, tid) is False


def test_admit_does_not_apply_to_ordinary_triage(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="rough idea", triage=True)
        assert kb.admit_block_loop_task(conn, tid) is False
        assert kb.get_task(conn, tid).status == "triage"


# ---------------------------------------------------------------------------
# Holdouts: active-PR guard + goal-mode flag remain intact
# ---------------------------------------------------------------------------


def test_active_pr_guard_still_defers_ready_lane(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    with kbc.connect_closing() as conn:
        ready_id = kb.create_task(conn, title="already PRed", assignee="worker")
        kb.add_comment(conn, ready_id, author="worker", body=f"Opened {PR_URL}")
        assert kbd.check_respawn_guard(conn, ready_id) == "active_pr"
        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 7, dry_run=True)
        spawned_ids = [s[0] for s in res.spawned]
        guarded = dict(res.respawn_guarded)

    assert ready_id not in spawned_ids
    assert guarded.get(ready_id) == "active_pr"


def test_goal_mode_round_trip_and_block_still_work(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="goal card", assignee="worker", goal_mode=True, goal_max_turns=8,
        )
        task = kb.get_task(conn, tid)
        assert task.goal_mode is True
        assert task.goal_max_turns == 8
        _claim_ready(conn, tid)
        assert kb.block_task(conn, tid, reason="need a parent", kind="dependency")
        waiting = kb.get_task(conn, tid)
        assert waiting.status == "todo"
        assert waiting.block_kind == "dependency"
        assert waiting.goal_mode is True
