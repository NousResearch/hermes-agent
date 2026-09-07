"""Human handoffs and root ownership survive automated decomposition."""

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()


@pytest.mark.parametrize("claimed", [False, True])
def test_repeated_human_input_stays_out_of_decomposition(board, claimed):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="Await human approval", assignee="owner")
        for recurrence in range(1, kb.BLOCK_RECURRENCE_LIMIT + 2):
            run_id = None
            if claimed:
                task = kb.claim_task(conn, tid, claimer="owner")
                assert task is not None
                run_id = task.current_run_id
            assert kb.block_task(
                conn, tid, reason="Await human GO", kind="needs_input",
                expected_run_id=run_id,
            )
            task = kb.get_task(conn, tid)
            assert task.status == "blocked"
            assert task.block_kind == "needs_input"
            assert task.block_recurrences == recurrence
            assert task.assignee == "owner"
            assert task.claim_lock is None
            assert tid not in decomp.list_triage_ids()
            assert not decomp.decompose_task(tid).ok
            assert kb.list_events(conn, tid)[-1].kind == "blocked"
            assert kb.unblock_task(conn, tid)


@pytest.mark.parametrize("owner", [None, "owner"])
@pytest.mark.parametrize("fanout", [False, True])
def test_decomposer_preserves_owner_but_explicit_routing_still_works(
    board, monkeypatch, owner, fanout,
):
    routing = decomp._Routing(
        orchestrator="orchestrator", default_assignee="specialist",
        auto_promote=True, roster=[], valid_names={"owner", "orchestrator", "specialist"},
    )
    monkeypatch.setattr(decomp, "_load_routing", lambda: routing)
    payload = {
        "fanout": fanout, "title": "Specified work", "body": "Concrete work",
        "assignee": "specialist",
        "tasks": [{"title": "Implement", "assignee": "specialist"}],
    }
    monkeypatch.setattr(decomp, "_call_aux", lambda *a, **kw: (json.dumps(payload), ""))
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="Work", assignee=owner, triage=True)
    outcome = decomp.decompose_task(tid, author="auto-decomposer")
    assert outcome.ok, outcome.reason
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).assignee == (
            owner or ("orchestrator" if fanout else "specialist")
        )
        for cid in outcome.child_ids or []:
            assert kb.get_task(conn, cid).assignee == "specialist"
        # Explicit DB routing is a deliberate reassignment, not a fallback.
        explicit = kb.create_task(conn, title="Explicit routing", assignee="owner", triage=True)
        assert kb.decompose_triage_task(
            conn, explicit, root_assignee="orchestrator",
            children=[{"title": "Child", "assignee": "specialist"}],
        )
        assert kb.get_task(conn, explicit).assignee == "orchestrator"
