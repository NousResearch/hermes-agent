"""Human decisions applied to a real runtime board.

These write to a real ``kanban.db`` through the runtime's own API rather than a fake, for
the reason the submission tests give and one more: **both defects this phase shipped with
were invisible to a fake.** A stub runtime returns whatever the stub was written to return,
so a stub cannot tell you that ``request_changes`` refuses a task nobody has claimed, or
that the audit names the process instead of the person. Only the live board said so.

They skip when the runtime is not importable, which is the normal state of NOVA's own test
environment: the layer is meant to install without it.
"""

from __future__ import annotations

import pytest

from nova.apply import apply_bundle
from nova.audit import AuditLog, new_correlation_id
from nova.spec import load_bundle

from .conftest import EXAMPLE_BUNDLE


@pytest.fixture(scope="module", autouse=True)
def _requires_runtime():
    pytest.importorskip(
        "hermes_cli.kanban_db", reason="work decisions need the runtime's work-store API"
    )


@pytest.fixture
def board(tmp_path, monkeypatch):
    """An applied bundle with one objective's steps on a fresh board."""
    from nova.runtime.hermes import HermesRuntime
    from nova.supervisor import submit_objective

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("NOVA_HOME", str(home))

    bundle = load_bundle(EXAMPLE_BUNDLE)
    runtime = HermesRuntime(home=home, tenant_id=bundle.tenant_id)
    audit = AuditLog(home / "audit.jsonl", tenant_id=bundle.tenant_id, actor="nova-control")
    apply_bundle(bundle, runtime, audit=audit)
    report = submit_objective(
        bundle.objectives[0], bundle.agents, runtime,
        audit=audit, tenant_id=bundle.tenant_id,
    )
    return bundle, runtime, audit, report


def decide(board, task_id, action, *, actor="priya-ops", **kwargs):
    _, runtime, audit, _ = board
    return runtime.decide_work(
        task_id, action, actor=actor, audit=audit.with_actor(actor),
        correlation_id=new_correlation_id(), **kwargs
    )


def task_ids(board):
    _, runtime, _, _ = board
    return {t.title: t.task_id for t in runtime.list_tasks(limit=100)}


def status_of(board, task_id):
    _, runtime, _, _ = board
    view = runtime.get_task(task_id)
    return view.runtime_status if view else ""


def comments(board, task_id):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc

    with kbc.connect_closing() as connection:
        return [(c.author, c.body) for c in kb.list_comments(connection, task_id)]


# -- the runtime refuses what would be inconsistent ---------------------------


def test_releasing_a_task_with_unfinished_parents_is_refused_with_the_reason(board):
    """The whole argument for asking the runtime for a transition rather than writing a
    status: a control plane that can write any state can write an inconsistent one."""
    _, runtime, _, report = board
    dependent = [i for i in report.result.items if runtime.get_task(i.task_id).runtime_status == "todo"]
    assert dependent, "the example objective should have a step that waits on another"

    decision = decide(board, dependent[0].task_id, "release")
    assert decision.applied is False
    assert "parent" in decision.reason.lower()
    assert status_of(board, dependent[0].task_id) == "todo"


# -- what only a live board could show ----------------------------------------


def test_a_human_can_reject_a_review_nobody_has_claimed(board):
    """The first live defect.

    ``request_changes`` closes an *active reviewer run* — a reviewer agent claimed the task
    and is handing it back. A human looking at a queue of tasks awaiting review has claimed
    nothing, and that call refuses. The operator's rejection is a different primitive, and
    a stub runtime would have happily reported success for either.
    """
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc

    _, runtime, _, report = board
    task_id = report.result.items[0].task_id
    with kbc.connect_closing() as connection:
        assert kb.request_review(connection, task_id, summary="done", force=True)
    assert status_of(board, task_id) == "review"

    decision = decide(board, task_id, "reject", reason="the threshold is GBP 25, not GBP 50")
    assert decision.applied is True, decision.reason
    assert status_of(board, task_id) != "review"

    bodies = [body for _, body in comments(board, task_id)]
    assert any("CHANGES REQUESTED" in body for body in bodies)
    assert any("GBP 25" in body for body in bodies), "the reason is what the worker re-runs against"


def test_the_audit_names_the_person_not_the_process(board):
    """The second live defect. The human was in ``detail`` and the top-level ``actor`` said
    ``nova-control``, so an auditor filtering on ``actor`` would have seen none of them."""
    _, _, audit, report = board
    task_id = report.result.items[0].task_id
    decide(board, task_id, "annotate", actor="priya-ops", note="use the Q3 export")

    events = [e for e in audit.read() if e.kind == "work.annotated"]
    assert events, "an annotation must be logged"
    assert {e.actor for e in events} == {"priya-ops"}


# -- the decisions themselves -------------------------------------------------


def test_a_note_reaches_the_board_attributed_to_its_author(board):
    _, _, _, report = board
    task_id = report.result.items[0].task_id
    assert decide(board, task_id, "annotate", note="use the Q3 export").applied

    assert ("priya-ops", "OPERATOR: use the Q3 export") in comments(board, task_id)


def test_a_blocked_task_can_be_resumed(board):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc

    _, _, _, report = board
    task_id = report.result.items[0].task_id
    with kbc.connect_closing() as connection:
        assert kb.block_task(connection, task_id, reason="source file unreadable")
    assert status_of(board, task_id) == "blocked"

    decision = decide(board, task_id, "resume")
    assert decision.applied is True
    assert status_of(board, task_id) == "ready"


def test_a_decision_about_a_task_that_does_not_exist_is_refused_not_raised(board):
    decision = decide(board, "t_does_not_exist", "resume")
    assert decision.applied is False
    assert "no such work item" in decision.reason


def test_an_anonymous_decision_is_refused(board):
    """A decision recorded against 'someone' is the exact failure identity exists to
    prevent, so it is refused rather than defaulted — twice over, as it turns out.

    The adapter refuses an empty actor, and the audit log refuses to be written as one, so
    a caller that skipped the first check still cannot produce an unattributed record. Both
    are asserted because they are independent: either could be removed by a future edit
    that looked reasonable on its own.
    """
    from nova.errors import AuditError, RuntimeAdapterError

    _, runtime, audit, report = board
    task_id = report.result.items[0].task_id

    with pytest.raises(RuntimeAdapterError, match="anonymous"):
        runtime.decide_work(
            task_id, "resume", actor="", audit=audit, correlation_id=new_correlation_id()
        )

    with pytest.raises(AuditError, match="cannot be empty"):
        audit.with_actor("")


def test_an_unknown_action_is_refused_before_the_board_is_opened(board):
    from nova.errors import RuntimeAdapterError

    _, _, _, report = board
    with pytest.raises(RuntimeAdapterError, match="not a work action"):
        decide(board, report.result.items[0].task_id, "detonate")
