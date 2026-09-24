"""``delegation.may_assign_to``, enforced inside the worker — on every route, not only NOVA's.

Until this, the declaration was enforced only on work NOVA's own supervisor submitted
(docs/platform/PHASE_5.md, "The boundary of that claim"). An agent creating a task for
another agent, and the runtime's decomposer routing a triage card's children, both put work
on any queue on the host. These tests run the *installed* policy plugin against a *real*
board written through the runtime's own API, because "who put this task here" is a fact
about the runtime's rows, and a fake board would assert my reading of them.
"""

from __future__ import annotations

import json

import pytest

from nova.apply import apply_bundle
from nova.policy.decide import (
    ALLOW,
    DENY,
    SUPERVISOR,
    decide_acceptance,
    decide_assignment,
)

from .test_policy_enforcement import load_installed_plugin

SUPPORT, OPS, VARIANT = "customer-support", "operations", "operations__acme-support-telegram"


# -- the decisions, pure --------------------------------------------------------


def _policy(agent, may=()):
    return {"agent_id": agent, "may_assign_to": list(may)}


@pytest.mark.parametrize(
    "assignee, effect",
    [(SUPPORT, ALLOW), (OPS, ALLOW), ("", ALLOW), (VARIANT, DENY), ("finance", DENY)],
)
def test_an_agent_may_create_work_only_for_itself_and_whom_it_declares(assignee, effect):
    assert decide_assignment(_policy(SUPPORT, [OPS]), assignee).effect == effect


def test_a_refused_assignment_names_the_fix():
    reason = decide_assignment(_policy(SUPPORT, [OPS]), "finance").reason
    assert "delegation.may_assign_to" in reason and "customer-support" in reason


@pytest.mark.parametrize(
    "authorizer, authorizer_policy, via_decomposer, effect, rule",
    [
        (SUPERVISOR, None, False, ALLOW, "supervisor-routed"),
        (OPS, _policy(OPS), False, ALLOW, "self-assigned"),
        (SUPPORT, _policy(SUPPORT, [OPS]), False, ALLOW, "delegation-declared"),
        (SUPPORT, _policy(SUPPORT, []), False, DENY, "delegation-not-declared"),
        (SUPPORT, _policy(SUPPORT, []), True, DENY, "delegation-not-declared"),
        ("alice", None, True, DENY, "decomposer-unowned"),
        ("", None, True, DENY, "decomposer-unowned"),
        ("alice", None, False, ALLOW, "operator-directed"),
    ],
)
def test_who_may_put_work_on_an_agents_queue(authorizer, authorizer_policy, via_decomposer, effect, rule):
    decision = decide_acceptance(
        _policy(OPS), authorizer=authorizer, authorizer_policy=authorizer_policy,
        via_decomposer=via_decomposer,
    )
    assert (decision.effect, decision.rule) == (effect, rule)


def test_the_compiled_policy_carries_the_declaration(bundle):
    from nova.policy import compile_policy

    document = compile_policy(bundle.agent(SUPPORT), bundle.policy).document
    assert document["may_assign_to"] == [OPS]


# -- the installed plugin, on a real board ---------------------------------------


@pytest.fixture
def board(bundle, runtime, audit, home, monkeypatch):
    kb = pytest.importorskip("hermes_cli.kanban_db", reason="reads the runtime's own board")
    from hermes_cli import kanban_db_connect as kbc

    monkeypatch.setenv("HERMES_HOME", str(home))
    apply_bundle(bundle, runtime, audit=audit)
    db = kb.kanban_db_path()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    kbc.init_db()

    class Board:
        def create(self, **kwargs):
            with kbc.connect_closing() as connection:
                return kb.create_task(connection, title=kwargs.pop("title", "work"), tenant="acme", **kwargs)

        def decompose(self, root, owner, children):
            from hermes_cli.kanban_db_graph import decompose_triage_task

            with kbc.connect_closing() as connection:
                return decompose_triage_task(
                    connection, root, root_assignee=owner, author="auto-decomposer",
                    children=[{"title": f"step {i}", "assignee": a} for i, a in enumerate(children)],
                )

        def task(self, task_id):
            with kbc.connect_closing() as connection:
                return kb.get_task(connection, task_id)

    return Board()


def worker(home, agent, task_id, monkeypatch):
    """The installed plugin, as a worker spawned for ``task_id`` would load it."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    return load_installed_plugin(home, agent, f"nova_policy_{agent.replace('-', '_')}")


def refused(plugin, tool="read_file"):
    result = plugin.pre_tool_call(tool_name=tool, args={})
    return bool(result and result.get("action") == "block" and "delegation" in result["message"])


def test_work_a_declared_delegator_handed_over_is_done(board, home, monkeypatch):
    task = board.create(assignee=OPS, created_by=SUPPORT)
    assert not refused(worker(home, OPS, task, monkeypatch), "kanban_complete")


def test_work_an_undeclared_delegator_handed_over_is_refused(board, home, monkeypatch):
    """customer-support declares operations, not the Telegram-scoped operations profile."""
    task = board.create(assignee=VARIANT, created_by=SUPPORT)
    plugin = worker(home, VARIANT, task, monkeypatch)
    assert refused(plugin)
    assert refused(plugin, "kanban_complete"), "refused work must not be reported as done"
    assert plugin.pre_tool_call(tool_name="kanban_block", args={}) is None, "it must be able to say why"


def test_a_refused_task_is_blocked_on_the_board_with_the_reason(board, home, monkeypatch):
    task = board.create(assignee=VARIANT, created_by=SUPPORT)
    worker(home, VARIANT, task, monkeypatch).on_session_start()
    row = board.task(task)
    assert row.status == "blocked"
    with __import__("hermes_cli.kanban_db_connect", fromlist=["x"]).connect_closing() as c:
        payload = c.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='blocked'", (task,)
        ).fetchone()[0]
    assert "may_assign_to" in json.loads(payload)["reason"]


def test_the_decomposer_is_held_to_the_card_owners_declaration(board, home, monkeypatch):
    root = board.create(assignee=SUPPORT, created_by="alice", triage=True)
    allowed, not_allowed = board.decompose(root, SUPPORT, [OPS, VARIANT])
    assert not refused(worker(home, OPS, allowed, monkeypatch))
    assert refused(worker(home, VARIANT, not_allowed, monkeypatch))


def test_the_decomposer_routing_a_card_no_agent_owns_is_refused(board, home, monkeypatch):
    root = board.create(created_by="alice", triage=True)
    (child,) = board.decompose(root, None, [OPS])
    assert refused(worker(home, OPS, child, monkeypatch))


def test_supervisor_and_operator_work_is_unaffected(board, home, monkeypatch):
    for creator in (SUPERVISOR, "alice"):
        task = board.create(assignee=OPS, created_by=creator)
        assert not refused(worker(home, OPS, task, monkeypatch)), creator


def test_an_unreadable_board_refuses_rather_than_assumes(board, home, monkeypatch, tmp_path):
    task = board.create(assignee=OPS, created_by=SUPPORT)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "missing.db"))
    assert refused(worker(home, OPS, task, monkeypatch))


def test_a_creator_name_cannot_point_outside_the_profiles(board, home, monkeypatch):
    """The creator is read from the board and used to find a policy file."""
    task = board.create(assignee=OPS, created_by="../../etc")
    plugin = worker(home, OPS, task, monkeypatch)
    assert plugin._sibling_policy("../../etc") is None


def test_the_tool_that_creates_work_is_held_to_the_declaration(board, home, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    plugin = load_installed_plugin(home, SUPPORT, "nova_policy_support_create")
    # The example grants no agent the tool, and its allow-list would refuse it first; grant
    # it here so what is under test is the delegation check, not the allow-list.
    plugin._load_policy()["allow"].append("kanban_create")
    ok = plugin.pre_tool_call(tool_name="kanban_create", args={"title": "x", "assignee": OPS})
    blocked = plugin.pre_tool_call(tool_name="kanban_create", args={"title": "x", "assignee": VARIANT})
    assert not (ok and ok.get("action") == "block")
    assert blocked and blocked["action"] == "block" and "may_assign_to" in blocked["message"]
