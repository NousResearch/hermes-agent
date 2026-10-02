"""Approval in task work, and the "Always" hole in chat approvals.

A task worker runs unattended, so the runtime's approval prompt refuses every escalation
there before any person sees it. NOVA's plugin asks on the board instead: it files the exact
call, holds the task, and lets that call — those arguments — through once after a person
approves it in the Control Centre. These drive the installed plugin and the control plane's
decisions against a real board, the way the worker and the Work screen do.

In chat, the runtime's prompt offers "always", which saves the escalation's rule key and
from then on lets every call carrying it run before any prompt or hook. The rule key is
therefore unique per call; the tests check that against the runtime's own allowlist.
"""

from __future__ import annotations

import json

import pytest

from nova.apply import apply_bundle
from nova.audit import new_correlation_id

from .test_policy_enforcement import load_installed_plugin

SUPPORT = "customer-support"
REFUND = {"order": "A-1001", "amount": 20}


@pytest.fixture
def applied(bundle, runtime, audit, home, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(home))
    apply_bundle(bundle, runtime, audit=audit)
    return home


# -- chat: the "Always" hole ---------------------------------------------------


def chat_plugin(home, monkeypatch, name):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    return load_installed_plugin(home, SUPPORT, name)


def test_each_escalation_has_its_own_rule_key(applied, monkeypatch):
    plugin = chat_plugin(applied, monkeypatch, "nova_always_1")
    first = plugin.pre_tool_call(tool_name="crm_refund", args=REFUND, tool_call_id="call_1")
    again = plugin.pre_tool_call(tool_name="crm_refund", args=REFUND, tool_call_id="call_2")
    unnamed = [plugin.pre_tool_call(tool_name="crm_refund", args=REFUND)["rule_key"] for _ in range(2)]
    assert first["action"] == "approve"
    assert first["rule_key"] == "nova:refund:call_1"
    assert again["rule_key"] == "nova:refund:call_2"
    assert unnamed[0] != unnamed[1], "without a call id the key must still never repeat"


def test_always_on_one_call_does_not_approve_the_next(applied, monkeypatch):
    """What "always" writes, checked with the runtime's own allowlist lookup."""
    approval = pytest.importorskip("tools.approval")
    plugin = chat_plugin(applied, monkeypatch, "nova_always_2")
    first = plugin.pre_tool_call(tool_name="crm_refund", args=REFUND, tool_call_id="call_1")
    later = plugin.pre_tool_call(tool_name="crm_refund", args=REFUND, tool_call_id="call_9")
    saved = f"plugin_rule:{first['rule_key']}"
    approval.approve_permanent(saved)
    try:
        assert approval.is_approved("chat", saved), "the runtime treats the saved key as approved"
        assert not approval.is_approved("chat", f"plugin_rule:{later['rule_key']}"), (
            "a later refund must be asked about again, not let through by an earlier 'always'"
        )
    finally:
        approval._permanent_set().discard(saved)


def test_an_escalation_record_carries_the_call_ids(applied, monkeypatch, tmp_path):
    plugin = chat_plugin(applied, monkeypatch, "nova_always_3")
    plugin.pre_tool_call(tool_name="crm_refund", args=REFUND, tool_call_id="call_7", session_id="s-1")
    log = json.loads((applied / "profiles" / SUPPORT / "nova-policy.json").read_text())["audit_log"]
    records = [json.loads(line) for line in open(log, encoding="utf-8") if line.strip()]
    detail = [r["detail"] for r in records if r["kind"] == "policy.decision"][-1]
    assert detail["tool_call_id"] == "call_7" and detail["session_id"] == "s-1"


# -- task work: asking on the board --------------------------------------------


@pytest.fixture
def board(applied, monkeypatch):
    kb = pytest.importorskip("hermes_cli.kanban_db")
    from hermes_cli import kanban_db_connect as kbc

    monkeypatch.setenv("HERMES_KANBAN_DB", str(applied / "kanban.db"))
    kbc.init_db()
    with kbc.connect_closing() as c:
        task_id = kb.create_task(c, title="refund A-1001", assignee=SUPPORT,
                                 created_by="nova-supervisor", tenant="acme")
    return task_id


class Worker:
    """One worker process: a fresh plugin load and a fresh claim, as the dispatcher spawns."""

    count = 0

    def __init__(self, home, task_id, monkeypatch):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        with kbc.connect_closing() as c:
            claimed = kb.claim_task(c, task_id)
        assert claimed is not None, f"task {task_id} was not claimable"
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
        Worker.count += 1
        self.plugin = load_installed_plugin(home, SUPPORT, f"nova_task_worker_{Worker.count}")

    def call(self, tool="crm_refund", args=None):
        return self.plugin.pre_tool_call(tool_name=tool, args=dict(REFUND if args is None else args))


def status(task_id):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as c:
        return kb.get_task(c, task_id).status


def comments(task_id):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as c:
        return [(x.author, x.body) for x in kb.list_comments(c, task_id)]


def decide(runtime, audit, task_id, action, actor="priya-ops", **kwargs):
    return runtime.decide_work(task_id, action, actor=actor, audit=audit.with_actor(actor),
                               correlation_id=new_correlation_id(), **kwargs)


def test_an_escalated_call_in_a_task_holds_the_task_and_files_the_exact_call(applied, board, monkeypatch, runtime):
    worker = Worker(applied, board, monkeypatch)
    directive = worker.call()
    assert directive["action"] == "block" and "HELD by NOVA policy" in directive["message"]
    assert status(board) == "blocked"
    # Nothing else runs in this process once the task is waiting for a person.
    assert worker.call(tool="crm_lookup", args={})["action"] == "block"

    view = runtime.get_task(board)
    held = view.detail["approval"]
    assert held["tool"] == "crm_refund" and held["action"] == "refund"
    assert '"A-1001"' in held["arguments"]
    assert view.detail["block_reason"].startswith(f"NOVA approval needed [{held['request_id']}]")


def test_the_work_screen_shows_a_held_call_as_an_approval(applied, board, monkeypatch, runtime, bundle):
    from nova.control import ControlAPI

    Worker(applied, board, monkeypatch).call()
    rows = ControlAPI(bundle, runtime).handle("/platform/v1/tasks").body["tasks"]
    row = next(r for r in rows if r["task_id"] == board)
    assert row["attention_kind"] == "approval"
    assert row["approval"]["tool"] == "crm_refund"
    assert row["error_summary"]["headline"] == "Waiting for your approval"


def test_approval_lets_that_exact_call_through_once(applied, board, monkeypatch, runtime, audit):
    Worker(applied, board, monkeypatch).call()

    resumed = decide(runtime, audit, board, "resume")
    assert not resumed.applied and "release" in resumed.reason, "resuming would only hold it again"

    approved = decide(runtime, audit, board, "release")
    assert approved.applied and approved.resulting_status == "ready"
    assert any(author == "priya-ops" and "APPROVED by priya-ops" in body for author, body in comments(board))

    worker = Worker(applied, board, monkeypatch)
    assert worker.call(args={"order": "A-1001", "amount": 2000})["action"] == "block", (
        "different arguments are a different call, and are asked about again"
    )


def test_a_grant_is_spent_by_one_call(applied, board, monkeypatch, runtime, audit):
    Worker(applied, board, monkeypatch).call()
    decide(runtime, audit, board, "release")

    worker = Worker(applied, board, monkeypatch)
    assert worker.call() is None, "the approved call runs"
    assert status(board) == "running"
    again = worker.call()
    assert again["action"] == "block" and "HELD" in again["message"], "a second refund is asked about again"


def test_a_refusal_needs_a_reason_and_the_worker_reads_it(applied, board, monkeypatch, runtime, audit):
    Worker(applied, board, monkeypatch).call()

    assert not decide(runtime, audit, board, "reject").applied
    refused = decide(runtime, audit, board, "reject", reason="order already refunded")
    assert refused.applied and refused.resulting_status == "ready"

    worker = Worker(applied, board, monkeypatch)
    directive = worker.call()
    assert directive["action"] == "block"
    assert "refused this exact call" in directive["message"]
    assert "order already refunded" in directive["message"]
    assert status(board) == "running", "a refusal does not hold the task again"
    assert worker.call(tool="crm_lookup", args={}) is None, "the rest of the task carries on"


def test_a_second_hold_in_one_task_is_still_answerable(applied, board, monkeypatch, runtime, audit):
    """The runtime's loop breaker sends a second same-kind block to triage."""
    Worker(applied, board, monkeypatch).call()
    decide(runtime, audit, board, "release")
    worker = Worker(applied, board, monkeypatch)
    assert worker.call() is None
    worker.call(args={"order": "A-2002", "amount": 5})

    view = runtime.get_task(board)
    assert view.state == "blocked", f"held for a person, whatever the runtime calls it ({view.runtime_status})"
    assert view.detail["approval"]["arguments"].count("A-2002") == 1
    assert decide(runtime, audit, board, "release").applied
    assert status(board) == "ready"


def test_the_decision_is_audited_under_the_person(applied, board, monkeypatch, runtime, audit):
    Worker(applied, board, monkeypatch).call()
    decide(runtime, audit, board, "release", actor="sam-lead")
    records = [json.loads(line) for line in audit.path.read_text().splitlines() if line.strip()]
    decided = [r for r in records if r["kind"] == "work.decided"][-1]
    assert decided["actor"] == "sam-lead"
    assert "approved crm_refund" in decided["detail"]["reason"]


def test_an_unwritable_store_refuses_the_call(applied, board, monkeypatch):
    (applied / "nova-approvals").write_text("not a directory")
    directive = Worker(applied, board, monkeypatch).call()
    assert directive["action"] == "block" and "could not be filed" in directive["message"]


def test_a_task_id_cannot_point_the_store_elsewhere():
    from nova.runtime.hermes.enforcement import request_path

    root = __import__("pathlib").Path("/x")
    assert request_path(root, "../etc", "ap_" + "0" * 20) is None
    assert request_path(root, "t_1", "../../passwd") is None
    assert request_path(root, "t_1", "ap_" + "a" * 20) == root / "t_1" / ("ap_" + "a" * 20 + ".json")
