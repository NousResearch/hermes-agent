"""The autonomy ledger (earned autonomy, phase 3): what triage said, what people decided.

The counting is checked on hand-built records, so every number has a known answer. Then
the joins are checked end to end — a chat answer recorded by the installed plugin's
``post_approval_response`` hook, and a board answer given through ``decide_work`` — because
a ledger that cannot join a verdict to its answer counts nothing.
"""

from __future__ import annotations

import json

import pytest

from nova.autonomy import ledger
from nova.audit import new_correlation_id

from .test_autonomy_triage import SUPPORT, Scripted, autonomy_block, call, install, plugin

A = "send_external_email"


def ev(kind, detail, *, tenant="acme", agent=SUPPORT, ts="2026-10-02T10:00:00Z", phase="record", corr="c"):
    return {"kind": kind, "detail": detail, "tenant_id": tenant, "subject": agent, "ts": ts,
            "phase": phase, "correlation_id": corr, "event_id": new_correlation_id()}


def triaged(n, verdict="auto_ok", *, tenant="acme", model="jev-1.13.0", ts=None):
    return ev("policy.triage", {"action": A, "verdict": verdict, "proceed": False, "mode": "shadow",
                                "model_version": model, "tool_call_id": f"t{n}", "failed": []},
              tenant=tenant, ts=ts or f"2026-10-02T10:00:{n:02d}Z")


def answered(n, outcome, *, tenant="acme"):
    return ev("policy.approval_outcome", {"action": A, "outcome": outcome, "tool_call_id": f"t{n}"}, tenant=tenant)


def escalated(n, *, tenant="acme"):
    return ev("policy.decision", {"action": A, "effect": "require_approval", "tool_call_id": f"t{n}"}, tenant=tenant)


def test_counts_and_the_two_numbers_that_matter():
    events = []
    for n in range(10):
        events += [escalated(n), triaged(n, "auto_ok" if n < 8 else "escalate")]
    outcomes = ["approved"] * 6 + ["rejected", "timed_out", "approved", "rejected"]
    events += [answered(n, o) for n, o in enumerate(outcomes)]
    book = ledger.build(events, [], tenant_id="acme")["actions"][A]
    t = book.all_time
    assert (t.escalations, t.approved_unchanged, t.rejected, t.timed_out) == (10, 7, 2, 1)
    # Safe verdicts a person reviewed: 0-7 minus the timed-out one = 7; one was rejected.
    assert (t.shadow_reviewed, t.shadow_agreed, t.false_safe) == (7, 6, 1)
    assert t.agreement == pytest.approx(6 / 7) and t.false_safe_rate == pytest.approx(1 / 7)
    assert t.approved_edited is None, "not observable, so not a number"


def test_the_window_is_the_newest_reviewed_safe_verdicts():
    events = [triaged(n) for n in range(6)] + [answered(n, "rejected" if n == 0 else "approved") for n in range(6)]
    book = ledger.build(events, [], tenant_id="acme", window=4)["actions"][A]
    assert book.all_time.false_safe == 1
    assert (book.window.shadow_reviewed, book.window.false_safe) == (4, 0), "the old rejection aged out"


def test_another_tenants_records_are_never_counted():
    mine = [triaged(1), answered(1, "approved")]
    theirs = [triaged(2, tenant="globex"), answered(2, "rejected", tenant="globex"),
              escalated(3, tenant="globex")]
    board = [{"request_id": "ap_x", "tenant_id": "globex", "status": "refused", "agent_id": SUPPORT, "action": A}]
    built = ledger.build(mine + theirs, board, tenant_id="acme")
    t = built["actions"][A].all_time
    assert (t.shadow_reviewed, t.false_safe, t.escalations) == (1, 0, 0)
    assert [d.key for d in built["decisions"]] == ["chat:t1"]


def test_autonomous_calls_and_incidents_are_counted():
    events = [
        ev("policy.autonomous_action", {"action": A}, phase="intent", corr="a1"),
        ev("policy.autonomous_action", {"status": "ok"}, phase="committed", corr="a1"),
        ev("policy.autonomous_action", {"action": A}, phase="intent", corr="a2"),  # never closed
        ev("autonomy.incident", {"action": A, "reason": "wrong customer", "autonomous": True}),
    ]
    book = ledger.build(events, [], tenant_id="acme")["actions"][A]
    assert (book.all_time.autonomous_executed, book.all_time.autonomous_open, book.all_time.incidents) == (1, 1, 1)
    assert book.last_autonomous_rejection["reason"] == "wrong customer"


def test_a_chat_answer_is_joined_to_its_verdict_through_the_installed_hook(tmp_path, monkeypatch):
    home, audit = install(tmp_path, autonomy_block(provider="typesafe", mode="shadow"))
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    directive = call(p, tool_call_id="call-42")
    p.post_approval_response(pattern_key=f"plugin_rule:{directive['rule_key']}", choice="once",
                             tool_call_id="call-42", session_id="s", surface="gateway")
    p.post_approval_response(pattern_key="dangerous:rm -rf", choice="once", tool_call_id="x")  # not NOVA's
    events = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert len([e for e in events if e["kind"] == "policy.approval_outcome"]) == 1
    built = ledger.build(events, [], tenant_id="acme")
    [decision] = built["decisions"]
    assert (decision.verdict, decision.outcome) == ("auto_ok", "approved")
    assert built["actions"][A].all_time.shadow_agreed == 1


def test_a_board_answer_is_joined_to_its_verdict(tmp_path, monkeypatch):
    kb = pytest.importorskip("hermes_cli.kanban_db")
    from hermes_cli import kanban_db_connect as kbc

    from nova.runtime.hermes import HermesRuntime

    home, audit = install(tmp_path, autonomy_block(provider="typesafe", mode="shadow"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    kbc.init_db()
    with kbc.connect_closing() as c:
        task_id = kb.create_task(c, title="reply", assignee=SUPPORT, created_by="nova-supervisor", tenant="acme")
        claimed = kb.claim_task(c, task_id)
    p = plugin(home, monkeypatch)
    p._PROVIDERS["typesafe"] = Scripted()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    call(p)
    runtime = HermesRuntime(home=home, tenant_id="acme")
    runtime.decide_work(task_id, "reject", actor="priya-ops", audit=audit.with_actor("priya-ops"),
                        correlation_id=new_correlation_id(), reason="wrong tone")
    events = [json.loads(line) for line in audit.path.read_text().splitlines()]
    built = ledger.build(events, runtime.approval_outcomes(), tenant_id="acme")
    [decision] = built["decisions"]
    assert decision.key.startswith("board:ap_")
    assert (decision.outcome, decision.decided_by) == ("rejected", "priya-ops")
    assert built["actions"][A].all_time.false_safe == 1
