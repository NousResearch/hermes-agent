from __future__ import annotations

from types import SimpleNamespace

from agent import eval_gate


def test_eval_dataset_fixture_requires_exact_policy_review_approval():
    action = {
        "audit_id": "eval-fixture-1",
        "actor": "hermes-agent",
        "delegated_user": "Kihoon",
        "resource": "synthetic_eval_fixtures",
        "scope": "eval dataset/red-team fixtures",
        "intent": "Introduce synthetic, non-operational policy fixtures.",
        "side_effect": "file_write",
        "approval_status": "explicit_approval",
        "approval_evidence": eval_gate.EVAL_DATASET_APPROVAL_PHRASE,
        "policy_review": True,
        "synthetic_only": True,
        "policy_version": eval_gate.POLICY_VERSION,
    }

    decision = eval_gate.evaluate_policy_action(action, surface="sandbox_eval_gate")

    assert decision.actual_decision == "allow"
    assert decision.passed is True


def test_live_gate_connection_requires_exact_approval_without_blocking_audit_mode():
    action = {
        "audit_id": "live-gate-1",
        "actor": "hermes-agent",
        "delegated_user": "Kihoon",
        "resource": "gateway_and_cron_pre_dispatch",
        "scope": "permission-enforcement pre-dispatch",
        "intent": "Connect deterministic eval gate before dispatch.",
        "side_effect": "runtime_behavior_change",
        "approval_status": "required_missing",
        "policy_version": eval_gate.POLICY_VERSION,
    }

    needs_approval = eval_gate.evaluate_policy_action(action, surface="gateway", enforce=True)
    assert needs_approval.actual_decision == "confirm"
    assert needs_approval.should_block is True

    approved = eval_gate.evaluate_policy_action(
        {**action, "approval_evidence": eval_gate.LIVE_GATE_APPROVAL_PHRASE},
        surface="gateway",
        enforce=True,
    )
    assert approved.actual_decision == "allow"
    assert approved.should_block is False


def test_gateway_event_disabled_by_default_allows_dispatch(monkeypatch):
    for key in ("HERMES_EVAL_GATE_ENABLED", "HERMES_EVAL_GATE_GATEWAY_ENABLED"):
        monkeypatch.delenv(key, raising=False)

    source = SimpleNamespace(platform=SimpleNamespace(value="discord"), user_id="u1", chat_id="c1")
    event = SimpleNamespace(text="hello", message_id="m1", source=source)

    decision = eval_gate.evaluate_gateway_event(event, config={})

    assert decision.actual_decision == "allow"
    assert decision.reason == "eval gate disabled"


def test_cron_gate_enforced_with_global_approval_evidence_allows_job(monkeypatch):
    monkeypatch.delenv("HERMES_EVAL_GATE_ENABLED", raising=False)
    job = {"id": "job1", "name": "safe job", "no_agent": True}
    config = {
        "eval_gate": {
            "cron_enabled": True,
            "enforce": True,
            "audit_only": False,
            "approval_evidence": eval_gate.LIVE_GATE_APPROVAL_PHRASE,
        }
    }

    decision = eval_gate.evaluate_cron_job(job, config=config)

    assert decision.actual_decision == "allow"
    assert decision.enforce is True
    assert decision.should_block is False
