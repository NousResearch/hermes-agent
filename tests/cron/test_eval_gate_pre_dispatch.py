from __future__ import annotations

from agent.eval_gate import EvalGateDecision
from cron import scheduler


def test_cron_eval_gate_skip_short_circuits_plugin(monkeypatch):
    plugin_called = {"value": False}

    def _fake_eval(job, config):
        return EvalGateDecision(
            surface="cron",
            audit_id="cron:test",
            actual_decision="reject",
            enforce=True,
            passed=False,
            reason="test cron block",
            action={},
        )

    def _fake_hook(name, **kwargs):
        plugin_called["value"] = True
        return []

    monkeypatch.setattr("agent.eval_gate.evaluate_cron_job", _fake_eval)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    result = scheduler._evaluate_cron_pre_dispatch_gate({"id": "job1", "name": "blocked"})

    assert result["action"] == "skip"
    assert result["reason"] == "test cron block"
    assert plugin_called["value"] is False


def test_cron_plugin_pre_dispatch_skip_after_eval_allows(monkeypatch):
    def _fake_eval(job, config):
        return EvalGateDecision(
            surface="cron",
            audit_id="cron:test",
            actual_decision="allow",
            enforce=False,
            passed=True,
            reason="ok",
            action={},
        )

    def _fake_hook(name, **kwargs):
        assert name == "pre_cron_dispatch"
        assert kwargs["job"]["id"] == "job1"
        return [{"action": "skip", "reason": "plugin block"}]

    monkeypatch.setattr("agent.eval_gate.evaluate_cron_job", _fake_eval)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    result = scheduler._evaluate_cron_pre_dispatch_gate({"id": "job1", "name": "blocked"})

    assert result == {"action": "skip", "reason": "plugin block"}
