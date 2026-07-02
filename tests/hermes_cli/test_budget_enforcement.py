"""Unit tests for the on_budget_check enforcement contract (PR1: contract + soft path)."""

from __future__ import annotations

from hermes_cli.plugins import VALID_HOOKS


def test_on_budget_check_is_a_valid_hook():
    assert "on_budget_check" in VALID_HOOKS


class TestGetBudgetCheckVerdict:
    """on_budget_check verdict aggregation — mirrors the pre_tool_call block path."""

    def test_soft_verdict_returned(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kw: [{"status": "soft", "message": "[BUDGET] 82%"}],
        )
        from hermes_cli.plugins import get_budget_check_verdict

        v = get_budget_check_verdict(session_id="s", scope_hint="global")
        assert v == {"status": "soft", "message": "[BUDGET] 82%"}

    def test_most_severe_wins(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kw: [
                {"status": "ok"},
                {"status": "hard", "message": "over"},
                {"status": "soft", "message": "near"},
            ],
        )
        from hermes_cli.plugins import get_budget_check_verdict

        assert get_budget_check_verdict()["status"] == "hard"

    def test_malformed_results_ignored(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kw: [
                "hard",                       # not a dict
                123,                          # not a dict
                {"message": "no status"},     # missing status
                {"status": "explode"},        # invalid status value
            ],
        )
        from hermes_cli.plugins import get_budget_check_verdict

        assert get_budget_check_verdict() is None

    def test_none_when_no_hooks(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kw: [],
        )
        from hermes_cli.plugins import get_budget_check_verdict

        assert get_budget_check_verdict() is None


class TestBootstrapNotice:
    def test_notice_when_no_hook_registered(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
        from hermes_cli.plugins import (
            BUDGET_ENFORCEMENT_BOOTSTRAP_NOTICE,
            budget_enforcement_bootstrap_notice,
        )

        assert budget_enforcement_bootstrap_notice() == BUDGET_ENFORCEMENT_BOOTSTRAP_NOTICE

    def test_none_when_hook_registered(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        from hermes_cli.plugins import budget_enforcement_bootstrap_notice

        assert budget_enforcement_bootstrap_notice() is None
