"""Account-binding guard for CLI /usage reset.

The destructive /usage reset must only proceed when the active agent's exact
api_key is available. Without it the helper would silently resolve
singleton/pool state and spend a reset belonging to an account the CLI session
never authenticated as in this turn. Bare /usage display fallback is unchanged.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import cli as cli_mod
from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "openai/codex-mini"
    cli_obj.provider = "openai-codex"
    cli_obj.base_url = "https://chatgpt.com/backend-api/codex"
    cli_obj.api_key = None
    cli_obj.agent = None
    return cli_obj


class TestCLIUsageResetAccountBinding:
    def test_reset_refuses_without_agent_api_key(self, monkeypatch, capsys):
        """No active agent, no explicit api_key → fail-closed before helper."""
        cli_obj = _make_cli()
        # provider is openai-codex (from self.provider) but no agent, no api_key.

        def _must_not_redeem(**kw):
            raise AssertionError("redeem must not run without session-bound credential")

        monkeypatch.setattr("agent.account_usage.redeem_codex_reset_credit", _must_not_redeem)

        cli_obj._usage_reset(force=False)
        out = capsys.readouterr().out

        assert "send a message" in out.lower() or "authenticate" in out.lower()

    def test_reset_refuses_with_agent_missing_api_key(self, monkeypatch, capsys):
        """Agent present but api_key None → still fail-closed."""
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key=None,
        )

        def _must_not_redeem(**kw):
            raise AssertionError("must not redeem without agent api_key")

        monkeypatch.setattr("agent.account_usage.redeem_codex_reset_credit", _must_not_redeem)

        cli_obj._usage_reset(force=False)
        out = capsys.readouterr().out

        assert "send a message" in out.lower() or "authenticate" in out.lower()

    def test_reset_forwards_agent_credential(self, monkeypatch, capsys):
        """Active agent with exact api_key → forwarded to helper."""
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="agent-session-token",
        )

        seen = {}

        def fake_redeem(*, base_url=None, api_key=None, force=False):
            seen.update(base_url=base_url, api_key=api_key, force=force)
            from agent.account_usage import CodexResetRedeemResult
            return CodexResetRedeemResult(status="reset", message="✅ redeemed", available_count=1)

        monkeypatch.setattr("agent.account_usage.redeem_codex_reset_credit", fake_redeem)

        cli_obj._usage_reset(force=False)
        out = capsys.readouterr().out

        assert "✅ redeemed" in out
        assert seen["api_key"] == "agent-session-token"
        assert seen["base_url"] == "https://chatgpt.com/backend-api/codex"

    def test_reset_force_does_not_bypass_credential_guard(self, monkeypatch, capsys):
        """--force bypasses exhaustion guard but NOT credential guard."""
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key=None,
        )

        def _must_not_redeem(**kw):
            raise AssertionError("force must not bypass credential guard")

        monkeypatch.setattr("agent.account_usage.redeem_codex_reset_credit", _must_not_redeem)

        cli_obj._usage_reset(force=True)
        out = capsys.readouterr().out

        assert "send a message" in out.lower() or "authenticate" in out.lower()


class TestCLIUsageResetArgParsing:
    """Unknown extra args must NOT silently trigger the destructive action."""

    def test_reset_with_unknown_extra_arg_refused(self, monkeypatch, capsys):
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="tok",
        )

        def _must_not_redeem(**kw):
            raise AssertionError("unknown extra arg must not trigger redeem")

        monkeypatch.setattr("agent.account_usage.redeem_codex_reset_credit", _must_not_redeem)

        cli_obj._handle_usage_command("/usage reset bogus")
        out = capsys.readouterr().out

        assert "unknown" in out.lower() or "try" in out.lower()

    def test_reset_force_with_extra_arg_refused(self, monkeypatch, capsys):
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="tok",
        )

        def _must_not_redeem(**kw):
            raise AssertionError("extra arg after --force must not trigger redeem")

        monkeypatch.setattr("agent.account_usage.redeem_codex_reset_credit", _must_not_redeem)

        cli_obj._handle_usage_command("/usage reset --force bogus")
        out = capsys.readouterr().out

        assert "unknown" in out.lower() or "try" in out.lower()

    def test_bare_usage_display_unchanged_without_agent(self, monkeypatch, capsys):
        """Bare /usage with no agent still shows Nous credits / fallback msg —
        the credential guard is scoped to the destructive reset path only."""
        cli_obj = _make_cli()
        cli_obj.agent = None

        # _print_nous_credits_block returns False → fallback message printed.
        monkeypatch.setattr(cli_obj, "_print_nous_credits_block", lambda: False)

        cli_obj._handle_usage_command("/usage")
        out = capsys.readouterr().out

        assert "send a message first" in out.lower() or "no active agent" in out.lower()
