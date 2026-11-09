"""``/usage`` shows the Nous balance and the ``/topup`` CTA only on a Nous route.

On a Claude subscription (DirectSDK) session the classic CLI printed the Nous Portal balance and
"/topup to add to your balance" right under the Claude plan limits, which read as a Claude balance.
"""
from __future__ import annotations

from tests.hermes_cli.test_cli_status_bar import _make_cli


def _usage(monkeypatch, capsys, provider: str) -> str:
    cli_obj = _make_cli()
    cli_obj.provider, cli_obj.base_url, cli_obj.api_key = provider, None, None
    cli_obj._print_nous_credits_block = lambda: print("  NOUS-BALANCE") or True
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", lambda *a, **kw: None)
    cli_obj._show_usage()
    return capsys.readouterr().out


def test_another_providers_usage_has_no_nous_balance_or_topup_cta(monkeypatch, capsys):
    out = _usage(monkeypatch, capsys, "claude-subscription-directsdk-experimental")
    assert "NOUS-BALANCE" not in out and "/topup" not in out


def test_a_nous_session_keeps_its_balance_and_cta(monkeypatch, capsys):
    out = _usage(monkeypatch, capsys, "nous")
    assert "NOUS-BALANCE" in out and "/topup" in out
