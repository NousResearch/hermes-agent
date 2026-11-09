"""Classic CLI ``/usage``: the active provider's block first, then the Nous balance when Nous is signed in,
and the ``/topup`` CTA only on a Nous route.

On a Claude subscription (DirectSDK) session the Nous balance and "/topup to add to your balance" printed
right under the Claude plan limits, which read as a Claude balance.
"""
from __future__ import annotations

from tests.hermes_cli.test_cli_status_bar import _make_cli

CLAUDE_LIMITS = ["📈 Claude plan limits", "Current session: 99% remaining (1% used)"]


def _usage(monkeypatch, capsys, provider: str, *, signed_in: bool) -> str:
    cli_obj = _make_cli()
    cli_obj.provider, cli_obj.base_url, cli_obj.api_key = provider, None, None
    cli_obj._print_nous_credits_block = lambda: signed_in and (print("  NOUS-BALANCE-BARS") or True)
    monkeypatch.setattr("agent.account_usage.account_limit_lines",
                        lambda p, **kw: CLAUDE_LIMITS if p != "nous" else [])
    monkeypatch.setattr("agent.account_usage_nous.nous_credits_lines",
                        lambda **kw: ["📈 Nous credits", "Top-up credits: $7.75"] if signed_in else [])
    cli_obj._show_usage()
    return capsys.readouterr().out


def test_claude_with_nous_signed_in_shows_both_blocks_claude_first_and_no_cta(monkeypatch, capsys):
    from agent.account_usage_nous import NOUS_OTHER_ROUTE_HEADING
    out = _usage(monkeypatch, capsys, "claude-subscription-directsdk-experimental", signed_in=True)
    assert out.index("Claude plan limits") < out.index(NOUS_OTHER_ROUTE_HEADING) < out.index("Top-up credits: $7.75")
    assert "/topup" not in out and "/subscription" not in out


def test_claude_without_nous_shows_the_claude_block_only(monkeypatch, capsys):
    out = _usage(monkeypatch, capsys, "claude-subscription-directsdk-experimental", signed_in=False)
    assert "Claude plan limits" in out
    assert "Nous" not in out and "/topup" not in out


def test_a_nous_session_shows_the_nous_block_and_the_cta(monkeypatch, capsys):
    out = _usage(monkeypatch, capsys, "nous", signed_in=True)
    assert "NOUS-BALANCE-BARS" in out and "/topup" in out
