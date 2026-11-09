"""Classic CLI ``/usage`` on a gateway session takes the route decision from the owner.

``GatewayAgent`` keeps its launch ``base_url`` while ``provider`` follows ``/model`` and the
``session.usage`` frames, so a CLI launched on Nous that moved to OpenRouter paired ``openrouter``
with the Nous endpoint: ``is_nous_inference_route`` accepted it and ``/usage`` printed the Nous
balance and ``/topup`` under OpenRouter. The owner resolves the session's route itself
(``session.account_usage``).
"""
from __future__ import annotations

from types import SimpleNamespace

from tests.hermes_cli.test_cli_status_bar import _make_cli

NOUS_URL = "https://inference-api.nousresearch.com/v1"


class _Owner:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def call(self, method, **params):
        self.calls.append(method)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _usage(monkeypatch, capsys, owner) -> str:
    cli_obj = _make_cli()
    cli_obj.session_id = "s"
    cli_obj.provider, cli_obj.base_url, cli_obj.api_key = "openrouter", "https://openrouter.ai/api/v1", None
    # After `/model ... --provider openrouter` on a Nous launch: provider moved, base_url did not.
    cli_obj.agent = SimpleNamespace(provider="openrouter", base_url=NOUS_URL, api_key="", session_api_calls=0)
    cli_obj._print_nous_credits_block = lambda: print("  NOUS-BALANCE") or True
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: owner)
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", lambda *a, **kw: None)
    cli_obj._show_usage()
    return capsys.readouterr().out


def test_the_owners_route_decides_the_nous_block(monkeypatch, capsys):
    owner = _Owner({"lines": ["📈 OpenRouter limits"], "nous_route": False})
    out = _usage(monkeypatch, capsys, owner)
    assert owner.calls == ["session.account_usage"]
    assert "📈 OpenRouter limits" in out
    assert "NOUS-BALANCE" not in out and "/topup" not in out


def test_an_owner_without_the_read_keeps_the_local_answer(monkeypatch, capsys):
    from hermes_cli.gateway_client import GatewayRPCError
    out = _usage(monkeypatch, capsys, _Owner(GatewayRPCError("unknown_method", code=-32601)))
    assert "NOUS-BALANCE" in out  # the pre-read CLI's answer for this pair
