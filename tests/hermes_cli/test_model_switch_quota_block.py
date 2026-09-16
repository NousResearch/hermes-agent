"""The CLI ``/model`` switch summary must show the NEW route's remaining quota / balance.

Both CLI switch paths — typed ``/model <name>`` and the picker — print through
``hermes_cli.cli_model_switch_mixin._print_switch_summary``, so that is the single CLI place the
block is wired in. Invariants pinned here:

  * the block is fetched for the route the session switched TO (provider + credentials of the
    pick), never the ambient one;
  * a provider without a limits API contributes NOTHING — no blank line, no empty header.
"""

from types import SimpleNamespace

import hermes_cli.cli_model_switch_mixin as switch_mixin


class _StubCLI:
    """Bare CLI stub: ``_print_switch_summary`` reads only the live route + agent."""

    model = "old/model"
    provider = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    api_key = "sk-old"
    agent = None
    _pending_model_switch_note = None


def _result(**overrides) -> SimpleNamespace:
    fields = dict(
        new_model="deepseek-v4-pro", target_provider="deepseek", provider_label="DeepSeek",
        base_url="https://api.deepseek.com/v1", api_key="sk-deepseek", api_mode="chat_completions",
        model_info=None, warning_message=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _print_summary(cli, result) -> str:
    import cli as cli_module
    printed: list[str] = []
    original = cli_module._cprint
    cli_module._cprint = lambda text: printed.append(text)
    try:
        switch_mixin._print_switch_summary(cli, result, cli.model, one_turn=False, strict_context=False)
    finally:
        cli_module._cprint = original
    return "\n".join(printed)


def test_cli_switch_summary_prints_the_new_routes_quota(monkeypatch):
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length", lambda *a, **k: None)
    seen = {}

    def _lines(provider, **kwargs):
        seen.update(provider=provider, **kwargs)
        return ["📈 DeepSeek balance", "Balance: 110.00 CNY"]

    monkeypatch.setattr("agent.account_usage.account_usage_lines", _lines)

    out = _print_summary(_StubCLI(), _result())

    assert seen["provider"] == "deepseek"
    assert seen["base_url"] == "https://api.deepseek.com/v1"
    assert seen["api_key"] == "sk-deepseek"
    assert "Model switched: deepseek-v4-pro" in out
    assert "DeepSeek balance" in out
    assert "Balance: 110.00 CNY" in out
    # The balance block belongs to the switch summary, under the route metadata.
    assert out.index("Provider: DeepSeek") < out.index("Balance: 110.00 CNY")


def test_cli_switch_summary_omits_the_block_for_a_route_without_a_limits_api(monkeypatch):
    """``ollama`` has no usage fetcher: the real chain contributes no lines and no blank divider."""
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length", lambda *a, **k: None)

    out = _print_summary(
        _StubCLI(), _result(new_model="llama3.1:8b", target_provider="ollama", provider_label="Ollama",
                            base_url="http://localhost:11434", api_key=""),
    )

    assert "Model switched: llama3.1:8b" in out
    assert "Balance" not in out
    assert "\n\n" not in out
