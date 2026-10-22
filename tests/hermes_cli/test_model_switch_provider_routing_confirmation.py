"""Regression coverage for #110784's CLI switch confirmation."""

from types import SimpleNamespace

from hermes_cli.model_switch import ModelSwitchResult


def test_cli_switch_summary_shows_effective_openrouter_pin(monkeypatch):
    import cli as cli_mod
    from hermes_cli.cli_model_switch_mixin import _print_switch_summary

    printed = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda line, *args, **kwargs: printed.append(line))
    monkeypatch.setattr(
        cli_mod,
        "CLI_CONFIG",
        {"provider_routing": {"only": ["together"], "models": {
            "z-ai/glm-5.3:exacto": {"only": ["fireworks"], "data_collection": "deny"},
        }}},
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length", lambda *args, **kwargs: None,
    )
    result = ModelSwitchResult(
        success=True,
        new_model="z-ai/glm-5.3:exacto",
        target_provider="openrouter",
        provider_label="OpenRouter",
    )
    cli = SimpleNamespace(agent=None, base_url="", api_key="")

    _print_switch_summary(cli, result, "old-model", one_turn=False, strict_context=False)

    assert "    Inference provider: Fireworks (pinned)" in printed
    assert "    Data collection: denied" in printed

