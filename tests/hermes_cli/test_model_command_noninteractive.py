from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from hermes_cli.model_switch import ModelSwitchResult
from hermes_cli.subcommands.model import build_model_parser


def test_verified_noninteractive_selection_skips_tty_and_persists(monkeypatch, capsys):
    import cli
    from hermes_cli import config, main, model_switch

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_model_parser(subparsers, cmd_model=main.cmd_model)
    args = parser.parse_args(
        ["model", "--provider", "anthropic", "--model", "claude-sonnet-4-6"])
    result = ModelSwitchResult(
        success=True, new_model="claude-sonnet-4-6", target_provider="anthropic",
        base_url="https://api.anthropic.com", api_mode="anthropic_messages",
        provider_label="Anthropic", is_global=True, model_verified=True)
    switch_calls, writes = [], []
    monkeypatch.setattr(
        main, "_require_tty",
        lambda *_: pytest.fail("flag-driven selection must not require a TTY"))
    monkeypatch.setattr(
        config, "load_config",
        lambda: {"model": {"default": "old", "provider": "openrouter"}})
    monkeypatch.setattr(config, "get_compatible_custom_providers", lambda _: [])
    monkeypatch.setattr(
        model_switch, "switch_model",
        lambda **kwargs: switch_calls.append(kwargs) or result)
    monkeypatch.setattr(cli, "save_config_value", lambda *values: writes.append(values))

    args.func(args)

    assert switch_calls[0]["explicit_provider"] == "anthropic"
    assert switch_calls[0]["raw_input"] == "claude-sonnet-4-6"
    assert writes == [
        ("model.default", "claude-sonnet-4-6"),
        ("model.provider", "anthropic"),
        ("model.base_url", "https://api.anthropic.com"),
        ("model.api_mode", "anthropic_messages"),
    ]
    assert "Default model set" in capsys.readouterr().out


def test_noninteractive_selection_fails_closed_without_writing(monkeypatch, capsys):
    import cli
    from hermes_cli import config, main, model_switch

    writes = []
    monkeypatch.setattr(config, "load_config", lambda: {})
    monkeypatch.setattr(config, "get_compatible_custom_providers", lambda _: [])
    monkeypatch.setattr(
        model_switch, "switch_model",
        lambda **_: ModelSwitchResult(
            success=True, new_model="typo-model", target_provider="anthropic",
            model_verified=False))
    monkeypatch.setattr(cli, "save_config_value", lambda *values: writes.append(values))

    with pytest.raises(SystemExit) as incomplete:
        main.cmd_model(
            SimpleNamespace(provider="anthropic", model_id=None, refresh=False))
    with pytest.raises(SystemExit) as unverified:
        main.cmd_model(
            SimpleNamespace(
                provider="anthropic", model_id="typo-model", refresh=False))

    assert incomplete.value.code == 2
    assert unverified.value.code == 1
    assert writes == []
    assert "could not be verified" in capsys.readouterr().err
