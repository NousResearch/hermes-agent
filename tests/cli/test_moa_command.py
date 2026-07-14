import queue
from unittest.mock import patch

from cli import HermesCLI
from hermes_cli.moa_config import decode_moa_turn


def _make_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {
        "moa": {
            "default_preset": "default",
            "presets": {
                "default": {
                    "reference_models": [{"provider": "openai-codex", "model": "gpt-5.5"}],
                    "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
                },
                "review": {
                    "reference_models": [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}],
                    "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
                },
            },
        }
    }
    cli._pending_input = queue.Queue()
    cli._pending_agent_seed = None
    cli._pending_moa_config = None
    cli._agent_running = False
    cli.agent = None
    return cli


def test_moa_bare_switches_to_default_preset_model():
    cli = _make_cli()
    with patch("cli._cprint"):
        assert cli.process_command("/moa") is True
    assert cli.provider == "moa"
    assert cli.requested_provider == "moa"
    assert cli.model == "default"
    assert cli.agent is None


def test_moa_exact_preset_switches_to_named_preset_model():
    cli = _make_cli()
    with patch("cli._cprint"):
        cli.process_command("/moa review")
    assert cli.provider == "moa"
    assert cli.model == "review"
    assert cli.agent is None


def test_moa_non_preset_is_one_shot_prompt():
    cli = _make_cli()
    with patch("cli._cprint"):
        cli.process_command("/moa inspect the flaky test")
    assert cli._pending_agent_seed == "inspect the flaky test"
    assert cli._pending_moa_disable_after_turn is True
    assert cli.provider == "moa"
    assert cli.model == "default"
    assert cli._pending_moa_restore_model["provider"] != "moa"


def test_decode_legacy_encoded_moa_turn_still_works():
    from hermes_cli.moa_config import build_moa_turn_prompt

    encoded = build_moa_turn_prompt("hello", _make_cli().config["moa"], preset="review")
    prompt, cfg = decode_moa_turn(encoded)
    assert prompt == "hello"
    assert cfg["reference_models"] == [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}]
