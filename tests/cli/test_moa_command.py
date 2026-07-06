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
    cli._pending_moa_disable_after_turn = False
    cli._pending_moa_restore_model = None
    cli._agent_running = False
    cli.agent = None
    cli.provider = "openrouter"
    cli.requested_provider = "openrouter"
    cli.model = "anthropic/claude-opus-4.8"
    cli.api_key = "test-key"
    cli.base_url = "https://openrouter.ai/api/v1"
    cli.api_mode = "chat_completions"
    return cli


def _fail_modal(**_kwargs):
    raise AssertionError("MoA preset picker should not open")


def test_moa_bare_shows_usage_no_switch():
    # /moa with no prompt is usage-only now; switching to a preset for the
    # session is done via the model picker, not /moa.
    cli = _make_cli()
    cli._pending_moa_disable_after_turn = False
    with patch("cli._cprint"):
        assert cli.process_command("/moa") is True
    assert cli.provider != "moa"
    assert cli._pending_agent_seed is None
    assert cli._pending_moa_disable_after_turn is False


def test_moa_explicit_preset_name_selects_preset_and_prompt():
    cli = _make_cli()
    cli._prompt_text_input_modal = _fail_modal
    with patch("cli._cprint"):
        cli.process_command("/moa review inspect this project")
    assert cli._pending_agent_seed == "inspect this project"
    assert cli._pending_moa_disable_after_turn is True
    assert cli.provider == "moa"
    assert cli.model == "review"


def test_moa_explicit_preset_flag_bypasses_picker():
    cli = _make_cli()
    cli.config["moa"]["prompt_preset"] = True
    cli._prompt_text_input_modal = _fail_modal
    with patch("cli._cprint"):
        cli.process_command("/moa --preset review inspect this project")
    assert cli._pending_agent_seed == "inspect this project"
    assert cli.provider == "moa"
    assert cli.model == "review"


def test_moa_non_preset_is_one_shot_prompt():
    cli = _make_cli()
    with patch("cli._cprint"):
        cli.process_command("/moa inspect the flaky test")
    assert cli._pending_agent_seed == "inspect the flaky test"
    assert cli._pending_moa_disable_after_turn is True
    assert cli.provider == "moa"
    assert cli.model == "default"
    assert cli._pending_moa_restore_model["provider"] != "moa"


def test_moa_prompt_preset_picker_preserves_natural_first_word():
    cli = _make_cli()
    cli.config["moa"]["prompt_preset"] = True
    captured = {}

    def _pick_review(**kwargs):
        captured.update(kwargs)
        return "review"

    cli._prompt_text_input_modal = _pick_review
    with patch("cli._cprint"):
        cli.process_command("/moa inspect the flaky test")

    assert cli._pending_agent_seed == "inspect the flaky test"
    assert cli.provider == "moa"
    assert cli.model == "review"
    assert captured["title"] == "Select MoA preset"
    assert any(choice[0] == "review" for choice in captured["choices"])


def test_decode_legacy_encoded_moa_turn_still_works():
    from hermes_cli.moa_config import build_moa_turn_prompt

    encoded = build_moa_turn_prompt("hello", _make_cli().config["moa"], preset="review")
    prompt, cfg = decode_moa_turn(encoded)
    assert prompt == "hello"
    assert cfg["reference_models"] == [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}]
