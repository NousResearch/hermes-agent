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


def test_moa_arg_is_always_one_shot_prompt():
    # Any argument (even a string that matches a preset name) is treated as a
    # one-shot prompt through the DEFAULT preset, then the model is restored.
    cli = _make_cli()
    with patch("cli._cprint"):
        cli.process_command("/moa review")
    assert cli._pending_agent_seed == "review"
    assert cli._pending_moa_disable_after_turn is True
    assert cli.provider == "moa"
    assert cli.model == "default"


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
    assert cfg["reference_models"] == [
        {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro", "enabled": True}
    ]


class TestNormalizeMoaModel:
    """#56828: `-Q -m moa:<preset>` must route through the MoA virtual provider.

    ``_normalize_moa_model`` maps the model string to (provider, preset); the
    __init__ wiring then forces ``requested_provider="moa"`` so the existing
    resolve_runtime_provider / agent_init MoA path runs in non-interactive mode.
    """

    def test_moa_prefix_maps_to_provider_and_preset(self):
        from cli import _normalize_moa_model
        assert _normalize_moa_model("moa:strategy") == ("moa", "strategy")

    def test_moa_prefix_is_case_insensitive_and_trims(self):
        from cli import _normalize_moa_model
        assert _normalize_moa_model("  MOA:code-review ") == ("moa", "code-review")

    def test_bare_moa_without_preset_is_not_treated_as_virtual(self):
        from cli import _normalize_moa_model
        # No preset after the colon → leave untouched (no provider override).
        assert _normalize_moa_model("moa:") == (None, "moa:")

    def test_non_moa_model_unchanged(self):
        from cli import _normalize_moa_model
        assert _normalize_moa_model("anthropic/claude-opus-4.8") == (None, "anthropic/claude-opus-4.8")

    def test_none_model_unchanged(self):
        from cli import _normalize_moa_model
        assert _normalize_moa_model(None) == (None, None)

    def test_colon_model_that_is_not_moa_unchanged(self):
        from cli import _normalize_moa_model
        # A provider:model form for a real provider must not be hijacked.
        assert _normalize_moa_model("openrouter:deepseek/deepseek-v4") == (
            None,
            "openrouter:deepseek/deepseek-v4",
        )

    def test_override_wins_over_explicit_provider(self):
        # __init__ resolves requested_provider as
        # ``_moa_provider_override or provider or ...``, so a moa: prefix must
        # take precedence over an explicit --provider (the #56828 deepseek case
        # where MoA was silently ignored).
        from cli import _normalize_moa_model
        override, model = _normalize_moa_model("moa:strategy")
        requested_provider = override or "deepseek" or "auto"
        assert requested_provider == "moa"
        assert model == "strategy"

