from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI
from hermes_cli.model_switch import ModelSwitchResult


def _result() -> ModelSwitchResult:
    model_info = SimpleNamespace(
        context_window=200_000,
        max_output=8_000,
        has_cost_data=lambda: False,
        format_capabilities=lambda: "chat",
    )
    return ModelSwitchResult(
        success=True,
        new_model="gpt-5.4",
        target_provider="openai",
        api_key="new-key",
        base_url="https://api.openai.com/v1",
        api_mode="chat_completions",
        provider_label="OpenAI",
        model_info=model_info,
    )


def test_handle_model_switch_passes_runtime_fallback_chain_to_live_agent():
    cli = object.__new__(HermesCLI)
    cli.model = "claude-sonnet-4-6"
    cli.provider = "anthropic"
    cli.base_url = "https://api.anthropic.com"
    cli.api_key = "old-key"
    cli.api_mode = "anthropic_messages"
    cli.requested_provider = "anthropic"
    cli._fallback_model = [{"provider": "zai", "model": "glm-5"}]
    cli.agent = MagicMock()

    with (
        patch("cli._cprint"),
        patch("hermes_cli.model_switch.parse_model_flags", return_value=("gpt-5.4", None, False)),
        patch("hermes_cli.model_switch.switch_model", return_value=_result()),
        patch("hermes_cli.config.load_config", return_value={}),
    ):
        cli._handle_model_switch("/model gpt-5.4")

    cli.agent.switch_model.assert_called_once_with(
        new_model="gpt-5.4",
        new_provider="openai",
        api_key="new-key",
        base_url="https://api.openai.com/v1",
        api_mode="chat_completions",
        fallback_model=cli._fallback_model,
    )


def test_apply_model_switch_result_passes_runtime_fallback_chain_to_live_agent():
    cli = object.__new__(HermesCLI)
    cli.model = "claude-sonnet-4-6"
    cli.provider = "anthropic"
    cli.base_url = "https://api.anthropic.com"
    cli.api_key = "old-key"
    cli.api_mode = "anthropic_messages"
    cli.requested_provider = "anthropic"
    cli._fallback_model = [{"provider": "zai", "model": "glm-5"}]
    cli.agent = MagicMock()

    with patch("cli._cprint"):
        cli._apply_model_switch_result(_result(), persist_global=False)

    cli.agent.switch_model.assert_called_once_with(
        new_model="gpt-5.4",
        new_provider="openai",
        api_key="new-key",
        base_url="https://api.openai.com/v1",
        api_mode="chat_completions",
        fallback_model=cli._fallback_model,
    )
