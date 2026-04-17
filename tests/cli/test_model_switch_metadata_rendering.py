"""Regression tests for /model confirmation metadata rendering."""

from unittest.mock import patch

import cli as cli_module
from agent.models_dev import ModelInfo
from cli import HermesCLI
from hermes_cli.model_switch import ModelSwitchResult


def _make_cli_stub() -> HermesCLI:
    cli = HermesCLI.__new__(HermesCLI)
    cli.model = "old-model"
    cli.provider = "openrouter"
    cli.requested_provider = "openrouter"
    cli.api_key = "test-key"
    cli._explicit_api_key = "test-key"
    cli.base_url = "https://openrouter.ai/api/v1"
    cli._explicit_base_url = "https://openrouter.ai/api/v1"
    cli.api_mode = "chat_completions"
    cli.agent = None
    return cli


def _switch_result_with_model_info() -> ModelSwitchResult:
    return ModelSwitchResult(
        success=True,
        new_model="qwen3.6",
        target_provider="openrouter",
        provider_label="OpenRouter",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        model_info=ModelInfo(
            id="qwen/qwen3.6",
            name="Qwen 3.6",
            family="qwen3.6",
            provider_id="qwen",
            context_window=128_000,
            max_output=8_192,
            tool_call=True,
        ),
        context_length=256_000,
    )


def test_apply_model_switch_result_keeps_model_info_metadata_when_context_override():
    cli = _make_cli_stub()
    result = _switch_result_with_model_info()

    with patch.object(cli_module, "_cprint") as mock_cprint:
        cli._apply_model_switch_result(result, persist_global=False)

    rendered = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
    assert "Context: 256,000 tokens" in rendered
    assert "Max output: 8,192 tokens" in rendered
    assert "Capabilities: tools" in rendered


def test_handle_model_switch_keeps_model_info_metadata_when_context_override():
    cli = _make_cli_stub()
    result = _switch_result_with_model_info()

    with (
        patch.object(cli_module, "_cprint") as mock_cprint,
        patch("hermes_cli.model_switch.parse_model_flags", return_value=("qwen3.6", "", False)),
        patch("hermes_cli.model_switch.switch_model", return_value=result),
    ):
        cli._handle_model_switch("/model qwen3.6")

    rendered = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
    assert "Context: 256,000 tokens" in rendered
    assert "Max output: 8,192 tokens" in rendered
    assert "Capabilities: tools" in rendered
