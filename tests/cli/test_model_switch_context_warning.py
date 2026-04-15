"""CLI regression tests for /model --global context-length warnings."""

import importlib
import sys
from unittest.mock import MagicMock, patch

from agent.models_dev import ModelInfo
from hermes_cli.model_switch import ModelSwitchResult


def _import_cli():
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs):
        import cli as cli_mod

        return importlib.reload(cli_mod)


def test_apply_model_switch_result_surfaces_context_mismatch_warning():
    cli_mod = _import_cli()
    cli_obj = object.__new__(cli_mod.HermesCLI)
    cli_obj.model = "kimi-k2.5"
    cli_obj.provider = "moonshotai"
    cli_obj.requested_provider = "moonshotai"
    cli_obj.api_key = "old-key"
    cli_obj._explicit_api_key = "old-key"
    cli_obj.base_url = "https://api.moonshot.ai/v1"
    cli_obj._explicit_base_url = "https://api.moonshot.ai/v1"
    cli_obj.api_mode = "chat_completions"
    cli_obj.agent = None

    result = ModelSwitchResult(
        success=True,
        new_model="glm-5-turbo",
        target_provider="zai",
        provider_changed=True,
        api_key="new-key",
        base_url="https://api.z.ai/v1",
        api_mode="chat_completions",
        provider_label="Z.AI",
        model_info=ModelInfo(
            id="glm-5-turbo",
            name="GLM-5 Turbo",
            family="glm-5",
            provider_id="zai",
            context_window=202752,
        ),
        is_global=True,
    )

    printed = []
    with (
        patch.object(cli_mod, "_cprint", side_effect=printed.append),
        patch.object(cli_mod, "save_config_value", return_value=True),
        patch(
            "hermes_cli.model_switch.get_context_length_mismatch_warning",
            return_value="config.yaml context_length (256,000) doesn't match this model.",
        ),
    ):
        cli_obj._apply_model_switch_result(result, persist_global=True)

    assert any("Context: 202,752 tokens" in line for line in printed)
    assert any("context_length (256,000) doesn't match this model" in line for line in printed)
    assert any("Saved to config.yaml (--global)" in line for line in printed)
