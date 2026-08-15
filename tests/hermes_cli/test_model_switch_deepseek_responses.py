"""DeepSeek model switches must recompute wire mode and official base URL."""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def _run_switch(raw_input: str, *, current_model: str, runtime_mode: str, runtime_url: str):
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "sk-test",
                "base_url": runtime_url,
                "api_mode": runtime_mode,
            },
        ),
        patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
    ):
        return switch_model(
            raw_input=raw_input,
            current_provider="deepseek",
            current_model=current_model,
            current_base_url=runtime_url,
            current_api_key="sk-test",
        )


def test_switch_pro_to_flash_uses_responses_root():
    result = _run_switch(
        "deepseek-v4-flash",
        current_model="deepseek-v4-pro",
        runtime_mode="chat_completions",
        runtime_url="https://api.deepseek.com/v1",
    )
    assert result.success
    assert result.api_mode == "codex_responses"
    assert result.base_url == "https://api.deepseek.com"


def test_switch_flash_to_pro_stays_on_responses_root():
    result = _run_switch(
        "deepseek-v4-pro",
        current_model="deepseek-v4-flash",
        runtime_mode="chat_completions",
        runtime_url="https://api.deepseek.com/v1",
    )
    assert result.success
    assert result.api_mode == "codex_responses"
    assert result.base_url == "https://api.deepseek.com"
