"""Tests for the canonical staged /model picker in the CLI."""

from unittest.mock import patch


def _make_cli():
    from cli import HermesCLI

    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "anthropic/claude-sonnet-4.6"
    cli_obj.provider = "openrouter"
    cli_obj.base_url = "https://openrouter.ai/api/v1"
    cli_obj.api_key = "sk-test"
    cli_obj.api_mode = "chat_completions"
    cli_obj.requested_provider = "openrouter"
    cli_obj._explicit_api_key = ""
    cli_obj._explicit_base_url = ""
    cli_obj._attached_images = []
    cli_obj._invalidate = lambda **_kwargs: None
    return cli_obj


def test_provider_selection_opens_openrouter_group_stage():
    cli_obj = _make_cli()
    cli_obj._model_picker_state = {
        "stage": "provider",
        "selected": 0,
        "providers": [
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "is_current": True,
                "models": ["anthropic/claude-sonnet-4.6"],
                "total_models": 2,
                "groups": [
                    {"id": "anthropic", "name": "Anthropic", "models": ["anthropic/claude-sonnet-4.6"], "total_models": 1},
                    {"id": "openai", "name": "OpenAI", "models": ["openai/gpt-5.4"], "total_models": 1},
                ],
            }
        ],
        "current_model": "anthropic/claude-sonnet-4.6",
        "current_provider": "OpenRouter",
        "user_provs": None,
        "custom_provs": None,
    }

    cli_obj._handle_model_picker_selection()

    assert cli_obj._model_picker_state["stage"] == "openrouter_group"
    assert cli_obj._model_picker_state["provider_data"]["slug"] == "openrouter"
    assert cli_obj._model_picker_state["group_list"][0]["id"] == "anthropic"
    assert cli_obj._model_picker_state["selected"] == 0


def test_group_selection_opens_model_stage():
    cli_obj = _make_cli()
    cli_obj._model_picker_state = {
        "stage": "openrouter_group",
        "selected": 0,
        "provider_data": {"slug": "openrouter", "name": "OpenRouter"},
        "group_list": [
            {"id": "anthropic", "name": "Anthropic", "models": ["anthropic/claude-sonnet-4.6"], "total_models": 1},
            {"id": "openai", "name": "OpenAI", "models": ["openai/gpt-5.4"], "total_models": 1},
        ],
        "user_provs": None,
        "custom_provs": None,
    }

    cli_obj._handle_model_picker_selection()

    assert cli_obj._model_picker_state["stage"] == "model"
    assert cli_obj._model_picker_state["group_data"]["id"] == "anthropic"
    assert cli_obj._model_picker_state["model_list"] == ["anthropic/claude-sonnet-4.6"]


def test_model_selection_uses_canonical_openrouter_provider():
    cli_obj = _make_cli()
    cli_obj._model_picker_state = {
        "stage": "model",
        "selected": 0,
        "provider_data": {"slug": "openrouter", "name": "OpenRouter"},
        "group_data": {"id": "anthropic", "name": "Anthropic"},
        "model_list": ["anthropic/claude-sonnet-4.6"],
        "user_provs": None,
        "custom_provs": None,
    }
    cli_obj._close_model_picker = lambda: None
    cli_obj._apply_model_switch_result = lambda result, persist_global: None

    with patch("hermes_cli.model_switch.switch_model") as mock_switch:
        mock_switch.return_value = object()
        cli_obj._handle_model_picker_selection()

    assert mock_switch.call_args.kwargs["explicit_provider"] == "openrouter"
