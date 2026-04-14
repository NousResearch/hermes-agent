"""Tests for the shared /model picker entry schema in the CLI."""

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
    return cli_obj


def test_handle_model_switch_opens_picker_with_shared_entries(monkeypatch):
    cli_obj = _make_cli()
    opened = {}

    monkeypatch.setattr("cli._cprint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("hermes_cli.config.get_compatible_custom_providers", lambda cfg: cfg.get("custom_providers"))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_picker_entries",
        lambda **_kwargs: [
            {
                "id": "openrouter:anthropic",
                "slug": "openrouter:anthropic",
                "switch_provider": "openrouter",
                "name": "OpenRouter / Anthropic",
                "models": ["anthropic/claude-sonnet-4.6"],
                "total_models": 1,
                "is_current": True,
            }
        ],
    )
    cli_obj._open_model_picker = lambda providers, current_model, current_provider, user_provs=None, custom_provs=None: opened.update({
        "providers": providers,
        "current_model": current_model,
        "current_provider": current_provider,
    })

    cli_obj._handle_model_switch("/model")

    assert opened["providers"][0]["slug"] == "openrouter:anthropic"
    assert opened["providers"][0]["switch_provider"] == "openrouter"
    assert opened["current_model"] == "anthropic/claude-sonnet-4.6"
    assert opened["current_provider"] == "OpenRouter"


def test_model_picker_selection_uses_switch_provider(monkeypatch):
    cli_obj = _make_cli()
    captured = {}

    cli_obj._model_picker_state = {
        "stage": "model",
        "selected": 0,
        "provider_data": {
            "id": "openrouter:anthropic",
            "slug": "openrouter:anthropic",
            "switch_provider": "openrouter",
            "name": "OpenRouter / Anthropic",
        },
        "model_list": ["anthropic/claude-sonnet-4.6"],
        "user_provs": None,
        "custom_provs": None,
    }
    cli_obj._close_model_picker = lambda: None
    cli_obj._apply_model_switch_result = lambda result, persist_global: captured.setdefault("applied", (result, persist_global))

    with patch("hermes_cli.model_switch.switch_model") as mock_switch:
        mock_switch.return_value = object()
        cli_obj._handle_model_picker_selection()

    assert mock_switch.call_args.kwargs["explicit_provider"] == "openrouter"
