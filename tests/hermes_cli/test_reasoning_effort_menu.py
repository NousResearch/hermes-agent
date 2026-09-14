import pytest

from hermes_cli.main_provider_setup import _prompt_reasoning_effort_selection


def test_reasoning_menu_orders_minimal_before_low(monkeypatch):
    captured = {}

    def _fake_radiolist(title, items, *, selected=0, cancel_returns=None, description=None):
        captured["items"] = items
        captured["selected"] = selected
        return selected  # pick the pre-selected (current) entry

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _fake_radiolist)

    selected = _prompt_reasoning_effort_selection(
        ["low", "minimal", "medium", "high"],
        current_effort="medium",
    )

    assert selected == "medium"
    assert captured["items"][:4] == [
        "minimal",
        "low",
        "medium  ← currently in use",
        "high",
    ]


def test_provider_switch_with_same_model_still_prompts_for_reasoning(tmp_path, monkeypatch):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.main import _PROVIDER_MODEL_FLOWS, select_provider_and_model

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = load_config()
    config["model"] = {"default": "shared-model", "provider": "openrouter"}
    save_config(config)
    prompted = []

    def _switch_provider(config, current_model, args):
        updated = load_config()
        updated["model"] = {"default": current_model, "provider": "anthropic"}
        save_config(updated)

    monkeypatch.setattr("hermes_cli.main._pick_provider", lambda *args: "anthropic")
    monkeypatch.setitem(_PROVIDER_MODEL_FLOWS, "anthropic", _switch_provider)
    monkeypatch.setattr(
        "hermes_cli.main_provider_setup._prompt_main_reasoning_effort",
        lambda model, provider: prompted.append((model, provider)),
    )
    monkeypatch.setattr("hermes_cli.main._clear_stale_openai_base_url", lambda: None)

    select_provider_and_model()

    assert prompted == [("shared-model", "anthropic")]


def test_legacy_custom_endpoint_switch_with_same_model_prompts_for_reasoning(
    tmp_path, monkeypatch
):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.main import select_provider_and_model

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = load_config()
    config["model"] = {
        "default": "shared-model",
        "provider": "custom",
        "base_url": "https://first.example/v1",
    }
    config["custom_providers"] = [
        {
            "name": "First Endpoint",
            "base_url": "https://first.example/v1",
            "model": "shared-model",
        },
        {
            "name": "Second Endpoint",
            "base_url": "https://second.example/v1",
            "model": "shared-model",
        },
    ]
    save_config(config)
    prompted = []

    def _switch_endpoint(config, provider_info):
        updated = load_config()
        updated["model"] = {
            "default": "shared-model",
            "provider": "custom",
            "base_url": provider_info["base_url"],
        }
        save_config(updated)

    monkeypatch.setattr(
        "hermes_cli.main._pick_provider", lambda *args: "custom:second-endpoint"
    )
    monkeypatch.setattr("hermes_cli.main._model_flow_named_custom", _switch_endpoint)
    monkeypatch.setattr(
        "hermes_cli.main_provider_setup._prompt_main_reasoning_effort",
        lambda model, provider: prompted.append((model, provider)),
    )
    monkeypatch.setattr("hermes_cli.main._clear_stale_openai_base_url", lambda: None)

    select_provider_and_model()

    assert prompted == [("shared-model", "custom:second-endpoint")]


def test_nested_provider_flow_uses_persisted_route_for_reasoning(tmp_path, monkeypatch):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.main import select_provider_and_model

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = load_config()
    config["model"] = {
        "default": "shared-model",
        "provider": "bedrock",
        "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
    }
    save_config(config)
    prompted = []

    def _select_bedrock_api_key(config, current_model):
        updated = load_config()
        updated["model"] = {
            "default": current_model,
            "provider": "custom:bedrock-mantle",
        }
        save_config(updated)

    monkeypatch.setattr("hermes_cli.main._pick_provider", lambda *args: "bedrock")
    monkeypatch.setattr(
        "hermes_cli.main._resolve_active_provider", lambda *args: "bedrock"
    )
    monkeypatch.setattr(
        "hermes_cli.main._model_flow_bedrock", _select_bedrock_api_key
    )
    monkeypatch.setattr(
        "hermes_cli.main_provider_setup._prompt_main_reasoning_effort",
        lambda model, provider: prompted.append((model, provider)),
    )
    monkeypatch.setattr("hermes_cli.main._clear_stale_openai_base_url", lambda: None)

    select_provider_and_model()

    assert prompted == [("shared-model", "custom:bedrock-mantle")]


def test_direct_custom_endpoint_switch_with_same_model_prompts_for_reasoning(
    tmp_path, monkeypatch
):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.main import _PROVIDER_MODEL_FLOWS, select_provider_and_model

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = load_config()
    config["model"] = {
        "default": "shared-model",
        "provider": "custom",
        "base_url": "https://first.example/v1",
    }
    save_config(config)
    prompted = []

    def _switch_endpoint(config, current_model, args):
        updated = load_config()
        updated["model"] = {
            "default": current_model,
            "provider": "custom",
            "base_url": "https://second.example/v1",
        }
        save_config(updated)

    monkeypatch.setattr("hermes_cli.main._pick_provider", lambda *args: "custom")
    monkeypatch.setattr(
        "hermes_cli.main._resolve_active_provider", lambda *args: "custom"
    )
    monkeypatch.setitem(_PROVIDER_MODEL_FLOWS, "custom", _switch_endpoint)
    monkeypatch.setattr(
        "hermes_cli.main_provider_setup._prompt_main_reasoning_effort",
        lambda model, provider: prompted.append((model, provider)),
    )

    select_provider_and_model()

    assert prompted == [("shared-model", "custom")]


@pytest.mark.parametrize(
    ("provider_before", "selected_provider", "providers"),
    [
        ("github-copilot", "copilot", None),
        ("OpenRouter", "openrouter", None),
        (
            "Local LLM",
            "custom:local-llm",
            {
                "local-llm": {
                    "name": "Local LLM",
                    "base_url": "http://127.0.0.1:1234/v1",
                }
            },
        ),
    ],
)
def test_reselecting_same_provider_alias_does_not_prompt_for_reasoning(
    tmp_path, monkeypatch, provider_before, selected_provider, providers
):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.main import _PROVIDER_MODEL_FLOWS, select_provider_and_model

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = load_config()
    config["model"] = {"default": "shared-model", "provider": provider_before}
    if providers is not None:
        config["providers"] = providers
    save_config(config)
    prompted = []

    def _repick_provider(config, current_model, args):
        updated = load_config()
        updated["model"] = {"default": current_model, "provider": selected_provider}
        save_config(updated)

    monkeypatch.setattr("hermes_cli.main._pick_provider", lambda *args: selected_provider)
    monkeypatch.setitem(_PROVIDER_MODEL_FLOWS, selected_provider, _repick_provider)
    monkeypatch.setattr(
        "hermes_cli.main_provider_setup._prompt_main_reasoning_effort",
        lambda model, provider: prompted.append((model, provider)),
    )
    monkeypatch.setattr("hermes_cli.main._clear_stale_openai_base_url", lambda: None)

    select_provider_and_model()

    assert prompted == []
