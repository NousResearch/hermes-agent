"""Regression tests for /model support of config.yaml custom_providers.

The terminal `hermes model` flow already exposes `custom_providers`, but the
shared slash-command pipeline (`/model` in CLI/gateway/Telegram) historically
only looked at `providers:`.
"""

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import list_authenticated_providers, switch_model
from hermes_cli.providers import resolve_provider_full


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def test_list_authenticated_providers_includes_custom_providers(monkeypatch):
    """No-args /model menus should include saved custom_providers entries."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        user_providers={},
        custom_providers=[
            {
                "name": "Local (127.0.0.1:4141)",
                "base_url": "http://127.0.0.1:4141/v1",
                "model": "rotator-openrouter-coding",
            }
        ],
        max_models=50,
    )

    assert any(
        p["slug"] == "custom:local-(127.0.0.1:4141)"
        and p["name"] == "Local (127.0.0.1:4141)"
        and p["models"] == ["rotator-openrouter-coding"]
        and p["api_url"] == "http://127.0.0.1:4141/v1"
        for p in providers
    )


def test_resolve_provider_full_finds_named_custom_provider():
    """Explicit /model --provider should resolve saved custom_providers entries."""
    resolved = resolve_provider_full(
        "custom:local-(127.0.0.1:4141)",
        user_providers={},
        custom_providers=[
            {
                "name": "Local (127.0.0.1:4141)",
                "base_url": "http://127.0.0.1:4141/v1",
            }
        ],
    )

    assert resolved is not None
    assert resolved.id == "custom:local-(127.0.0.1:4141)"
    assert resolved.name == "Local (127.0.0.1:4141)"
    assert resolved.base_url == "http://127.0.0.1:4141/v1"
    assert resolved.source == "user-config"


def test_switch_model_accepts_explicit_named_custom_provider(monkeypatch):
    """Shared /model switch pipeline should accept --provider for custom_providers."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested: {
            "api_key": "no-key-required",
            "base_url": "http://127.0.0.1:4141/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="rotator-openrouter-coding",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        current_base_url="https://chatgpt.com/backend-api/codex",
        current_api_key="",
        explicit_provider="custom:local-(127.0.0.1:4141)",
        user_providers={},
        custom_providers=[
            {
                "name": "Local (127.0.0.1:4141)",
                "base_url": "http://127.0.0.1:4141/v1",
                "model": "rotator-openrouter-coding",
            }
        ],
    )

    assert result.success is True
    assert result.target_provider == "custom:local-(127.0.0.1:4141)"
    assert result.provider_label == "Local (127.0.0.1:4141)"
    assert result.new_model == "rotator-openrouter-coding"
    assert result.base_url == "http://127.0.0.1:4141/v1"
    assert result.api_key == "no-key-required"


def test_list_groups_same_name_custom_providers_into_one_row(monkeypatch):
    """Multiple custom_providers entries sharing a name should produce one row
    with all models collected, not N duplicate rows."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openrouter",
        user_providers={},
        custom_providers=[
            {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "qwen3-coder:480b-cloud"},
            {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "glm-5.1:cloud"},
            {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "kimi-k2.5"},
            {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "minimax-m2.7:cloud"},
            {"name": "Moonshot", "base_url": "https://api.moonshot.ai/v1", "model": "kimi-k2-thinking"},
        ],
        max_models=50,
    )

    ollama_rows = [p for p in providers if p["name"] == "Ollama Cloud"]
    assert len(ollama_rows) == 1, f"Expected 1 Ollama Cloud row, got {len(ollama_rows)}"
    assert ollama_rows[0]["models"] == [
        "qwen3-coder:480b-cloud", "glm-5.1:cloud", "kimi-k2.5", "minimax-m2.7:cloud"
    ]
    assert ollama_rows[0]["total_models"] == 4

    moonshot_rows = [p for p in providers if p["name"] == "Moonshot"]
    assert len(moonshot_rows) == 1
    assert moonshot_rows[0]["models"] == ["kimi-k2-thinking"]


def test_list_deduplicates_same_model_in_group(monkeypatch):
    """Duplicate model entries under the same provider name should not produce
    duplicate entries in the models list."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openrouter",
        user_providers={},
        custom_providers=[
            {"name": "MyProvider", "base_url": "http://localhost:11434/v1", "model": "llama3"},
            {"name": "MyProvider", "base_url": "http://localhost:11434/v1", "model": "llama3"},
            {"name": "MyProvider", "base_url": "http://localhost:11434/v1", "model": "mistral"},
        ],
        max_models=50,
    )

    my_rows = [p for p in providers if p["name"] == "MyProvider"]
    assert len(my_rows) == 1
    assert my_rows[0]["models"] == ["llama3", "mistral"]
    assert my_rows[0]["total_models"] == 2
