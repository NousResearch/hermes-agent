"""``resolve_runtime_provider`` honors ``providers.<id>.models.<model>.transport``.

The startup / resume / gateway path resolves the wire from the configured entry. With a
per-model transport the SAME named entry must yield a different ``api_mode`` depending on the
target model, and the entry-level ``transport`` must keep serving every model that declares none.
"""

import pytest

from hermes_cli import runtime_provider as rp

_BASE = "https://gateway.example.com/v1"


@pytest.fixture
def gateway_config(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = {
        "model": {"provider": "custom:gateway", "default": "glm-5.3"},
        "providers": {"gateway": {
            "api": _BASE, "api_key": "test-key", "transport": "chat_completions",
            "default_model": "glm-5.3",
            "models": {
                "claude-sonnet-5": {"transport": "anthropic_messages"},
                "gpt-5.4-mini": {"transport": "codex_responses"},
                "glm-5.3": {"context_length": 200000},
            },
        }},
    }
    monkeypatch.setattr(rp, "load_config", lambda: config)
    return config


def test_target_model_selects_its_declared_transport(gateway_config):
    messages = rp.resolve_runtime_provider(requested="custom:gateway", target_model="claude-sonnet-5")
    responses = rp.resolve_runtime_provider(requested="custom:gateway", target_model="gpt-5.4-mini")
    chat = rp.resolve_runtime_provider(requested="custom:gateway", target_model="glm-5.3")
    assert (messages["provider"], messages["model"], messages["api_mode"]) == ("custom", "claude-sonnet-5", "anthropic_messages")
    assert (responses["model"], responses["api_mode"]) == ("gpt-5.4-mini", "codex_responses")
    assert (chat["model"], chat["api_mode"]) == ("glm-5.3", "chat_completions")
    # Same route, same credential, same base URL — only the wire differs.
    assert messages["base_url"] == responses["base_url"] == chat["base_url"] == _BASE
    assert messages["api_key"] == responses["api_key"] == chat["api_key"] == "test-key"


def test_default_model_without_target_keeps_entry_transport(gateway_config):
    resolved = rp.resolve_runtime_provider(requested="custom:gateway")
    assert (resolved["model"], resolved["api_mode"]) == ("glm-5.3", "chat_completions")


def test_default_model_with_declared_transport_is_honored(gateway_config):
    gateway_config["providers"]["gateway"]["default_model"] = "claude-sonnet-5"
    resolved = rp.resolve_runtime_provider(requested="custom:gateway")
    assert (resolved["model"], resolved["api_mode"]) == ("claude-sonnet-5", "anthropic_messages")


def test_codex_app_server_opt_in_keeps_precedence(gateway_config):
    """``model.openai_runtime: codex_app_server`` is a runtime, not a wire; a per-model transport
    must not downgrade it (#75186 contract)."""
    gateway_config["model"]["openai_runtime"] = "codex_app_server"
    resolved = rp.resolve_runtime_provider(requested="custom:gateway", target_model="gpt-5.4-mini")
    assert resolved["api_mode"] == "codex_app_server"
