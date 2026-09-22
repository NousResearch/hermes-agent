"""Per-model ``transport`` on a named custom provider (``providers.<id>.models.<model>.transport``).

One endpoint can serve models behind different API surfaces — a gateway that routes ``claude-*``
through a native Anthropic Messages door and ``gpt-*`` through the Responses API while every
other model stays on chat/completions. Before this, a ``providers:`` entry had exactly one wire,
so such a gateway needed one entry per transport plus a hand-maintained alias table.
"""

from hermes_cli.config_providers import (
    _normalize_custom_provider_entry,
    custom_provider_model_api_mode,
    get_custom_provider_model_api_mode,
)

_BASE = "https://gateway.example.com/v1"


def _entry(models):
    return _normalize_custom_provider_entry(
        {"api": _BASE, "key_env": "GATEWAY_API_KEY", "transport": "chat_completions", "models": models},
        provider_key="gateway")


def test_dict_models_declare_transport_per_model():
    entry = _entry({
        "claude-sonnet-5": {"transport": "anthropic_messages"},
        "gpt-5.4-mini": {"transport": "codex_responses"},
        "glm-5.3": {"context_length": 200000},
    })
    assert custom_provider_model_api_mode(entry, "claude-sonnet-5") == "anthropic_messages"
    assert custom_provider_model_api_mode(entry, "gpt-5.4-mini") == "codex_responses"
    # No per-model transport → "" so the entry-level wire applies unchanged.
    assert custom_provider_model_api_mode(entry, "glm-5.3") == ""
    assert custom_provider_model_api_mode(entry, "unknown-model") == ""


def test_list_models_rows_are_normalized_to_the_same_shape():
    entry = _entry([
        {"id": "claude-sonnet-5", "transport": "anthropic_messages"},
        "glm-5.3",
    ])
    assert custom_provider_model_api_mode(entry, "claude-sonnet-5") == "anthropic_messages"
    assert custom_provider_model_api_mode(entry, "glm-5.3") == ""


def test_api_mode_alias_and_legacy_spellings_are_canonicalized():
    entry = _entry({
        "claude-sonnet-5": {"api_mode": "anthropic"},   # legacy alias key + alias spelling
        "gpt-5.4-mini": {"transport": "responses"},
    })
    assert custom_provider_model_api_mode(entry, "claude-sonnet-5") == "anthropic_messages"
    assert custom_provider_model_api_mode(entry, "gpt-5.4-mini") == "codex_responses"


def test_exact_model_id_match_only():
    entry = _entry({"claude-sonnet-5": {"transport": "anthropic_messages"}})
    assert custom_provider_model_api_mode(entry, "Claude-Sonnet-5") == ""
    assert custom_provider_model_api_mode(entry, "") == ""
    assert custom_provider_model_api_mode({}, "claude-sonnet-5") == ""
    assert custom_provider_model_api_mode(None, "claude-sonnet-5") == ""


def test_route_lookup_matches_the_entry_by_base_url():
    entries = [
        _entry({"claude-sonnet-5": {"transport": "anthropic_messages"}}),
        _normalize_custom_provider_entry(
            {"api": "https://other.example.com/v1", "models": {"claude-sonnet-5": {"transport": "codex_responses"}}},
            provider_key="other"),
    ]
    assert get_custom_provider_model_api_mode("claude-sonnet-5", _BASE, custom_providers=entries) == "anthropic_messages"
    assert get_custom_provider_model_api_mode("claude-sonnet-5", _BASE + "/", custom_providers=entries) == "anthropic_messages"
    assert get_custom_provider_model_api_mode("claude-sonnet-5", "https://other.example.com/v1", custom_providers=entries) == "codex_responses"
    assert get_custom_provider_model_api_mode("claude-sonnet-5", "https://nowhere.example.com/v1", custom_providers=entries) == ""
    assert get_custom_provider_model_api_mode("", _BASE, custom_providers=entries) == ""
    assert get_custom_provider_model_api_mode("claude-sonnet-5", "", custom_providers=entries) == ""


def test_route_lookup_reads_config_when_no_entries_are_passed(monkeypatch):
    config = {"providers": {"gateway": {
        "api": _BASE, "key_env": "GATEWAY_API_KEY",
        "models": {"claude-sonnet-5": {"transport": "anthropic_messages"}}}}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    assert get_custom_provider_model_api_mode("claude-sonnet-5", _BASE) == "anthropic_messages"
    assert get_custom_provider_model_api_mode("glm-5.3", _BASE) == ""
