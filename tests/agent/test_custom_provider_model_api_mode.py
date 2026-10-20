"""Per-model wire routing for custom providers (#126322).

``determine_api_mode`` previously saw only the provider, so a model whose catalog entry
declares ``anthropic_messages`` on an otherwise OpenAI-compatible aggregator was still sent
down ``/v1/chat/completions`` and the upstream 500'd. The per-model ``api_mode`` /
``transport`` in the provider's ``models.<id>`` entry now wins over the provider default.
"""

import pytest

from hermes_cli import config as config_mod
from hermes_cli.providers import determine_api_mode, model_specific_api_mode


AGGREGATOR_CFG = {
    "providers": {
        "custom-aggregator": {
            "name": "custom-aggregator",
            "base_url": "https://api.example-gateway.test/v1",
            "key_env": "EXAMPLE_GATEWAY_API_KEY",
            "models": {
                "claude-opus-5-5": {"api_mode": "anthropic"},
                "gpt-6": {"api_mode": "chat_completions"},
                "gemini-3": {"transport": "openai_chat"},
                "typo-model": {"api_mode": "not-a-wire"},
                "bare-model": {},
            },
        }
    },
}


@pytest.fixture(autouse=True)
def _stub_config(monkeypatch):
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: AGGREGATOR_CFG)


class TestModelSpecificApiMode:
    def test_declared_wire_wins_over_provider_default(self):
        assert determine_api_mode(
            "custom:custom-aggregator", "https://api.example-gateway.test/v1",
            model="claude-opus-5-5") == "anthropic_messages"

    def test_alias_spelling_is_canonicalized(self):
        # ``anthropic`` is a config alias for the canonical ``anthropic_messages`` wire.
        assert model_specific_api_mode("custom:custom-aggregator", "claude-opus-5-5") == \
            "anthropic_messages"

    def test_explicit_same_as_default_wire_is_kept(self):
        assert determine_api_mode(
            "custom:custom-aggregator", "https://api.example-gateway.test/v1",
            model="gpt-6") == "chat_completions"

    def test_transport_key_is_honored(self):
        assert model_specific_api_mode("custom:custom-aggregator", "gemini-3") == \
            "chat_completions"

    def test_model_without_declaration_keeps_default(self):
        # No entry for this model: fall through to the provider-level default.
        assert determine_api_mode(
            "custom:custom-aggregator", "https://api.example-gateway.test/v1",
            model="unknown-model") == "chat_completions"

    def test_unrecognized_wire_falls_back_to_default(self):
        # A config typo must not pin a wire no transport serves.
        assert model_specific_api_mode("custom:custom-aggregator", "typo-model") is None
        assert determine_api_mode(
            "custom:custom-aggregator", "https://api.example-gateway.test/v1",
            model="typo-model") == "chat_completions"

    def test_entry_without_wire_falls_back_to_default(self):
        assert model_specific_api_mode("custom:custom-aggregator", "bare-model") is None

    def test_no_model_keeps_provider_default(self):
        assert determine_api_mode(
            "custom:custom-aggregator", "https://api.example-gateway.test/v1") == \
            "chat_completions"

    def test_unconfigured_provider_is_unaffected(self):
        assert model_specific_api_mode("custom:other-gateway", "claude-opus-5-5") is None
        assert determine_api_mode(
            "custom:other-gateway", "https://other.test/v1",
            model="claude-opus-5-5") == "chat_completions"

    def test_host_mandated_wire_still_wins(self):
        # api.actual.inc only speaks chat/completions; the endpoint's wire beats config.
        assert determine_api_mode(
            "custom:custom-aggregator", "https://api.actual.inc/v1",
            model="claude-opus-5-5") == "chat_completions"

    def test_legacy_custom_providers_row_matches_by_name(self, monkeypatch):
        monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {
            "custom_providers": [
                {"name": "Old Gateway", "base_url": "https://old.test/v1",
                 "models": [{"id": "m1", "api_mode": "anthropic_messages"}]},
            ],
        })
        assert determine_api_mode("custom:old-gateway", "https://old.test/v1", model="m1") == \
            "anthropic_messages"

    def test_model_variants_are_spelling_tolerant(self):
        # Dot/dash variants resolve to the same catalog entry (x-4.5 vs x.4.5).
        assert model_specific_api_mode("custom:custom-aggregator", "claude-opus-5.5") == \
            "anthropic_messages"
