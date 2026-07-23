"""Tests for named user-defined provider entries in the ACP model selector.

Named endpoints from the ``providers:`` mapping (and legacy
``custom_providers:`` list) are invisible to canonical provider enumeration,
so ``_build_model_state`` must append them explicitly for ACP clients to
offer them — the TUI ``/model`` picker already renders these entries
(#47039 implemented named endpoints for the TUI surface only).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acp_adapter.server import HermesACPAgent, _named_custom_provider_catalogs
from acp_adapter.session import SessionManager
from acp.schema import SessionModelState


MANTLE_URL = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"


def _cfg(providers=None, custom_providers=None):
    cfg = {}
    if providers is not None:
        cfg["providers"] = providers
    if custom_providers is not None:
        cfg["custom_providers"] = custom_providers
    return cfg


class TestNamedCustomProviderCatalogs:
    def test_declared_default_model_survives_failed_discovery(self, monkeypatch):
        """Endpoints without a /models route keep their declared models."""
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "test-key")
        cfg = _cfg(
            providers={
                "bedrock-mantle": {
                    "name": "AWS Bedrock Mantle",
                    "base_url": MANTLE_URL,
                    "key_env": "BEDROCK_MANTLE_API_KEY",
                    "api_mode": "codex_responses",
                    "default_model": "openai.gpt-5.5",
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.fetch_api_models", return_value=None
        ):
            catalogs = _named_custom_provider_catalogs()

        assert catalogs == [
            (
                "custom:bedrock-mantle",
                "AWS Bedrock Mantle",
                [("openai.gpt-5.5", "")],
            )
        ]

    def test_live_discovery_extends_declared_models(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "k")
        cfg = _cfg(
            providers={
                "relay": {
                    "name": "Relay",
                    "base_url": "https://relay.example/v1",
                    "key_env": "SOME_KEY",
                    "default_model": "model-a",
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["model-a", "model-b"],
        ):
            catalogs = _named_custom_provider_catalogs()

        assert len(catalogs) == 1
        slug, label, models = catalogs[0]
        assert slug == "custom:relay"
        assert [m for m, _ in models] == ["model-a", "model-b"]

    def test_declared_models_dict_included(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "k")
        cfg = _cfg(
            providers={
                "relay": {
                    "name": "Relay",
                    "base_url": "https://relay.example/v1",
                    "key_env": "SOME_KEY",
                    "default_model": "model-a",
                    "models": {"model-b": {}, "model-c": {}},
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.fetch_api_models", return_value=None
        ):
            catalogs = _named_custom_provider_catalogs()

        assert [m for m, _ in catalogs[0][2]] == ["model-a", "model-b", "model-c"]

    def test_disabled_provider_skipped(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "k")
        cfg = _cfg(
            providers={
                "off": {
                    "name": "Disabled Endpoint",
                    "base_url": "https://off.example/v1",
                    "key_env": "SOME_KEY",
                    "default_model": "m",
                    "enabled": False,
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.fetch_api_models", return_value=None
        ):
            assert _named_custom_provider_catalogs() == []

    def test_no_credential_and_no_declared_models_skipped(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        cfg = _cfg(
            providers={
                "bare": {
                    "name": "Bare",
                    "base_url": "https://bare.example/v1",
                    "key_env": "MISSING_KEY",
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.fetch_api_models", return_value=None
        ):
            assert _named_custom_provider_catalogs() == []

    def test_legacy_custom_providers_list_included(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "k")
        cfg = _cfg(
            custom_providers=[
                {
                    "name": "Legacy Endpoint",
                    "base_url": "https://legacy.example/v1",
                    "key_env": "SOME_KEY",
                    "model": "legacy-model",
                }
            ]
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.fetch_api_models", return_value=None
        ):
            catalogs = _named_custom_provider_catalogs()

        assert catalogs == [
            ("custom:legacy-endpoint", "Legacy Endpoint", [("legacy-model", "")])
        ]


class TestModelStateIncludesNamedProviders:
    @pytest.mark.asyncio
    async def test_named_provider_models_appear_in_model_state(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(
                model="gpt-5.4", provider="openai-codex"
            )
        )
        acp_agent = HermesACPAgent(session_manager=manager)

        with patch(
            "hermes_cli.models.curated_models_for_provider",
            return_value=[("gpt-5.4", "recommended")],
        ), patch(
            "acp_adapter.server._named_custom_provider_catalogs",
            return_value=[
                (
                    "custom:bedrock-mantle",
                    "AWS Bedrock Mantle",
                    [("openai.gpt-5.5", "")],
                )
            ],
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        ids = [m.model_id for m in resp.models.available_models]
        # Current provider's models come first, named endpoints after.
        assert ids[0] == "openai-codex:gpt-5.4"
        assert "custom:bedrock-mantle:openai.gpt-5.5" in ids
        named = next(
            m
            for m in resp.models.available_models
            if m.model_id == "custom:bedrock-mantle:openai.gpt-5.5"
        )
        assert "AWS Bedrock Mantle" in (named.description or "")

    def test_selector_choice_id_round_trips_through_parse_model_input(self):
        """The encoded choice id must resolve back to the named provider."""
        from hermes_cli.models import parse_model_input

        choice_id = "custom:bedrock-mantle:openai.gpt-5.5"
        provider, model = parse_model_input(choice_id, "bedrock")
        assert provider == "custom:bedrock-mantle"
        assert model == "openai.gpt-5.5"
