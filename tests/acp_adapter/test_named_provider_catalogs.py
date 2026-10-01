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

from acp_adapter.model_catalog import _named_custom_provider_catalogs
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from acp.schema import SessionModelState
from models.catalog_endpoint import EndpointModels


MANTLE_URL = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"


def _cfg(providers=None, custom_providers=None):
    cfg = {}
    if providers is not None:
        cfg["providers"] = providers
    if custom_providers is not None:
        cfg["custom_providers"] = custom_providers
    return cfg


class TestNamedCustomProviderCatalogs:

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
            "models.catalog_endpoint.discover_endpoint_models",
            return_value=EndpointModels(("model-a", "model-b")),
        ):
            catalogs = _named_custom_provider_catalogs()

        assert len(catalogs) == 1
        slug, label, models = catalogs[0]
        assert slug == "custom:relay"
        assert [m for m, _ in models] == ["model-a", "model-b"]


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
            "models.catalog_endpoint.discover_endpoint_models", return_value=None
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
            "models.catalog_endpoint.discover_endpoint_models", return_value=None
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
            "models.catalog_endpoint.discover_endpoint_models", return_value=None
        ):
            catalogs = _named_custom_provider_catalogs()

        assert catalogs == [
            ("custom:legacy-endpoint", "Legacy Endpoint", [("legacy-model", "")])
        ]
    def test_no_key_ollama_provider_discovers_native_catalog(self):
        cfg = _cfg(
            providers={
                "custom:ollama": {
                    "name": "Ollama",
                    "base_url": "http://127.0.0.1:11434/v1",
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "models.catalog_endpoint.discover_endpoint_models",
            return_value=EndpointModels(("qwen3:1.7b",), native_ollama=True),
        ):
            catalogs = _named_custom_provider_catalogs()

        assert [m for m, _ in catalogs[0][2]] == ["qwen3:1.7b"]
    def test_legacy_credentialless_ollama_discovers_native_catalog(self):
        cfg = _cfg(
            custom_providers=[
                {
                    "name": "Local Ollama",
                    "base_url": "http://127.0.0.1:11434/v1",
                }
            ]
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "models.catalog_endpoint.discover_endpoint_models",
            return_value=EndpointModels(("qwen3:1.7b",), native_ollama=True),
        ) as fetch:
            catalogs = _named_custom_provider_catalogs()

        assert [m for m, _ in catalogs[0][2]] == ["qwen3:1.7b"]
        fetch.assert_called_once()

    def test_native_empty_catalog_is_authoritative_over_default_model(self):
        cfg = _cfg(
            providers={
                "custom:ollama": {
                    "name": "Ollama",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "default_model": "saved:model",
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "models.catalog_endpoint.discover_endpoint_models",
            return_value=EndpointModels((), native_ollama=True),
        ):
            assert _named_custom_provider_catalogs() == [
                ("custom:ollama", "Ollama", [])
            ]


class TestModelStateIncludesNamedProviders:
    @pytest.mark.asyncio
    async def test_authoritative_empty_named_catalog_does_not_resurrect_current_model(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(
                model="saved:model", provider="ollama"
            )
        )
        acp_agent = HermesACPAgent(session_manager=manager)

        with patch(
            "acp_adapter.model_catalog._named_custom_provider_catalogs",
            return_value=[("custom:ollama", "Ollama", [])],
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        assert resp.models.current_model_id == ""
        assert all(
            not item.model_id.startswith(("ollama:", "custom:ollama:"))
            for item in resp.models.available_models
        )

    @pytest.mark.asyncio
    async def test_named_provider_models_appear_in_model_state(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(
                model="gpt-5.4", provider="openai-codex"
            )
        )
        acp_agent = HermesACPAgent(session_manager=manager)

        with patch(
            "acp_adapter.model_catalog._named_custom_provider_catalogs",
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

    @pytest.mark.asyncio
    async def test_configured_provider_inventory_row_uses_custom_choice_id(self):
        """A ``providers:`` row must not expose its raw config key to ACP."""
        from models import parse_model_ref

        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(model="model-a", provider="relay")
        )
        acp_agent = HermesACPAgent(session_manager=manager)
        cfg = {
            "providers": {
                "relay": {
                    "name": "Relay",
                    "base_url": "https://relay.example/v1",
                }
            }
        }
        inventory = {
            "providers": [{"slug": "relay", "name": "Relay", "is_user_defined": True, "models": ["model-a"]}]
        }

        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.inventory.build_models_payload", return_value=inventory
        ), patch(
            "acp_adapter.model_catalog._named_custom_provider_catalogs",
            return_value=[("custom:relay", "Relay", [("model-a", "")])],
        ):
            resp = await acp_agent.new_session(cwd="/tmp")
            choice_ids = [item.model_id for item in resp.models.available_models]
            ref = parse_model_ref(
                resp.models.current_model_id,
                "relay",
                known_provider_ids={"custom"},
                named_custom_provider_ids={"custom:relay"},
            )

        assert choice_ids == ["custom:relay:model-a"]
        assert ref.provider == "custom:relay"
        assert ref.model == "model-a"

    def test_selector_choice_id_round_trips_through_model_ref(self):
        """The encoded choice id must resolve back to the named provider."""
        from models import parse_model_ref

        choice_id = "custom:bedrock-mantle:openai/gpt-5.5:beta"
        cfg = {
            "providers": {
                "bedrock-mantle": {
                    "name": "AWS Bedrock Mantle",
                    "base_url": "https://bedrock.example/v1",
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            ref = parse_model_ref(
                choice_id,
                "bedrock",
                known_provider_ids={"custom"},
                named_custom_provider_ids={"custom:bedrock-mantle"},
            )
        assert ref.provider == "custom:bedrock-mantle"
        assert ref.model == "openai/gpt-5.5:beta"

    def test_selector_choice_id_round_trips_colon_bearing_custom_identity(self):
        """Configured provider and model IDs may both contain colons."""
        from models import parse_model_ref

        cfg = {
            "providers": {
                "local-127.0.0.1:11434": {
                    "name": "Local Ollama",
                    "base_url": "http://127.0.0.1:11434/v1",
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            ref = parse_model_ref(
                "custom:local-127.0.0.1:11434:qwen3:1.7b",
                "custom",
                known_provider_ids={"custom"},
                named_custom_provider_ids={"custom:local-127.0.0.1:11434"},
            )
        assert ref.provider == "custom:local-127.0.0.1:11434"
        assert ref.model == "qwen3:1.7b"

    @pytest.mark.asyncio
    async def test_named_entry_shadowing_a_canonical_provider_keeps_the_canonical_session(self):
        """``providers.openrouter:`` (a proxy) must not swallow the canonical OpenRouter rows nor
        relabel a session running on openrouter.ai as ``custom:openrouter`` (that id resolves to
        the proxy base_url)."""
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(
                model="model-c", provider="openrouter", base_url="https://openrouter.ai/api/v1")
        )
        acp_agent = HermesACPAgent(session_manager=manager)
        inventory = {"providers": [
            {"slug": "openrouter", "name": "OpenRouter", "is_user_defined": False, "models": ["model-c"]},
            {"slug": "custom:openrouter", "name": "openrouter", "is_user_defined": True,
             "api_url": "https://or.example/api/v1", "models": ["model-a"]},
        ]}

        with patch("hermes_cli.inventory.build_models_payload", return_value=inventory), patch(
            "acp_adapter.model_catalog._named_custom_provider_catalogs",
            return_value=[("custom:openrouter", "openrouter", [("model-a", "")])],
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert resp.models.current_model_id == "openrouter:model-c"
        assert [m.model_id for m in resp.models.available_models] == ["openrouter:model-c", "custom:openrouter:model-a"]

def test_named_catalogue_reads_the_profile_scoped_key_not_launch_environment(monkeypatch):
    from agent import secret_scope

    monkeypatch.setenv("PROFILE_KEY", "launch-profile-key")
    config = _cfg(providers={
        "relay": {
            "name": "Scoped Relay", "base_url": "https://relay.example/v1",
            "key_env": "PROFILE_KEY", "default_model": "relay-model",
        }
    })
    seen = []

    def scoped_key(name, default=""):
        seen.append(name)
        return "routed-profile-key"

    def fetch(**kwargs):
        assert kwargs["api_key"] == "routed-profile-key"
        return EndpointModels(("relay-model", "other"))

    with patch("hermes_cli.config.load_config", return_value=config), patch.object(
        secret_scope, "get_secret", scoped_key,
    ), patch("models.catalog_endpoint.discover_endpoint_models", fetch):
        catalogs = _named_custom_provider_catalogs()
    assert seen == ["PROFILE_KEY"]
    assert catalogs[0][2] == [("relay-model", ""), ("other", "")]


def test_catalogue_refuses_unscoped_multiplex_secret_even_if_launch_env_has_it(monkeypatch):
    from agent import secret_scope

    monkeypatch.setenv("PROFILE_KEY", "launch-profile-key")
    config = _cfg(providers={
        "relay": {
            "name": "Scoped Relay", "base_url": "https://relay.example/v1",
            "key_env": "PROFILE_KEY", "default_model": "relay-model",
        }
    })

    def fail_closed(*_args, **_kwargs):
        raise secret_scope.UnscopedSecretError("PROFILE_KEY")

    with patch("hermes_cli.config.load_config", return_value=config), patch.object(
        secret_scope, "get_secret", fail_closed,
    ):
        with pytest.raises(secret_scope.UnscopedSecretError):
            _named_custom_provider_catalogs()


def test_native_allowlisted_models_survive_successfully_empty_probe(monkeypatch):
    config = _cfg(providers={
        "custom:ollama": {
            "name": "Ollama", "base_url": "http://127.0.0.1:11434/v1",
            "default_model": "chosen:latest", "models": ["chosen:latest"],
        }
    })
    seen = {}

    def fetch(**kwargs):
        seen.update(kwargs)
        return EndpointModels((), native_ollama=True)

    with patch("hermes_cli.config.load_config", return_value=config), patch(
        "models.catalog_endpoint.discover_endpoint_models", fetch,
    ):
        catalogs = _named_custom_provider_catalogs()
    assert seen["preserve_native_models"] is True
    assert catalogs == [("custom:ollama", "Ollama", [("chosen:latest", "")])]
