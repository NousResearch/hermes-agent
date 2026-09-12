"""Bare configured custom-provider keys in ACP ``session/set_model``.

``parse_model_input`` gated the ``provider:model`` split on the hardcoded
``_KNOWN_PROVIDER_NAMES``, so the bare key of a user-configured
``providers:`` entry fell through to ``(current_provider, <full string>)``.
``set_session_model`` then kept the old endpoint and inference hit the
previous relay.
"""

from types import SimpleNamespace
from unittest.mock import patch

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


def _cfg(providers=None, custom_providers=None):
    cfg = {}
    if providers is not None:
        cfg["providers"] = providers
    if custom_providers is not None:
        cfg["custom_providers"] = custom_providers
    return cfg


RELAY_PROVIDERS = {
    "relay": {
        "name": "Relay",
        "base_url": "https://relay.example/v1",
    }
}


class TestBareConfiguredCustomProviderKey:

    def test_routes_to_named_provider(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.models.detect_provider_for_model", lambda m, c: None)
        with patch("hermes_cli.config.load_config", return_value=_cfg(providers=RELAY_PROVIDERS)):
            assert HermesACPAgent._resolve_model_selection(
                "relay:meta/llama-3.3-70b", "opencode"
            ) == ("custom:relay", "meta/llama-3.3-70b")

    def test_rebuilds_endpoint_across_switch(self, monkeypatch):
        """``keep_endpoint`` must not carry the old relay over the switch."""
        monkeypatch.setattr("hermes_cli.models.detect_provider_for_model", lambda m, c: None)
        calls = []
        real_make_agent = SessionManager._make_agent

        def spy(self, **kwargs):
            calls.append(kwargs)
            return real_make_agent(self, **kwargs)

        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(provider=None, model=None)
        )
        acp_agent = HermesACPAgent(session_manager=manager)
        state = manager.create_session(cwd="/tmp")
        state.agent.provider = "opencode"
        state.agent.base_url = "https://opencode.example/v1"
        state.agent.api_mode = "responses"

        with patch("hermes_cli.config.load_config", return_value=_cfg(providers=RELAY_PROVIDERS)), patch.object(
            SessionManager, "_make_agent", spy
        ):
            old, new, model = acp_agent._switch_model(
                state, "relay:meta/llama-3.3-70b", keep_endpoint=True
            )

        assert (old, new, model) == ("opencode", "custom:relay", "meta/llama-3.3-70b")
        assert calls[-1]["requested_provider"] == "custom:relay"
        assert calls[-1]["model"] == "meta/llama-3.3-70b"
        assert "base_url" not in calls[-1]
        assert "api_mode" not in calls[-1]

    def test_canonical_provider_name_wins_over_same_named_user_key(self):
        """A ``providers:`` key named like a canonical provider must not hijack it."""
        from hermes_cli.models import parse_model_input

        cfg = _cfg(
            providers={
                "anthropic": {"name": "anthropic", "base_url": "https://relay.example/v1"},
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert parse_model_input("anthropic:claude-3.5-sonnet", "opencode") == (
                "anthropic",
                "claude-3.5-sonnet",
            )

    def test_unknown_prefix_keeps_legacy_fall_through(self):
        from hermes_cli.models import parse_model_input

        with patch("hermes_cli.config.load_config", return_value=_cfg(providers=RELAY_PROVIDERS)):
            assert parse_model_input("foo:bar", "opencode") == ("opencode", "foo:bar")

    def test_colon_bearing_provider_key_only_routes_via_custom_prefix(self):
        from hermes_cli.models import parse_model_input

        cfg = _cfg(
            providers={
                "local-127.0.0.1:11434": {
                    "name": "Local Ollama",
                    "base_url": "http://127.0.0.1:11434/v1",
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert parse_model_input("local-127.0.0.1:11434:qwen3:1.7b", "opencode") == (
                "opencode",
                "local-127.0.0.1:11434:qwen3:1.7b",
            )
            assert parse_model_input("custom:local-127.0.0.1:11434:qwen3:1.7b", "opencode") == (
                "custom:local-127.0.0.1:11434",
                "qwen3:1.7b",
            )
