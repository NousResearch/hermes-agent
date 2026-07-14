"""Tests for Honcho base_context injection config."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


class TestBaseContextConfigParsing:
    def test_defaults_preserve_existing_behavior(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({"apiKey": "test-key"}))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.base_context_include_session_summary is True
        assert cfg.base_context_user_include_observations is True
        assert cfg.base_context_user_include_peer_card is True
        assert cfg.base_context_ai_include_observations is True
        assert cfg.base_context_ai_include_peer_card is True

    def test_parses_legacy_honcho_json_snake_case(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "base_context": {
                "include_session_summary": False,
                "user_representation": {
                    "include_observations": False,
                    "include_peer_card": True,
                },
                "ai_representation": {
                    "include_observations": False,
                    "include_peer_card": False,
                },
            },
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.base_context_include_session_summary is False
        assert cfg.base_context_user_include_observations is False
        assert cfg.base_context_user_include_peer_card is True
        assert cfg.base_context_ai_include_observations is False
        assert cfg.base_context_ai_include_peer_card is False

    def test_parses_legacy_honcho_json_camel_case(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "baseContext": {
                "includeSessionSummary": False,
                "userRepresentation": {
                    "includeObservations": False,
                    "includePeerCard": True,
                },
                "aiRepresentation": {
                    "includeObservations": False,
                    "includePeerCard": False,
                },
            },
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.base_context_include_session_summary is False
        assert cfg.base_context_user_include_observations is False
        assert cfg.base_context_user_include_peer_card is True
        assert cfg.base_context_ai_include_observations is False
        assert cfg.base_context_ai_include_peer_card is False

    def test_host_block_overrides_root_per_leaf(self, tmp_path):
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "base_context": {
                "user_representation": {
                    "include_observations": False,
                    "include_peer_card": False,
                },
            },
            "hosts": {
                "hermes": {
                    "base_context": {
                        "user_representation": {"include_peer_card": True}
                    }
                }
            },
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.base_context_user_include_observations is False
        assert cfg.base_context_user_include_peer_card is True

    def test_host_block_camel_case_overrides_root(self, tmp_path):
        """Host-level camelCase baseContext overrides root snake_case per leaf.

        The parser accepts both camelCase and snake_case at every level
        (root and host).  This test covers the host-level camelCase path
        so the 'accepted everywhere' contract is exercised distinctly from
        the snake_case host override test above.
        """
        config_path = tmp_path / "honcho.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "baseContext": {
                "includeSessionSummary": False,
                "aiRepresentation": {
                    "includeObservations": False,
                    "includePeerCard": False,
                },
            },
            "hosts": {
                "hermes": {
                    "baseContext": {
                        "aiRepresentation": {"includePeerCard": True}
                    }
                }
            },
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.base_context_include_session_summary is False
        assert cfg.base_context_ai_include_observations is False
        assert cfg.base_context_ai_include_peer_card is True


class TestBaseContextFetchGates:
    def _manager(self, cfg: HonchoClientConfig) -> HonchoSessionManager:
        manager = HonchoSessionManager(honcho=MagicMock(), config=cfg)
        manager._cache["session"] = HonchoSession(
            key="session",
            user_peer_id="user-peer",
            assistant_peer_id="ai-peer",
            honcho_session_id="honcho-session",
        )
        return manager

    def test_card_only_uses_peer_card_without_peer_context(self):
        cfg = HonchoClientConfig(
            base_context_include_session_summary=False,
            base_context_user_include_observations=False,
            base_context_user_include_peer_card=True,
            base_context_ai_include_observations=False,
            base_context_ai_include_peer_card=False,
        )
        manager = self._manager(cfg)
        peer = MagicMock()
        peer.get_card.return_value = ["Name: Daniel"]
        manager._get_or_create_peer = MagicMock(return_value=peer)

        result = manager.get_prefetch_context("session", "current query")

        assert result == {"card": "Name: Daniel"}
        peer.context.assert_not_called()
        peer.representation.assert_not_called()
        peer.get_card.assert_called_once_with(target="user-peer")
        manager._get_or_create_peer.assert_any_call("user-peer")
        assert manager._get_or_create_peer.call_count == 1

    def test_observations_and_card_use_single_peer_context_call(self):
        cfg = HonchoClientConfig(
            base_context_include_session_summary=False,
            base_context_user_include_observations=True,
            base_context_user_include_peer_card=True,
            base_context_ai_include_observations=False,
            base_context_ai_include_peer_card=False,
        )
        manager = self._manager(cfg)
        peer = MagicMock()
        peer.context.return_value = SimpleNamespace(
            representation="topic relevant representation",
            peer_card=["Name: Daniel"],
        )
        manager._get_or_create_peer = MagicMock(return_value=peer)

        result = manager.get_prefetch_context("session", "current query")

        assert result == {
            "representation": "topic relevant representation",
            "card": "Name: Daniel",
        }
        peer.context.assert_called_once_with(target="user-peer", search_query="current query")
        peer.get_card.assert_not_called()

    def test_all_peer_fields_disabled_skips_peer_fetch(self):
        cfg = HonchoClientConfig(
            base_context_include_session_summary=False,
            base_context_user_include_observations=False,
            base_context_user_include_peer_card=False,
            base_context_ai_include_observations=False,
            base_context_ai_include_peer_card=False,
        )
        manager = self._manager(cfg)
        manager._get_or_create_peer = MagicMock()

        result = manager.get_prefetch_context("session", "current query")

        assert result == {}
        manager._get_or_create_peer.assert_not_called()

    def test_session_summary_can_be_disabled(self):
        cfg = HonchoClientConfig(
            base_context_include_session_summary=False,
            base_context_user_include_observations=False,
            base_context_user_include_peer_card=False,
            base_context_ai_include_observations=False,
            base_context_ai_include_peer_card=False,
        )
        manager = self._manager(cfg)
        honcho_session = MagicMock()
        manager._sessions_cache["honcho-session"] = honcho_session

        result = manager.get_prefetch_context("session")

        assert result == {}
        honcho_session.context.assert_not_called()

    def test_session_summary_included_by_default_when_available(self):
        cfg = HonchoClientConfig(
            base_context_user_include_observations=False,
            base_context_user_include_peer_card=False,
            base_context_ai_include_observations=False,
            base_context_ai_include_peer_card=False,
        )
        manager = self._manager(cfg)
        honcho_session = MagicMock()
        honcho_session.context.return_value = SimpleNamespace(
            summary=SimpleNamespace(content="Previously on Honcho..."),
        )
        manager._sessions_cache["honcho-session"] = honcho_session

        result = manager.get_prefetch_context("session")

        assert result == {"summary": "Previously on Honcho..."}
        honcho_session.context.assert_called_once_with(summary=True)
