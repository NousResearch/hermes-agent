"""Tests for honcho_profile's empty-card hint (#5137 follow-up)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from plugins.memory.honcho import HonchoMemoryProvider


def _make_provider(**cfg_overrides) -> HonchoMemoryProvider:
    provider = HonchoMemoryProvider()
    provider._manager = MagicMock()
    provider._manager.get_peer_card.return_value = []  # empty card
    provider._manager.resolve_peer_id.side_effect = lambda _key, peer: "runi" if peer == "user" else peer
    provider._session_key = "agent:main:test"
    provider._session_initialized = True  # bypass the lazy _ensure_session() gate
    provider._cron_skipped = False
    provider._platform = cfg_overrides.get("platform", "cli")

    cfg = MagicMock()
    # Defaults match HonchoClientConfig defaults
    cfg.user_observe_me = cfg_overrides.get("user_observe_me", True)
    cfg.user_observe_others = cfg_overrides.get("user_observe_others", True)
    cfg.ai_observe_me = cfg_overrides.get("ai_observe_me", True)
    cfg.ai_observe_others = cfg_overrides.get("ai_observe_others", True)
    cfg.message_max_chars = 25000
    provider._config = cfg

    provider._dialectic_cadence = cfg_overrides.get("dialectic_cadence", 1)
    provider._turn_count = cfg_overrides.get("turn_count", 5)
    return provider


class TestEmptyProfileHint:
    def test_returns_hint_not_bare_error_message(self, monkeypatch):
        monkeypatch.delenv("HONCHO_PROFILE_RESOURCE_URI", raising=False)
        provider = _make_provider(platform="api_server")
        raw = provider.handle_tool_call("honcho_profile", {})
        payload = json.loads(raw)
        assert payload["result"] == "No profile facts available yet."
        assert "resolvedPeer" not in payload
        assert "_meta" not in payload
        assert "hint" in payload
        assert "not an error" in payload["hint"].lower()

    def test_empty_card_emits_opt_in_resource_metadata(self, monkeypatch):
        monkeypatch.setenv("HONCHO_PROFILE_RESOURCE_URI", "ui://hugin/peer-card")
        provider = _make_provider(platform="api_server")
        payload = json.loads(provider.handle_tool_call("honcho_profile", {}))
        assert payload["resolvedPeer"] == "runi"
        assert payload["_meta"]["ui"]["resourceUri"] == "ui://hugin/peer-card"

    def test_hint_mentions_warmup_when_turn_count_below_cadence(self):
        provider = _make_provider(turn_count=1, dialectic_cadence=3)
        raw = provider.handle_tool_call("honcho_profile", {})
        payload = json.loads(raw)
        assert "turn" in payload["hint"].lower()
        assert "cadence" in payload["hint"].lower()


    def test_resource_metadata_is_api_server_only(self, monkeypatch):
        monkeypatch.setenv("HONCHO_PROFILE_RESOURCE_URI", "ui://hugin/peer-card")
        provider = _make_provider(platform="cli")
        payload = json.loads(provider.handle_tool_call("honcho_profile", {}))
        assert "_meta" not in payload

    def test_dynamic_manager_resolver_falls_back_to_requested_peer(self, monkeypatch):
        monkeypatch.setenv("HONCHO_PROFILE_RESOURCE_URI", "ui://hugin/peer-card")
        provider = _make_provider(platform="api_server")
        provider._manager = MagicMock()
        provider._manager.get_peer_card.return_value = ["Fact"]
        payload = json.loads(provider.handle_tool_call("honcho_profile", {"peer": "runi_user"}))
        assert payload["resolvedPeer"] == "runi_user"

    def test_populated_card_returns_card_without_hint(self, monkeypatch):
        """Regression: a populated card should NOT trigger the hint path."""
        monkeypatch.setenv("HONCHO_PROFILE_RESOURCE_URI", "ui://hugin/peer-card")
        provider = _make_provider(platform="api_server")
        provider._manager.get_peer_card.return_value = ["Fact 1", "Fact 2"]
        raw = provider.handle_tool_call("honcho_profile", {})
        payload = json.loads(raw)
        assert payload["result"] == ["Fact 1", "Fact 2"]
        assert payload["peer"] == "user"
        assert payload["resolvedPeer"] == "runi"
        assert payload["_meta"] == {
            "ui": {"resourceUri": "ui://hugin/peer-card"},
            "mimeType": "text/html;profile=mcp-app",
        }
        assert "hint" not in payload
