"""Tests for honcho_profile's empty-card hint (#5137 follow-up)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from plugins.memory.honcho import HonchoMemoryProvider


def _make_provider(**cfg_overrides) -> HonchoMemoryProvider:
    provider = HonchoMemoryProvider()
    provider._manager = MagicMock()
    provider._manager.get_peer_card.return_value = []  # empty card
    provider._session_key = "agent:main:test"
    provider._session_initialized = True  # bypass the lazy _ensure_session() gate
    provider._cron_skipped = False

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
    def test_returns_hint_not_bare_error_message(self):
        provider = _make_provider()
        raw = provider.handle_tool_call("honcho_profile", {})
        payload = json.loads(raw)
        assert payload["result"] == "No profile facts available yet."
        assert "hint" in payload
        assert "not an error" in payload["hint"].lower()

    def test_hint_mentions_warmup_when_turn_count_below_cadence(self):
        provider = _make_provider(turn_count=1, dialectic_cadence=3)
        raw = provider.handle_tool_call("honcho_profile", {})
        payload = json.loads(raw)
        assert "turn" in payload["hint"].lower()
        assert "cadence" in payload["hint"].lower()


    def test_populated_card_returns_card_without_hint(self):
        """Regression: a populated card should NOT trigger the hint path."""
        provider = _make_provider()
        provider._manager.get_peer_card.return_value = ["Fact 1", "Fact 2"]
        raw = provider.handle_tool_call("honcho_profile", {})
        payload = json.loads(raw)
        assert payload["result"] == ["Fact 1", "Fact 2"]
        assert "hint" not in payload
