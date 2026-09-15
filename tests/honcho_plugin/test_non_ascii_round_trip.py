"""Regression: honcho tool payloads must keep non-ASCII literal (#99806)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from plugins.memory.honcho import HonchoMemoryProvider

_KO = "한국어 관측"
_JA = "日本語のテスト"


def _make_provider() -> HonchoMemoryProvider:
    provider = HonchoMemoryProvider()
    provider._manager = MagicMock()
    provider._session_key = "agent:main:test"
    provider._session_initialized = True  # bypass the lazy _ensure_session() gate
    provider._cron_skipped = False

    cfg = MagicMock()
    cfg.user_observe_me = True
    cfg.user_observe_others = True
    cfg.ai_observe_me = True
    cfg.ai_observe_others = True
    cfg.message_max_chars = 25000
    provider._config = cfg

    provider._dialectic_cadence = 1
    provider._turn_count = 5
    return provider


class TestNonAsciiRoundTrip:
    def test_peer_card_is_not_escaped(self):
        provider = _make_provider()
        provider._manager.get_peer_card.return_value = [_KO, _JA]

        raw = provider.handle_tool_call("honcho_profile", {})

        assert "\\u" not in raw
        assert _KO in raw
        assert json.loads(raw)["result"] == [_KO, _JA]

    def test_search_result_is_not_escaped(self):
        provider = _make_provider()
        provider._manager.search_context.return_value = _KO

        raw = provider.handle_tool_call("honcho_search", {"query": "q"})

        assert "\\u" not in raw
        assert json.loads(raw)["result"] == _KO
