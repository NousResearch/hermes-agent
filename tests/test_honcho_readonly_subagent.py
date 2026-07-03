"""Read-only profile delegation for the Honcho provider (issue #41889).

A profile-backed subagent reads the target profile's Honcho memory but must
NOT write it back. The provider stays active for recall while every write
surface — automatic turn sync, the conclusion mirror, the session-end flush,
and the explicit peer-card / conclusion tools — is gated on ``_write_enabled``,
which ``initialize`` derives from ``agent_context="subagent"``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from plugins.memory.honcho import HonchoMemoryProvider


class _Cfg(SimpleNamespace):
    def resolve_session_name(self, **kwargs):
        return "test-session"


def _disabled_cfg() -> _Cfg:
    # enabled=False makes initialize() return right after it sets _write_enabled,
    # so we exercise the flag without standing up a real Honcho backend.
    return _Cfg(enabled=False, api_key=None, base_url=None)


def test_initialize_subagent_context_disables_writes(monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: _disabled_cfg(),
    )
    p = HonchoMemoryProvider()
    p.initialize("s", agent_context="subagent")
    assert p._write_enabled is False


def test_initialize_primary_context_keeps_writes_enabled(monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: _disabled_cfg(),
    )
    p = HonchoMemoryProvider()
    p.initialize("s", agent_context="primary")
    assert p._write_enabled is True


def _read_only_provider() -> HonchoMemoryProvider:
    p = HonchoMemoryProvider()
    p._cron_skipped = False
    p._write_enabled = False
    p._session_initialized = True
    p._session_key = "k"
    p._manager = MagicMock()
    return p


def test_sync_turn_skipped_when_read_only():
    p = _read_only_provider()
    p.sync_turn("user said", "assistant said")
    p._manager.get_or_create.assert_not_called()


def test_on_memory_write_skipped_when_read_only():
    p = _read_only_provider()
    p.on_memory_write("add", "user", "a durable fact")
    p._manager.create_conclusion.assert_not_called()


def test_conclude_tool_refused_when_read_only():
    p = _read_only_provider()
    out = json.loads(p.handle_tool_call("honcho_conclude", {"conclusion": "x"}))
    assert "read-only" in out.get("error", "")
    p._manager.create_conclusion.assert_not_called()


def test_peer_card_update_refused_but_read_allowed_when_read_only():
    p = _read_only_provider()

    # Write (card update) is refused, and set_peer_card is never called.
    out = json.loads(
        p.handle_tool_call("honcho_profile", {"peer": "user", "card": ["fact"]})
    )
    assert "read-only" in out.get("error", "")
    p._manager.set_peer_card.assert_not_called()

    # Read (no card) still works.
    p._manager.get_peer_card.return_value = "Jordan prefers concise docs"
    out = json.loads(p.handle_tool_call("honcho_profile", {"peer": "user"}))
    assert out["result"] == "Jordan prefers concise docs"
    p._manager.get_peer_card.assert_called_once()
