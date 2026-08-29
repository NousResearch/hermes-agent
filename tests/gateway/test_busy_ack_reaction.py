"""Regression tests for reaction-based busy acks (busy_ack_reaction).

When display.platforms.<platform>.busy_ack_reaction is enabled and the
adapter implements send_reaction(chat_id, message_id, emoji), a busy-input
acknowledgment is delivered as a mode-aware emoji reaction on the user's
message instead of the usual text bubble. Failure or absence falls back to
the text bubble.
"""
from __future__ import annotations

import sys
import threading
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Minimal telegram stubs so gateway imports cleanly (mirrors sibling tests).
_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (  # noqa: E402
    MessageEvent,
    MessageType,
    SessionSource,
    build_session_key,
)
from gateway.run import GatewayRunner  # noqa: E402


def _make_event() -> MessageEvent:
    source = SessionSource(
        platform=MagicMock(value="buzz"),
        chat_id="chan-1",
        chat_type="group",
        user_id="user1",
    )
    return MessageEvent(
        text="turn left instead",
        message_type=MessageType.TEXT,
        source=source,
        message_id="ev-42",
    )


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._busy_input_mode = "interrupt"
    runner._session_db = MagicMock()
    runner._session_db._db = MagicMock()
    runner._session_db._db.get_compression_lock_holder.return_value = None
    return runner


def _make_adapter(with_reaction: bool) -> MagicMock:
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value="buzz")
    if with_reaction:
        adapter.send_reaction = AsyncMock(return_value=True)
    else:
        del adapter.send_reaction  # base adapters lack the method
    return adapter


def _make_running_parent() -> MagicMock:
    parent = MagicMock()
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 4,
        "max_iterations": 150,
        "current_tool": "terminal",
    }
    # Advertise active-turn redirect support so interrupt-mode messages are
    # redirected (the "↪️" ack path) rather than plain interrupted ("⚡").
    parent._supports_active_turn_redirect = True
    parent.redirect = MagicMock(return_value=True)
    return parent


async def _drive_busy_path(
    monkeypatch, *, with_reaction: bool, display_cfg: dict, reaction_mock=None
) -> tuple[MagicMock, MagicMock]:
    """Run the busy handler end-to-end and return (adapter, parent)."""
    runner = _make_runner()
    adapter = _make_adapter(with_reaction=with_reaction)
    if reaction_mock is not None:
        adapter.send_reaction = reaction_mock
    event = _make_event()
    sk = build_session_key(event.source)
    parent = _make_running_parent()
    runner._running_agents[sk] = parent
    runner._running_agents_ts[sk] = 1.0
    runner.adapters[event.source.platform] = adapter
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"display": display_cfg},
    )
    handled = await runner._handle_active_session_busy_message(event, sk)
    assert handled is True
    return adapter, parent


@pytest.mark.asyncio
async def test_redirect_ack_uses_reaction_when_enabled(monkeypatch) -> None:
    adapter, _ = await _drive_busy_path(
        monkeypatch,
        with_reaction=True,
        display_cfg={"platforms": {"buzz": {"busy_ack_reaction": True}}},
    )
    adapter.send_reaction.assert_awaited_once()
    kwargs = adapter.send_reaction.await_args.kwargs
    assert kwargs["message_id"] == "ev-42"
    assert kwargs["emoji"] == "↪️"
    # No text bubble may follow a successful reaction ack.
    adapter._send_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_reaction_ack_falls_back_to_text_when_unsupported(monkeypatch) -> None:
    adapter, _ = await _drive_busy_path(
        monkeypatch,
        with_reaction=False,
        display_cfg={"platforms": {"buzz": {"busy_ack_reaction": True}}},
    )
    adapter._send_with_retry.assert_awaited_once()  # text bubble used


@pytest.mark.asyncio
async def test_reaction_ack_skipped_when_disabled(monkeypatch) -> None:
    adapter, _ = await _drive_busy_path(
        monkeypatch,
        with_reaction=True,
        display_cfg={},
    )
    adapter.send_reaction.assert_not_awaited()
    adapter._send_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_reaction_ack_failure_falls_back_to_text(monkeypatch) -> None:
    adapter, _ = await _drive_busy_path(
        monkeypatch,
        with_reaction=True,
        display_cfg={"platforms": {"buzz": {"busy_ack_reaction": True}}},
        reaction_mock=AsyncMock(side_effect=RuntimeError("relay down")),
    )
    adapter._send_with_retry.assert_awaited_once()  # fallback bubble sent
