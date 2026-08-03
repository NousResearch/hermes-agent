"""Tests for the Matrix adapter's agent-facing add_reaction/remove_reaction.

``send_message(action="react")`` dispatches by duck-typing public
``add_reaction``/``remove_reaction`` coroutines on the live adapter
(tools/send_message_tool.py). Photon ships the pair; Matrix had only the
private ``_send_reaction`` machinery, so reacting on Matrix errored
"Platform 'matrix' does not support message reactions" despite the adapter
being able to post native ``m.reaction`` annotations. These tests pin the
public pair: default-target resolution, explicit ``message_id``, unreact
redaction of our own annotation, and the no-target error path.
"""

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Stub mautrix so plugins.platforms.matrix.adapter can be imported without the SDK.
# ---------------------------------------------------------------------------

def _stub_mautrix():
    stub = types.ModuleType("mautrix")
    for sub in ("mautrix.types", "mautrix.client", "mautrix.client.api",
                "mautrix.errors", "mautrix.crypto", "mautrix.util",
                "mautrix.util.config"):
        sys.modules.setdefault(sub, types.ModuleType(sub))
    sys.modules.setdefault("mautrix", stub)
    m = sys.modules["mautrix.types"]

    class EventType:
        ROOM_MESSAGE = "m.room.message"
        REACTION = "m.reaction"
        ROOM_ENCRYPTED = "m.room.encrypted"
        ROOM_NAME = "m.room.name"

    class PaginationDirection:
        BACKWARD = "b"
        FORWARD = "f"

    class PresenceState:
        ONLINE = "online"
        OFFLINE = "offline"
        UNAVAILABLE = "unavailable"

    class RoomCreatePreset:
        PRIVATE = "private_chat"
        PUBLIC = "public_chat"
        TRUSTED_PRIVATE = "trusted_private_chat"

    class TrustState:
        UNVERIFIED = 0
        VERIFIED = 1

    for attr in ("ContentURI", "EventID", "RoomID", "SyncToken", "UserID"):
        setattr(m, attr, str)
    m.EventType = EventType
    m.PaginationDirection = PaginationDirection
    m.PresenceState = PresenceState
    m.RoomCreatePreset = RoomCreatePreset
    m.TrustState = TrustState


_stub_mautrix()

from plugins.platforms.matrix.adapter import MatrixAdapter  # noqa: E402


ROOM = "!testroom:matrix.org"


def _make_adapter():
    """Construct a MatrixAdapter with only the reaction state under test."""
    adapter = object.__new__(MatrixAdapter)
    adapter._reactions_enabled = True
    adapter._pending_reactions = {}
    adapter._last_inbound_by_room = {}
    adapter._agent_reactions = {}
    adapter._send_reaction = AsyncMock(return_value="$reaction-1")
    adapter._redact_reaction = AsyncMock(return_value=True)
    return adapter


def _run(coro):
    return asyncio.run(coro)


def test_add_reaction_defaults_to_last_inbound():
    adapter = _make_adapter()
    adapter._last_inbound_by_room[ROOM] = "$inbound-42"
    result = _run(adapter.add_reaction(chat_id=ROOM, emoji="❤️"))
    assert result == {"success": True, "message_id": "$inbound-42"}
    adapter._send_reaction.assert_awaited_once_with(ROOM, "$inbound-42", "❤️")
    assert adapter._agent_reactions[(ROOM, "$inbound-42")] == "$reaction-1"


def test_add_reaction_explicit_message_id_wins():
    adapter = _make_adapter()
    adapter._last_inbound_by_room[ROOM] = "$inbound-42"
    result = _run(adapter.add_reaction(chat_id=ROOM, emoji="🔥", message_id="$older-7"))
    assert result["success"] is True
    adapter._send_reaction.assert_awaited_once_with(ROOM, "$older-7", "🔥")


def test_add_reaction_without_target_errors():
    adapter = _make_adapter()
    result = _run(adapter.add_reaction(chat_id=ROOM, emoji="👍"))
    assert result["success"] is False
    assert "message_id" in result["error"]
    adapter._send_reaction.assert_not_awaited()


def test_add_reaction_send_failure_is_reported():
    adapter = _make_adapter()
    adapter._last_inbound_by_room[ROOM] = "$inbound-42"
    adapter._send_reaction = AsyncMock(return_value=None)
    result = _run(adapter.add_reaction(chat_id=ROOM, emoji="❤️"))
    assert result["success"] is False
    assert (ROOM, "$inbound-42") not in adapter._agent_reactions


def test_remove_reaction_redacts_own_annotation():
    adapter = _make_adapter()
    adapter._last_inbound_by_room[ROOM] = "$inbound-42"
    _run(adapter.add_reaction(chat_id=ROOM, emoji="❤️"))
    result = _run(adapter.remove_reaction(chat_id=ROOM))
    assert result == {"success": True, "message_id": "$inbound-42"}
    adapter._redact_reaction.assert_awaited_once()
    args = adapter._redact_reaction.await_args.args
    assert args[0] == ROOM
    assert args[1] == "$reaction-1"
    assert adapter._agent_reactions == {}


def test_remove_reaction_without_own_reaction_errors():
    adapter = _make_adapter()
    adapter._last_inbound_by_room[ROOM] = "$inbound-42"
    result = _run(adapter.remove_reaction(chat_id=ROOM))
    assert result["success"] is False
    adapter._redact_reaction.assert_not_awaited()


def test_processing_start_records_target_even_when_reactions_muted():
    """MATRIX_REACTIONS=false mutes the lifecycle tapbacks, not agent intent."""
    adapter = _make_adapter()
    adapter._reactions_enabled = False
    event = SimpleNamespace(
        message_id="$inbound-99",
        source=SimpleNamespace(chat_id=ROOM),
    )
    _run(adapter.on_processing_start(event))
    assert adapter._last_inbound_by_room[ROOM] == "$inbound-99"
    adapter._send_reaction.assert_not_awaited()


def test_send_message_react_dispatches_to_matrix_adapter():
    """The generic dispatch resolves matrix's new public pair end to end."""
    import json
    from unittest.mock import patch

    import tools.send_message_tool as smt
    from gateway.config import Platform

    adapter = _make_adapter()
    adapter._last_inbound_by_room[ROOM] = "$inbound-42"
    runner = SimpleNamespace(adapters={Platform("matrix"): adapter})
    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        result = json.loads(
            smt.send_message_tool(
                {"action": "react", "target": f"matrix:{ROOM}", "emoji": "❤️"}
            )
        )
    assert result["success"] is True
    assert result["message_id"] == "$inbound-42"
