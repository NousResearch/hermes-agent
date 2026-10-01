"""Tests for Matrix adapter fail-closed approval reaction auth.

When MATRIX_ALLOWED_USERS is not configured, _on_reaction must deny
approval reactions by default unless GATEWAY_ALLOW_ALL_USERS=true.
Mirrors the Telegram _is_callback_user_authorized fix (commit 89d32052e,
PR #28494).
"""

import asyncio
import sys
import types
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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

from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(allowed_user_ids=None):
    """Construct a MatrixAdapter with only the state needed by _on_reaction."""
    adapter = object.__new__(MatrixAdapter)
    adapter._user_id = "@bot:matrix.org"
    adapter._allowed_user_ids = set(allowed_user_ids) if allowed_user_ids else set()
    adapter._approval_reaction_map = {"✅": "once", "❎": "deny"}
    adapter._approval_prompts_by_event = {}
    adapter._approval_prompt_by_session = {}
    adapter._processed_events = deque(maxlen=512)
    adapter._processed_events_set = set()
    return adapter


def _make_event(sender, reacts_to, key="✅"):
    """Minimal Matrix reaction event."""
    return SimpleNamespace(
        sender=sender,
        event_id=f"$reaction-{sender.split(':')[0]}",
        room_id="!testroom:matrix.org",
        content={"m.relates_to": {"event_id": reacts_to, "key": key}},
    )


def _make_prompt(chat_id="!testroom:matrix.org"):
    return _MatrixApprovalPrompt(
        session_key="session-abc",
        chat_id=chat_id,
        message_id="$prompt-event-1",
    )


def _run(adapter, event):
    """Run _on_reaction and return whether the prompt was resolved."""
    prompt_event_id = "$prompt-event-1"
    prompt = _make_prompt()
    adapter._approval_prompts_by_event[prompt_event_id] = prompt
    adapter._redact_bot_approval_reactions = AsyncMock()

    fake_approval = types.ModuleType("tools.approval")
    fake_approval.resolve_gateway_approval = lambda session_key, choice: 1
    with patch.dict(sys.modules, {"tools.approval": fake_approval}):
        asyncio.run(adapter._on_reaction(event))

    return prompt.resolved


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestApprovalReactionFailClosed:
    """_on_reaction approval auth must be fail-closed (parity with Telegram)."""

    def test_no_allowlist_no_allow_all_denies(self, monkeypatch):
        """No MATRIX_ALLOWED_USERS + no GATEWAY_ALLOW_ALL_USERS → deny."""
        monkeypatch.delenv("MATRIX_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        adapter = _make_adapter(allowed_user_ids=None)
        event = _make_event("@stranger:matrix.org", "$prompt-event-1")
        assert _run(adapter, event) is False


def _gateway_wired_adapter():
    """Adapter with the real gateway auth callback, as ``GatewayRunner`` wires it."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.pairing import PairingStore
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.config.platforms = {Platform.MATRIX: PlatformConfig(enabled=True, extra={})}
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {"default": runner.pairing_store}
    runner._primary_profile_name = "default"
    adapter = _make_adapter(allowed_user_ids=None)
    adapter._is_dm_room = AsyncMock(return_value=False)
    runner.adapters = {Platform.MATRIX: adapter}
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.MATRIX))
    return adapter, runner.pairing_store


@pytest.mark.parametrize("grant, reactor, resolves", [
    ("pairing", "@paired:matrix.org", True),
    ("MATRIX_ALLOW_ALL_USERS", "@anyone:matrix.org", True),
    ("none", "@stranger:matrix.org", False),
])
def test_prompt_reaction_follows_the_gateway_authorization(monkeypatch, grant, reactor, resolves):
    """A user the gateway lets send commands can answer their prompt by reaction; others cannot."""
    for key in ("MATRIX_ALLOWED_USERS", "MATRIX_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS",
                "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    adapter, pairing_store = _gateway_wired_adapter()
    if grant == "pairing":
        pairing_store.approve_code("matrix", pairing_store.generate_code("matrix", reactor))
    elif grant == "MATRIX_ALLOW_ALL_USERS":
        monkeypatch.setenv("MATRIX_ALLOW_ALL_USERS", "true")

    assert _run(adapter, _make_event(reactor, "$prompt-event-1")) is resolves


