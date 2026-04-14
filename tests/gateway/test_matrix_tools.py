"""Tests for Matrix LLM-callable tools and registrations."""

import json
from unittest.mock import AsyncMock

import pytest

from agent.prompt_builder import PLATFORM_HINTS
from model_tools import get_all_tool_names, get_toolset_for_tool
from tools.matrix_tools import (
    _handle_create_room,
    _handle_fetch_history,
    _handle_invite_user,
    _handle_redact_message,
    _handle_send_reaction,
    _handle_set_presence,
    set_matrix_adapter,
)
from tools.registry import registry
from toolsets import TOOLSETS, resolve_toolset


class DummyAdapter:
    def __init__(self):
        self.send_reaction = AsyncMock(return_value="$reaction")
        self.redact_message = AsyncMock(return_value=True)
        self.create_room = AsyncMock(return_value="!new:example.org")
        self.invite_user = AsyncMock(return_value=True)
        self.fetch_room_history = AsyncMock(return_value=[{"event_id": "$e1", "body": "hello"}])
        self.set_presence = AsyncMock(return_value=True)
        self._loop = None


@pytest.fixture(autouse=True)
def _clear_matrix_adapter():
    set_matrix_adapter(None)
    yield
    set_matrix_adapter(None)


class TestMatrixToolRegistration:
    def test_model_tools_discovers_matrix_tools(self):
        names = set(get_all_tool_names())
        for name in {
            "matrix_send_reaction",
            "matrix_redact_message",
            "matrix_create_room",
            "matrix_invite_user",
            "matrix_fetch_history",
            "matrix_set_presence",
        }:
            assert name in names
            assert get_toolset_for_tool(name) == "matrix"

    def test_toolset_wiring_is_matrix_specific(self):
        assert "matrix" in TOOLSETS
        assert "matrix" in TOOLSETS["hermes-matrix"]["includes"]
        resolved = set(resolve_toolset("hermes-matrix"))
        assert "matrix_send_reaction" in resolved
        assert "matrix_fetch_history" in resolved

    def test_platform_hint_includes_matrix(self):
        assert "matrix" in PLATFORM_HINTS
        assert "matrix_send_reaction" in PLATFORM_HINTS["matrix"]

    def test_registry_hides_matrix_tools_without_live_adapter(self):
        defs = registry.get_definitions({"matrix_send_reaction", "matrix_fetch_history"}, quiet=True)
        assert defs == []

    def test_registry_exposes_matrix_tools_with_live_adapter(self):
        set_matrix_adapter(DummyAdapter())
        defs = registry.get_definitions({"matrix_send_reaction", "matrix_fetch_history"}, quiet=True)
        fn_names = {d["function"]["name"] for d in defs}
        assert fn_names == {"matrix_send_reaction", "matrix_fetch_history"}


class TestMatrixToolHandlers:
    def test_send_reaction_requires_live_adapter(self):
        result = json.loads(_handle_send_reaction({"room_id": "!r:ex", "event_id": "$e", "emoji": "👍"}))
        assert "error" in result
        assert "not connected" in result["error"].lower()

    def test_send_reaction_uses_public_adapter_api(self):
        adapter = DummyAdapter()
        set_matrix_adapter(adapter)
        result = json.loads(_handle_send_reaction({"room_id": "!r:ex", "event_id": "$e", "emoji": "👍"}))
        assert result["success"] is True
        assert result["reaction_event_id"] == "$reaction"
        adapter.send_reaction.assert_awaited_once_with("!r:ex", "$e", "👍")

    def test_redact_message_validates_ids(self):
        result = json.loads(_handle_redact_message({"room_id": "bad", "event_id": "$e"}))
        assert "error" in result
        result = json.loads(_handle_redact_message({"room_id": "!r:ex", "event_id": "bad"}))
        assert "error" in result

    def test_create_room_blocks_public_rooms_by_default(self, monkeypatch):
        adapter = DummyAdapter()
        set_matrix_adapter(adapter)
        monkeypatch.delenv("MATRIX_ALLOW_PUBLIC_ROOMS", raising=False)
        result = json.loads(_handle_create_room({"name": "Public", "preset": "public_chat"}))
        assert "error" in result
        adapter.create_room.assert_not_called()

    def test_create_room_allows_public_rooms_when_enabled(self, monkeypatch):
        adapter = DummyAdapter()
        set_matrix_adapter(adapter)
        monkeypatch.setenv("MATRIX_ALLOW_PUBLIC_ROOMS", "true")
        result = json.loads(_handle_create_room({"name": "Public", "preset": "public_chat"}))
        assert result["success"] is True
        adapter.create_room.assert_awaited_once()

    def test_invite_user_validates_user_id(self):
        adapter = DummyAdapter()
        set_matrix_adapter(adapter)
        result = json.loads(_handle_invite_user({"room_id": "!r:ex", "user_id": "nope"}))
        assert "error" in result
        adapter.invite_user.assert_not_called()

    def test_fetch_history_delegates_to_existing_adapter_method(self):
        adapter = DummyAdapter()
        set_matrix_adapter(adapter)
        result = json.loads(_handle_fetch_history({"room_id": "!r:ex", "limit": 25, "start": "tok"}))
        assert result["count"] == 1
        adapter.fetch_room_history.assert_awaited_once_with("!r:ex", limit=25, start="tok")

    def test_fetch_history_clamps_invalid_limit(self):
        adapter = DummyAdapter()
        set_matrix_adapter(adapter)
        _handle_fetch_history({"room_id": "!r:ex", "limit": 999})
        adapter.fetch_room_history.assert_awaited_once_with("!r:ex", limit=200, start="")

    def test_set_presence_validates_state(self):
        adapter = DummyAdapter()
        set_matrix_adapter(adapter)
        result = json.loads(_handle_set_presence({"state": "busy"}))
        assert "error" in result
        adapter.set_presence.assert_not_called()

    def test_set_presence_calls_adapter(self):
        adapter = DummyAdapter()
        set_matrix_adapter(adapter)
        result = json.loads(_handle_set_presence({"state": "online", "status_msg": "ready"}))
        assert result["success"] is True
        adapter.set_presence.assert_awaited_once_with(state="online", status_msg="ready")
