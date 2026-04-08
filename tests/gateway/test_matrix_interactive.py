"""Tests for Matrix interactive features — thinking fields, approval buttons, model picker.

Covers:
  - matrix_thinking.py — ThinkingManager for collapsible introspection fields
  - matrix.py — thinking methods, send_exec_approval, send_model_picker,
    _on_reaction dispatch for approval and model selection
"""

import asyncio
import importlib
import re
import sys
import time
import types
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform, PlatformConfig


def _make_fake_nio():
    """Create a lightweight fake ``nio`` module for isinstance checks."""
    mod = types.ModuleType("nio")

    class RoomSendResponse:
        def __init__(self, event_id="$fake_event"):
            self.event_id = event_id

    class RoomRedactResponse:
        pass

    mod.RoomSendResponse = RoomSendResponse
    mod.RoomRedactResponse = RoomRedactResponse
    return mod


def _make_adapter():
    """Create a MatrixAdapter with mocked config for testing."""
    from gateway.platforms.matrix import MatrixAdapter
    config = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        },
    )
    return MatrixAdapter(config)


# ===========================================================================
# ThinkingManager lifecycle tests
# ===========================================================================

class TestThinkingManager:
    """Tests for the collapsible introspection field manager."""

    def setup_method(self):
        self.nio = _make_fake_nio()
        self._nio_patcher = patch.dict("sys.modules", {"nio": self.nio})
        self._nio_patcher.start()
        if "gateway.platforms.matrix_thinking" in sys.modules:
            importlib.reload(sys.modules["gateway.platforms.matrix_thinking"])
        self.adapter = MagicMock()
        self.adapter._client = MagicMock()
        self.adapter._client.room_send = AsyncMock(
            return_value=self.nio.RoomSendResponse("$thinking_evt")
        )

    def teardown_method(self):
        self._nio_patcher.stop()

    def _make_manager(self):
        from gateway.platforms.matrix_thinking import ThinkingManager
        return ThinkingManager(self.adapter)

    @pytest.mark.asyncio
    async def test_start_creates_session(self):
        mgr = self._make_manager()
        evt_id = await mgr.start("!room:ex", "task1", "Working...")
        assert evt_id == "$thinking_evt"
        assert mgr.has_session("task1")

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        mgr = self._make_manager()
        evt1 = await mgr.start("!room:ex", "task1")
        evt2 = await mgr.start("!room:ex", "task1")
        assert evt1 == evt2
        assert self.adapter._client.room_send.call_count == 1

    @pytest.mark.asyncio
    async def test_finalize_removes_session(self):
        mgr = self._make_manager()
        await mgr.start("!room:ex", "task1")
        await mgr.finalize("task1", "Done")
        assert not mgr.has_session("task1")

    @pytest.mark.asyncio
    async def test_abort_marks_warning(self):
        mgr = self._make_manager()
        await mgr.start("!room:ex", "task1")
        await mgr.abort("task1", "Timed out")
        assert not mgr.has_session("task1")
        assert self.adapter._client.room_send.call_count >= 2

    @pytest.mark.asyncio
    async def test_update_increments_step(self):
        mgr = self._make_manager()
        await mgr.start("!room:ex", "task1")
        await mgr.update("task1", "Step 1", "doing stuff")
        key = mgr._session_key("task1", "thinking")
        session = mgr._sessions.get(key)
        assert session is not None
        assert session.step_count == 1

    @pytest.mark.asyncio
    async def test_tool_activity_field_kind(self):
        mgr = self._make_manager()
        evt_id = await mgr.start("!room:ex", "task1", field_kind="tools")
        assert evt_id == "$thinking_evt"
        assert mgr.has_session("task1", "tools")

    @pytest.mark.asyncio
    async def test_abort_all(self):
        mgr = self._make_manager()
        self.adapter._client.room_send = AsyncMock(
            side_effect=[
                self.nio.RoomSendResponse("$evt1"),
                self.nio.RoomSendResponse("$evt2"),
                self.nio.RoomSendResponse("$edit1"),
                self.nio.RoomSendResponse("$edit2"),
            ]
        )
        await mgr.start("!room:ex", "task1", field_kind="thinking")
        await mgr.start("!room:ex", "task1", field_kind="tools")
        await mgr.abort_all("shutdown")
        assert len(mgr._sessions) == 0

    @pytest.mark.asyncio
    async def test_cleanup_stale(self):
        mgr = self._make_manager()
        await mgr.start("!room:ex", "task1")
        key = mgr._session_key("task1", "thinking")
        mgr._sessions[key].started_at = time.time() - 3600
        await mgr.cleanup_stale(max_age=1800)
        assert not mgr.has_session("task1")


# ===========================================================================
# ThinkingManager HTML generation
# ===========================================================================

class TestThinkingManagerHtml:
    """Tests for HTML generation — especially width parity fix."""

    def _make_manager(self):
        from gateway.platforms.matrix_thinking import ThinkingManager
        return ThinkingManager(MagicMock())

    def test_details_has_width_style(self):
        mgr = self._make_manager()
        html = mgr._build_html("test", 1, time.time(), "content")
        assert 'style="' in html
        assert "width:100%" in html

    def test_thinking_and_tools_same_style(self):
        """Both field kinds must have identical width styling."""
        mgr = self._make_manager()
        thinking = mgr._build_html("test", 1, time.time(), "content", field_kind="thinking")
        tools = mgr._build_html("test", 1, time.time(), "content", field_kind="tools")
        thinking_style = re.search(r'style="([^"]*)"', thinking)
        tools_style = re.search(r'style="([^"]*)"', tools)
        assert thinking_style and tools_style
        assert thinking_style.group(1) == tools_style.group(1)

    def test_pre_has_overflow_style(self):
        mgr = self._make_manager()
        html = mgr._build_html("test", 1, time.time(), "some code")
        assert "overflow-x:auto" in html
        assert "max-width:100%" in html

    def test_model_label_shown(self):
        mgr = self._make_manager()
        html = mgr._build_html("test", 1, time.time(), "", model_label="gpt-5.4")
        assert "gpt-5.4" in html

    def test_truncation_at_max_body(self):
        mgr = self._make_manager()
        html = mgr._build_html("test", 1, time.time(), "x" * 70_000)
        assert "truncated" in html


# ===========================================================================
# MatrixAdapter — thinking method wiring
# ===========================================================================

class TestAdapterThinkingWiring:
    def setup_method(self):
        self.adapter = _make_adapter()

    @pytest.mark.asyncio
    async def test_thinking_disabled(self):
        self.adapter._thinking_enabled = False
        result = await self.adapter.start_thinking("!room:ex", "task1")
        assert result is None

    @pytest.mark.asyncio
    async def test_thinking_enabled_lazy_init(self):
        self.adapter._thinking_enabled = True
        with patch("gateway.platforms.matrix_thinking.ThinkingManager") as MockMgr:
            mock_mgr = MagicMock()
            mock_mgr.start = AsyncMock(return_value="$evt")
            MockMgr.return_value = mock_mgr
            result = await self.adapter.start_thinking("!room:ex", "task1")
            assert result == "$evt"
            assert self.adapter._thinking_manager is mock_mgr

    @pytest.mark.asyncio
    async def test_abort_thinking_noop_when_no_manager(self):
        self.adapter._thinking_enabled = True
        self.adapter._thinking_manager = None
        await self.adapter.abort_thinking("task1")  # should not raise


# ===========================================================================
# MatrixAdapter — approval buttons
# ===========================================================================

class TestAdapterApprovalButtons:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._client = MagicMock()

    @pytest.mark.asyncio
    async def test_send_approval_posts_message(self):
        from gateway.platforms.matrix import SendResult
        with patch.object(self.adapter, "send", new_callable=AsyncMock) as mock_send, \
             patch.object(self.adapter, "_send_reaction", new_callable=AsyncMock):
            mock_send.return_value = SendResult(success=True, message_id="$approval_msg")
            result = await self.adapter.send_exec_approval(
                chat_id="!room:ex", command="rm -rf /",
                session_key="session_123", description="destructive command",
            )
            assert result.success is True
            assert "$approval_msg" in self.adapter._approval_state

    @pytest.mark.asyncio
    async def test_approval_state_tracks_session_key(self):
        from gateway.platforms.matrix import SendResult
        with patch.object(self.adapter, "send", new_callable=AsyncMock) as mock_send, \
             patch.object(self.adapter, "_send_reaction", new_callable=AsyncMock):
            mock_send.return_value = SendResult(success=True, message_id="$msg1")
            await self.adapter.send_exec_approval(
                chat_id="!room:ex", command="echo test", session_key="sess_abc",
            )
            state = self.adapter._approval_state["$msg1"]
            assert state["session_key"] == "sess_abc"

    @pytest.mark.asyncio
    async def test_reaction_resolves_approval(self):
        self.adapter._approval_state["$target_evt"] = {
            "session_key": "sess_xyz", "command": "dangerous cmd", "chat_id": "!room:ex",
        }
        event = MagicMock(sender="@user:example.org", event_id="$react_evt",
                          reacts_to="$target_evt", key="✅")
        room = MagicMock(room_id="!room:ex")

        with patch.object(self.adapter, "_is_duplicate_event", return_value=False), \
             patch("tools.approval.resolve_gateway_approval") as mock_resolve, \
             patch.object(self.adapter, "edit_message", new_callable=AsyncMock):
            self.adapter._user_id = "@bot:example.org"
            await self.adapter._on_reaction(room, event)
            mock_resolve.assert_called_once_with("sess_xyz", "once")
            assert "$target_evt" not in self.adapter._approval_state

    @pytest.mark.asyncio
    async def test_deny_reaction(self):
        self.adapter._approval_state["$target_evt"] = {
            "session_key": "sess_xyz", "command": "bad cmd", "chat_id": "!room:ex",
        }
        event = MagicMock(sender="@user:example.org", event_id="$react_evt2",
                          reacts_to="$target_evt", key="❌")
        room = MagicMock(room_id="!room:ex")

        with patch.object(self.adapter, "_is_duplicate_event", return_value=False), \
             patch("tools.approval.resolve_gateway_approval") as mock_resolve, \
             patch.object(self.adapter, "edit_message", new_callable=AsyncMock):
            self.adapter._user_id = "@bot:example.org"
            await self.adapter._on_reaction(room, event)
            mock_resolve.assert_called_once_with("sess_xyz", "deny")

    @pytest.mark.asyncio
    async def test_send_approval_reacts_with_4_emoji(self):
        from gateway.platforms.matrix import SendResult
        with patch.object(self.adapter, "send", new_callable=AsyncMock) as mock_send, \
             patch.object(self.adapter, "_send_reaction", new_callable=AsyncMock) as mock_react:
            mock_send.return_value = SendResult(success=True, message_id="$msg")
            await self.adapter.send_exec_approval(
                chat_id="!room:ex", command="test", session_key="s1",
            )
            assert mock_react.call_count == 4


# ===========================================================================
# MatrixAdapter — model picker
# ===========================================================================

class TestAdapterModelPicker:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._client = MagicMock()

    @pytest.mark.asyncio
    async def test_send_picker_posts_numbered_list(self):
        from gateway.platforms.matrix import SendResult
        providers = [
            {"name": "OpenAI", "slug": "openai", "models": [{"id": "gpt-5.4"}]},
            {"name": "Anthropic", "slug": "anthropic", "models": [{"id": "claude-4"}]},
        ]
        with patch.object(self.adapter, "send", new_callable=AsyncMock) as mock_send, \
             patch.object(self.adapter, "_send_reaction", new_callable=AsyncMock):
            mock_send.return_value = SendResult(success=True, message_id="$picker_msg")
            result = await self.adapter.send_model_picker(
                chat_id="!room:ex", providers=providers,
                current_model="gpt-5.4", current_provider="openai",
                session_key="sess_123", on_model_selected=AsyncMock(),
            )
            assert result.success is True
            assert "$picker_msg" in self.adapter._model_picker_state

    @pytest.mark.asyncio
    async def test_cancel_picker(self):
        self.adapter._model_picker_state["$picker"] = {
            "stage": "provider", "providers": [{"name": "Test"}],
            "session_key": "s1", "chat_id": "!room:ex",
            "on_model_selected": AsyncMock(),
            "current_model": "m1", "current_provider": "p1", "metadata": None,
        }
        with patch.object(self.adapter, "edit_message", new_callable=AsyncMock):
            await self.adapter._handle_model_picker_reaction(
                "$picker", "❌", "!room:ex", "@user:ex",
            )
        assert "$picker" not in self.adapter._model_picker_state

    @pytest.mark.asyncio
    async def test_provider_selection_advances_to_model_stage(self):
        self.adapter._model_picker_state["$picker"] = {
            "stage": "provider",
            "providers": [{"name": "OpenAI", "slug": "openai", "models": [{"id": "gpt-5.4"}]}],
            "session_key": "s1", "chat_id": "!room:ex",
            "on_model_selected": AsyncMock(),
            "current_model": "m1", "current_provider": "p1", "metadata": None,
        }
        with patch.object(self.adapter, "edit_message", new_callable=AsyncMock):
            await self.adapter._handle_model_picker_reaction(
                "$picker", "1️⃣", "!room:ex", "@user:ex",
            )
        assert self.adapter._model_picker_state["$picker"]["stage"] == "model"

    @pytest.mark.asyncio
    async def test_model_selection_calls_callback(self):
        callback = AsyncMock(return_value="Switched!")
        self.adapter._model_picker_state["$picker"] = {
            "stage": "model",
            "selected_provider": {"name": "OpenAI", "slug": "openai"},
            "display_models": [{"id": "gpt-5.4"}],
            "session_key": "s1", "chat_id": "!room:ex",
            "on_model_selected": callback,
            "current_model": "m1", "current_provider": "p1",
            "metadata": None, "providers": [],
        }
        with patch.object(self.adapter, "edit_message", new_callable=AsyncMock):
            await self.adapter._handle_model_picker_reaction(
                "$picker", "1️⃣", "!room:ex", "@user:ex",
            )
        callback.assert_called_once_with("!room:ex", "gpt-5.4", "openai")
        assert "$picker" not in self.adapter._model_picker_state

    @pytest.mark.asyncio
    async def test_unknown_reaction_ignored(self):
        """Reactions not matching any state should be silently ignored."""
        event = MagicMock(sender="@user:ex", event_id="$evt",
                          reacts_to="$unknown", key="🎉")
        room = MagicMock(room_id="!room:ex")
        with patch.object(self.adapter, "_is_duplicate_event", return_value=False):
            self.adapter._user_id = "@bot:example.org"
            await self.adapter._on_reaction(room, event)  # should not raise
