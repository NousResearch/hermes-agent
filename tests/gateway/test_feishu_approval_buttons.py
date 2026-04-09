"""Tests for Feishu interactive card approval buttons."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Feishu mock so FeishuAdapter can be imported without lark-oapi
# ---------------------------------------------------------------------------
def _ensure_feishu_mocks():
    """Provide stubs for lark-oapi / aiohttp.web so the import succeeds."""
    if "lark_oapi" not in sys.modules:
        mod = MagicMock()
        for name in (
            "lark_oapi", "lark_oapi.api.im.v1",
            "lark_oapi.event", "lark_oapi.event.callback_type",
        ):
            sys.modules.setdefault(name, mod)
    if "aiohttp" not in sys.modules:
        aio = MagicMock()
        sys.modules.setdefault("aiohttp", aio)
        sys.modules.setdefault("aiohttp.web", aio.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig
from gateway.platforms.feishu import FeishuAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter() -> FeishuAdapter:
    """Create a FeishuAdapter with mocked internals."""
    config = PlatformConfig(enabled=True)
    adapter = FeishuAdapter(config)
    adapter._client = MagicMock()
    return adapter


def _make_card_action_data(
    action_value: dict,
    chat_id: str = "oc_12345",
    open_id: str = "ou_user1",
    token: str = "tok_abc",
) -> SimpleNamespace:
    """Create a mock Feishu card action callback data object."""
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=SimpleNamespace(open_chat_id=chat_id),
            operator=SimpleNamespace(open_id=open_id),
            action=SimpleNamespace(
                tag="button",
                value=action_value,
            ),
        ),
    )


# ===========================================================================
# send_exec_approval — interactive card with buttons
# ===========================================================================

class TestFeishuExecApproval:
    """Test send_exec_approval sends an interactive card."""

    @pytest.mark.asyncio
    async def test_sends_interactive_card(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_001"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_exec_approval(
                chat_id="oc_12345",
                command="rm -rf /important",
                session_key="agent:main:feishu:group:oc_12345",
                description="dangerous deletion",
            )

        assert result.success is True
        assert result.message_id == "msg_001"

        mock_send.assert_called_once()
        kwargs = mock_send.call_args[1]
        assert kwargs["chat_id"] == "oc_12345"
        assert kwargs["msg_type"] == "interactive"

        # Verify card payload contains the command and buttons
        card = json.loads(kwargs["payload"])
        assert card["header"]["template"] == "orange"
        assert "rm -rf /important" in card["elements"][0]["content"]
        assert "dangerous deletion" in card["elements"][0]["content"]

        # Check buttons
        actions = card["elements"][1]["actions"]
        assert len(actions) == 4
        action_names = [a["value"]["hermes_action"] for a in actions]
        assert action_names == [
            "approve_once", "approve_session", "approve_always", "deny"
        ]

    @pytest.mark.asyncio
    async def test_stores_approval_state(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_002"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await adapter.send_exec_approval(
                chat_id="oc_12345",
                command="echo test",
                session_key="my-session-key",
            )

        assert len(adapter._approval_state) == 1
        approval_id = list(adapter._approval_state.keys())[0]
        state = adapter._approval_state[approval_id]
        assert state["session_key"] == "my-session-key"
        assert state["message_id"] == "msg_002"
        assert state["chat_id"] == "oc_12345"

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.send_exec_approval(
            chat_id="oc_12345", command="ls", session_key="s"
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_truncates_long_command(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_003"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            long_cmd = "x" * 5000
            await adapter.send_exec_approval(
                chat_id="oc_12345", command=long_cmd, session_key="s"
            )

        card = json.loads(mock_send.call_args[1]["payload"])
        content = card["elements"][0]["content"]
        assert "..." in content
        assert len(content) < 5000

    @pytest.mark.asyncio
    async def test_multiple_approvals_get_unique_ids(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_x"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await adapter.send_exec_approval(
                chat_id="oc_1", command="cmd1", session_key="s1"
            )
            await adapter.send_exec_approval(
                chat_id="oc_2", command="cmd2", session_key="s2"
            )

        assert len(adapter._approval_state) == 2
        ids = list(adapter._approval_state.keys())
        assert ids[0] != ids[1]


# ===========================================================================
# _handle_card_action_event — approval button clicks
# ===========================================================================

class TestFeishuApprovalCallback:
    """Test the approval intercept in _handle_card_action_event."""

    @pytest.mark.asyncio
    async def test_resolves_approval_on_click(self):
        adapter = _make_adapter()
        adapter._approval_state[1] = {
            "session_key": "agent:main:feishu:group:oc_12345",
            "message_id": "msg_001",
            "chat_id": "oc_12345",
        }

        data = _make_card_action_data(
            action_value={"hermes_action": "approve_once", "approval_id": 1},
        )

        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_user1", "user_name": "Norbert", "user_id_alt": None},
            ),
            patch.object(adapter, "_update_approval_card", new_callable=AsyncMock) as mock_update,
            patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve,
        ):
            await adapter._handle_card_action_event(data)

        mock_resolve.assert_called_once_with("agent:main:feishu:group:oc_12345", "once")
        mock_update.assert_called_once_with("msg_001", "Approved once", "Norbert", "once")

        # State should be cleaned up
        assert 1 not in adapter._approval_state

    @pytest.mark.asyncio
    async def test_deny_button(self):
        adapter = _make_adapter()
        adapter._approval_state[2] = {
            "session_key": "some-session",
            "message_id": "msg_002",
            "chat_id": "oc_12345",
        }

        data = _make_card_action_data(
            action_value={"hermes_action": "deny", "approval_id": 2},
            token="tok_deny",
        )

        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_alice", "user_name": "Alice", "user_id_alt": None},
            ),
            patch.object(adapter, "_update_approval_card", new_callable=AsyncMock) as mock_update,
            patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve,
        ):
            await adapter._handle_card_action_event(data)

        mock_resolve.assert_called_once_with("some-session", "deny")
        mock_update.assert_called_once_with("msg_002", "Denied", "Alice", "deny")

    @pytest.mark.asyncio
    async def test_session_approval(self):
        adapter = _make_adapter()
        adapter._approval_state[3] = {
            "session_key": "sess-3",
            "message_id": "msg_003",
            "chat_id": "oc_99",
        }

        data = _make_card_action_data(
            action_value={"hermes_action": "approve_session", "approval_id": 3},
            token="tok_ses",
        )

        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_u", "user_name": "Bob", "user_id_alt": None},
            ),
            patch.object(adapter, "_update_approval_card", new_callable=AsyncMock) as mock_update,
            patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve,
        ):
            await adapter._handle_card_action_event(data)

        mock_resolve.assert_called_once_with("sess-3", "session")
        mock_update.assert_called_once_with("msg_003", "Approved for session", "Bob", "session")

    @pytest.mark.asyncio
    async def test_always_approval(self):
        adapter = _make_adapter()
        adapter._approval_state[4] = {
            "session_key": "sess-4",
            "message_id": "msg_004",
            "chat_id": "oc_55",
        }

        data = _make_card_action_data(
            action_value={"hermes_action": "approve_always", "approval_id": 4},
            token="tok_alw",
        )

        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_u", "user_name": "Carol", "user_id_alt": None},
            ),
            patch.object(adapter, "_update_approval_card", new_callable=AsyncMock),
            patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve,
        ):
            await adapter._handle_card_action_event(data)

        mock_resolve.assert_called_once_with("sess-4", "always")

    @pytest.mark.asyncio
    async def test_already_resolved_drops_silently(self):
        adapter = _make_adapter()
        # No state for approval_id 99 — already resolved

        data = _make_card_action_data(
            action_value={"hermes_action": "approve_once", "approval_id": 99},
            token="tok_gone",
        )

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._handle_card_action_event(data)

        # Should NOT resolve — already handled
        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_approval_actions_route_normally(self):
        """Non-approval card actions should still become synthetic commands."""
        adapter = _make_adapter()

        data = _make_card_action_data(
            action_value={"custom_action": "something_else"},
            token="tok_normal",
        )

        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_u", "user_name": "Dave", "user_id_alt": None},
            ),
            patch.object(adapter, "get_chat_info", new_callable=AsyncMock, return_value={"name": "Test Chat"}),
            patch.object(adapter, "_handle_message_with_guards", new_callable=AsyncMock) as mock_handle,
            patch("tools.approval.resolve_gateway_approval") as mock_resolve,
        ):
            await adapter._handle_card_action_event(data)

        # Should NOT resolve any approval
        mock_resolve.assert_not_called()
        # Should have routed as synthetic command
        mock_handle.assert_called_once()
        event = mock_handle.call_args[0][0]
        assert "/card button" in event.text


# ===========================================================================
# _update_approval_card — card replacement after resolution
# ===========================================================================

class TestFeishuUpdateApprovalCard:
    """Test the card update after approval resolution."""

    @pytest.mark.asyncio
    async def test_updates_card_on_approve(self):
        adapter = _make_adapter()

        mock_update = AsyncMock()
        adapter._client.im.v1.message.update = MagicMock()

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            await adapter._update_approval_card(
                "msg_001", "Approved once", "Norbert", "once"
            )

        mock_thread.assert_called_once()
        # Verify the update request was built
        call_args = mock_thread.call_args
        assert call_args[0][0] == adapter._client.im.v1.message.update

    @pytest.mark.asyncio
    async def test_updates_card_on_deny(self):
        adapter = _make_adapter()

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            await adapter._update_approval_card(
                "msg_002", "Denied", "Alice", "deny"
            )

        mock_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_update_when_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            await adapter._update_approval_card(
                "msg_001", "Approved", "Bob", "once"
            )

        mock_thread.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_update_when_no_message_id(self):
        adapter = _make_adapter()

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            await adapter._update_approval_card(
                "", "Approved", "Bob", "once"
            )

        mock_thread.assert_not_called()

    @pytest.mark.asyncio
    async def test_swallows_update_errors(self):
        adapter = _make_adapter()

        with patch("asyncio.to_thread", new_callable=AsyncMock, side_effect=Exception("API error")):
            # Should not raise
            await adapter._update_approval_card(
                "msg_001", "Approved", "Bob", "once"
            )
