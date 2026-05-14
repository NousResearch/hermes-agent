"""Tests for Feishu interactive card approval buttons."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
    if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
        mod = MagicMock()
        for name in (
            "lark_oapi", "lark_oapi.api.im.v1",
            "lark_oapi.event", "lark_oapi.event.callback_type",
        ):
            sys.modules.setdefault(name, mod)
    if importlib.util.find_spec("aiohttp") is None and "aiohttp" not in sys.modules:
        aio = MagicMock()
        sys.modules.setdefault("aiohttp", aio)
        sys.modules.setdefault("aiohttp.web", aio.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig
import plugins.platforms.feishu.adapter as feishu_module
from plugins.platforms.feishu.adapter import FeishuAdapter, FeishuGroupRule


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
    user_id: str = "",
    union_id: str = "",
    token: str = "tok_abc",
) -> SimpleNamespace:
    """Create a mock Feishu card action callback data object."""
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=SimpleNamespace(open_chat_id=chat_id),
            operator=SimpleNamespace(open_id=open_id, user_id=user_id, union_id=union_id),
            action=SimpleNamespace(
                tag="button",
                value=action_value,
            ),
        ),
    )


def _close_submitted_coro(coro, _loop):
    """Close scheduled coroutines in sync-handler tests to avoid unawaited warnings."""
    coro.close()
    return SimpleNamespace(add_done_callback=lambda *_args, **_kwargs: None)


def _register_slash_confirm(session_key: str, confirm_id: str, result=None) -> None:
    """Register gateway slash-confirm state for Feishu callback tests."""
    from tools import slash_confirm

    async def _handler(_choice: str):
        return result

    slash_confirm.clear(session_key)
    slash_confirm.register(session_key, confirm_id, "test", _handler)


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
# send_update_prompt — interactive card with buttons
# ===========================================================================

class TestFeishuUpdatePrompt:
    """Test send_update_prompt sends an interactive card."""

    @pytest.mark.asyncio
    async def test_sends_interactive_card(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_up_001"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_update_prompt(
                chat_id="oc_12345",
                prompt="Restore stashed changes after update?",
                default="y",
                session_key="agent:main:feishu:group:oc_12345",
                metadata={"thread_id": "th_1"},
            )

        assert result.success is True
        assert result.message_id == "msg_up_001"

        kwargs = mock_send.call_args[1]
        assert kwargs["chat_id"] == "oc_12345"
        assert kwargs["msg_type"] == "interactive"
        assert kwargs["metadata"] == {"thread_id": "th_1"}

        card = json.loads(kwargs["payload"])
        assert card["header"]["template"] == "orange"
        assert "Restore stashed changes after update?" in card["elements"][0]["content"]
        assert "Default: `y`" in card["elements"][0]["content"]
        actions = card["elements"][1]["actions"]
        assert [a["value"]["hermes_update_prompt_action"] for a in actions] == ["y", "n"]

    @pytest.mark.asyncio
    async def test_stores_prompt_state(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_up_002"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await adapter.send_update_prompt(
                chat_id="oc_12345",
                prompt="Continue update?",
                session_key="my-session-key",
            )

        assert len(adapter._update_prompt_state) == 1
        prompt_id = list(adapter._update_prompt_state.keys())[0]
        state = adapter._update_prompt_state[prompt_id]
        assert state["session_key"] == "my-session-key"
        assert state["message_id"] == "msg_up_002"
        assert state["chat_id"] == "oc_12345"

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.send_update_prompt(
            chat_id="oc_12345",
            prompt="Continue update?",
            session_key="s",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_send_failure_returns_error(self):
        adapter = _make_adapter()
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            side_effect=TimeoutError("timed out"),
        ):
            result = await adapter.send_update_prompt(
                chat_id="oc_12345",
                prompt="Continue update?",
                session_key="s",
            )

        assert result.success is False
        assert "timed out" in (result.error or "")


# ===========================================================================
# send_slash_confirm — interactive card with three confirmation buttons
# ===========================================================================

class TestFeishuSlashConfirm:
    """Test send_slash_confirm sends and resolves an interactive card."""

    @pytest.mark.asyncio
    async def test_sends_interactive_card(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_sc_001"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_slash_confirm(
                chat_id="oc_12345",
                title="Confirm /reload-mcp",
                message="Reload MCP tools?",
                session_key="agent:main:feishu:group:oc_12345",
                confirm_id="confirm-1",
                metadata={"thread_id": "th_1"},
            )

        assert result.success is True
        assert result.message_id == "msg_sc_001"

        kwargs = mock_send.call_args[1]
        assert kwargs["chat_id"] == "oc_12345"
        assert kwargs["msg_type"] == "interactive"
        assert kwargs["metadata"] == {"thread_id": "th_1"}

        card = json.loads(kwargs["payload"])
        assert card["header"]["template"] == "orange"
        assert card["header"]["title"]["content"] == "Confirm /reload-mcp"
        assert "Reload MCP tools?" in card["elements"][0]["content"]
        actions = card["elements"][1]["actions"]
        assert [a["value"]["hermes_slash_confirm_action"] for a in actions] == [
            "once", "always", "cancel",
        ]
        assert [a["value"]["slash_confirm_id"] for a in actions] == [
            "confirm-1", "confirm-1", "confirm-1",
        ]

    @pytest.mark.asyncio
    async def test_stores_slash_confirm_state(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_sc_002"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await adapter.send_slash_confirm(
                chat_id="oc_12345",
                title="Confirm",
                message="Proceed?",
                session_key="my-session-key",
                confirm_id="42",
                metadata={"thread_id": "th_2"},
            )

        state = adapter._slash_confirm_state["42"]
        assert state["session_key"] == "my-session-key"
        assert state["message_id"] == "msg_sc_002"
        assert state["chat_id"] == "oc_12345"
        assert state["metadata"] == {"thread_id": "th_2"}

    @pytest.mark.asyncio
    async def test_new_confirm_supersedes_old_confirm_for_same_session(self):
        adapter = _make_adapter()

        first_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_sc_old"),
        )
        second_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_sc_new"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            side_effect=[first_response, second_response],
        ):
            await adapter.send_slash_confirm(
                chat_id="oc_12345",
                title="Confirm",
                message="Old prompt?",
                session_key="same-session",
                confirm_id="old-confirm",
            )
            await adapter.send_slash_confirm(
                chat_id="oc_12345",
                title="Confirm",
                message="New prompt?",
                session_key="same-session",
                confirm_id="new-confirm",
            )

        assert "old-confirm" not in adapter._slash_confirm_state
        assert adapter._slash_confirm_state["new-confirm"]["message_id"] == "msg_sc_new"

    @pytest.mark.asyncio
    async def test_failed_new_confirm_still_drops_old_confirm_for_same_session(self):
        adapter = _make_adapter()

        first_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_sc_old"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            side_effect=[first_response, TimeoutError("timed out")],
        ):
            await adapter.send_slash_confirm(
                chat_id="oc_12345",
                title="Confirm",
                message="Old prompt?",
                session_key="same-session-fail",
                confirm_id="old-confirm",
            )
            result = await adapter.send_slash_confirm(
                chat_id="oc_12345",
                title="Confirm",
                message="New prompt?",
                session_key="same-session-fail",
                confirm_id="new-confirm",
            )

        assert result.success is False
        assert "old-confirm" not in adapter._slash_confirm_state
        assert "new-confirm" not in adapter._slash_confirm_state

    @pytest.mark.asyncio
    async def test_resolves_and_sends_followup(self):
        adapter = _make_adapter()
        adapter._slash_confirm_state["confirm-2"] = {
            "session_key": "sess-sc-2",
            "message_id": "msg_sc_003",
            "chat_id": "oc_12345",
            "metadata": {"thread_id": "th_3"},
        }

        async def _resolve(session_key, confirm_id, choice):
            assert session_key == "sess-sc-2"
            assert confirm_id == "confirm-2"
            assert choice == "always"
            return "Reloaded."

        with (
            patch("tools.slash_confirm.resolve", side_effect=_resolve) as mock_resolve,
            patch.object(adapter, "send", new_callable=AsyncMock) as mock_send,
        ):
            await adapter._resolve_slash_confirm("confirm-2", "always", "Alice")

        assert mock_resolve.called
        mock_send.assert_called_once_with(
            chat_id="oc_12345",
            content="Reloaded.",
            metadata={"thread_id": "th_3"},
        )
        assert "confirm-2" not in adapter._slash_confirm_state

    @pytest.mark.asyncio
    async def test_resolves_without_followup(self):
        adapter = _make_adapter()
        adapter._slash_confirm_state["confirm-3"] = {
            "session_key": "sess-sc-3",
            "message_id": "msg_sc_004",
            "chat_id": "oc_12345",
            "metadata": {},
        }

        with (
            patch("tools.slash_confirm.resolve", new_callable=AsyncMock, return_value=None) as mock_resolve,
            patch.object(adapter, "send", new_callable=AsyncMock) as mock_send,
        ):
            await adapter._resolve_slash_confirm("confirm-3", "cancel", "Alice")

        mock_resolve.assert_awaited_once_with("sess-sc-3", "confirm-3", "cancel")
        mock_send.assert_not_called()
        assert "confirm-3" not in adapter._slash_confirm_state

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.send_slash_confirm(
            chat_id="oc_12345",
            title="Confirm",
            message="Proceed?",
            session_key="s",
            confirm_id="c",
        )
        assert result.success is False

# ===========================================================================
# _resolve_approval — approval state pop + gateway resolution
# ===========================================================================

class TestResolveApproval:
    """Test _resolve_approval pops state and calls resolve_gateway_approval."""

    @pytest.mark.asyncio
    async def test_resolves_once(self):
        adapter = _make_adapter()
        adapter._approval_state[1] = {
            "session_key": "agent:main:feishu:group:oc_12345",
            "message_id": "msg_001",
            "chat_id": "oc_12345",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._resolve_approval(1, "once", "Norbert", open_id="ou_user1", chat_id="oc_12345")

        mock_resolve.assert_called_once_with("agent:main:feishu:group:oc_12345", "once")
        assert 1 not in adapter._approval_state

    @pytest.mark.asyncio
    async def test_resolves_deny(self):
        adapter = _make_adapter()
        adapter._approval_state[2] = {
            "session_key": "some-session",
            "message_id": "msg_002",
            "chat_id": "oc_12345",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._resolve_approval(2, "deny", "Alice", open_id="ou_user1", chat_id="oc_12345")

        mock_resolve.assert_called_once_with("some-session", "deny")

    @pytest.mark.asyncio
    async def test_resolves_session(self):
        adapter = _make_adapter()
        adapter._approval_state[3] = {
            "session_key": "sess-3",
            "message_id": "msg_003",
            "chat_id": "oc_99",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._resolve_approval(3, "session", "Bob", open_id="ou_user1", chat_id="oc_99")

        mock_resolve.assert_called_once_with("sess-3", "session")

    @pytest.mark.asyncio
    async def test_resolves_always(self):
        adapter = _make_adapter()
        adapter._approval_state[4] = {
            "session_key": "sess-4",
            "message_id": "msg_004",
            "chat_id": "oc_55",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._resolve_approval(4, "always", "Carol", open_id="ou_user1", chat_id="oc_55")

        mock_resolve.assert_called_once_with("sess-4", "always")

    @pytest.mark.asyncio
    async def test_already_resolved_drops_silently(self):
        adapter = _make_adapter()

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._resolve_approval(99, "once", "Nobody", open_id="ou_user1", chat_id="oc_12345")

        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_click_does_not_resolve(self):
        adapter = _make_adapter()
        adapter._admins = {"ou_admin"}
        adapter._approval_state[5] = {
            "session_key": "sess-5",
            "message_id": "msg_005",
            "chat_id": "oc_12345",
        }

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._resolve_approval(5, "once", "Mallory", open_id="ou_intruder", chat_id="oc_12345")

        mock_resolve.assert_not_called()
        assert 5 in adapter._approval_state

    @pytest.mark.asyncio
    async def test_chat_mismatch_does_not_resolve(self):
        adapter = _make_adapter()
        adapter._approval_state[6] = {
            "session_key": "sess-6",
            "message_id": "msg_006",
            "chat_id": "oc_expected",
        }

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._resolve_approval(6, "session", "Norbert", open_id="ou_user1", chat_id="oc_wrong")

        mock_resolve.assert_not_called()
        assert 6 in adapter._approval_state

# ===========================================================================
# _handle_card_action_event — non-approval card actions
# ===========================================================================

class TestNonApprovalCardAction:
    """Non-approval card actions should still route as synthetic commands."""

    @pytest.mark.asyncio
    async def test_routes_as_synthetic_command(self):
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
        ):
            await adapter._handle_card_action_event(data)

        mock_handle.assert_called_once()
        event = mock_handle.call_args[0][0]
        assert "/card button" in event.text


# ===========================================================================
# _on_card_action_trigger — inline card response for approval actions
# ===========================================================================

class _FakeCallBackCard:
    def __init__(self):
        self.type = None
        self.data = None


class _FakeP2Response:
    def __init__(self):
        self.card = None


@pytest.fixture(autouse=False)
def _patch_callback_card_types(monkeypatch):
    """Provide real-ish P2CardActionTriggerResponse / CallBackCard for tests."""
    monkeypatch.setattr(feishu_module, "P2CardActionTriggerResponse", _FakeP2Response)
    monkeypatch.setattr(feishu_module, "CallBackCard", _FakeCallBackCard)


class TestCardActionCallbackResponse:
    """Test that _on_card_action_trigger returns updated card inline."""

    def test_drops_action_when_loop_not_ready(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = None
        data = _make_card_action_data({"hermes_action": "approve_once", "approval_id": 1})

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_returns_card_for_approve_action(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_bob"}
        adapter._approval_state[1] = {
            "session_key": "sess-1",
            "message_id": "msg-1",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {"hermes_action": "approve_once", "approval_id": 1},
            open_id="ou_bob",
        )
        adapter._sender_name_cache["ou_bob"] = ("Bob", 9999999999)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None
        assert response.card.type == "raw"
        card = response.card.data
        assert card["header"]["template"] == "green"
        assert "Approved once" in card["header"]["title"]["content"]
        assert "Bob" in card["elements"][0]["content"]

    def test_returns_card_for_deny_action(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        adapter._approval_state[2] = {
            "session_key": "sess-2",
            "message_id": "msg-2",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {"hermes_action": "deny", "approval_id": 2},
        )

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response.card is not None
        card = response.card.data
        assert card["header"]["template"] == "red"
        assert "Denied" in card["header"]["title"]["content"]

    def test_ignores_missing_approval_id(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data({"hermes_action": "approve_once"})

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_no_card_for_non_approval_action(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data({"some_other": "value"})

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None

    def test_falls_back_to_open_id_when_name_not_cached(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_unknown"}
        adapter._approval_state[3] = {
            "session_key": "sess-3",
            "message_id": "msg-3",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {"hermes_action": "approve_session", "approval_id": 3},
            open_id="ou_unknown",
        )

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        card = response.card.data
        assert "ou_unknown" in card["elements"][0]["content"]

    def test_ignores_expired_cached_name(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_expired"}
        adapter._approval_state[4] = {
            "session_key": "sess-4",
            "message_id": "msg-4",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {"hermes_action": "approve_once", "approval_id": 4},
            open_id="ou_expired",
        )
        adapter._sender_name_cache["ou_expired"] = ("Old Name", 1)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        card = response.card.data
        assert "Old Name" not in card["elements"][0]["content"]
        assert "ou_expired" in card["elements"][0]["content"]

    def test_rejects_approval_click_from_unauthorized_user(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_allowed"}
        adapter._approval_state[5] = {
            "session_key": "sess-5",
            "message_id": "msg-5",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {"hermes_action": "approve_once", "approval_id": 5},
            open_id="ou_attacker",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_rejects_approval_click_when_callback_chat_mismatches(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_bob"}
        adapter._approval_state[6] = {
            "session_key": "sess-6",
            "message_id": "msg-6",
            "chat_id": "oc_expected",
        }
        data = _make_card_action_data(
            {"hermes_action": "approve_once", "approval_id": 6},
            chat_id="oc_mismatch",
            open_id="ou_bob",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_returns_card_for_update_prompt_yes(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_bob"}
        adapter._update_prompt_state[1] = {
            "session_key": "sess-up-1",
            "message_id": "msg_up_003",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {"hermes_update_prompt_action": "y", "update_prompt_id": 1},
            open_id="ou_bob",
        )
        adapter._sender_name_cache["ou_bob"] = ("Bob", 9999999999)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None
        card = response.card.data
        assert card["header"]["template"] == "green"
        assert "answered: Yes" in card["header"]["title"]["content"]
        assert "Bob" in card["elements"][0]["content"]

    def test_returns_card_for_update_prompt_no(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        adapter._update_prompt_state[2] = {
            "session_key": "sess-up-2",
            "message_id": "msg_up_004",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {"hermes_update_prompt_action": "n", "update_prompt_id": 2},
        )

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None
        card = response.card.data
        assert card["header"]["template"] == "red"
        assert "answered: No" in card["header"]["title"]["content"]

    def test_ignores_missing_update_prompt_id(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data({"hermes_update_prompt_action": "y"})

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_already_resolved_update_prompt_returns_no_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data(
            {"hermes_update_prompt_action": "y", "update_prompt_id": 99},
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_update_prompt_schedule_failure_returns_no_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        adapter._update_prompt_state[1] = {
            "session_key": "sess-up-1",
            "message_id": "msg_up_005",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {"hermes_update_prompt_action": "y", "update_prompt_id": 1},
        )

        with patch("asyncio.run_coroutine_threadsafe", side_effect=RuntimeError("loop closed")):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None

    def test_update_prompt_unauthorized_operator_returns_no_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._update_prompt_state[1] = {
            "session_key": "sess-up-1",
            "message_id": "msg_up_006",
            "chat_id": "oc_12345",
        }
        adapter._allowed_group_users = {"ou_allowed"}
        data = _make_card_action_data(
            {"hermes_update_prompt_action": "y", "update_prompt_id": 1},
            open_id="ou_intruder",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_update_prompt_empty_allowlists_fail_closed(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._update_prompt_state[7] = {
            "session_key": "sess-up-7",
            "message_id": "msg_up_007",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {"hermes_update_prompt_action": "y", "update_prompt_id": 7},
            open_id="ou_intruder",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        assert 7 in adapter._update_prompt_state
        mock_submit.assert_not_called()

    def test_update_prompt_chat_mismatch_returns_no_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_bob"}
        adapter._update_prompt_state[8] = {
            "session_key": "sess-up-8",
            "message_id": "msg_up_008",
            "chat_id": "oc_expected",
        }
        data = _make_card_action_data(
            {"hermes_update_prompt_action": "y", "update_prompt_id": 8},
            chat_id="oc_mismatch",
            open_id="ou_bob",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        assert 8 in adapter._update_prompt_state
        mock_submit.assert_not_called()


    def test_returns_card_for_slash_confirm_always(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_bob"}
        _register_slash_confirm("sess-sc-1", "confirm-1")
        adapter._slash_confirm_state["confirm-1"] = {
            "session_key": "sess-sc-1",
            "message_id": "msg_sc_004",
            "chat_id": "oc_12345",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "always",
                "slash_confirm_id": "confirm-1",
            },
            open_id="ou_bob",
        )
        adapter._sender_name_cache["ou_bob"] = ("Bob", 9999999999)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None
        card = response.card.data
        assert card["header"]["template"] == "green"
        assert "Always approved" in card["header"]["title"]["content"]
        assert "Bob" in card["elements"][0]["content"]

    def test_returns_card_for_slash_confirm_cancel(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        _register_slash_confirm("sess-sc-2", "confirm-2")
        adapter._slash_confirm_state["confirm-2"] = {
            "session_key": "sess-sc-2",
            "message_id": "msg_sc_005",
            "chat_id": "oc_12345",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "cancel",
                "slash_confirm_id": "confirm-2",
            },
        )

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None
        card = response.card.data
        assert card["header"]["template"] == "red"
        assert "Cancelled" in card["header"]["title"]["content"]

    def test_unknown_slash_confirm_returns_no_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "missing",
            },
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_gateway_missing_slash_confirm_returns_no_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        adapter._slash_confirm_state["confirm-missing-pending"] = {
            "session_key": "sess-sc-missing-pending",
            "message_id": "msg_sc_missing_pending",
            "chat_id": "oc_12345",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "confirm-missing-pending",
            },
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()
        assert "confirm-missing-pending" not in adapter._slash_confirm_state

    def test_gateway_mismatched_slash_confirm_returns_no_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        _register_slash_confirm("sess-sc-mismatch", "new-confirm")
        adapter._slash_confirm_state["old-confirm"] = {
            "session_key": "sess-sc-mismatch",
            "message_id": "msg_sc_old",
            "chat_id": "oc_12345",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "old-confirm",
            },
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()
        assert "old-confirm" not in adapter._slash_confirm_state

    def test_expired_gateway_slash_confirm_returns_no_card(self, _patch_callback_card_types):
        from tools import slash_confirm

        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        _register_slash_confirm("sess-sc-expired", "confirm-expired")
        slash_confirm._pending["sess-sc-expired"]["created_at"] = 0
        adapter._slash_confirm_state["confirm-expired"] = {
            "session_key": "sess-sc-expired",
            "message_id": "msg_sc_expired",
            "chat_id": "oc_12345",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "confirm-expired",
            },
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()
        assert "confirm-expired" not in adapter._slash_confirm_state
        assert slash_confirm.get_pending("sess-sc-expired") is None

    def test_superseded_slash_confirm_returns_no_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._slash_confirm_state["new-confirm"] = {
            "session_key": "same-session",
            "message_id": "msg_sc_new",
            "chat_id": "oc_12345",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "old-confirm",
            },
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_slash_confirm_unauthorized_operator_returns_no_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_allowed"}
        _register_slash_confirm("sess-sc-3", "confirm-3")
        adapter._slash_confirm_state["confirm-3"] = {
            "session_key": "sess-sc-3",
            "message_id": "msg_sc_006",
            "chat_id": "oc_12345",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "confirm-3",
            },
            open_id="ou_intruder",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_slash_confirm_schedule_failure_returns_no_card(self, _patch_callback_card_types):
        from tools import slash_confirm

        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        _register_slash_confirm("sess-sc-4", "confirm-4")
        adapter._slash_confirm_state["confirm-4"] = {
            "session_key": "sess-sc-4",
            "message_id": "msg_sc_007",
            "chat_id": "oc_12345",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "confirm-4",
            },
        )

        with patch("asyncio.run_coroutine_threadsafe", side_effect=RuntimeError("loop closed")):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        assert "confirm-4" in adapter._slash_confirm_state
        assert slash_confirm.get_pending("sess-sc-4") is not None

    def test_invalid_slash_confirm_action_returns_no_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._slash_confirm_state["confirm-5"] = {
            "session_key": "sess-sc-5",
            "message_id": "msg_sc_008",
            "chat_id": "oc_12345",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "bogus",
                "slash_confirm_id": "confirm-5",
            },
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_duplicate_slash_confirm_click_is_reserved_synchronously(self, _patch_callback_card_types):
        from tools import slash_confirm

        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        _register_slash_confirm("sess-sc-dup", "confirm-dup")
        adapter._slash_confirm_state["confirm-dup"] = {
            "session_key": "sess-sc-dup",
            "message_id": "msg_sc_dup",
            "chat_id": "oc_12345",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "confirm-dup",
            },
        )

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro) as mock_submit:
            first_response = adapter._on_card_action_trigger(data)
            second_response = adapter._on_card_action_trigger(data)

        assert first_response.card is not None
        assert second_response.card is None
        assert mock_submit.call_count == 1
        assert slash_confirm.get_pending("sess-sc-dup") is None

    def test_slash_confirm_duplicate_during_claim_does_not_steal_pending(self, _patch_callback_card_types):
        from tools import slash_confirm

        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        _register_slash_confirm("sess-sc-race", "confirm-race")
        adapter._slash_confirm_state["confirm-race"] = {
            "session_key": "sess-sc-race",
            "message_id": "msg_sc_race",
            "chat_id": "oc_12345",
            "chat_type": "group",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "confirm-race",
            },
        )
        nested_responses = []
        original_claim = adapter._claim_slash_confirm_pending

        def _claim_with_nested_click(**kwargs):
            nested_responses.append(adapter._on_card_action_trigger(data))
            return original_claim(**kwargs)

        with (
            patch.object(adapter, "_claim_slash_confirm_pending", side_effect=_claim_with_nested_click),
            patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro) as mock_submit,
        ):
            response = adapter._on_card_action_trigger(data)

        assert response.card is not None
        assert nested_responses[0].card is None
        assert mock_submit.call_count == 1
        assert slash_confirm.get_pending("sess-sc-race") is None

    def test_slash_confirm_dm_ignores_group_allowlist(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_someone_else"}
        _register_slash_confirm("agent:main:feishu:dm:oc_dm", "confirm-dm")
        adapter._slash_confirm_state["confirm-dm"] = {
            "session_key": "agent:main:feishu:dm:oc_dm",
            "message_id": "msg_sc_dm",
            "chat_id": "oc_dm",
            "chat_type": "dm",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "confirm-dm",
            },
            chat_id="oc_dm",
            open_id="ou_dm_user",
        )

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None

    @pytest.mark.asyncio
    async def test_valid_slash_confirm_callback_resolves_and_sends_followup(
        self,
        _patch_callback_card_types,
    ):
        from tools import slash_confirm

        adapter = _make_adapter()
        adapter._loop = asyncio.get_running_loop()
        adapter._allowed_group_users = {"ou_user1"}
        _register_slash_confirm("sess-sc-e2e", "confirm-e2e", result="Resolved follow-up")
        adapter._slash_confirm_state["confirm-e2e"] = {
            "session_key": "sess-sc-e2e",
            "message_id": "msg_sc_e2e",
            "chat_id": "oc_12345",
            "metadata": {"thread_id": "th_e2e"},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "always",
                "slash_confirm_id": "confirm-e2e",
            },
        )

        with patch.object(adapter, "send", new_callable=AsyncMock) as mock_send:
            response = adapter._on_card_action_trigger(data)
            await asyncio.sleep(0.01)

        assert response is not None
        assert response.card is not None
        mock_send.assert_awaited_once_with(
            chat_id="oc_12345",
            content="Resolved follow-up",
            metadata={"thread_id": "th_e2e"},
        )
        assert slash_confirm.get_pending("sess-sc-e2e") is None
        assert "confirm-e2e" not in adapter._slash_confirm_state

    def test_slash_confirm_respects_per_chat_group_rule_allowlist(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = set()
        adapter._group_rules["oc_12345"] = FeishuGroupRule(
            policy="allowlist",
            allowlist={"ou_allowed"},
        )
        _register_slash_confirm("sess-sc-rule", "confirm-rule")
        adapter._slash_confirm_state["confirm-rule"] = {
            "session_key": "sess-sc-rule",
            "message_id": "msg_sc_rule",
            "chat_id": "oc_12345",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "confirm-rule",
            },
            open_id="ou_intruder",
            chat_id="oc_12345",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()
        assert "confirm-rule" in adapter._slash_confirm_state

    def test_slash_confirm_group_rule_allows_operator_user_id(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = set()
        adapter._group_rules["oc_12345"] = FeishuGroupRule(
            policy="allowlist",
            allowlist={"u_allowed"},
        )
        _register_slash_confirm("sess-sc-user-id-rule", "confirm-user-id-rule")
        adapter._slash_confirm_state["confirm-user-id-rule"] = {
            "session_key": "sess-sc-user-id-rule",
            "message_id": "msg_sc_user_id_rule",
            "chat_id": "oc_12345",
            "chat_type": "group",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "confirm-user-id-rule",
            },
            open_id="ou_other",
            user_id="u_allowed",
            chat_id="oc_12345",
        )

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None

    def test_slash_confirm_group_rule_allows_operator_union_id(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = set()
        adapter._group_rules["oc_12345"] = FeishuGroupRule(
            policy="allowlist",
            allowlist={"on_allowed"},
        )
        _register_slash_confirm("sess-sc-union-rule", "confirm-union-rule")
        adapter._slash_confirm_state["confirm-union-rule"] = {
            "session_key": "sess-sc-union-rule",
            "message_id": "msg_sc_union_rule",
            "chat_id": "oc_12345",
            "chat_type": "group",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "confirm-union-rule",
            },
            open_id="ou_other",
            union_id="on_allowed",
            chat_id="oc_12345",
        )

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None

    def test_slash_confirm_group_rule_rejects_blacklisted_union_id(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = set()
        adapter._group_rules["oc_12345"] = FeishuGroupRule(
            policy="blacklist",
            blacklist={"on_blocked"},
        )
        _register_slash_confirm("sess-sc-union-blacklist", "confirm-union-blacklist")
        adapter._slash_confirm_state["confirm-union-blacklist"] = {
            "session_key": "sess-sc-union-blacklist",
            "message_id": "msg_sc_union_blacklist",
            "chat_id": "oc_12345",
            "chat_type": "group",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "confirm-union-blacklist",
            },
            open_id="ou_other",
            union_id="on_blocked",
            chat_id="oc_12345",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_named_profile_group_session_rejects_different_participant(self, _patch_callback_card_types):
        from tools import slash_confirm

        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"alice_id", "bob_id"}
        session_key = "agent:research:feishu:group:oc_12345:alice_id"
        _register_slash_confirm(session_key, "confirm-alice")
        adapter._slash_confirm_state["confirm-alice"] = {
            "session_key": session_key,
            "message_id": "msg_sc_alice",
            "chat_id": "oc_12345",
            "chat_type": "group",
            "participant_id": adapter._slash_confirm_participant_id(
                session_key=session_key,
            ),
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "confirm-alice",
            },
            open_id="bob_id",
            chat_id="oc_12345",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()
        assert "confirm-alice" in adapter._slash_confirm_state
        assert slash_confirm.get_pending(session_key) is not None

    @pytest.mark.asyncio
    async def test_send_slash_confirm_stores_group_participant_from_session_key(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_sc_participant"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await adapter.send_slash_confirm(
                chat_id="oc_12345",
                title="Confirm",
                message="Proceed?",
                session_key="agent:research:feishu:group:oc_12345:alice_id",
                confirm_id="confirm-participant",
            )

        assert adapter._slash_confirm_state["confirm-participant"]["participant_id"] == "alice_id"

    def test_slash_confirm_rejects_callback_from_different_chat(self, _patch_callback_card_types):
        from tools import slash_confirm

        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = set()
        adapter._group_rules["oc_b"] = FeishuGroupRule(
            policy="allowlist",
            allowlist={"ou_user1"},
        )
        _register_slash_confirm("sess-sc-chat-mismatch", "confirm-chat-mismatch")
        adapter._slash_confirm_state["confirm-chat-mismatch"] = {
            "session_key": "sess-sc-chat-mismatch",
            "message_id": "msg_sc_chat_mismatch",
            "chat_id": "oc_a",
            "chat_type": "group",
            "metadata": {},
        }
        data = _make_card_action_data(
            {
                "hermes_slash_confirm_action": "once",
                "slash_confirm_id": "confirm-chat-mismatch",
            },
            chat_id="oc_b",
            open_id="ou_user1",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()
        assert "confirm-chat-mismatch" in adapter._slash_confirm_state
        assert slash_confirm.get_pending("sess-sc-chat-mismatch") is not None

class TestResolveUpdatePrompt:
    """Test update prompt resolution persists the response file."""

    @pytest.mark.asyncio
    async def test_writes_response_file(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        adapter._update_prompt_state[1] = {
            "session_key": "sess-up-1",
            "message_id": "msg_up_003",
            "chat_id": "oc_12345",
        }

        await adapter._resolve_update_prompt(1, "y", "Alice")

        assert (tmp_path / ".hermes" / ".update_response").read_text() == "y"
        assert 1 not in adapter._update_prompt_state

    @pytest.mark.asyncio
    async def test_overwrites_existing_response_file(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / ".update_response").write_text("n")
        adapter._update_prompt_state[2] = {
            "session_key": "sess-up-2",
            "message_id": "msg_up_004",
            "chat_id": "oc_12345",
        }

        await adapter._resolve_update_prompt(2, "y", "Alice")

        assert (home / ".update_response").read_text() == "y"

    @pytest.mark.asyncio
    async def test_unknown_prompt_id_drops_silently(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()

        await adapter._resolve_update_prompt(99, "n", "Nobody")

        assert not (tmp_path / ".hermes" / ".update_response").exists()

    @pytest.mark.asyncio
    async def test_chat_mismatch_does_not_write_response_file(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        adapter._allowed_group_users = {"ou_bob"}
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        adapter._update_prompt_state[10] = {
            "session_key": "sess-up-10",
            "message_id": "msg_up_010",
            "chat_id": "oc_expected",
        }

        await adapter._resolve_update_prompt(
            10,
            "y",
            "Bob",
            open_id="ou_bob",
            chat_id="oc_wrong",
        )

        assert not (tmp_path / ".hermes" / ".update_response").exists()
        assert 10 in adapter._update_prompt_state
