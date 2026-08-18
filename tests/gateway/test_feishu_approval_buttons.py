"""Tests for Feishu interactive card approval buttons."""

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
from plugins.platforms.feishu.adapter import FeishuAdapter


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


def _close_submitted_coro(coro, _loop):
    """Close scheduled coroutines in sync-handler tests to avoid unawaited warnings."""
    coro.close()
    return SimpleNamespace(add_done_callback=lambda *_args, **_kwargs: None)


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

    def test_rejects_approval_click_when_group_policy_open(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_allowed"}
        adapter._group_policy = "open"
        adapter._default_group_policy = "open"
        adapter._approval_state[6] = {
            "session_key": "sess-6",
            "message_id": "msg-6",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(
            {"hermes_action": "approve_once", "approval_id": 6},
            open_id="ou_attacker",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()


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

    def test_update_prompt_unauthorized_click_rejected_when_group_policy_open(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_allowed"}
        adapter._group_policy = "open"
        adapter._default_group_policy = "open"
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

    # Scenarios below are adapted from @liuliu0223's regression suite in
    # #99021: DM paired-mode (empty allowlist) positive paths, fail-closed
    # rejection of missing operator identity, and forwarded-card rejection.

    def test_paired_mode_participant_can_approve(self, _patch_callback_card_types):
        """Empty allowlist (DM paired mode): the card recipient can still approve."""
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._admins = set()
        adapter._allowed_group_users = set()
        adapter._approval_state[15] = {
            "session_key": "sess-15",
            "message_id": "msg-15",
            "chat_id": "oc_dm_chat",
        }
        data = _make_card_action_data(
            {"hermes_action": "approve_once", "approval_id": 15},
            chat_id="oc_dm_chat",
            open_id="ou_dm_user",
        )
        adapter._sender_name_cache["ou_dm_user"] = ("DM User", 9999999999)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None
        assert "Approved once" in response.card.data["header"]["title"]["content"]

    def test_paired_mode_participant_can_confirm_update_prompt(self, _patch_callback_card_types):
        """Empty allowlist (DM paired mode): the prompt recipient can still confirm."""
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._admins = set()
        adapter._allowed_group_users = set()
        adapter._update_prompt_state[23] = {
            "session_key": "sess-up-23",
            "message_id": "msg_up_023",
            "chat_id": "oc_dm_chat",
        }
        data = _make_card_action_data(
            {"hermes_update_prompt_action": "y", "update_prompt_id": 23},
            chat_id="oc_dm_chat",
            open_id="ou_dm_user",
        )
        adapter._sender_name_cache["ou_dm_user"] = ("DM User", 9999999999)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None

    def test_empty_open_id_rejected_on_approval(self, _patch_callback_card_types):
        """Approval click without an operator identity is rejected (fail-closed)."""
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._admins = set()
        adapter._allowed_group_users = set()
        adapter._approval_state[24] = {
            "session_key": "sess-24",
            "message_id": "msg-24",
            "chat_id": "oc_dm_chat",
        }
        data = _make_card_action_data(
            {"hermes_action": "approve_once", "approval_id": 24},
            chat_id="oc_dm_chat",
            open_id="",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()
        assert 24 in adapter._approval_state

    def test_empty_open_id_rejected_on_update_prompt(self, _patch_callback_card_types):
        """Update prompt click without an operator identity is rejected (fail-closed)."""
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._admins = set()
        adapter._allowed_group_users = set()
        adapter._update_prompt_state[25] = {
            "session_key": "sess-up-25",
            "message_id": "msg_up_025",
            "chat_id": "oc_dm_chat",
        }
        data = _make_card_action_data(
            {"hermes_update_prompt_action": "y", "update_prompt_id": 25},
            chat_id="oc_dm_chat",
            open_id="",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()
        assert 25 in adapter._update_prompt_state

    def test_approval_card_forwarded_to_different_chat_rejected(self, _patch_callback_card_types):
        """Approval card forwarded out of its DM: chat mismatch rejects the click."""
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._admins = set()
        adapter._allowed_group_users = set()
        adapter._approval_state[26] = {
            "session_key": "sess-26",
            "message_id": "msg-26",
            "chat_id": "oc_dm_chat",
        }
        data = _make_card_action_data(
            {"hermes_action": "approve_once", "approval_id": 26},
            chat_id="oc_forwarded_group",
            open_id="ou_dm_user",
        )

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()
        assert 26 in adapter._approval_state


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

        await adapter._resolve_update_prompt(1, "y", "Alice", open_id="ou_user1", chat_id="oc_12345")

        assert (tmp_path / ".hermes" / ".update_response").read_text() == "y"
        assert 1 not in adapter._update_prompt_state

    @pytest.mark.asyncio
    async def test_unauthorized_operator_does_not_write_response(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        adapter._allowed_group_users = {"ou_allowed"}
        adapter._group_policy = "open"
        adapter._default_group_policy = "open"
        adapter._update_prompt_state[2] = {
            "session_key": "sess-up-2",
            "message_id": "msg_up_004",
            "chat_id": "oc_12345",
        }

        await adapter._resolve_update_prompt(2, "y", "Mallory", open_id="ou_intruder", chat_id="oc_12345")

        assert not (tmp_path / ".hermes" / ".update_response").exists()
        assert 2 in adapter._update_prompt_state

    @pytest.mark.asyncio
    async def test_missing_operator_identity_does_not_write_response(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        adapter._allowed_group_users = {"ou_allowed"}
        adapter._update_prompt_state[3] = {
            "session_key": "sess-up-3",
            "message_id": "msg_up_005",
            "chat_id": "oc_12345",
        }

        await adapter._resolve_update_prompt(3, "y", "Anonymous", open_id="", chat_id="oc_12345")

        assert not (tmp_path / ".hermes" / ".update_response").exists()
        assert 3 in adapter._update_prompt_state


# ===========================================================================
# send_model_picker — interactive model picker card + callback validation
# ===========================================================================

def _make_select_action_data(
    option_value: str,
    chat_id: str = "oc_12345",
    open_id: str = "ou_user1",
    token: str = "tok_mp",
) -> SimpleNamespace:
    """Mock a Feishu select_static card action callback (option.value form)."""
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=SimpleNamespace(open_chat_id=chat_id),
            operator=SimpleNamespace(open_id=open_id),
            action=SimpleNamespace(
                tag="select_static",
                value={},
                option=SimpleNamespace(value=option_value),
            ),
        ),
    )


class TestPickerValueScanShapes:
    """The recursive scan must find the picker value in every callback shape."""

    PREFIX = "hermes_model_picker:3:0"
    TARGET = "hermes_model_picker:3:0"

    def test_scan_finds_nested_string_in_action_value_dict(self):
        # shape: action.value = {"selected_option": {"value": "<prefix...>"}}
        adapter = _make_adapter()
        action = SimpleNamespace(tag="select_static", value={"selected_option": {"value": self.TARGET}})
        assert adapter._find_model_picker_option_value(action, action.value) == self.TARGET

    def test_scan_finds_value_in_key_field(self):
        # shape: action.value = {"key": "<prefix...>"}
        adapter = _make_adapter()
        action = SimpleNamespace(tag="select_static", value={"key": self.TARGET})
        assert adapter._find_model_picker_option_value(action, action.value) == self.TARGET

    def test_scan_finds_option_value_str(self):
        # shape: action.option.value = "<prefix...>"
        adapter = _make_adapter()
        action = SimpleNamespace(
            tag="select_static", value={}, option=SimpleNamespace(value=self.TARGET),
        )
        assert adapter._find_model_picker_option_value(action, action.value) == self.TARGET

    def test_scan_finds_option_value_dict(self):
        # shape: action.option = {"value": "<prefix...>"}
        adapter = _make_adapter()
        action = SimpleNamespace(tag="select_static", value={}, option={"value": self.TARGET})
        assert adapter._find_model_picker_option_value(action, action.value) == self.TARGET

    def test_scan_finds_bare_string_value(self):
        # shape: action.value = "<prefix...>" directly
        adapter = _make_adapter()
        action = SimpleNamespace(tag="select_static", value=self.TARGET)
        assert adapter._find_model_picker_option_value(action, action.value) == self.TARGET

    def test_scan_finds_deeply_nested_list_shape(self):
        # shape: action.value = {"options": [{"value": "<prefix...>"}]}
        adapter = _make_adapter()
        action = SimpleNamespace(tag="select_static", value={"options": [{"value": self.TARGET}]})
        assert adapter._find_model_picker_option_value(action, action.value) == self.TARGET

    def test_scan_returns_none_for_unrelated_payload(self):
        adapter = _make_adapter()
        action = SimpleNamespace(tag="select_static", value={"key": "some_other_value"})
        assert adapter._find_model_picker_option_value(action, action.value) is None


class TestFeishuModelPicker:
    """Feishu model picker: card payload, state, callback validation."""

    @pytest.mark.asyncio
    async def test_curated_model_picker_config_wins(self):
        """platforms.feishu.extra.model_picker replaces the full catalog."""
        from gateway.config import PlatformConfig

        config = PlatformConfig(enabled=True)
        config.extra = {
            "model_picker": {
                "enabled": True,
                "models": [
                    {"label": "GPT-5.6 Terra", "model": "gpt-5.6-terra", "provider": "cc-switch-gpt-5-5"},
                    {"label": "DeepSeek V4 Pro 0813", "model": "deepseek-v4-pro-0813", "provider": "cc-switch-deepseek-pro"},
                    {"label": "GLM 5.3", "model": "glm-5.3", "provider": "cc-switch-glm"},
                ],
            }
        }
        adapter = FeishuAdapter(config)
        adapter._client = MagicMock()

        providers = [{
            "slug": "cc-switch-gpt-5-5",
            "name": "cc-switch gpt",
            "models": ["gpt-5.6", "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.6-luna"],
        }]
        mock_response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_mp1"))
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock, return_value=mock_response
        ) as mock_send:
            result = await adapter.send_model_picker(
                chat_id="oc_1",
                providers=providers,
                current_model="glm-5.3",
                current_provider="cc-switch-glm",
                session_key="sk",
                on_model_selected=AsyncMock(),
            )

        assert result.success is True
        card = json.loads(mock_send.call_args[1]["payload"])
        options = card["elements"][1]["actions"][0]["options"]
        labels = [o["text"]["content"] for o in options]
        # Curated three only — no full GPT catalog.
        assert labels == ["GPT-5.6 Terra", "DeepSeek V4 Pro 0813", "GLM 5.3"]

    @pytest.mark.asyncio
    async def test_send_model_picker_sends_dropdown_and_records_state(self):
        adapter = _make_adapter()
        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_mp_1"),
        )
        providers = [
            {
                "slug": "cc-switch-glm",
                "name": "cc-switch glm",
                "models": ["glm-5.3", "glm-5.2"],
            },
            {
                "slug": "cc-switch-gpt-5-5",
                "name": "cc-switch gpt",
                "models": [{"model": "gpt-5.6-terra"}],
            },
            {
                "slug": "cc-switch-deepseek-pro",
                "name": "cc-switch deepseek",
                "models": ["cc/deepseek-v4-pro"],
            },
        ]

        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_model_picker(
                chat_id="oc_12345",
                providers=providers,
                current_model="glm-5.3",
                current_provider="cc-switch-glm",
                session_key="sess-mp",
                on_model_selected=AsyncMock(),
            )

        assert result.success is True
        kwargs = mock_send.call_args[1]
        assert kwargs["msg_type"] == "interactive"

        card = json.loads(kwargs["payload"])
        assert "glm-5.3" in card["elements"][0]["content"]
        select = card["elements"][1]["actions"][0]
        assert select["tag"] == "select_static"
        options = select["options"]
        assert len(options) == 4  # 2 glm + 1 gpt + 1 cc/-prefixed deepseek
        # Option values must be opaque refs (prefix:picker_id:index), never raw model ids
        assert options[0]["value"].startswith("hermes_model_picker:1:")
        # Labels must show the FULL model id — "cc/"-prefixed aliases stay
        # distinguishable instead of collapsing to their last path segment.
        labels = [o["text"]["content"] for o in options]
        assert labels == [
            "cc-switch glm · glm-5.3",
            "cc-switch glm · glm-5.2",
            "cc-switch gpt · gpt-5.6-terra",
            "cc-switch deepseek · cc/deepseek-v4-pro",
        ]

        # State records server-side model entries for callback resolution
        assert len(adapter._model_picker_state) == 1
        state = adapter._model_picker_state[1]
        assert state["chat_id"] == "oc_12345"
        assert state["session_key"] == "sess-mp"
        assert [e[1] for e in state["model_entries"]] == [
            "glm-5.3", "glm-5.2", "gpt-5.6-terra", "cc/deepseek-v4-pro",
        ]
        assert [e[2] for e in state["model_entries"]] == [
            "cc-switch-glm", "cc-switch-glm", "cc-switch-gpt-5-5",
            "cc-switch-deepseek-pro",
        ]

    @pytest.mark.asyncio
    async def test_send_model_picker_returns_error_when_no_models(self):
        adapter = _make_adapter()
        result = await adapter.send_model_picker(
            chat_id="oc_12345",
            providers=[{"slug": "x", "name": "X", "models": []}],
            current_model="m",
            current_provider="x",
            session_key="s",
            on_model_selected=AsyncMock(),
        )
        assert result.success is False

    def test_callback_resolves_selection_and_returns_inline_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        adapter._model_picker_state[3] = {
            "chat_id": "oc_12345",
            "session_key": "sess-mp",
            "on_model_selected": AsyncMock(return_value="switched to glm-5.3"),
            "model_entries": [
                ("cc-switch glm · glm-5.3", "glm-5.3", "cc-switch-glm"),
                ("cc-switch gpt · gpt-5.6-terra", "gpt-5.6-terra", "cc-switch-gpt-5-5"),
            ],
        }
        data = _make_select_action_data("hermes_model_picker:3:0", open_id="ou_user1")

        submitted = {}

        def _capture_submit(coro, loop):
            submitted["coro"] = coro
            return SimpleNamespace(add_done_callback=lambda *_a, **_k: None)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_capture_submit):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None
        assert response.card.type == "raw"
        assert "glm-5.3" in response.card.data["elements"][0]["content"]

        # One-time: picker state consumed immediately
        assert 3 not in adapter._model_picker_state
        # Scheduled coroutine carries the server-resolved model + provider
        assert submitted, "selection dispatch was not scheduled"

    def test_callback_rejects_tampered_index(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        adapter._model_picker_state[7] = {
            "chat_id": "oc_12345",
            "session_key": "sess-mp",
            "on_model_selected": AsyncMock(),
            "model_entries": [("lbl", "glm-5.3", "cc-switch-glm")],
        }
        data = _make_select_action_data("hermes_model_picker:7:99", open_id="ou_user1")

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        assert 7 in adapter._model_picker_state  # not consumed
        mock_submit.assert_not_called()

    def test_callback_rejects_unauthorized_user(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_admin"}
        adapter._model_picker_state[9] = {
            "chat_id": "oc_12345",
            "session_key": "sess-mp",
            "on_model_selected": AsyncMock(),
            "model_entries": [("lbl", "glm-5.3", "cc-switch-glm")],
        }
        data = _make_select_action_data("hermes_model_picker:9:0", open_id="ou_stranger")

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        assert 9 in adapter._model_picker_state
        mock_submit.assert_not_called()

    def test_callback_rejects_chat_mismatch(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        adapter._model_picker_state[11] = {
            "chat_id": "oc_expected",
            "session_key": "sess-mp",
            "on_model_selected": AsyncMock(),
            "model_entries": [("lbl", "glm-5.3", "cc-switch-glm")],
        }
        data = _make_select_action_data("hermes_model_picker:11:0", chat_id="oc_other", open_id="ou_user1")

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        assert 11 in adapter._model_picker_state
        mock_submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_invokes_callback_and_posts_confirmation(self):
        adapter = _make_adapter()
        on_selected = AsyncMock(return_value="✅ 已切换到 glm-5.3")

        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
        ) as mock_send:
            await adapter._dispatch_model_picker_selection(
                on_model_selected=on_selected,
                chat_id="oc_12345",
                model_id="glm-5.3",
                provider_slug="cc-switch-glm",
            )

        on_selected.assert_awaited_once_with("oc_12345", "glm-5.3", "cc-switch-glm")
        mock_send.assert_awaited_once()
        sent = json.loads(mock_send.call_args[1]["payload"])
        assert "glm-5.3" in sent["text"]

    @pytest.mark.asyncio
    async def test_dispatch_failure_still_reports_error(self):
        adapter = _make_adapter()
        on_selected = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
        ) as mock_send:
            await adapter._dispatch_model_picker_selection(
                on_model_selected=on_selected,
                chat_id="oc_12345",
                model_id="glm-5.3",
                provider_slug="cc-switch-glm",
            )

        mock_send.assert_awaited_once()
        sent = json.loads(mock_send.call_args[1]["payload"])
        assert "失败" in sent["text"]


