"""Tests for the Feishu /list quick-command menu card.

/list must send an interactive card with allowlisted buttons, and button
clicks must dispatch the corresponding slash command through the normal
gateway pipeline — with NO Feishu reply anchor (99992354 regression).
"""

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_feishu_mocks():
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

from gateway.config import PlatformConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType, SessionSource, _reply_anchor_for_event
from plugins.platforms.feishu.adapter import FeishuAdapter


def _make_adapter() -> FeishuAdapter:
    config = PlatformConfig(enabled=True)
    adapter = FeishuAdapter(config)
    adapter._client = MagicMock()
    return adapter


def _make_card_data(action_value: dict, chat_id: str = "oc_123", open_id: str = "ou_u1") -> SimpleNamespace:
    return SimpleNamespace(
        event=SimpleNamespace(
            token="tok_l",
            context=SimpleNamespace(open_chat_id=chat_id),
            operator=SimpleNamespace(open_id=open_id),
            action=SimpleNamespace(tag="button", value=action_value),
        ),
    )


class TestFeishuLCommandCard:
    @pytest.mark.asyncio
    async def test_send_command_list_sends_interactive_card(self):
        adapter = _make_adapter()
        mock_response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_l1"))
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock, return_value=mock_response
        ) as mock_send:
            result = await adapter.send_command_list(chat_id="oc_123", session_key="sk-1")

        assert result.success is True
        mock_send.assert_called_once()
        kwargs = mock_send.call_args[1]
        assert kwargs["chat_id"] == "oc_123"
        assert kwargs["msg_type"] == "interactive"
        assert kwargs["reply_to"] is None

        card = json.loads(kwargs["payload"])
        buttons = [
            btn
            for element in card["elements"]
            if element.get("tag") == "action"
            for btn in element["actions"]
        ]
        actions = [b["value"]["hermes_list_action"] for b in buttons]
        # Required allowlist entries.
        for required in ("model", "status", "sessions", "reasoning", "fast", "title", "compact", "commands"):
            assert required in actions, f"missing action {required}"
        # No sensitive commands on the card.
        for forbidden in ("restart", "update", "yolo", "stop"):
            assert forbidden not in actions

    @pytest.mark.asyncio
    async def test_list_card_click_dispatches_allowlisted_command(self):
        adapter = _make_adapter()
        data = _make_card_data({"hermes_list_action": "status"})

        with (
            patch.object(adapter, "_allow_group_message", return_value=True),
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_u1", "user_name": "U", "user_id_alt": None},
            ),
            patch.object(adapter, "get_chat_info", new_callable=AsyncMock, return_value={"name": "C"}),
            patch.object(adapter, "_handle_message_with_guards", new_callable=AsyncMock) as guards,
        ):
            await adapter._handle_list_card_action(data)

        guards.assert_awaited_once()
        dispatched = guards.call_args[0][0]
        assert dispatched.text == "/status"
        assert dispatched.message_type == MessageType.COMMAND
        # Synthetic card event must NOT carry a Feishu reply anchor.
        assert _reply_anchor_for_event(dispatched) is None
        # And its message_id must not be the callback token.
        assert dispatched.message_id != "tok_l"

    @pytest.mark.asyncio
    async def test_list_card_click_rejects_non_allowlisted_action(self):
        adapter = _make_adapter()
        data = _make_card_data({"hermes_list_action": "restart"})

        with patch.object(adapter, "_handle_message_with_guards", new_callable=AsyncMock) as guards:
            await adapter._handle_list_card_action(data)

        guards.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_list_card_click_rejects_unauthorized_user(self):
        adapter = _make_adapter()
        data = _make_card_data({"hermes_list_action": "status"})

        with (
            patch.object(adapter, "_allow_group_message", return_value=False),
            patch.object(adapter, "_handle_message_with_guards", new_callable=AsyncMock) as guards,
        ):
            await adapter._handle_list_card_action(data)

        guards.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_card_action_trigger_routes_list_action(self):
        adapter = _make_adapter()
        data = _make_card_data({"hermes_list_action": "title"})

        submitted = []

        def fake_submit(loop, coro):
            submitted.append(coro)
            coro.close()
            return True

        with patch.object(adapter, "_loop_accepts_callbacks", return_value=True), \
             patch.object(adapter, "_submit_on_loop", side_effect=fake_submit):
            adapter._on_card_action_trigger(data)

        assert len(submitted) == 1


class TestReplyAnchorSuppression:
    def test_feishu_card_callback_yields_no_reply_anchor(self):
        event = MessageEvent(
            text="/status",
            message_type=MessageType.COMMAND,
            source=SessionSource(
                platform=Platform.FEISHU,
                chat_id="oc_123",
                chat_name="C",
                chat_type="group",
                user_id="ou_u1",
                user_name="U",
            ),
            message_id=str(uuid.uuid4()),
            raw_message=SimpleNamespace(action=SimpleNamespace(value={"hermes_list_action": "status"})),
        )
        assert _reply_anchor_for_event(event) is None

    def test_normal_feishu_message_keeps_reply_anchor(self):
        event = MessageEvent(
            text="hi",
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=Platform.FEISHU,
                chat_id="oc_123",
                chat_name="C",
                chat_type="group",
                user_id="ou_u1",
                user_name="U",
            ),
            message_id="om_real_message_id",
            raw_message=None,
        )
        assert _reply_anchor_for_event(event) == "om_real_message_id"


class TestLCommandRegistry:
    def test_l_is_gateway_known_command_and_list_alias(self):
        from hermes_cli.commands import resolve_command, is_gateway_known_command

        cmd = resolve_command("l")
        assert cmd is not None and cmd.name == "l"
        assert "list" in cmd.aliases
        assert is_gateway_known_command("l")
        # The alias resolves to the same canonical command.
        assert resolve_command("list").name == "l"
