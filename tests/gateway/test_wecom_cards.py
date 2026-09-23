"""Tests for WeCom template-card interactivity (cards.py): exec-approval and model-picker
buttons, inbound tap routing, and DM-only enforcement."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import ExecApprovalPrompt, SendResult
from plugins.platforms.wecom.cards import CARD_BUTTON_MAX
from plugins.platforms.wecom.adapter import WeComAdapter


def _make_adapter(monkeypatch, dm_allowed=True):
    """Bare adapter instance with the card mixin state and minimal transport stubs, so card
    logic can be exercised without a real WebSocket."""
    adapter = WeComAdapter.__new__(WeComAdapter)
    adapter.platform = Platform.WECOM  # feeds the read-only ``name`` property used by log lines
    adapter._init_card_state()
    adapter._group_chat_ids = set()
    adapter._stream_expired_chats = set()
    adapter._stream_turns = {}
    adapter._last_chat_req_ids = {}
    adapter._reply_req_ids = {}
    adapter._chat_queues, adapter._chat_workers = {}, {}
    adapter._control_queues, adapter._control_workers = {}, {}
    adapter._chat_token_usage = {}
    adapter._dm_policy = "open"
    adapter._allow_from = []
    # Instance-attr shadow of the mixin method (card handlers call it for DM auth).
    adapter._is_dm_intake_allowed = lambda sender_id: dm_allowed and bool(sender_id)
    return adapter


class TestApprovalCard:
    def test_declares_native_approval_buttons(self):
        assert WeComAdapter.supports_exec_approval_buttons() is True

    def test_sends_card_and_registers_task_id(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        captured = {}
        adapter._cached_reply_req_id = lambda *a: None
        adapter._find_active_turn_for_chat = lambda *a: False

        async def _fake_send(chat_id, body, *, reply_req_id=None, is_control=False):
            captured["chat_id"], captured["body"] = chat_id, body
            return SendResult(success=True, message_id="m1", raw_response=body)

        adapter._send_card = _fake_send
        sent = asyncio.run(adapter._send_exec_approval_prompt(ExecApprovalPrompt(
            chat_id="zhangsan", session_key="sess-1",
            text="⚠️ please approve: rm -rf /", actions=[("Allow Once", "once", ""), ("Deny", "deny", "")],
            command="rm -rf /", description="dangerous command", smart_denied=False)))

        assert sent.success is True
        assert len(adapter._approval_state) == 1
        task_id = next(iter(adapter._approval_state))
        assert adapter._approval_state[task_id]["session_key"] == "sess-1"
        card = captured["body"]
        assert card["card_type"] == "button_interaction"
        assert len(card["button_list"]) == 2
        keys = [b["key"] for b in card["button_list"]]
        assert keys == ["once", "deny"]
        assert card["task_id"] == task_id

    def test_group_chat_falls_back_to_text(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._group_chat_ids.add("group_123")
        adapter.send = AsyncMock(return_value=SendResult(success=True))
        result = asyncio.run(adapter._send_exec_approval_prompt(ExecApprovalPrompt(
            chat_id="group_123", session_key="sess-g", text="⚠️ please approve",
            actions=[("Allow Once", "once", "")], command="x", description="d", smart_denied=False)))

        assert result.success is True
        adapter.send.assert_awaited_once()
        assert adapter._approval_state == {}
        assert "please approve" in adapter.send.call_args[0][1]

    def test_failed_card_send_falls_back_to_text_and_clears_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._send_card_inner = AsyncMock(return_value=SendResult(success=False, error="boom"))
        adapter.send = AsyncMock(return_value=SendResult(success=True))
        result = asyncio.run(adapter._send_exec_approval_prompt(ExecApprovalPrompt(
            chat_id="zhangsan", session_key="sess-x", text="⚠️ fallback text",
            actions=[("Deny", "deny", "")], command="x", description="d", smart_denied=False)))

        assert result.success is False  # card failure surfaced; text fallback attempted
        assert adapter._approval_state == {}
        adapter.send.assert_awaited_once_with("zhangsan", "⚠️ fallback text")


def _tap_payload(task_id, event_key, *, userid="zhangsan", chattype="single", chatid=""):
    return {
        "cmd": "aibot_event_callback",
        "headers": {"req_id": "req-1"},
        "body": {
            "msgid": f"EVT-{task_id}-{event_key}",
            "aibotid": "BOT", "chattype": chattype, "chatid": chatid,
            "from": {"userid": userid},
            "msgtype": "event",
            "event": {"eventtype": "template_card_event", "template_card_event": {
                "card_type": "button_interaction", "event_key": event_key, "task_id": task_id}},
        },
    }


class TestModelPicker:
    def _providers(self):
        return [
            {"slug": "alibaba-token-plan-cn", "name": "千问 Token Plan", "models": ["qwen3.8-flash", "qwen3-max"]},
            {"slug": "deepseek", "name": "DeepSeek", "models": ["deepseek-flash", "deepseek-v3"]},
        ]

    def test_sends_provider_page_with_two_level_nav(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        captured = {}
        adapter._cached_reply_req_id = lambda *a: None
        adapter._find_active_turn_for_chat = lambda *a: False

        async def _fake_send(chat_id, body, *, reply_req_id=None, is_control=False):
            captured["chat_id"], captured["body"] = chat_id, body
            return SendResult(success=True, message_id="m2", raw_response=body)

        adapter._send_card = _fake_send
        result = asyncio.run(adapter.send_model_picker(
            chat_id="zhangsan", providers=self._providers(), current_model="deepseek-flash",
            current_provider="deepseek", session_key="sess-m", on_model_selected=AsyncMock()))
        assert result.success is True
        card = captured["body"]
        assert card["card_type"] == "button_interaction"
        # First page = providers, not models
        keys = [b["key"] for b in card["button_list"]]
        assert keys == ["p:alibaba-token-plan-cn", "p:deepseek"]
        texts = [b["text"] for b in card["button_list"]]
        assert texts[1].startswith("✓")  # deepseek is current provider
        assert len(adapter._model_picker_state) == 1
        state = next(iter(adapter._model_picker_state.values()))
        assert state["stage"] == "providers"
        # Names flow through the real get_label mapping (English labels under test env);
        # pin slugs/models/flags, only require a non-empty display name.
        assert [p["slug"] for p in state["providers"]] == ["alibaba-token-plan-cn", "deepseek"]
        assert [p["models"] for p in state["providers"]] == [["qwen3.8-flash", "qwen3-max"], ["deepseek-flash", "deepseek-v3"]]
        assert [p["is_current"] for p in state["providers"]] == [False, True]
        assert all(p["name"] for p in state["providers"])

    def test_provider_tap_drills_into_model_page(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._update_card = AsyncMock()
        adapter._model_picker_state["mp-1"] = {
            "session_key": "sess-m", "chat_id": "zhangsan",
            "current_model": "deepseek-flash", "current_provider": "deepseek",
            "providers": [
                {"slug": "alibaba-token-plan-cn", "name": "千问 Token Plan",
                 "models": ["qwen3.8-flash", "qwen3-max"], "is_current": False},
                {"slug": "deepseek", "name": "DeepSeek", "models": ["deepseek-flash", "deepseek-v3"],
                 "is_current": True},
            ],
            "stage": "providers", "selected_provider": "", "model_page": 0,
            "on_model_selected": AsyncMock(),
        }
        asyncio.run(adapter._handle_template_card_event(_tap_payload("mp-1", "p:deepseek")))
        state = adapter._model_picker_state["mp-1"]
        assert state["stage"] == "models"
        assert state["selected_provider"] == "deepseek"
        adapter._update_card.assert_awaited_once()
        # The card re-rendered for deepseek's models: 2 models + back (+next nav absent w/ 1 page)
        card = adapter._update_card.call_args[0][1]
        assert [b["key"] for b in card["button_list"]] == ["m:0", "m:1", "back"]
        # state retained — drilling is not resolution
        assert "mp-1" in adapter._model_picker_state

    def test_back_returns_to_provider_page(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._update_card = AsyncMock()
        adapter._model_picker_state["mp-2"] = {
            "session_key": "s", "chat_id": "zhangsan", "current_model": "m", "current_provider": "p",
            "providers": [{"slug": "a", "name": "A", "models": ["a1", "a2", "a3", "a4"], "is_current": True}],
            "stage": "models", "selected_provider": "a", "model_page": 1,
            "on_model_selected": AsyncMock(),
        }
        asyncio.run(adapter._handle_template_card_event(_tap_payload("mp-2", "back")))
        state = adapter._model_picker_state["mp-2"]
        assert state["stage"] == "providers"
        assert state["selected_provider"] == ""

    def test_model_page_paginates(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._model_picker_state["mp-3"] = {
            "session_key": "s", "chat_id": "zhangsan", "current_model": "a1", "current_provider": "a",
            "providers": [{"slug": "a", "name": "A", "models": [f"model-{i}" for i in range(7)], "is_current": True}],
            "stage": "models", "selected_provider": "a", "model_page": 0,
            "on_model_selected": AsyncMock(),
        }
        # Page 0: 3 models + back + next (no prev)
        card = adapter._build_model_card("mp-3")
        keys = [b["key"] for b in card["button_list"]]
        assert keys == ["m:0", "m:1", "m:2", "back", "pg:1"]
        # Page 1: 3 models + back + both navs
        adapter._model_picker_state["mp-3"]["model_page"] = 1
        card = adapter._build_model_card("mp-3")
        keys = [b["key"] for b in card["button_list"]]
        assert keys == ["m:3", "m:4", "m:5", "back", "pg:0", "pg:2"]
        assert len(keys) <= 6
        # Last page: leftover model + back + prev
        adapter._model_picker_state["mp-3"]["model_page"] = 2
        card = adapter._build_model_card("mp-3")
        assert [b["key"] for b in card["button_list"]] == ["m:6", "back", "pg:1"]

    def test_model_tap_calls_callback_and_forwards_result(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._update_card = AsyncMock()
        callback = AsyncMock(return_value="✅ 已切换到 deepseek-flash")
        adapter.send = AsyncMock(return_value=SendResult(success=True))
        adapter._model_picker_state["mp-9"] = {
            "session_key": "sess-m", "chat_id": "zhangsan", "current_model": "x", "current_provider": "p",
            "providers": [{"slug": "deepseek", "name": "DeepSeek",
                           "models": ["deepseek-flash", "deepseek-v3"], "is_current": True}],
            "stage": "models", "selected_provider": "deepseek", "model_page": 0,
            "on_model_selected": callback}
        asyncio.run(adapter._handle_template_card_event(_tap_payload("mp-9", "m:0")))
        callback.assert_awaited_once_with("zhangsan", "deepseek-flash", "deepseek")
        adapter.send.assert_awaited_once_with("zhangsan", "✅ 已切换到 deepseek-flash")
        assert "mp-9" not in adapter._model_picker_state  # popped on selection
        adapter._update_card.assert_awaited_once()

    def test_group_chat_returns_not_supported(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._group_chat_ids.add("group_9")
        result = asyncio.run(adapter.send_model_picker(
            chat_id="group_9", providers=self._providers(), current_model="x", current_provider="p",
            session_key="s", on_model_selected=AsyncMock()))
        assert result.success is False
        assert result.error == "Not supported"

    def test_no_providers_returns_error(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        result = asyncio.run(adapter.send_model_picker(
            chat_id="zhangsan", providers=[], current_model="x", current_provider="p",
            session_key="s", on_model_selected=AsyncMock()))
        assert result.success is False


class TestInboundTaps:
    def test_approval_tap_resolves(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._update_card = AsyncMock()
        adapter._approval_state["ea-abc"] = {"session_key": "sess-1", "chat_id": "zhangsan", "desc": "d"}
        with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
            asyncio.run(adapter._handle_template_card_event(_tap_payload("ea-abc", "deny")))
        resolve.assert_called_once_with("sess-1", "deny")
        assert "ea-abc" not in adapter._approval_state  # popped
        adapter._update_card.assert_awaited_once()

    def test_approval_tap_repeat_is_ignored(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._approval_state["ea-abc"] = {"session_key": "sess-1", "chat_id": "zhangsan", "desc": "d"}
        with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
            asyncio.run(adapter._handle_template_card_event(_tap_payload("ea-abc", "deny")))
            asyncio.run(adapter._handle_template_card_event(_tap_payload("ea-abc", "deny")))
        resolve.assert_called_once()

    def test_model_tap_calls_callback_and_forwards_result(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._update_card = AsyncMock()
        callback = AsyncMock(return_value="✅ 已切换到 deepseek-flash")
        adapter.send = AsyncMock(return_value=SendResult(success=True))
        adapter._model_picker_state["mp-9"] = {
            "session_key": "sess-m", "chat_id": "zhangsan", "current_model": "x", "current_provider": "p",
            "providers": [{"slug": "deepseek", "name": "DeepSeek",
                           "models": ["deepseek-flash", "deepseek-v3"], "is_current": True}],
            "stage": "models", "selected_provider": "deepseek", "model_page": 0,
            "on_model_selected": callback}
        asyncio.run(adapter._handle_template_card_event(_tap_payload("mp-9", "m:0")))
        callback.assert_awaited_once_with("zhangsan", "deepseek-flash", "deepseek")
        adapter.send.assert_awaited_once_with("zhangsan", "✅ 已切换到 deepseek-flash")
        assert "mp-9" not in adapter._model_picker_state
        adapter._update_card.assert_awaited_once()

    def test_unauthorized_sender_rejected(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, dm_allowed=False)
        adapter._approval_state["ea-abc"] = {"session_key": "sess-1", "chat_id": "zhangsan", "desc": "d"}
        with patch("tools.approval.resolve_gateway_approval") as resolve:
            asyncio.run(adapter._handle_template_card_event(_tap_payload("ea-abc", "deny", userid="mallory")))
        resolve.assert_not_called()
        assert "ea-abc" in adapter._approval_state  # untouched

    def test_group_tap_ignored(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._approval_state["ea-abc"] = {"session_key": "sess-1", "chat_id": "g", "desc": "d"}
        with patch("tools.approval.resolve_gateway_approval") as resolve:
            asyncio.run(adapter._handle_template_card_event(
                _tap_payload("ea-abc", "deny", chattype="group", chatid="group_1")))
        resolve.assert_not_called()
        assert "ea-abc" in adapter._approval_state  # not popped

    def test_unknown_task_id_logged_not_raised(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        asyncio.run(adapter._handle_template_card_event(_tap_payload("nope", "m:0")))  # must not raise


class TestCardBodies:
    def test_button_cap_at_six(self):
        adapter = _make_adapter(None)
        buttons = [{"text": f"b{i}", "key": f"k{i}"} for i in range(10)]
        card = adapter._button_interaction_card(title="t", desc="d", sub_title="s", buttons=buttons, task_id="t1")
        assert len(card["button_list"]) == CARD_BUTTON_MAX

    def test_text_notice_has_no_buttons(self):
        adapter = _make_adapter(None)
        card = adapter._text_notice_card(title="✅ 已允许一次", desc="决策人：zhangsan", task_id="ea-1")
        assert card["card_type"] == "text_notice"
        assert card["task_id"] == "ea-1"
        assert "button_list" not in card