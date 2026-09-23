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


def _tap_payload(task_id, event_key, *, userid="zhangsan", chattype="single", chatid="", selected=None):
    """One template_card_event. ``selected`` mirrors WeCom's dropdown state on a tap:
    {question_key: chosen option id} → selected_items.selected_item[].option_ids.option_id."""
    tce = {"card_type": "button_interaction", "event_key": event_key, "task_id": task_id}
    if selected:
        tce["selected_items"] = {"selected_item": [
            {"question_key": key, "option_ids": {"option_id": [value]}}
            for key, value in selected.items()]}
    return {
        "cmd": "aibot_event_callback",
        "headers": {"req_id": "req-1"},
        "body": {
            "msgid": f"EVT-{task_id}-{event_key}",
            "aibotid": "BOT", "chattype": chattype, "chatid": chatid,
            "from": {"userid": userid},
            "msgtype": "event",
            "event": {"eventtype": "template_card_event", "template_card_event": tce},
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
        # First page = providers, not models. Buttons are only paging + Next (providers page has
        # no Back); the names live in the dropdown.
        assert [b["key"] for b in card["button_list"]] == ["ppg:0", "ppg:0", "pick"]
        sel = card["button_selection"]
        assert sel["question_key"] == "provider"
        assert sel["title"] == "提供商"
        assert len(sel["option_list"]) == 2
        # Regression: dropdown options get the full row width, so names are NOT button-clipped.
        # (Names come from the adapter's get_label mapping, which may differ from the raw fixture.)
        assert len(adapter._model_picker_state) == 1
        state = next(iter(adapter._model_picker_state.values()))
        assert [o["text"] for o in sel["option_list"]] == [
            adapter._option_text(p["name"]) for p in state["providers"]]
        assert [o["id"] for o in sel["option_list"]] == ["alibaba-token-plan-cn", "deepseek"]
        # The active provider is preselected so one tap on '下一步' keeps it.
        assert sel["selected_id"] == "deepseek"
        assert state["stage"] == "providers"
        # Names flow through the real get_label mapping (English labels under test env);
        # pin slugs/models/flags, only require a non-empty display name.
        assert [p["slug"] for p in state["providers"]] == ["alibaba-token-plan-cn", "deepseek"]
        assert [p["models"] for p in state["providers"]] == [["qwen3.8-flash", "qwen3-max"], ["deepseek-flash", "deepseek-v3"]]
        assert [p["is_current"] for p in state["providers"]] == [False, True]
        assert all(p["name"] for p in state["providers"])

    def test_dropdown_options_carry_full_names(self):
        """Regression: a WeCom button renders ~6 ASCII chars on a 3-per-row line (the client
        ellipsises a name), so names go in a dropdown, whose options get the full row width.
        Keep them inside the docs' ≤10 字 — 20 ASCII chars, cut at exactly 20."""
        adapter = _make_adapter(None)
        # A date stamp costs width without distinguishing anything a user picks on, so it goes
        # first — which is what lets a real id fit at all (22 chars → 17).
        assert adapter._option_text("deepseek-v4-flash-0731") == "deepseek-v4-flash"
        # Anything still over 20 is cut at exactly 20 chars (no ellipsis slot).
        assert adapter._option_text("Alibaba Token Plan (China)") == "Alibaba Token Plan ("
        assert adapter._option_text("x" * 40) == "x" * 20
        card = adapter._button_interaction_card(
            title="t", desc="d", sub_title="s", task_id="mx:1",
            buttons=[{"text": "切换", "key": "apply"}],
            selection=adapter._selection("model", "模型", [
                {"id": "deepseek-v4-flash-0731", "text": adapter._option_text("deepseek-v4-flash-0731")}]))
        # The option text is the trimmed label; the id keeps the real model id for the callback.
        option = card["button_selection"]["option_list"][0]
        assert option["text"] == "deepseek-v4-flash" and option["id"] == "deepseek-v4-flash-0731"

    def test_no_selection_key_when_absent(self):
        adapter = _make_adapter(None)
        card = adapter._button_interaction_card(
            title="t", desc="d", sub_title="s", task_id="a1",
            buttons=[{"text": "批准一次", "key": "once"}])
        # The approval card has no dropdown; the key must be absent, not an empty object.
        assert "button_selection" not in card

    def test_parses_selected_items_payload(self):
        """Wire shape: selected_items.selected_item[] → {question_key: option_ids.option_id[0]}.
        The action button is '切换'/'下一步', so the chosen value must survive parsing."""
        adapter = _make_adapter(None)
        assert adapter._parse_selected_items({"selected_items": {"selected_item": [
            {"question_key": "model", "option_ids": {"option_id": ["deepseek-flash"]}}]}}) == {"model": "deepseek-flash"}
        # A lone item may arrive unwrapped, and a single id may arrive as a bare string.
        assert adapter._parse_selected_items({"selected_items": {"selected_item": {
            "question_key": "provider", "option_ids": {"option_id": "deepseek"}}}}) == {"provider": "deepseek"}
        # No dropdown interaction at all (the tap only pressed a button).
        assert adapter._parse_selected_items(None) == {}
        assert adapter._parse_selected_items({"selected_items": {}}) == {}

    def test_provider_pick_drills_into_model_page(self, monkeypatch):
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
        # The chosen provider arrives in the dropdown payload, not on the button key.
        asyncio.run(adapter._handle_template_card_event(
            _tap_payload("mp-1", "pick", selected={"provider": "deepseek"})))
        state = adapter._model_picker_state["mp-1"]
        assert state["stage"] == "models"
        assert state["selected_provider"] == "deepseek"
        adapter._update_card.assert_awaited_once()
        card = adapter._update_card.call_args[0][1]
        assert [b["key"] for b in card["button_list"]] == ["back", "pg:0", "pg:0", "apply"]
        sel = card["button_selection"]
        assert sel["question_key"] == "model"
        assert [o["id"] for o in sel["option_list"]] == ["deepseek-flash", "deepseek-v3"]
        assert sel["selected_id"] == "deepseek-flash"  # the active model is preselected
        # state retained — drilling is not resolution
        assert "mp-1" in adapter._model_picker_state

    def test_pick_without_a_selection_uses_the_active_provider(self, monkeypatch):
        """Tapping '下一步' without touching the dropdown must still work (no dead end)."""
        adapter = _make_adapter(monkeypatch)
        adapter._update_card = AsyncMock()
        adapter._model_picker_state["mp-6"] = {
            "session_key": "s", "chat_id": "zhangsan", "current_model": "m", "current_provider": "b",
            "providers": [{"slug": "a", "name": "A", "models": ["m1"], "is_current": False},
                          {"slug": "b", "name": "B", "models": ["m2"], "is_current": True}],
            "stage": "providers", "selected_provider": "", "model_page": 0,
            "on_model_selected": AsyncMock(),
        }
        asyncio.run(adapter._handle_template_card_event(_tap_payload("mp-6", "pick")))
        assert adapter._model_picker_state["mp-6"]["selected_provider"] == "b"

    def test_provider_page_paginates(self, monkeypatch):
        """More than 10 providers spill to a second dropdown page (SelectionItem caps at 10)."""
        adapter = _make_adapter(monkeypatch)
        adapter._update_card = AsyncMock()
        adapter._model_picker_state["mp-5"] = {
            "session_key": "s", "chat_id": "zhangsan", "current_model": "m", "current_provider": "a12",
            "providers": [{"slug": f"a{i}", "name": f"Brand{i}", "models": ["m1", "m2"], "is_current": False}
                          for i in range(12)],
            "stage": "providers", "selected_provider": "", "model_page": 0, "provider_page": 0,
            "on_model_selected": AsyncMock(),
        }
        card = adapter._build_provider_card("mp-5")
        assert [b["key"] for b in card["button_list"]] == ["ppg:0", "ppg:1", "pick"]
        sel = card["button_selection"]
        assert len(sel["option_list"]) == 10  # the SelectionItem cap
        assert [o["id"] for o in sel["option_list"]] == [f"a{i}" for i in range(10)]
        asyncio.run(adapter._handle_template_card_event(_tap_payload("mp-5", "ppg:1")))
        assert adapter._model_picker_state["mp-5"]["provider_page"] == 1
        card = adapter._update_card.call_args[0][1]
        sel = card["button_selection"]
        assert [o["id"] for o in sel["option_list"]] == ["a10", "a11"]  # the remainder
        # The active provider (a12) is only on page 1, so this page has no preselection
        assert "selected_id" not in sel
        assert "mp-5" in adapter._model_picker_state  # paging never resolves the picker

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
        assert state["model_page"] == 0

    def test_model_page_paginates(self, monkeypatch):
        """25 models ⇒ 3 dropdown pages of 10, each with 返回/上页/下页/切换 and its own options."""
        adapter = _make_adapter(monkeypatch)
        adapter._model_picker_state["mp-3"] = {
            "session_key": "s", "chat_id": "zhangsan", "current_model": "model-21", "current_provider": "a",
            "providers": [{"slug": "a", "name": "A", "models": [f"model-{i}" for i in range(25)], "is_current": True}],
            "stage": "models", "selected_provider": "a", "model_page": 0,
            "on_model_selected": AsyncMock(),
        }
        card = adapter._build_model_card("mp-3")
        assert [b["key"] for b in card["button_list"]] == ["back", "pg:0", "pg:1", "apply"]
        sel = card["button_selection"]
        assert [o["id"] for o in sel["option_list"]] == [f"model-{i}" for i in range(10)]
        assert "selected_id" not in sel  # the active model is not on page 0
        # Last page: the 5 remaining models, and the active model is now preselected
        adapter._model_picker_state["mp-3"]["model_page"] = 2
        card = adapter._build_model_card("mp-3")
        sel = card["button_selection"]
        assert [o["id"] for o in sel["option_list"]] == [f"model-{i}" for i in range(20, 25)]
        assert sel["selected_id"] == "model-21"
        # Never more than 10 options, and never more than 6 buttons (both protocol caps)
        assert len(sel["option_list"]) <= 10
        assert len(card["button_list"]) <= CARD_BUTTON_MAX

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
        # The tapped button is '切换'; the model id comes from the dropdown payload.
        asyncio.run(adapter._handle_template_card_event(
            _tap_payload("mp-9", "apply", selected={"model": "deepseek-flash"})))
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
        # The tapped button is '切换'; the model id comes from the dropdown payload.
        asyncio.run(adapter._handle_template_card_event(
            _tap_payload("mp-9", "apply", selected={"model": "deepseek-flash"})))
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