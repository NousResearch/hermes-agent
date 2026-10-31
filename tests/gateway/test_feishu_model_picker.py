"""Tests for Feishu interactive card model picker."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure repo root is importable
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_feishu_mocks():
    """Provide stubs for lark-oapi / aiohttp.web so import succeeds."""
    if "lark_oapi" not in sys.modules:
        try:
            has_lark = importlib.util.find_spec("lark_oapi") is not None
        except ValueError:
            has_lark = False
        if not has_lark:
            mod = MagicMock()
            for name in (
                "lark_oapi", "lark_oapi.api.im.v1",
                "lark_oapi.event", "lark_oapi.event.callback_type",
            ):
                sys.modules.setdefault(name, mod)
    if "aiohttp" not in sys.modules:
        try:
            has_aio = importlib.util.find_spec("aiohttp") is not None
        except ValueError:
            has_aio = False
        if not has_aio:
            aio = MagicMock()
            sys.modules.setdefault("aiohttp", aio)
            sys.modules.setdefault("aiohttp.web", aio.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig
import plugins.platforms.feishu.adapter as feishu_module
from plugins.platforms.feishu.adapter import FeishuAdapter


class _FakeCallBackCard:
    def __init__(self):
        self.type = None
        self.data = None


class _FakeP2Response:
    def __init__(self):
        self.card = None


@pytest.fixture(autouse=True)
def _patch_callback_card_types(monkeypatch):
    """Provide real-ish P2CardActionTriggerResponse / CallBackCard for tests."""
    monkeypatch.setattr(feishu_module, "P2CardActionTriggerResponse", _FakeP2Response)
    monkeypatch.setattr(feishu_module, "CallBackCard", _FakeCallBackCard)


def _make_adapter() -> FeishuAdapter:
    config = PlatformConfig(enabled=True)
    adapter = FeishuAdapter(config)
    adapter._client = MagicMock()
    return adapter


def _make_card_action_data(
    action_value: dict,
    chat_id: str = "oc_12345",
    open_id: str = "ou_user1",
    message_id: str = "om_picker_1",
) -> SimpleNamespace:
    return SimpleNamespace(
        event=SimpleNamespace(
            token="tok_abc",
            context=SimpleNamespace(open_chat_id=chat_id, open_message_id=message_id),
            operator=SimpleNamespace(open_id=open_id),
            action=SimpleNamespace(value=action_value),
        )
    )


class TestFeishuModelPicker:
    @pytest.mark.asyncio
    async def test_send_model_picker_success(self):
        adapter = _make_adapter()
        sent_payloads = []

        async def mock_send(chat_id, msg_type, payload, **kwargs):
            sent_payloads.append((chat_id, msg_type, json.loads(payload)))
            resp = MagicMock()
            resp.code = 0
            resp.data = MagicMock(message_id="om_picker_1")
            return resp

        adapter._feishu_send_with_retry = AsyncMock(side_effect=mock_send)

        providers = [
            {"slug": "magpie", "name": "Magpie", "models": ["gpt-5.4", "gemini-3.8"], "is_current": True},
            {"slug": "cliproxy", "name": "CLIProxy", "models": ["opus-4.6"], "is_current": False},
        ]
        on_selected = AsyncMock(return_value="OK")

        result = await adapter.send_model_picker(
            chat_id="oc_12345",
            providers=providers,
            current_model="gemini-3.8",
            current_provider="magpie",
            session_key="test_session",
            on_model_selected=on_selected,
        )

        assert result.success is True
        assert result.message_id == "om_picker_1"
        assert len(adapter._model_picker_state) == 1

        picker_id = next(iter(adapter._model_picker_state.keys()))
        state = adapter._model_picker_state[picker_id]
        assert state["chat_id"] == "oc_12345"
        assert state["current_model"] == "gemini-3.8"

        # Check card content
        _, msg_type, card = sent_payloads[0]
        assert msg_type == "interactive"
        buttons = []
        for elem in card["elements"]:
            if elem.get("tag") == "action":
                buttons.extend(elem.get("actions", []))
        labels = [b["text"]["content"] for b in buttons]
        assert "✓ Magpie (2)" in labels
        assert "CLIProxy (1)" in labels

    @pytest.mark.asyncio
    async def test_select_provider_drilldown(self):
        adapter = _make_adapter()
        adapter._loop = asyncio.get_running_loop()

        providers = [
            {"slug": "magpie", "name": "Magpie", "models": ["gpt-5.4", "gemini-3.8"], "is_current": True},
        ]
        on_selected = AsyncMock()
        adapter._model_picker_state[10] = {
            "chat_id": "oc_12345",
            "session_key": "s1",
            "providers": providers,
            "current_model": "gemini-3.8",
            "current_provider": "magpie",
            "on_model_selected": on_selected,
        }

        action_data = _make_card_action_data({
            "hermes_model_picker_action": "select_provider",
            "picker_id": 10,
            "provider_slug": "magpie",
        })

        resp = adapter._on_card_action_trigger(action_data)
        assert resp is not None
        new_card = resp.card.data
        assert "Magpie" in new_card["header"]["title"]["content"]

        buttons = []
        for elem in new_card["elements"]:
            if elem.get("tag") == "action":
                buttons.extend(elem.get("actions", []))
        labels = [b["text"]["content"] for b in buttons]
        assert "gpt-5.4" in labels
        assert "✓ gemini-3.8" in labels

    @pytest.mark.asyncio
    async def test_back_button(self):
        adapter = _make_adapter()
        adapter._loop = asyncio.get_running_loop()

        providers = [
            {"slug": "magpie", "name": "Magpie", "models": ["gpt-5.4"], "is_current": True},
        ]
        adapter._model_picker_state[11] = {
            "chat_id": "oc_12345",
            "session_key": "s1",
            "providers": providers,
            "current_model": "gpt-5.4",
            "current_provider": "magpie",
            "on_model_selected": AsyncMock(),
        }

        action_data = _make_card_action_data({
            "hermes_model_picker_action": "back",
            "picker_id": 11,
        })
        resp = adapter._on_card_action_trigger(action_data)
        new_card = resp.card.data
        assert "Providers" in new_card["header"]["title"]["content"] or "服务商" in new_card["header"]["title"]["content"]

    @pytest.mark.asyncio
    async def test_cancel_button(self):
        adapter = _make_adapter()
        adapter._loop = asyncio.get_running_loop()

        adapter._model_picker_state[12] = {
            "chat_id": "oc_12345",
            "session_key": "s1",
            "providers": [],
            "current_model": "m",
            "current_provider": "p",
            "on_model_selected": AsyncMock(),
        }

        action_data = _make_card_action_data({
            "hermes_model_picker_action": "cancel",
            "picker_id": 12,
        })
        resp = adapter._on_card_action_trigger(action_data)
        assert 12 not in adapter._model_picker_state
        new_card = resp.card.data
        assert "Cancelled" in new_card["header"]["title"]["content"] or "已取消" in new_card["header"]["title"]["content"]

    @pytest.mark.asyncio
    async def test_select_model_returns_switching_and_patches_card(self):
        adapter = _make_adapter()
        adapter._loop = asyncio.get_running_loop()

        on_selected = AsyncMock(return_value="Switched session model to gpt-5.4")
        adapter._patch_card_message = AsyncMock(return_value=True)

        adapter._model_picker_state[13] = {
            "chat_id": "oc_12345",
            "message_id": "om_picker_13",
            "session_key": "s1",
            "providers": [{"slug": "magpie", "name": "Magpie", "models": ["gpt-5.4"]}],
            "current_model": "old-model",
            "current_provider": "magpie",
            "on_model_selected": on_selected,
        }

        action_data = _make_card_action_data(
            {
                "hermes_model_picker_action": "select_model",
                "picker_id": "13",  # string type normalization test
                "provider_slug": "magpie",
                "model_id": "gpt-5.4",
            },
            message_id="om_picker_13",
        )

        resp = adapter._on_card_action_trigger(action_data)
        # 1. State must be popped immediately to prevent double click
        assert 13 not in adapter._model_picker_state

        # 2. Immediate response shows switching card (avoid 3s timeout)
        switching_card = resp.card.data
        assert "Switching" in switching_card["header"]["title"]["content"] or "正在切换" in switching_card["header"]["title"]["content"]
        assert "gpt-5.4" in switching_card["elements"][0]["content"]

        # 3. Allow background task to complete
        await asyncio.sleep(0.05)
        on_selected.assert_called_once_with("oc_12345", "gpt-5.4", "magpie")

        # 4. Card should have been patched with success card
        adapter._patch_card_message.assert_called_once()
        patch_msg_id, patched_card = adapter._patch_card_message.call_args[0]
        assert patch_msg_id == "om_picker_13"
        assert patched_card["header"]["template"] == "green"
        assert "gpt-5.4" in patched_card["elements"][0]["content"]

    @pytest.mark.asyncio
    async def test_select_model_failure_fallback(self):
        adapter = _make_adapter()
        adapter._loop = asyncio.get_running_loop()

        on_selected = AsyncMock(side_effect=RuntimeError("Provider offline"))
        adapter._patch_card_message = AsyncMock(return_value=True)

        adapter._model_picker_state[14] = {
            "chat_id": "oc_12345",
            "message_id": "om_picker_14",
            "session_key": "s1",
            "providers": [{"slug": "magpie", "name": "Magpie", "models": ["gpt-5.4"]}],
            "current_model": "old-model",
            "current_provider": "magpie",
            "on_model_selected": on_selected,
        }

        action_data = _make_card_action_data(
            {
                "hermes_model_picker_action": "select_model",
                "picker_id": 14,
                "provider_slug": "magpie",
                "model_id": "gpt-5.4",
            },
            message_id="om_picker_14",
        )

        resp = adapter._on_card_action_trigger(action_data)
        assert resp is not None

        await asyncio.sleep(0.05)
        adapter._patch_card_message.assert_called_once()
        _, patched_card = adapter._patch_card_message.call_args[0]
        assert patched_card["header"]["template"] == "red"
        assert "Provider offline" in patched_card["elements"][0]["content"]

    @pytest.mark.asyncio
    async def test_invalid_picker_id_returns_empty_or_expired(self):
        adapter = _make_adapter()
        adapter._loop = asyncio.get_running_loop()

        # Non-numeric picker id
        data = _make_card_action_data({"hermes_model_picker_action": "cancel", "picker_id": "bad_id"})
        resp = adapter._on_card_action_trigger(data)
        assert resp.card is None

        # Expired numeric picker id
        data_expired = _make_card_action_data({"hermes_model_picker_action": "cancel", "picker_id": 9999})
        resp_expired = adapter._on_card_action_trigger(data_expired)
        assert "Expired" in resp_expired.card.data["header"]["title"]["content"] or "过期" in resp_expired.card.data["header"]["title"]["content"]
