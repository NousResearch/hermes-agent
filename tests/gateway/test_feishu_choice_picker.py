"""Tests for the Feishu choice-picker card (second-step selections).

/l menu commands that need a second selection step (reasoning strength,
fast mode) reuse the gateway's generic choice picker; these tests cover
the Feishu adapter implementation of send_choice_picker + callback.
"""

import importlib.util
import json
import sys
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

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


def _make_adapter() -> FeishuAdapter:
    config = PlatformConfig(enabled=True)
    adapter = FeishuAdapter(config)
    adapter._client = MagicMock()
    return adapter


REASONING_CHOICES = [
    {"value": "none", "label": "关闭", "is_current": False},
    {"value": "low", "label": "Low", "is_current": False},
    {"value": "medium", "label": "Medium", "is_current": True},
    {"value": "high", "label": "High", "is_current": False},
]


class TestFeishuChoicePickerCard:
    @pytest.mark.asyncio
    async def test_send_choice_picker_renders_buttons_and_stores_state(self):
        adapter = _make_adapter()
        on_choice = AsyncMock(return_value="Reasoning set to high")
        mock_response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_cp1"))

        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock, return_value=mock_response
        ) as mock_send:
            result = await adapter.send_choice_picker(
                chat_id="oc_123",
                title="推理强度：当前 Medium",
                choices=REASONING_CHOICES,
                session_key="sk-1",
                on_choice_selected=on_choice,
            )

        assert result.success is True
        kwargs = mock_send.call_args[1]
        card = json.loads(kwargs["payload"])
        buttons = [
            b
            for el in card["elements"] if el.get("tag") == "action"
            for b in el["actions"]
        ]
        labels = [b["text"]["content"] for b in buttons]
        assert labels == ["关闭", "Low", "Medium", "High"]
        # Current choice highlighted as primary.
        medium = next(b for b in buttons if b["text"]["content"] == "Medium")
        assert medium["type"] == "primary"
        # Values are opaque refs — never raw values.
        assert all(b["value"]["hermes_choice_picker"].startswith("1:") for b in buttons)

        # State stored server-side for callback resolution.
        assert len(adapter._choice_picker_state) == 1
        assert adapter._choice_picker_state[1]["choices"] == REASONING_CHOICES

    @pytest.mark.asyncio
    async def test_callback_invokes_choice_handler_and_resolves_card(self):
        adapter = _make_adapter()
        on_choice = AsyncMock(return_value="Reasoning effort set to high for this session.")
        adapter._choice_picker_state[1] = {
            "chat_id": "oc_123",
            "session_key": "sk-1",
            "choices": REASONING_CHOICES,
            "on_choice_selected": on_choice,
        }

        event = SimpleNamespace(
            operator=SimpleNamespace(open_id="ou_u1", user_id=""),
            context=SimpleNamespace(open_chat_id="oc_123"),
        )
        submitted = []

        def fake_submit(loop, coro):
            submitted.append(coro)
            return True

        with (
            patch.object(adapter, "_loop_accepts_callbacks", return_value=True),
            patch.object(adapter, "_allow_group_message", return_value=True),
            patch.object(adapter, "_submit_on_loop", side_effect=fake_submit),
            patch.object(
                adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
                return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_ok")),
            ) as mock_send,
        ):
            resp = adapter._handle_choice_picker_card_action(
                event=event,
                action_value={"hermes_choice_picker": "1:3"},  # High
                loop=object(),
            )
            # Execute the scheduled callback while the send mock is active.
            assert len(submitted) == 1
            await submitted[0]

        # Resolved card returned synchronously.
        assert resp is not None
        assert "High" in resp.card.data["header"]["title"]["content"]
        on_choice.assert_awaited_once_with("oc_123", "high")
        # Result text sent as a fresh message (no reply anchor).
        mock_send.assert_awaited_once()
        send_kwargs = mock_send.call_args[1]
        assert send_kwargs["reply_to"] is None
        # Picker consumed — one-time use.
        assert 1 not in adapter._choice_picker_state

    @pytest.mark.asyncio
    async def test_callback_rejects_tampered_ref(self):
        adapter = _make_adapter()
        adapter._choice_picker_state[1] = {
            "chat_id": "oc_123",
            "session_key": "sk-1",
            "choices": REASONING_CHOICES,
            "on_choice_selected": AsyncMock(),
        }
        event = SimpleNamespace(
            operator=SimpleNamespace(open_id="ou_u1", user_id=""),
            context=SimpleNamespace(open_chat_id="oc_123"),
        )
        for bad in ("1:99", "1:x", "bogus", "2:0"):
            with patch.object(adapter, "_submit_on_loop") as submit:
                adapter._handle_choice_picker_card_action(
                    event=event,
                    action_value={"hermes_choice_picker": bad},
                    loop=object(),
                )
                submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_rejects_unauthorized_user(self):
        adapter = _make_adapter()
        adapter._choice_picker_state[1] = {
            "chat_id": "oc_123",
            "session_key": "sk-1",
            "choices": REASONING_CHOICES,
            "on_choice_selected": AsyncMock(),
        }
        event = SimpleNamespace(
            operator=SimpleNamespace(open_id="ou_stranger", user_id=""),
            context=SimpleNamespace(open_chat_id="oc_123"),
        )
        with (
            patch.object(adapter, "_allow_group_message", return_value=False),
            patch.object(adapter, "_submit_on_loop") as submit,
        ):
            adapter._handle_choice_picker_card_action(
                event=event,
                action_value={"hermes_choice_picker": "1:0"},
                loop=object(),
            )
            submit.assert_not_called()
