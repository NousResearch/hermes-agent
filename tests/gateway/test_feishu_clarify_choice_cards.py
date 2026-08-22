"""Tests for Feishu clarify cards and the generic choice picker."""

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
from plugins.platforms.feishu.adapter import FeishuAdapter
from tools import clarify_gateway


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter() -> FeishuAdapter:
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
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=SimpleNamespace(open_chat_id=chat_id),
            operator=SimpleNamespace(open_id=open_id, user_id="u_1"),
            action=SimpleNamespace(tag="button", value=action_value),
        ),
    )


class _FakeCallBackCard:
    def __init__(self):
        self.type = None
        self.data = None


class _FakeP2Response:
    def __init__(self):
        self.card = None


@pytest.fixture
def _patch_callback_card_types(monkeypatch):
    monkeypatch.setattr(feishu_module, "P2CardActionTriggerResponse", _FakeP2Response)
    monkeypatch.setattr(feishu_module, "CallBackCard", _FakeCallBackCard)


@pytest.fixture(autouse=True)
def _clean_clarify_registry():
    yield
    with clarify_gateway._lock:
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()


_OK_RESPONSE = SimpleNamespace(
    success=lambda: True,
    data=SimpleNamespace(message_id="msg_cl_1"),
)


# ===========================================================================
# send_clarify — interactive card rendering
# ===========================================================================

class TestFeishuSendClarify:
    @pytest.mark.asyncio
    async def test_multi_choice_sends_interactive_card(self):
        adapter = _make_adapter()
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=_OK_RESPONSE,
        ) as mock_send:
            result = await adapter.send_clarify(
                chat_id="oc_12345",
                question="Pick a color",
                choices=["red", "green ish option that is quite long indeed"],
                clarify_id="cl_1",
                session_key="sess-1",
            )

        assert result.success is True
        kwargs = mock_send.call_args[1]
        assert kwargs["msg_type"] == "interactive"
        card = json.loads(kwargs["payload"])
        body = card["elements"][0]["content"]
        assert "Pick a color" in body
        assert "1. red" in body
        actions = card["elements"][1]["actions"]
        # 2 numeric buttons + Other
        assert len(actions) == 3
        assert actions[0]["value"] == {"hermes_clarify_action": "0", "clarify_id": "cl_1"}
        assert actions[-1]["value"]["hermes_clarify_action"] == "other"
        # State stored for the callback path
        assert adapter._clarify_state["cl_1"]["choices"][0] == "red"

    @pytest.mark.asyncio
    async def test_open_ended_sends_plain_text(self):
        adapter = _make_adapter()
        with patch.object(
            adapter, "send", new_callable=AsyncMock,
            return_value=SimpleNamespace(success=True),
        ) as mock_send:
            result = await adapter.send_clarify(
                chat_id="oc_12345",
                question="What now?",
                choices=None,
                clarify_id="cl_2",
                session_key="sess-2",
            )
        assert result.success is True
        assert "What now?" in mock_send.call_args[0][1]
        assert "cl_2" not in adapter._clarify_state


# ===========================================================================
# _handle_clarify_card_action — button taps
# ===========================================================================

class TestClarifyCardAction:
    def _register(self, clarify_id="cl_1", choices=("red", "green")):
        clarify_gateway.register(
            clarify_id=clarify_id,
            session_key="sess-1",
            question="Pick a color",
            choices=list(choices),
        )

    def test_numeric_tap_resolves_entry(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        self._register()
        adapter._clarify_state["cl_1"] = {
            "session_key": "sess-1", "chat_id": "oc_12345",
            "choices": ["red", "green"],
        }
        data = _make_card_action_data(
            {"hermes_clarify_action": "1", "clarify_id": "cl_1"},
        )

        response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None
        assert "green" in json.dumps(response.card.data, ensure_ascii=False)
        # Entry resolved with the choice text
        with clarify_gateway._lock:
            entry = clarify_gateway._entries["cl_1"]
        assert entry.response == "green"
        assert entry.event.is_set()
        assert "cl_1" not in adapter._clarify_state

    def test_other_tap_flips_awaiting_text(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        self._register()
        adapter._clarify_state["cl_1"] = {
            "session_key": "sess-1", "chat_id": "oc_12345",
            "choices": ["red", "green"],
        }
        data = _make_card_action_data(
            {"hermes_clarify_action": "other", "clarify_id": "cl_1"},
        )

        response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None
        with clarify_gateway._lock:
            entry = clarify_gateway._entries["cl_1"]
        assert entry.awaiting_text is True
        assert not entry.event.is_set()
        # State kept so the entry can still be validated later
        assert "cl_1" in adapter._clarify_state

    def test_unauthorized_tap_does_not_resolve(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._admins = {"ou_admin"}
        self._register()
        adapter._clarify_state["cl_1"] = {
            "session_key": "sess-1", "chat_id": "oc_12345",
            "choices": ["red", "green"],
        }
        data = _make_card_action_data(
            {"hermes_clarify_action": "0", "clarify_id": "cl_1"},
            open_id="ou_intruder",
        )

        response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        with clarify_gateway._lock:
            entry = clarify_gateway._entries["cl_1"]
        assert not entry.event.is_set()
        assert "cl_1" in adapter._clarify_state

    def test_expired_entry_returns_plain_response(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        # No clarify_gateway entry registered — simulates timeout eviction.
        adapter._clarify_state["cl_gone"] = {
            "session_key": "sess-1", "chat_id": "oc_12345",
            "choices": ["red"],
        }
        data = _make_card_action_data(
            {"hermes_clarify_action": "0", "clarify_id": "cl_gone"},
        )

        response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        assert "cl_gone" not in adapter._clarify_state

    def test_chat_mismatch_rejected(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        self._register()
        adapter._clarify_state["cl_1"] = {
            "session_key": "sess-1", "chat_id": "oc_expected",
            "choices": ["red", "green"],
        }
        data = _make_card_action_data(
            {"hermes_clarify_action": "0", "clarify_id": "cl_1"},
            chat_id="oc_other",
        )

        response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        with clarify_gateway._lock:
            entry = clarify_gateway._entries["cl_1"]
        assert not entry.event.is_set()


# ===========================================================================
# send_choice_picker + _handle_choice_picker_card_action
# ===========================================================================

class TestFeishuChoicePicker:
    @pytest.mark.asyncio
    async def test_sends_interactive_card_with_choices(self):
        adapter = _make_adapter()
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=_OK_RESPONSE,
        ) as mock_send:
            result = await adapter.send_choice_picker(
                chat_id="oc_12345",
                title="Reasoning effort",
                choices=[
                    {"value": "low", "label": "low", "is_current": False},
                    {"value": "high", "label": "high", "is_current": True},
                ],
                session_key="sess-1",
                on_choice_selected=AsyncMock(),
            )

        assert result.success is True
        kwargs = mock_send.call_args[1]
        assert kwargs["msg_type"] == "interactive"
        card = json.loads(kwargs["payload"])
        assert card["header"]["title"]["content"] == "Reasoning effort"
        actions = card["elements"][0]["actions"]
        assert len(actions) == 2
        assert actions[1]["text"]["content"] == "✓ high"
        assert actions[1]["type"] == "primary"
        assert len(adapter._choice_picker_state) == 1

    @pytest.mark.asyncio
    async def test_empty_choices_fail(self):
        adapter = _make_adapter()
        result = await adapter.send_choice_picker(
            chat_id="oc_12345", title="t", choices=[],
            session_key="s", on_choice_selected=AsyncMock(),
        )
        assert result.success is False

    def test_tap_schedules_applier_and_returns_card(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._allowed_group_users = {"ou_user1"}
        applier = AsyncMock(return_value="Applied: high")
        adapter._choice_picker_state[1] = {
            "chat_id": "oc_12345",
            "session_key": "sess-1",
            "choices": [
                {"value": "low", "label": "low"},
                {"value": "high", "label": "high"},
            ],
            "on_choice_selected": applier,
        }
        data = _make_card_action_data(
            {"hermes_choice_picker_action": "1", "picker_id": 1},
        )

        scheduled = []

        def _capture(coro, _loop, **_kwargs):
            scheduled.append(coro)
            return SimpleNamespace(add_done_callback=lambda *_a, **_k: None)

        with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=_capture):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None
        assert "high" in json.dumps(response.card.data, ensure_ascii=False)
        assert 1 not in adapter._choice_picker_state
        assert len(scheduled) == 1

        # Drive the scheduled applier coroutine and verify the applier and
        # the follow-up send both fire with the selected value.
        with patch.object(
            adapter, "send", new_callable=AsyncMock,
            return_value=SimpleNamespace(success=True),
        ) as mock_send:
            asyncio.run(scheduled[0])
        applier.assert_awaited_once_with("oc_12345", "high")
        assert "Applied: high" in mock_send.call_args[0][1]

    def test_unauthorized_tap_keeps_state(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._admins = {"ou_admin"}
        adapter._choice_picker_state[2] = {
            "chat_id": "oc_12345", "session_key": "s",
            "choices": [{"value": "low", "label": "low"}],
            "on_choice_selected": AsyncMock(),
        }
        data = _make_card_action_data(
            {"hermes_choice_picker_action": "0", "picker_id": 2},
            open_id="ou_intruder",
        )

        response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        assert 2 in adapter._choice_picker_state
