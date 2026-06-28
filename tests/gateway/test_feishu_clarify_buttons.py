"""Tests for Feishu interactive clarify card buttons.

Mirrors test_feishu_approval_buttons.py — the clarify card reuses the same
``_on_card_action_trigger`` / inline-card-response plumbing the approval and
update-prompt cards use, so the tests follow the same shape.
"""

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
from plugins.platforms.feishu.adapter import FeishuAdapter, _CLARIFY_OTHER_ACTION


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
            operator=SimpleNamespace(open_id=open_id),
            action=SimpleNamespace(tag="button", value=action_value),
        ),
    )


def _close_submitted_coro(coro, _loop):
    coro.close()
    return SimpleNamespace(add_done_callback=lambda *_a, **_kw: None)


# ===========================================================================
# send_clarify — interactive card with buttons
# ===========================================================================

class TestSendClarify:
    @pytest.mark.asyncio
    async def test_sends_interactive_card_with_choice_buttons(self):
        adapter = _make_adapter()
        mock_response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="msg_001"))
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock, return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_clarify(
                chat_id="oc_12345",
                question="Which deployment target?",
                choices=["staging", "prod"],
                clarify_id="cid_abc",
                session_key="agent:main:feishu:group:oc_12345",
            )

        assert result.success is True
        assert result.message_id == "msg_001"
        kwargs = mock_send.call_args[1]
        assert kwargs["chat_id"] == "oc_12345"
        assert kwargs["msg_type"] == "interactive"

        card = json.loads(kwargs["payload"])
        assert card["header"]["template"] == "blue"
        assert "Which deployment target?" in card["elements"][0]["content"]

        actions = card["elements"][1]["actions"]
        # one button per choice + a trailing "Other"
        assert len(actions) == 3
        values = [a["value"]["hermes_clarify_action"] for a in actions]
        assert values == ["0", "1", _CLARIFY_OTHER_ACTION]
        assert all(a["value"]["clarify_id"] == "cid_abc" for a in actions)
        # button labels carry the choice text; first is highlighted
        assert actions[0]["text"]["content"] == "staging"
        assert actions[0]["type"] == "primary"
        assert actions[1]["text"]["content"] == "prod"

    @pytest.mark.asyncio
    async def test_stores_clarify_state(self):
        adapter = _make_adapter()
        mock_response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="msg_002"))
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock, return_value=mock_response,
        ):
            await adapter.send_clarify(
                chat_id="oc_12345",
                question="Pick one",
                choices=["a", "b", "c"],
                clarify_id="cid_state",
                session_key="my-session-key",
            )

        assert "cid_state" in adapter._clarify_state
        state = adapter._clarify_state["cid_state"]
        assert state["session_key"] == "my-session-key"
        assert state["message_id"] == "msg_002"
        assert state["chat_id"] == "oc_12345"
        assert state["choices"] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_open_ended_falls_back_to_text(self):
        adapter = _make_adapter()
        with (
            patch.object(adapter, "send", new_callable=AsyncMock) as mock_send_text,
            patch.object(adapter, "_feishu_send_with_retry", new_callable=AsyncMock) as mock_card,
        ):
            mock_send_text.return_value = SimpleNamespace(success=True, message_id="m")
            await adapter.send_clarify(
                chat_id="oc_12345",
                question="What is your name?",
                choices=None,
                clarify_id="cid_open",
                session_key="s",
            )

        mock_card.assert_not_called()
        mock_send_text.assert_called_once()
        assert "What is your name?" in mock_send_text.call_args[1]["content"]
        assert "cid_open" not in adapter._clarify_state

    @pytest.mark.asyncio
    async def test_truncates_long_choice_label(self):
        adapter = _make_adapter()
        mock_response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="msg_t"))
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock, return_value=mock_response,
        ) as mock_send:
            long_choice = "x" * 200
            await adapter.send_clarify(
                chat_id="oc_12345", question="Q", choices=[long_choice], clarify_id="cid_l", session_key="s",
            )

        card = json.loads(mock_send.call_args[1]["payload"])
        label = card["elements"][1]["actions"][0]["text"]["content"]
        assert len(label) < 200
        assert label.endswith("…")
        # value still maps to the index, so resolution stays exact
        assert card["elements"][1]["actions"][0]["value"]["hermes_clarify_action"] == "0"

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.send_clarify(
            chat_id="oc_12345", question="Q", choices=["a", "b"], clarify_id="c", session_key="s",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_send_failure_returns_error(self):
        adapter = _make_adapter()
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock, side_effect=TimeoutError("timed out"),
        ):
            result = await adapter.send_clarify(
                chat_id="oc_12345", question="Q", choices=["a"], clarify_id="c", session_key="s",
            )
        assert result.success is False
        assert "timed out" in (result.error or "")


# ===========================================================================
# _resolve_clarify — state pop + gateway resolution
# ===========================================================================

class TestResolveClarify:
    @pytest.mark.asyncio
    async def test_resolves_once(self):
        adapter = _make_adapter()
        adapter._clarify_state["cid"] = {
            "session_key": "sess", "message_id": "m", "chat_id": "oc_12345", "choices": ["a", "b"],
        }
        with patch("tools.clarify_gateway.resolve_gateway_clarify", return_value=True) as mock_resolve:
            await adapter._resolve_clarify(
                clarify_id="cid", choice="b", user_name="Bob", open_id="ou_user1", chat_id="oc_12345",
            )
        mock_resolve.assert_called_once_with("cid", "b")
        assert "cid" not in adapter._clarify_state

    @pytest.mark.asyncio
    async def test_unknown_id_drops_silently(self):
        adapter = _make_adapter()
        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            await adapter._resolve_clarify(clarify_id="nope", choice="x", user_name="N", open_id="ou_user1")
        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_click_does_not_resolve(self):
        adapter = _make_adapter()
        adapter._admins = {"ou_admin"}
        adapter._clarify_state["cid"] = {
            "session_key": "sess", "message_id": "m", "chat_id": "oc_12345", "choices": ["a"],
        }
        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            await adapter._resolve_clarify(
                clarify_id="cid", choice="a", user_name="Mallory", open_id="ou_intruder", chat_id="oc_12345",
            )
        mock_resolve.assert_not_called()
        assert "cid" in adapter._clarify_state

    @pytest.mark.asyncio
    async def test_chat_mismatch_does_not_resolve(self):
        adapter = _make_adapter()
        adapter._clarify_state["cid"] = {
            "session_key": "sess", "message_id": "m", "chat_id": "oc_expected", "choices": ["a"],
        }
        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            await adapter._resolve_clarify(
                clarify_id="cid", choice="a", user_name="Bob", open_id="ou_user1", chat_id="oc_wrong",
            )
        mock_resolve.assert_not_called()
        assert "cid" in adapter._clarify_state


# ===========================================================================
# _on_card_action_trigger — inline card response for clarify actions
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
    monkeypatch.setattr(feishu_module, "P2CardActionTriggerResponse", _FakeP2Response)
    monkeypatch.setattr(feishu_module, "CallBackCard", _FakeCallBackCard)


class TestClarifyCardActionResponse:
    def _ready_adapter(self):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        return adapter

    def test_returns_card_for_choice(self, _patch_callback_card_types):
        adapter = self._ready_adapter()
        adapter._allowed_group_users = {"ou_bob"}
        adapter._clarify_state["cid"] = {
            "session_key": "sess", "message_id": "m", "chat_id": "oc_12345", "choices": ["staging", "prod"],
        }
        adapter._sender_name_cache["ou_bob"] = ("Bob", 9999999999)
        data = _make_card_action_data(
            {"hermes_clarify_action": "1", "clarify_id": "cid"}, open_id="ou_bob",
        )
        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None
        assert response.card.type == "raw"
        card = response.card.data
        assert card["header"]["template"] == "green"
        assert "prod" in card["elements"][0]["content"]
        assert "Bob" in card["elements"][0]["content"]

    def test_other_button_enters_text_capture(self, _patch_callback_card_types):
        adapter = self._ready_adapter()
        adapter._allowed_group_users = {"ou_bob"}
        adapter._clarify_state["cid"] = {
            "session_key": "sess", "message_id": "m", "chat_id": "oc_12345", "choices": ["a", "b"],
        }
        adapter._sender_name_cache["ou_bob"] = ("Bob", 9999999999)
        data = _make_card_action_data(
            {"hermes_clarify_action": _CLARIFY_OTHER_ACTION, "clarify_id": "cid"}, open_id="ou_bob",
        )
        with patch("tools.clarify_gateway.mark_awaiting_text") as mock_mark:
            response = adapter._on_card_action_trigger(data)

        mock_mark.assert_called_once_with("cid")
        assert response.card is not None
        assert response.card.data["header"]["template"] == "blue"
        assert "type your response" in response.card.data["elements"][0]["content"].lower()
        # state cleared — the next text message resolves it via the gateway intercept
        assert "cid" not in adapter._clarify_state

    def test_ignores_missing_clarify_id(self, _patch_callback_card_types):
        adapter = self._ready_adapter()
        data = _make_card_action_data({"hermes_clarify_action": "0"})
        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)
        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_unknown_clarify_id_returns_no_card(self, _patch_callback_card_types):
        adapter = self._ready_adapter()
        data = _make_card_action_data({"hermes_clarify_action": "0", "clarify_id": "ghost"})
        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)
        assert response.card is None
        mock_submit.assert_not_called()

    def test_rejects_unauthorized_user(self, _patch_callback_card_types):
        adapter = self._ready_adapter()
        adapter._allowed_group_users = {"ou_allowed"}
        adapter._clarify_state["cid"] = {
            "session_key": "sess", "message_id": "m", "chat_id": "oc_12345", "choices": ["a"],
        }
        data = _make_card_action_data(
            {"hermes_clarify_action": "0", "clarify_id": "cid"}, open_id="ou_attacker",
        )
        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)
        assert response.card is None
        mock_submit.assert_not_called()
        assert "cid" in adapter._clarify_state

    def test_rejects_chat_mismatch(self, _patch_callback_card_types):
        adapter = self._ready_adapter()
        adapter._allowed_group_users = {"ou_bob"}
        adapter._clarify_state["cid"] = {
            "session_key": "sess", "message_id": "m", "chat_id": "oc_expected", "choices": ["a"],
        }
        data = _make_card_action_data(
            {"hermes_clarify_action": "0", "clarify_id": "cid"}, chat_id="oc_mismatch", open_id="ou_bob",
        )
        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)
        assert response.card is None
        mock_submit.assert_not_called()

    def test_invalid_choice_index_returns_no_card(self, _patch_callback_card_types):
        adapter = self._ready_adapter()
        adapter._allowed_group_users = {"ou_bob"}
        adapter._clarify_state["cid"] = {
            "session_key": "sess", "message_id": "m", "chat_id": "oc_12345", "choices": ["a", "b"],
        }
        data = _make_card_action_data(
            {"hermes_clarify_action": "9", "clarify_id": "cid"}, open_id="ou_bob",
        )
        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)
        assert response.card is None
        mock_submit.assert_not_called()
