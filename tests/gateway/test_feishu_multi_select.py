"""Feishu checker-form coverage for select_many."""

from __future__ import annotations

import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest  # ty: ignore[unresolved-import]


def _ensure_feishu_mocks() -> None:
    if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
        mod = MagicMock()
        for name in (
            "lark_oapi",
            "lark_oapi.api.application.v6",
            "lark_oapi.api.im.v1",
            "lark_oapi.core",
            "lark_oapi.core.const",
            "lark_oapi.core.model",
            "lark_oapi.event.callback.model.p2_card_action_trigger",
            "lark_oapi.event.dispatcher_handler",
            "lark_oapi.ws",
            "lark_oapi.event",
            "lark_oapi.event.callback_type",
        ):
            sys.modules.setdefault(name, mod)
    if importlib.util.find_spec("aiohttp") is None and "aiohttp" not in sys.modules:
        aio = MagicMock()
        sys.modules.setdefault("aiohttp", aio)
        sys.modules.setdefault("aiohttp.web", aio.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
import plugins.platforms.feishu.adapter as feishu_module
from plugins.platforms.feishu.adapter import FeishuAdapter
from tools import clarify_gateway


def _make_adapter() -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter._client = MagicMock()
    return adapter


def _make_card_action_data(
    action_value: dict,
    *,
    form_value: dict | None = None,
    chat_id: str = "oc_12345",
    open_id: str = "ou_user1",
    token: str = "tok_select_many",
) -> SimpleNamespace:
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=SimpleNamespace(open_chat_id=chat_id),
            operator=SimpleNamespace(
                open_id=open_id,
                user_id=None,
                union_id=None,
            ),
            action=SimpleNamespace(
                tag="button",
                value=action_value,
                form_value=form_value,
            ),
        ),
    )


class _FakeCallBackCard:
    def __init__(self):
        self.type = None
        self.data = None


class _FakeCallBackToast:
    def __init__(self):
        self.type = None
        self.content = None


class _FakeP2Response:
    def __init__(self):
        self.card = None
        self.toast = None


@pytest.fixture(autouse=True)
def _clear_gateway_state():
    with clarify_gateway._lock:
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()
    yield
    with clarify_gateway._lock:
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()


@pytest.fixture
def _patch_callback_types(monkeypatch):
    monkeypatch.setattr(feishu_module, "P2CardActionTriggerResponse", _FakeP2Response)
    monkeypatch.setattr(feishu_module, "CallBackCard", _FakeCallBackCard)
    monkeypatch.setattr(feishu_module, "CallBackToast", _FakeCallBackToast)


def _prepare_callback_adapter() -> FeishuAdapter:
    adapter = _make_adapter()
    adapter._loop = MagicMock()
    adapter._loop.is_closed = MagicMock(return_value=False)
    adapter._allowed_group_users = {"ou_user1"}
    adapter._sender_name_cache["ou_user1"] = ("Alice", 9999999999)
    return adapter


def _store_state(
    adapter: FeishuAdapter,
    *,
    select_id: str,
    session_key: str,
    choices: list[str],
    initiator_identities: tuple[str, ...] = (),
) -> None:
    adapter._multi_select_state[select_id] = {
        "session_key": session_key,
        "message_id": "msg_select_many",
        "chat_id": "oc_12345",
        "question": "Pick directories",
        "choices": choices,
        "initiator_identities": frozenset(initiator_identities),
    }


class TestFeishuSendMultiSelect:
    @pytest.mark.asyncio
    async def test_sends_card_v2_with_visible_checkers_and_submit(self):
        adapter = _make_adapter()
        response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg-many-1"),
        )

        with patch.object(
            adapter,
            "_feishu_send_with_retry",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_send:
            result = await adapter.send_multi_select(
                chat_id="oc_12345",
                question="Which directories should be removed?",
                choices=["cache", "dist", "node_modules"],
                select_id="many-1",
                session_key="agent:main:feishu:dm:oc_12345",
            )

        assert result.success is True
        assert result.message_id == "msg-many-1"
        payload = json.loads(mock_send.call_args.kwargs["payload"])
        assert mock_send.call_args.kwargs["msg_type"] == "interactive"
        assert payload["schema"] == "2.0"
        assert [element["tag"] for element in payload["body"]["elements"]] == [
            "markdown",
            "form",
            "button",
        ]
        assert "multi_select_static" not in json.dumps(payload)
        form = payload["body"]["elements"][1]
        checkers = form["elements"][:-1]
        assert [checker["tag"] for checker in checkers] == [
            "checker",
            "checker",
            "checker",
        ]
        assert [checker["name"] for checker in checkers] == [
            "selected_choice_0",
            "selected_choice_1",
            "selected_choice_2",
        ]
        assert [checker["text"]["content"] for checker in checkers] == [
            "cache",
            "dist",
            "node_modules",
        ]
        assert all(checker["checked"] is False for checker in checkers)
        submit = form["elements"][-1]
        assert submit["form_action_type"] == "submit"
        assert submit["behaviors"] == [{
            "type": "callback",
            "value": {
                "hermes_action": "select_many_submit",
                "select_id": "many-1",
            },
        }]
        cancel = payload["body"]["elements"][2]
        assert cancel["behaviors"] == [{
            "type": "callback",
            "value": {
                "hermes_action": "select_many_cancel",
                "select_id": "many-1",
            },
        }]
        assert adapter._multi_select_state["many-1"]["choices"] == [
            "cache",
            "dist",
            "node_modules",
        ]

    @pytest.mark.asyncio
    async def test_checker_names_stay_unique_for_each_visible_choice(self):
        adapter = _make_adapter()
        response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg-many-duplicates"),
        )

        with patch.object(
            adapter,
            "_feishu_send_with_retry",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_send:
            await adapter.send_multi_select(
                chat_id="oc_12345",
                question="Pick duplicates",
                choices=["same", "same"],
                select_id="many-duplicates",
                session_key="agent:main:feishu:dm:oc_12345",
            )

        payload = json.loads(mock_send.call_args.kwargs["payload"])
        form = payload["body"]["elements"][1]
        assert [checker["name"] for checker in form["elements"][:-1]] == [
            "selected_choice_0",
            "selected_choice_1",
        ]
        assert [checker["text"]["content"] for checker in form["elements"][:-1]] == [
            "same",
            "same",
        ]

    @pytest.mark.asyncio
    async def test_card_rejection_falls_back_to_numbered_text(self):
        adapter = _make_adapter()
        clarify_gateway.register_select_many(
            "many-fallback",
            "session-fallback",
            "Pick directories",
            ["cache", "dist"],
        )
        rejected = SimpleNamespace(
            success=lambda: False,
            code=230099,
            msg="invalid interactive card",
        )

        with (
            patch.object(
                adapter,
                "_feishu_send_with_retry",
                new_callable=AsyncMock,
                return_value=rejected,
            ),
            patch.object(
                adapter,
                "send",
                new_callable=AsyncMock,
                return_value=SendResult(success=True, message_id="msg-text"),
            ) as mock_text_send,
        ):
            result = await adapter.send_multi_select(
                chat_id="oc_12345",
                question="Pick directories",
                choices=["cache", "dist"],
                select_id="many-fallback",
                session_key="session-fallback",
            )

        assert result.success is True
        content = mock_text_send.call_args.kwargs["content"]
        assert "1. cache" in content
        assert "2. dist" in content
        assert "1 2" in content
        assert "cancel" in content.lower()
        assert "many-fallback" not in adapter._multi_select_state
        pending = clarify_gateway.get_pending_for_session("session-fallback")
        assert pending is not None
        assert pending.awaiting_text is True


class TestFeishuMultiSelectCallback:
    def test_dm_submit_resolves_selected_list(self, _patch_callback_types):
        adapter = _prepare_callback_adapter()
        adapter._allowed_group_users = {"on_other_user"}
        choices = ["cache", "dist", "node_modules"]
        session_key = "agent:coder:feishu:dm:oc_12345"
        entry = clarify_gateway.register_select_many(
            "many-submit", session_key, "Pick directories", choices
        )
        _store_state(
            adapter,
            select_id="many-submit",
            session_key=session_key,
            choices=choices,
        )
        data = _make_card_action_data(
            {
                "hermes_action": "select_many_submit",
                "select_id": "many-submit",
            },
            form_value={
                "selected_choice_0": True,
                "selected_choice_1": False,
                "selected_choice_2": True,
            },
        )

        response = adapter._on_card_action_trigger(data)

        assert entry.response == ["cache", "node_modules"]
        assert entry.event.is_set()
        assert "many-submit" not in adapter._multi_select_state
        assert response.card is not None
        assert response.card.data["header"]["template"] == "green"
        assert response.card.data["schema"] == "2.0"
        summary = response.card.data["body"]["elements"][0]["content"]
        assert "cache" in summary
        assert "node_modules" in summary
        assert "Alice" in summary

    def test_legacy_multi_select_form_value_still_resolves(
        self,
        _patch_callback_types,
    ):
        adapter = _prepare_callback_adapter()
        choices = ["cache", "dist"]
        session_key = "agent:main:feishu:dm:oc_12345"
        entry = clarify_gateway.register_select_many(
            "many-legacy", session_key, "Pick directories", choices
        )
        _store_state(
            adapter,
            select_id="many-legacy",
            session_key=session_key,
            choices=choices,
        )
        data = _make_card_action_data(
            {
                "hermes_action": "select_many_submit",
                "select_id": "many-legacy",
            },
            form_value={"selected_choices": ["1"]},
        )

        adapter._on_card_action_trigger(data)

        assert entry.response == ["dist"]
        assert entry.event.is_set()

    def test_cancel_resolves_empty_list(self, _patch_callback_types):
        adapter = _prepare_callback_adapter()
        choices = ["cache", "dist"]
        session_key = "agent:main:feishu:dm:oc_12345"
        entry = clarify_gateway.register_select_many(
            "many-cancel", session_key, "Pick directories", choices
        )
        _store_state(
            adapter,
            select_id="many-cancel",
            session_key=session_key,
            choices=choices,
        )
        data = _make_card_action_data(
            {
                "hermes_action": "select_many_cancel",
                "select_id": "many-cancel",
            }
        )

        response = adapter._on_card_action_trigger(data)

        assert entry.response == []
        assert entry.event.is_set()
        assert "many-cancel" not in adapter._multi_select_state
        assert response.card is not None
        assert "Cancelled" in response.card.data["header"]["title"]["content"]

    def test_invalid_form_value_does_not_resolve(self, _patch_callback_types):
        adapter = _prepare_callback_adapter()
        choices = ["cache", "dist"]
        entry = clarify_gateway.register_select_many(
            "many-invalid", "session-invalid", "Pick directories", choices
        )
        _store_state(
            adapter,
            select_id="many-invalid",
            session_key="session-invalid",
            choices=choices,
        )
        data = _make_card_action_data(
            {
                "hermes_action": "select_many_submit",
                "select_id": "many-invalid",
            },
            form_value={
                "selected_choice_0": True,
                "selected_choice_99": True,
            },
        )

        response = adapter._on_card_action_trigger(data)

        assert not entry.event.is_set()
        assert "many-invalid" in adapter._multi_select_state
        assert response.toast is not None
        assert response.toast.type == "error"

    def test_callback_chat_mismatch_does_not_resolve(self, _patch_callback_types):
        adapter = _prepare_callback_adapter()
        choices = ["cache"]
        entry = clarify_gateway.register_select_many(
            "many-chat", "agent:main:feishu:dm:oc_12345", "Pick", choices
        )
        _store_state(
            adapter,
            select_id="many-chat",
            session_key="agent:main:feishu:dm:oc_12345",
            choices=choices,
        )
        data = _make_card_action_data(
            {
                "hermes_action": "select_many_submit",
                "select_id": "many-chat",
            },
            form_value={"selected_choice_0": True},
            chat_id="oc_other",
        )

        response = adapter._on_card_action_trigger(data)

        assert not entry.event.is_set()
        assert response.toast is not None
        assert "different chat" in response.toast.content.lower()

    def test_group_unauthorized_operator_does_not_resolve(self, _patch_callback_types):
        adapter = _prepare_callback_adapter()
        adapter._allowed_group_users = {"ou_allowed"}
        choices = ["cache"]
        session_key = "agent:main:feishu:group:oc_12345:ou_allowed"
        entry = clarify_gateway.register_select_many(
            "many-auth", session_key, "Pick", choices
        )
        _store_state(
            adapter,
            select_id="many-auth",
            session_key=session_key,
            choices=choices,
        )
        data = _make_card_action_data(
            {
                "hermes_action": "select_many_submit",
                "select_id": "many-auth",
            },
            form_value={"selected_choice_0": True},
            open_id="ou_intruder",
        )

        response = adapter._on_card_action_trigger(data)

        assert not entry.event.is_set()
        assert response.toast is not None
        assert "not authorized" in response.toast.content.lower()

    def test_other_allowed_group_member_cannot_submit_initiators_form(
        self,
        _patch_callback_types,
    ):
        adapter = _prepare_callback_adapter()
        adapter._allowed_group_users = {"ou_alice", "ou_bob"}
        choices = ["cache"]
        session_key = "agent:coder:feishu:group:oc_12345:on_alice"
        entry = clarify_gateway.register_select_many(
            "many-owner", session_key, "Pick", choices
        )
        _store_state(
            adapter,
            select_id="many-owner",
            session_key=session_key,
            choices=choices,
            initiator_identities=("ou_alice", "u_alice", "on_alice"),
        )
        data = _make_card_action_data(
            {
                "hermes_action": "select_many_submit",
                "select_id": "many-owner",
            },
            form_value={"selected_choice_0": True},
            open_id="ou_bob",
        )

        response = adapter._on_card_action_trigger(data)

        assert not entry.event.is_set()
        assert response.card is None
        assert response.toast is not None
        assert "authorized" in response.toast.content.lower()
