"""Tests for Feishu interactive card clarify buttons."""

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest  # ty: ignore[unresolved-import]

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_feishu_mocks():
    """Provide stubs so FeishuAdapter can be imported without optional deps."""
    if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
        mod = MagicMock()
        for name in (
            "lark_oapi",
            "lark_oapi.api.im.v1",
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
    chat_id: str = "oc_12345",
    open_id: str = "ou_user1",
    token: str = "tok_clarify",
) -> SimpleNamespace:
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=SimpleNamespace(open_chat_id=chat_id),
            operator=SimpleNamespace(open_id=open_id, user_id=None, union_id=None),
            action=SimpleNamespace(tag="button", value=action_value),
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
def _clear_clarify_gateway_state():
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


def _store_clarify_state(
    adapter: FeishuAdapter,
    *,
    clarify_id: str,
    session_key: str,
    choices: list[str],
    chat_id: str = "oc_12345",
    initiator_identities: tuple[str, ...] = (),
) -> None:
    adapter._clarify_state[clarify_id] = {
        "session_key": session_key,
        "message_id": "msg_clarify",
        "chat_id": chat_id,
        "question": "Pick one",
        "choices": choices,
        "initiator_identities": frozenset(initiator_identities),
    }


class TestFeishuSendClarify:
    @pytest.mark.asyncio
    async def test_multiple_choice_sends_interactive_card(self):
        adapter = _make_adapter()
        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_clarify_1"),
        )

        with patch.object(
            adapter,
            "_feishu_send_with_retry",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_clarify(
                chat_id="oc_12345",
                question="Which environment?",
                choices=["Development", "Staging", "Production", "Disaster Recovery"],
                clarify_id="clarify-1",
                session_key="agent:main:feishu:group:oc_12345:ou_user1",
            )

        assert result.success is True
        assert result.message_id == "msg_clarify_1"
        kwargs = mock_send.call_args.kwargs
        assert kwargs["msg_type"] == "interactive"
        card = json.loads(kwargs["payload"])
        assert set(card) == {"config", "header", "elements"}
        assert card["config"] == {"wide_screen_mode": True}
        assert [element["tag"] for element in card["elements"]] == [
            "markdown",
            "action",
        ]
        assert "Which environment?" in card["elements"][0]["content"]
        assert "Reply with the number" not in card["elements"][0]["content"]

        actions = card["elements"][1]["actions"]
        assert len(actions) == 5
        assert all(
            set(action) == {"tag", "text", "type", "value"}
            and action["tag"] == "button"
            and action["text"]["tag"] == "plain_text"
            for action in actions
        )
        assert [action["value"]["hermes_action"] for action in actions] == [
            "clarify",
            "clarify",
            "clarify",
            "clarify",
            "clarify",
        ]
        assert [action["value"]["choice_index"] for action in actions] == [
            0,
            1,
            2,
            3,
            "other",
        ]
        assert all(
            action["value"]["clarify_id"] == "clarify-1" for action in actions
        )
        assert [action["text"]["content"] for action in actions[:-1]] == [
            "1",
            "2",
            "3",
            "4",
        ]
        assert "Other" in actions[-1]["text"]["content"]

        state = adapter._clarify_state["clarify-1"]
        assert state["session_key"] == "agent:main:feishu:group:oc_12345:ou_user1"
        assert state["message_id"] == "msg_clarify_1"
        assert state["chat_id"] == "oc_12345"
        assert state["choices"] == [
            "Development",
            "Staging",
            "Production",
            "Disaster Recovery",
        ]

    @pytest.mark.asyncio
    async def test_api_rejection_logs_clarify_id_and_feishu_error(self, caplog):
        adapter = _make_adapter()
        mock_response = SimpleNamespace(
            success=lambda: False,
            code=230099,
            msg="invalid interactive card",
        )

        with (
            patch.object(
                adapter,
                "_feishu_send_with_retry",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            caplog.at_level(logging.ERROR),
        ):
            result = await adapter.send_clarify(
                chat_id="oc_12345",
                question="Which environment?",
                choices=["Development", "Production"],
                clarify_id="clarify-rejected",
                session_key="session-rejected",
            )

        assert result.success is False
        assert result.error == "[230099] invalid interactive card"
        assert "clarify-rejected" not in adapter._clarify_state
        assert "clarify-rejected" in caplog.text
        assert "[230099] invalid interactive card" in caplog.text

    @pytest.mark.asyncio
    async def test_open_ended_uses_text_fallback(self):
        adapter = _make_adapter()
        with patch.object(
            adapter,
            "send",
            new_callable=AsyncMock,
            return_value=SendResult(success=True, message_id="msg_text"),
        ) as mock_send:
            result = await adapter.send_clarify(
                chat_id="oc_12345",
                question="Describe the expected behavior",
                choices=None,
                clarify_id="clarify-open",
                session_key="session-open",
            )

        assert result.success is True
        mock_send.assert_awaited_once()
        assert "Describe the expected behavior" in mock_send.call_args.kwargs["content"]
        assert "clarify-open" not in adapter._clarify_state


class TestFeishuClarifyCardAction:
    @pytest.mark.parametrize(
        "session_key",
        [
            "agent:main:feishu:dm:oc_12345",
            "agent:coder:feishu:dm:oc_12345",
        ],
    )
    def test_dm_owner_click_is_not_gated_by_group_allowlist(
        self,
        _patch_callback_types,
        session_key,
    ):
        adapter = _prepare_callback_adapter()
        adapter._allowed_group_users = {"on_union_user1"}
        choices = ["Red", "Green"]
        clarify_gateway.register(
            "clarify-dm-owner",
            session_key,
            "Pick one",
            choices,
        )
        _store_clarify_state(
            adapter,
            clarify_id="clarify-dm-owner",
            session_key=session_key,
            choices=choices,
        )
        data = _make_card_action_data(
            {
                "hermes_action": "clarify",
                "clarify_id": "clarify-dm-owner",
                "choice_index": 0,
            },
            open_id="ou_user1",
        )

        response = adapter._on_card_action_trigger(data)

        with clarify_gateway._lock:
            entry = clarify_gateway._entries["clarify-dm-owner"]
        assert entry.response == "Red"
        assert entry.event.is_set()
        assert response.card is not None

    @pytest.mark.parametrize(
        "allowlisted_id",
        ["ou_user1", "u_tenant_user1", "on_union_user1"],
    )
    def test_group_click_matches_any_cached_feishu_id_alias(
        self,
        _patch_callback_types,
        allowlisted_id,
    ):
        adapter = _prepare_callback_adapter()
        adapter._allowed_group_users = {allowlisted_id}
        adapter._remember_identity_aliases(
            SimpleNamespace(
                open_id="ou_user1",
                user_id="u_tenant_user1",
                union_id="on_union_user1",
            )
        )
        choices = ["Red", "Green"]
        session_key = "agent:main:feishu:group:oc_12345:on_union_user1"
        clarify_gateway.register(
            "clarify-id-alias",
            session_key,
            "Pick one",
            choices,
        )
        _store_clarify_state(
            adapter,
            clarify_id="clarify-id-alias",
            session_key=session_key,
            choices=choices,
        )
        data = _make_card_action_data(
            {
                "hermes_action": "clarify",
                "clarify_id": "clarify-id-alias",
                "choice_index": 1,
            },
            open_id="ou_user1",
        )

        response = adapter._on_card_action_trigger(data)

        with clarify_gateway._lock:
            entry = clarify_gateway._entries["clarify-id-alias"]
        assert entry.response == "Green"
        assert entry.event.is_set()
        assert response.card is not None

    def test_other_allowed_group_member_cannot_answer_initiators_prompt(
        self,
        _patch_callback_types,
    ):
        adapter = _prepare_callback_adapter()
        adapter._allowed_group_users = {"ou_alice", "ou_bob"}
        session_key = "agent:coder:feishu:group:oc_12345:on_alice"
        entry = clarify_gateway.register(
            "clarify-owner",
            session_key,
            "Pick one",
            ["A", "B"],
        )
        _store_clarify_state(
            adapter,
            clarify_id="clarify-owner",
            session_key=session_key,
            choices=["A", "B"],
            initiator_identities=("ou_alice", "u_alice", "on_alice"),
        )
        data = _make_card_action_data(
            {
                "hermes_action": "clarify",
                "clarify_id": "clarify-owner",
                "choice_index": 0,
            },
            open_id="ou_bob",
        )

        response = adapter._on_card_action_trigger(data)

        assert not entry.event.is_set()
        assert response.card is None
        assert response.toast is not None
        assert "authorized" in response.toast.content.lower()

    def test_partial_identity_event_does_not_forget_cached_aliases(self):
        adapter = _prepare_callback_adapter()
        adapter._remember_identity_aliases(
            SimpleNamespace(
                open_id="ou_user1",
                user_id="u_tenant_user1",
                union_id="on_union_user1",
            )
        )

        adapter._remember_identity_aliases(
            SimpleNamespace(open_id="ou_user1", user_id=None, union_id=None)
        )

        assert adapter._known_sender_identities("ou_user1") == frozenset(
            {"ou_user1", "u_tenant_user1", "on_union_user1"}
        )

    def test_choice_resolves_and_replaces_card(self, _patch_callback_types):
        adapter = _prepare_callback_adapter()
        choices = ["Red", "Green", "Blue"]
        clarify_gateway.register("clarify-choice", "session-choice", "Pick one", choices)
        _store_clarify_state(
            adapter,
            clarify_id="clarify-choice",
            session_key="session-choice",
            choices=choices,
        )
        data = _make_card_action_data(
            {
                "hermes_action": "clarify",
                "clarify_id": "clarify-choice",
                "choice_index": 1,
            }
        )

        response = adapter._on_card_action_trigger(data)

        with clarify_gateway._lock:
            entry = clarify_gateway._entries["clarify-choice"]
        assert entry.response == "Green"
        assert entry.event.is_set()
        assert "clarify-choice" not in adapter._clarify_state
        assert response.card is not None
        assert response.card.data["header"]["template"] == "green"
        assert "Green" in response.card.data["elements"][0]["content"]
        assert "Alice" in response.card.data["elements"][0]["content"]
        assert all(element.get("tag") != "action" for element in response.card.data["elements"])

    def test_other_marks_awaiting_text_and_removes_buttons(self, _patch_callback_types):
        adapter = _prepare_callback_adapter()
        choices = ["Red", "Green"]
        clarify_gateway.register("clarify-other", "session-other", "Pick one", choices)
        _store_clarify_state(
            adapter,
            clarify_id="clarify-other",
            session_key="session-other",
            choices=choices,
        )
        data = _make_card_action_data(
            {
                "hermes_action": "clarify",
                "clarify_id": "clarify-other",
                "choice_index": "other",
            }
        )

        response = adapter._on_card_action_trigger(data)

        pending = clarify_gateway.get_pending_for_session("session-other")
        assert pending is not None
        assert pending.awaiting_text is True
        assert not pending.event.is_set()
        assert "clarify-other" not in adapter._clarify_state
        assert response.card is not None
        assert "type" in response.card.data["elements"][0]["content"].lower()
        assert all(element.get("tag") != "action" for element in response.card.data["elements"])

        assert clarify_gateway.resolve_text_response_for_session(
            "session-other",
            "A custom answer",
        )
        assert pending.response == "A custom answer"
        assert pending.event.is_set()

    def test_expired_prompt_returns_friendly_toast(self, _patch_callback_types, caplog):
        adapter = _prepare_callback_adapter()
        _store_clarify_state(
            adapter,
            clarify_id="clarify-expired",
            session_key="session-expired",
            choices=["A"],
        )
        data = _make_card_action_data(
            {
                "hermes_action": "clarify",
                "clarify_id": "clarify-expired",
                "choice_index": 0,
            }
        )

        with caplog.at_level(logging.WARNING):
            response = adapter._on_card_action_trigger(data)

        assert response.card is None
        assert response.toast is not None
        assert "expired" in response.toast.content.lower()
        assert "clarify-expired" not in adapter._clarify_state
        assert "clarify-expired" in caplog.text

    def test_session_mismatch_does_not_resolve(self, _patch_callback_types):
        adapter = _prepare_callback_adapter()
        entry = clarify_gateway.register(
            "clarify-session",
            "actual-session",
            "Pick one",
            ["A"],
        )
        _store_clarify_state(
            adapter,
            clarify_id="clarify-session",
            session_key="different-session",
            choices=["A"],
        )
        data = _make_card_action_data(
            {
                "hermes_action": "clarify",
                "clarify_id": "clarify-session",
                "choice_index": 0,
            }
        )

        response = adapter._on_card_action_trigger(data)

        assert not entry.event.is_set()
        assert response.toast is not None
        assert "expired" in response.toast.content.lower()

    def test_duplicate_token_only_resolves_once(self, _patch_callback_types):
        adapter = _prepare_callback_adapter()
        choices = ["A"]
        clarify_gateway.register("clarify-dedup", "session-dedup", "Pick one", choices)
        _store_clarify_state(
            adapter,
            clarify_id="clarify-dedup",
            session_key="session-dedup",
            choices=choices,
        )
        data = _make_card_action_data(
            {
                "hermes_action": "clarify",
                "clarify_id": "clarify-dedup",
                "choice_index": 0,
            },
            token="same-token",
        )

        with patch(
            "tools.clarify_gateway.resolve_gateway_clarify",
            wraps=clarify_gateway.resolve_gateway_clarify,
        ) as mock_resolve:
            first = adapter._on_card_action_trigger(data)
            second = adapter._on_card_action_trigger(data)

        assert first.card is not None
        assert second.card is None
        assert second.toast is not None
        assert "already" in second.toast.content.lower()
        mock_resolve.assert_called_once_with("clarify-dedup", "A")
