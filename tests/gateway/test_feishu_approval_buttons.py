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
from plugins.platforms.feishu.adapter import FeishuAdapter, FeishuGroupRule


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
    card_id: str = "om_card",
) -> SimpleNamespace:
    """Create a mock Feishu card action callback data object."""
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=SimpleNamespace(
                open_chat_id=chat_id,
                open_message_id=card_id,
            ),
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


def test_pinned_sdk_card_callback_and_send_receipt_fields_match_adapter_contract():
    callback_model = pytest.importorskip(
        "lark_oapi.event.callback.model.p2_card_action_trigger"
    )
    reply_model = pytest.importorskip(
        "lark_oapi.api.im.v1.model.reply_message_response_body"
    )
    create_model = pytest.importorskip(
        "lark_oapi.api.im.v1.model.create_message_response_body"
    )

    assert {"open_chat_id", "open_message_id"}.issubset(
        callback_model.CallBackContext._types
    )
    assert {"open_id", "user_id", "union_id"}.issubset(
        callback_model.CallBackOperator._types
    )
    assert "message_id" in reply_model.ReplyMessageResponseBody._types
    assert "message_id" in create_model.CreateMessageResponseBody._types

    callback = callback_model.P2CardActionTrigger({
        "event": {
            "operator": {
                "open_id": "ou_operator",
                "user_id": "operator-stable-id",
                "union_id": "on_operator",
            },
            "context": {
                "open_chat_id": "oc_chat",
                "open_message_id": "om_card",
            },
            "action": {
                "value": {"hermes_interactive_action_id": "ia_opaque"},
            },
        }
    })
    assert callback.event.operator.user_id == "operator-stable-id"
    assert callback.event.context.open_chat_id == "oc_chat"
    assert callback.event.context.open_message_id == "om_card"


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


class TestFeishuPluginInteractiveCard:
    """Plugin cards use opaque Hermes action-instance values only."""

    def test_webhook_mode_uses_generic_fallback_instead_of_native_ledger(self):
        adapter = FeishuAdapter(
            PlatformConfig(
                enabled=True,
                extra={"connection_mode": "webhook"},
            )
        )

        assert adapter.supports_interactive_cards is False

    @pytest.mark.asyncio
    async def test_native_render_contains_no_business_payload_in_button_values(self):
        from gateway.interactive_actions import (
            InteractiveCardAction,
            InteractiveCardEnvelope,
            InteractiveCardFact,
            InteractiveCardSection,
        )

        adapter = _make_adapter()
        envelope = InteractiveCardEnvelope(
            version=1,
            title="Apply proposal?",
            summary="Review the commercial change.",
            facts=(InteractiveCardFact("Customer", "Acme"),),
            sections=(InteractiveCardSection("Terms", "CNY 120,000 / year"),),
            fallback_text="Apply proposal for Acme in the admin console.",
            expires_in_seconds=900,
            actions=(
                InteractiveCardAction(
                    label="Apply",
                    action="proposal-plugin/apply",
                    external_action_id="proposal-acme-v7",
                    payload={"proposal_id": "prop_7", "revision": 7},
                    style="primary",
                ),
            ),
        )
        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="om_card"),
        )

        with patch.object(
            adapter,
            "_feishu_send_with_retry",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_interactive_card(
                chat_id="oc_chat",
                envelope=envelope,
                action_instance_ids=("ia_opaque_123",),
                reply_to="om_trigger",
                metadata={"thread_id": "om_root"},
            )

        assert result.success is True
        kwargs = mock_send.await_args.kwargs
        assert kwargs["msg_type"] == "interactive"
        assert kwargs["reply_to"] == "om_trigger"
        card = json.loads(kwargs["payload"])
        buttons = [
            action
            for element in card["elements"]
            if element.get("tag") == "action"
            for action in element["actions"]
        ]
        assert [button["value"] for button in buttons] == [
            {"hermes_interactive_action_id": "ia_opaque_123"}
        ]
        serialized_values = json.dumps(
            [button["value"] for button in buttons],
            ensure_ascii=False,
        )
        assert "prop_7" not in serialized_values
        assert "proposal-acme-v7" not in serialized_values
        assert "proposal-plugin/apply" not in serialized_values

    @pytest.mark.asyncio
    async def test_callback_uses_same_stable_user_identity_order_as_inbound_messages(
        self,
    ):
        from gateway.interactive_actions import InteractiveActionResult

        adapter = _make_adapter()
        adapter._resolve_sender_name_from_api = AsyncMock(return_value="Alice")
        adapter.get_chat_info = AsyncMock(
            return_value={"name": "Sales", "type": "group"}
        )
        runner = SimpleNamespace(
            _profile_name_for_source=lambda _source: None,
            _begin_interactive_action_from_adapter=AsyncMock(
                return_value=InteractiveActionResult.processing()
            ),
        )
        adapter.gateway_runner = runner
        event = SimpleNamespace(
            context=SimpleNamespace(
                open_chat_id="oc_chat",
                open_message_id="om_card",
            ),
            operator=SimpleNamespace(
                open_id="ou_app_scoped",
                user_id="tenant-stable-user",
                union_id="on_developer_scoped",
            ),
        )

        result = await adapter._begin_plugin_interactive_action(
            event=event,
            action_instance_id="ia_opaque",
        )

        assert result.status == "processing"
        kwargs = runner._begin_interactive_action_from_adapter.await_args.kwargs
        assert kwargs["callback"].operator_id == "tenant-stable-user"
        assert kwargs["source"].user_id == "tenant-stable-user"
        assert kwargs["source"].user_id_alt == "on_developer_scoped"
        assert kwargs["source"].message_id == "om_card"
        assert kwargs["adapter"] is adapter

    @pytest.mark.parametrize(
        ("status", "title_fragment", "template"),
        [
            ("processing", "Processing", "blue"),
            ("succeeded", "Applied", "green"),
            ("downstream_replay", "Already applied", "green"),
            ("already_processed", "Already processed", "blue"),
            ("denied", "Denied", "red"),
            ("conflict", "Conflict", "orange"),
            ("retryable_failure", "Retryable failure", "orange"),
            ("unknown_outcome", "Outcome unknown", "grey"),
        ],
    )
    def test_status_cards_distinguish_truthful_states(
        self,
        status,
        title_fragment,
        template,
    ):
        from gateway.interactive_actions import InteractiveActionResult

        result = getattr(InteractiveActionResult, status)()
        card = FeishuAdapter._build_interactive_action_status_card(result)

        assert title_fragment in card["header"]["title"]["content"]
        assert card["header"]["template"] == template
        if status != "succeeded":
            assert card["header"]["title"]["content"] != "✅ Applied"

    @pytest.mark.asyncio
    async def test_final_result_updates_native_card(self):
        from gateway.interactive_actions import InteractiveActionResult

        adapter = _make_adapter()
        response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="om_card"),
        )
        with (
            patch.object(
                adapter._client.im.v1.message,
                "update",
                return_value=response,
            ),
            patch.object(
                adapter,
                "_run_blocking",
                new_callable=AsyncMock,
                return_value=response,
            ) as run_blocking,
        ):
            result = await adapter.update_interactive_card(
                chat_id="oc_chat",
                card_id="om_card",
                result=InteractiveActionResult.succeeded(),
            )

        assert result.success is True
        assert result.message_id == "om_card"
        request = run_blocking.await_args.args[1]
        assert request.message_id == "om_card"
        assert request.request_body.msg_type == "interactive"
        card = json.loads(request.request_body.content)
        assert card["header"]["template"] == "green"

    @pytest.mark.asyncio
    async def test_retryable_update_keeps_same_opaque_action_reclaimable(self):
        from gateway.interactive_actions import InteractiveActionResult

        adapter = _make_adapter()
        response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="om_card"),
        )
        with (
            patch.object(
                adapter._client.im.v1.message,
                "update",
                return_value=response,
            ),
            patch.object(
                adapter,
                "_run_blocking",
                new_callable=AsyncMock,
                return_value=response,
            ) as run_blocking,
        ):
            result = await adapter.update_interactive_card(
                chat_id="oc_chat",
                card_id="om_card",
                result=InteractiveActionResult.retryable_failure(),
                action_instance_id="ia_opaque_123",
            )

        assert result.success is True
        request = run_blocking.await_args.args[1]
        card = json.loads(request.request_body.content)
        assert card["elements"][1]["actions"][0]["value"] == {
            "hermes_interactive_action_id": "ia_opaque_123"
        }


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
        adapter._allowed_group_users = {"ou_user1"}
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

    @pytest.mark.parametrize(
        ("action_value", "state_attr"),
        [
            (
                {"hermes_action": "approve_once", "approval_id": 9},
                "_approval_state",
            ),
            (
                {
                    "hermes_update_prompt_action": "y",
                    "update_prompt_id": 9,
                },
                "_update_prompt_state",
            ),
        ],
        ids=("approval", "update-prompt"),
    )
    @pytest.mark.parametrize(
        ("open_id", "is_allowed"),
        [
            ("ou_outsider", False),
            ("ou_group_operator", True),
        ],
        ids=("outsider-denied", "group-operator-allowed"),
    )
    def test_group_allowlist_controls_globally_unlisted_operators(
        self,
        _patch_callback_card_types,
        action_value,
        state_attr,
        open_id,
        is_allowed,
    ):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter._group_rules = {
            "oc_12345": FeishuGroupRule(
                policy="allowlist",
                allowlist={"ou_group_operator"},
            )
        }
        getattr(adapter, state_attr)[9] = {
            "session_key": "sess-9",
            "message_id": "msg-9",
            "chat_id": "oc_12345",
        }
        data = _make_card_action_data(action_value, open_id=open_id)

        assert adapter._admins == set()
        assert adapter._allowed_group_users == set()
        with patch(
            "asyncio.run_coroutine_threadsafe",
            side_effect=_close_submitted_coro,
        ) as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert (response.card is not None) is is_allowed
        if is_allowed:
            mock_submit.assert_called_once()
        else:
            mock_submit.assert_not_called()

    def test_drops_action_when_loop_not_ready(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = None
        data = _make_card_action_data({"hermes_action": "approve_once", "approval_id": 1})

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_registered_plugin_action_returns_processing_and_bypasses_synthetic_path(
        self,
        _patch_callback_card_types,
    ):
        from gateway.interactive_actions import InteractiveActionResult

        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        adapter.gateway_runner = MagicMock()
        data = _make_card_action_data(
            {"hermes_interactive_action_id": "ia_opaque_123"},
            open_id="ou_bob",
        )
        submitted = []

        def submit(coro, _loop):
            submitted.append(coro)
            coro.close()
            return SimpleNamespace(
                result=lambda timeout: InteractiveActionResult.processing()
            )

        with (
            patch("asyncio.run_coroutine_threadsafe", side_effect=submit),
            patch.object(adapter, "_submit_on_loop") as mock_background_submit,
        ):
            response = adapter._on_card_action_trigger(data)

        assert len(submitted) == 1
        mock_background_submit.assert_not_called()
        assert response.card is not None
        assert response.card.data["header"]["template"] == "blue"
        assert "Processing" in response.card.data["header"]["title"]["content"]

    def test_registered_plugin_claim_timeout_cancels_pending_work(
        self,
        _patch_callback_card_types,
    ):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data(
            {"hermes_interactive_action_id": "ia_opaque_123"},
        )
        future = MagicMock()
        future.result.side_effect = TimeoutError

        def submit(coro, _loop):
            coro.close()
            return future

        with patch("asyncio.run_coroutine_threadsafe", side_effect=submit):
            response = adapter._on_card_action_trigger(data)

        future.cancel.assert_called_once_with()
        assert response.card is None

    @pytest.mark.parametrize("reserved_value", ["", 0, None])
    def test_reserved_plugin_action_key_never_falls_through_to_synthetic_command(
        self,
        _patch_callback_card_types,
        reserved_value,
    ):
        from gateway.interactive_actions import InteractiveActionResult

        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data(
            {"hermes_interactive_action_id": reserved_value},
        )

        with (
            patch.object(
                adapter,
                "_handle_plugin_interactive_action",
                return_value=adapter._build_interactive_action_callback_response(
                    InteractiveActionResult.denied()
                ),
            ) as plugin_handler,
            patch.object(adapter, "_submit_on_loop") as synthetic_submit,
        ):
            response = adapter._on_card_action_trigger(data)

        plugin_handler.assert_called_once()
        assert plugin_handler.call_args.kwargs["action_instance_id"] == ""
        synthetic_submit.assert_not_called()
        assert response.card is None

    def test_denied_plugin_click_does_not_replace_shared_card(
        self,
        _patch_callback_card_types,
    ):
        from gateway.interactive_actions import InteractiveActionResult

        adapter = _make_adapter()

        response = adapter._build_interactive_action_callback_response(
            InteractiveActionResult.denied()
        )

        assert response.card is None

    def test_unknown_outcome_replay_replaces_card_with_terminal_warning(
        self,
        _patch_callback_card_types,
    ):
        from gateway.interactive_actions import InteractiveActionResult

        adapter = _make_adapter()
        result = InteractiveActionResult.unknown_outcome()

        response = adapter._build_interactive_action_callback_response(result)

        assert response.card is not None
        assert "Outcome unknown" in response.card.data["header"]["title"]["content"]
        assert response.card.data["elements"][0]["content"] == result.user_message

    def test_unknown_non_hermes_action_preserves_synthetic_card_path(
        self,
        _patch_callback_card_types,
    ):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data({"custom_action": "legacy"})

        with patch.object(adapter, "_submit_on_loop", return_value=True) as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        mock_submit.assert_called_once()
        submitted_coro = mock_submit.call_args.args[1]
        submitted_coro.close()

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


class TestResolveUpdatePrompt:
    """Test update prompt resolution persists the response file."""

    @pytest.mark.asyncio
    async def test_writes_response_file(self, tmp_path, monkeypatch):
        adapter = _make_adapter()
        adapter._allowed_group_users = {"ou_user1"}
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        adapter._update_prompt_state[1] = {
            "session_key": "sess-up-1",
            "message_id": "msg_up_003",
            "chat_id": "oc_12345",
        }

        await adapter._resolve_update_prompt(
            1,
            "y",
            "Alice",
            open_id="ou_user1",
            chat_id="oc_12345",
        )

        assert (tmp_path / ".hermes" / ".update_response").read_text() == "y"
        assert 1 not in adapter._update_prompt_state
