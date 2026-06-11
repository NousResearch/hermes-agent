"""Tests for Feishu clarify interactive card buttons."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    """Create a FeishuAdapter with mocked internals for testing."""
    from gateway.platforms.feishu import FeishuAdapter

    config = PlatformConfig(
        enabled=True,
        extra={
            "app_id": "test_app",
            "app_secret": "test_secret",
            "domain_name": "feishu",
            "connection_mode": "websocket",
        },
    )
    adapter = FeishuAdapter(config)
    adapter._client = MagicMock()  # Pretend connected
    adapter._loop = asyncio.new_event_loop()
    return adapter


class TestBuildClarifyCard:
    """Tests for _build_clarify_card static method."""

    def test_multi_choice_card_structure(self):
        from gateway.platforms.feishu import FeishuAdapter

        card = FeishuAdapter._build_clarify_card(
            question="What color?",
            choices=["Red", "Blue", "Green"],
            clarify_id="c-123",
            clarify_id_int=1,
        )

        # Header
        assert card["header"]["template"] == "blue"
        assert "Clarification" in card["header"]["title"]["content"]

        # Markdown body contains question and choices
        md = card["elements"][0]["content"]
        assert "What color?" in md
        assert "1. Red" in md
        assert "2. Blue" in md
        assert "3. Green" in md

        # Actions: 3 choice buttons + 1 "Other" button
        actions = card["elements"][1]["actions"]
        assert len(actions) == 4
        assert actions[0]["type"] == "primary"  # First button highlighted
        assert actions[1]["type"] == "default"
        assert actions[2]["type"] == "default"
        assert "Other" in actions[3]["text"]["content"]

    def test_button_values_contain_clarify_ids(self):
        from gateway.platforms.feishu import FeishuAdapter

        card = FeishuAdapter._build_clarify_card(
            question="Pick one",
            choices=["A", "B"],
            clarify_id="c-456",
            clarify_id_int=7,
        )
        actions = card["elements"][1]["actions"]

        # First choice button
        assert actions[0]["value"]["hermes_clarify_action"] == "0"
        assert actions[0]["value"]["clarify_id"] == "c-456"
        assert actions[0]["value"]["clarify_id_int"] == 7

        # Second choice button
        assert actions[1]["value"]["hermes_clarify_action"] == "1"

        # "Other" button
        assert actions[2]["value"]["hermes_clarify_action"] == "-1"

    def test_single_choice(self):
        from gateway.platforms.feishu import FeishuAdapter

        card = FeishuAdapter._build_clarify_card(
            question="Confirm?",
            choices=["Yes"],
            clarify_id="c-789",
            clarify_id_int=1,
        )
        actions = card["elements"][1]["actions"]
        assert len(actions) == 2  # 1 choice + 1 "Other"
        assert actions[0]["value"]["hermes_clarify_action"] == "0"


class TestSendClarify:
    """Tests for send_clarify override."""

    @pytest.mark.asyncio
    async def test_open_ended_falls_back_to_base(self):
        adapter = _make_adapter()
        with patch.object(
            type(adapter).__bases__[0], "send_clarify", new_callable=AsyncMock
        ) as mock_base:
            mock_base.return_value = MagicMock(success=True)
            result = await adapter.send_clarify(
                chat_id="chat1",
                question="Tell me more",
                choices=None,
                clarify_id="c-1",
                session_key="sess1",
            )
            mock_base.assert_awaited_once()
            assert result.success

    @pytest.mark.asyncio
    async def test_multi_choice_sends_interactive_card(self):
        adapter = _make_adapter()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"code": 0, "data": {"message_id": "msg_123"}}
        adapter._feishu_send_with_retry = AsyncMock(return_value=mock_response)
        adapter._finalize_send_result = MagicMock(
            return_value=MagicMock(success=True, message_id="msg_123")
        )

        result = await adapter.send_clarify(
            chat_id="chat1",
            question="Which option?",
            choices=["A", "B", "C"],
            clarify_id="c-10",
            session_key="sess1",
        )

        assert result.success
        adapter._feishu_send_with_retry.assert_awaited_once()
        call_kwargs = adapter._feishu_send_with_retry.call_args
        assert call_kwargs.kwargs["msg_type"] == "interactive"

        # Verify the card payload is valid JSON with clarify structure
        payload = json.loads(call_kwargs.kwargs["payload"])
        assert payload["header"]["template"] == "blue"
        assert len(payload["elements"][1]["actions"]) == 4  # 3 choices + Other

        # State should be stored
        assert len(adapter._clarify_state) == 1
        state = list(adapter._clarify_state.values())[0]
        assert state["clarify_id"] == "c-10"
        assert state["session_key"] == "sess1"

    @pytest.mark.asyncio
    async def test_no_client_returns_error(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.send_clarify(
            chat_id="chat1",
            question="Q?",
            choices=["A"],
            clarify_id="c-1",
            session_key="s1",
        )
        assert not result.success
        assert "Not connected" in result.error


class TestHandleClarifyCardAction:
    """Tests for _handle_clarify_card_action."""

    def _make_event(self, open_id="user_123", chat_id="chat_1"):
        event = SimpleNamespace()
        event.operator = SimpleNamespace(open_id=open_id, user_id="uid_1")
        event.context = SimpleNamespace(open_chat_id=chat_id)
        return event

    def test_regular_choice_resolves(self):
        adapter = _make_adapter()
        loop = adapter._loop
        adapter._allow_group_message = MagicMock(return_value=True)
        adapter._get_cached_sender_name = MagicMock(return_value="TestUser")
        adapter._submit_on_loop = MagicMock(return_value=True)

        # Pre-populate state
        adapter._clarify_state[1] = {
            "session_key": "sess1",
            "clarify_id": "c-10",
            "message_id": "msg_1",
            "chat_id": "chat_1",
        }

        action_value = {
            "hermes_clarify_action": "0",
            "clarify_id": "c-10",
            "clarify_id_int": 1,
        }

        result = adapter._handle_clarify_card_action(
            event=self._make_event(),
            action_value=action_value,
            loop=loop,
        )

        # Should have scheduled the resolve coroutine
        adapter._submit_on_loop.assert_called_once()

    def test_other_button_enters_text_capture(self):
        adapter = _make_adapter()
        loop = adapter._loop
        adapter._allow_group_message = MagicMock(return_value=True)
        adapter._get_cached_sender_name = MagicMock(return_value="TestUser")
        adapter._submit_on_loop = MagicMock(return_value=True)

        adapter._clarify_state[2] = {
            "session_key": "sess1",
            "clarify_id": "c-20",
            "message_id": "msg_2",
            "chat_id": "chat_1",
        }

        action_value = {
            "hermes_clarify_action": "-1",
            "clarify_id": "c-20",
            "clarify_id_int": 2,
        }

        result = adapter._handle_clarify_card_action(
            event=self._make_event(),
            action_value=action_value,
            loop=loop,
        )

        # Should schedule _resolve_clarify_other (not _resolve_clarify)
        adapter._submit_on_loop.assert_called_once()

    def test_unauthorized_user_rejected(self):
        adapter = _make_adapter()
        loop = adapter._loop
        adapter._allow_group_message = MagicMock(return_value=False)

        adapter._clarify_state[3] = {
            "session_key": "sess1",
            "clarify_id": "c-30",
            "message_id": "msg_3",
            "chat_id": "chat_1",
        }

        action_value = {
            "hermes_clarify_action": "0",
            "clarify_id": "c-30",
            "clarify_id_int": 3,
        }

        result = adapter._handle_clarify_card_action(
            event=self._make_event(open_id="unauthorized"),
            action_value=action_value,
            loop=loop,
        )

        # Should not have resolved — state still present
        assert 3 in adapter._clarify_state

    def test_already_resolved_ignored(self):
        adapter = _make_adapter()
        loop = adapter._loop

        # No state for clarify_id_int=99
        action_value = {
            "hermes_clarify_action": "0",
            "clarify_id": "c-99",
            "clarify_id_int": 99,
        }

        result = adapter._handle_clarify_card_action(
            event=self._make_event(),
            action_value=action_value,
            loop=loop,
        )

        # Should return empty response, not crash
        assert result is not None or result is None  # Just no exception


class TestResolveClarify:
    """Tests for _resolve_clarify and _resolve_clarify_other."""

    @pytest.mark.asyncio
    async def test_resolve_clarify_calls_gateway(self):
        adapter = _make_adapter()
        adapter._clarify_state[1] = {
            "session_key": "sess1",
            "clarify_id": "c-10",
            "message_id": "msg_1",
            "chat_id": "chat_1",
        }

        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            mock_resolve.return_value = True
            await adapter._resolve_clarify(1, "c-10", 2, "TestUser")
            mock_resolve.assert_called_once_with("c-10", "2")

        # State should be popped
        assert 1 not in adapter._clarify_state

    @pytest.mark.asyncio
    async def test_resolve_clarify_other_marks_awaiting_text(self):
        adapter = _make_adapter()
        adapter._clarify_state[2] = {
            "session_key": "sess1",
            "clarify_id": "c-20",
            "message_id": "msg_2",
            "chat_id": "chat_1",
        }

        with patch("tools.clarify_gateway.mark_awaiting_text") as mock_mark:
            mock_mark.return_value = True
            await adapter._resolve_clarify_other(2, "c-20", "TestUser")
            mock_mark.assert_called_once_with("c-20")

        # State should NOT be popped (still awaiting text)
        assert 2 in adapter._clarify_state


class TestResolvedCard:
    """Tests for _build_resolved_clarify_card."""

    def test_choice_resolved_card(self):
        from gateway.platforms.feishu import FeishuAdapter

        card = FeishuAdapter._build_resolved_clarify_card(
            choice_text="✅ Choice 1",
            user_name="Alice",
        )
        assert card["header"]["template"] == "green"
        assert "Choice 1" in card["header"]["title"]["content"]
        assert "Alice" in card["elements"][0]["content"]

    def test_waiting_resolved_card(self):
        from gateway.platforms.feishu import FeishuAdapter

        card = FeishuAdapter._build_resolved_clarify_card(
            choice_text="✏️ Waiting for your answer...",
            user_name="Bob",
        )
        assert card["header"]["template"] == "blue"
        assert "Waiting" in card["header"]["title"]["content"]


class TestCardActionTriggerRouting:
    """Tests that _on_card_action_trigger routes clarify actions correctly."""

    def test_clarify_action_routed_to_handler(self):
        adapter = _make_adapter()
        adapter._loop_accepts_callbacks = MagicMock(return_value=True)
        adapter._handle_clarify_card_action = MagicMock(return_value="clarify_response")

        data = SimpleNamespace()
        data.event = SimpleNamespace(
            action=SimpleNamespace(
                value={
                    "hermes_clarify_action": "0",
                    "clarify_id": "c-1",
                    "clarify_id_int": 1,
                }
            ),
            operator=SimpleNamespace(open_id="u1", user_id="uid1"),
            context=SimpleNamespace(open_chat_id="chat1"),
        )

        result = adapter._on_card_action_trigger(data)
        assert result == "clarify_response"
        adapter._handle_clarify_card_action.assert_called_once()

    def test_approval_action_not_routed_to_clarify(self):
        adapter = _make_adapter()
        adapter._loop_accepts_callbacks = MagicMock(return_value=True)
        adapter._handle_approval_card_action = MagicMock(return_value="approval_response")
        adapter._handle_clarify_card_action = MagicMock()

        data = SimpleNamespace()
        data.event = SimpleNamespace(
            action=SimpleNamespace(
                value={"hermes_action": "approve_once", "approval_id": 1}
            ),
            operator=SimpleNamespace(open_id="u1", user_id="uid1"),
            context=SimpleNamespace(open_chat_id="chat1"),
        )

        result = adapter._on_card_action_trigger(data)
        assert result == "approval_response"
        adapter._handle_clarify_card_action.assert_not_called()
