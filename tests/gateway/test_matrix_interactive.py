"""Tests for Matrix interactive approval and model picker controls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


def _make_adapter():
    from gateway.platforms.matrix import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        },
    )
    return MatrixAdapter(config)


def _reaction_event(event_id: str, sender: str, target_event_id: str, key: str, room_id: str = "!room:example.org"):
    return SimpleNamespace(
        event_id=event_id,
        sender=sender,
        room_id=room_id,
        content={
            "m.relates_to": {
                "event_id": target_event_id,
                "key": key,
            }
        },
    )


@pytest.mark.asyncio
class TestMatrixExecApproval:
    async def test_send_exec_approval_stores_control_state(self):
        adapter = _make_adapter()
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="$approval"))

        result = await adapter.send_exec_approval(
            chat_id="!room:example.org",
            command="rm -rf /tmp/x",
            session_key="session-1",
            description="dangerous command",
            metadata={"thread_id": "$thread1", "sender_id": "@chris:example.org"},
        )

        assert result.success is True
        assert "$approval" in adapter._approval_state
        state = adapter._approval_state["$approval"]
        assert state["session_key"] == "session-1"
        assert state["thread_id"] == "$thread1"
        assert state["authorized_actor"] == "@chris:example.org"
        assert adapter._event_roles["$approval"]["role"] == "interactive_control"
        assert adapter._event_roles["$approval"]["control_type"] == "approval"
        assert adapter._pending_reactions == {}

    async def test_send_exec_approval_preserves_thread_metadata(self):
        adapter = _make_adapter()
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="$approval"))

        await adapter.send_exec_approval(
            chat_id="!room:example.org",
            command="ls",
            session_key="session-1",
            metadata={"thread_id": "$thread2"},
        )

        adapter.send.assert_awaited_once()
        assert adapter.send.call_args.kwargs["metadata"]["thread_id"] == "$thread2"

    async def test_send_exec_approval_seeds_reaction_choices(self):
        adapter = _make_adapter()
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="$approval"))
        adapter.send_reaction = AsyncMock(return_value=SendResult(success=True, message_id="$rxn"))

        await adapter.send_exec_approval(
            chat_id="!room:example.org",
            command="rm -rf /tmp/x",
            session_key="session-1",
        )

        seeded = [call.args[2] for call in adapter.send_reaction.await_args_list]
        assert seeded == ["✅", "🔁", "♾️", "❌"]

    async def test_send_model_picker_seeds_number_reactions_and_cancel(self):
        adapter = _make_adapter()
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="$picker"))
        adapter.send_reaction = AsyncMock(return_value=SendResult(success=True, message_id="$rxn"))

        async def _on_selected(_chat_id, _model_id, _provider_slug):
            return "done"

        providers = [
            {"slug": "openai-codex", "name": "OpenAI Codex", "models": ["gpt-5.4"], "total_models": 1, "is_current": True},
            {"slug": "custom-vllm", "name": "vLLM", "models": ["m1", "m2"], "total_models": 2},
        ]
        await adapter.send_model_picker(
            chat_id="!room:example.org",
            providers=providers,
            current_model="gpt-5.4",
            current_provider="openai-codex",
            session_key="session-1",
            on_model_selected=_on_selected,
        )

        seeded = [call.args[2] for call in adapter.send_reaction.await_args_list]
        assert seeded == ["1️⃣", "2️⃣", "❌"]

    async def test_approval_reaction_resolves_once(self):
        adapter = _make_adapter()
        adapter._approval_state["$approval"] = {
            "session_key": "session-1",
            "room_id": "!room:example.org",
            "thread_id": None,
            "authorized_actor": "@chris:example.org",
            "resolved": False,
        }
        adapter._event_roles["$approval"] = {"role": "interactive_control", "control_type": "approval"}
        adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="$edit"))

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            event = _reaction_event("$r1", "@chris:example.org", "$approval", "✅")
            await adapter._on_reaction(event)
            await adapter._on_reaction(event)

        mock_resolve.assert_called_once_with("session-1", "once")
        assert "$approval" not in adapter._approval_state

    async def test_approval_wrong_user_ignored(self):
        adapter = _make_adapter()
        adapter._approval_state["$approval"] = {
            "session_key": "session-1",
            "room_id": "!room:example.org",
            "thread_id": None,
            "authorized_actor": "@chris:example.org",
            "resolved": False,
        }
        adapter._event_roles["$approval"] = {"role": "interactive_control", "control_type": "approval"}
        adapter.edit_message = AsyncMock()

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._on_reaction(_reaction_event("$r1", "@intruder:example.org", "$approval", "✅"))

        mock_resolve.assert_not_called()
        assert "$approval" in adapter._approval_state

    async def test_approval_self_reaction_ignored(self):
        adapter = _make_adapter()
        adapter._approval_state["$approval"] = {
            "session_key": "session-1",
            "room_id": "!room:example.org",
            "thread_id": None,
            "authorized_actor": None,
            "resolved": False,
        }
        adapter._event_roles["$approval"] = {"role": "interactive_control", "control_type": "approval"}
        adapter.edit_message = AsyncMock()

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._on_reaction(_reaction_event("$r1", "@bot:example.org", "$approval", "✅"))

        mock_resolve.assert_not_called()

    async def test_disconnect_denies_pending_approvals_once(self):
        adapter = _make_adapter()
        adapter._approval_state["$approval"] = {
            "session_key": "session-1",
            "room_id": "!room:example.org",
            "thread_id": None,
            "authorized_actor": None,
            "resolved": False,
        }
        adapter._model_picker_state["$picker"] = {"resolved": False}
        adapter._event_roles["$approval"] = {"role": "interactive_control", "control_type": "approval"}
        adapter._sync_task = None
        adapter._client = None

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter.disconnect()

        mock_resolve.assert_called_once_with("session-1", "deny")
        assert adapter._approval_state == {}
        assert adapter._model_picker_state == {}
        assert adapter._event_roles == {}


@pytest.mark.asyncio
class TestMatrixModelPicker:
    async def test_send_model_picker_stores_control_state(self):
        adapter = _make_adapter()
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="$picker"))

        async def _on_selected(_chat_id, _model_id, _provider_slug):
            return "done"

        providers = [
            {"slug": "openai-codex", "name": "OpenAI Codex", "models": ["gpt-5.4"], "total_models": 1, "is_current": True},
            {"slug": "custom-vllm", "name": "vLLM", "models": ["m1", "m2"], "total_models": 2},
        ]
        result = await adapter.send_model_picker(
            chat_id="!room:example.org",
            providers=providers,
            current_model="gpt-5.4",
            current_provider="openai-codex",
            session_key="session-1",
            on_model_selected=_on_selected,
            metadata={"thread_id": "$thread1", "sender_id": "@chris:example.org"},
        )

        assert result.success is True
        state = adapter._model_picker_state["$picker"]
        assert state["stage"] == "provider"
        assert state["thread_id"] == "$thread1"
        assert state["authorized_actor"] == "@chris:example.org"
        assert adapter._event_roles["$picker"]["control_type"] == "model_picker"

    async def test_model_picker_provider_then_model_selection(self):
        adapter = _make_adapter()
        callback = AsyncMock(return_value="Model switched")
        adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="$edit"))
        adapter._model_picker_state["$picker"] = {
            "control_type": "model_picker",
            "stage": "provider",
            "room_id": "!room:example.org",
            "thread_id": "$thread1",
            "session_key": "session-1",
            "authorized_actor": "@chris:example.org",
            "providers": [
                {"slug": "openai-codex", "name": "OpenAI Codex", "models": ["gpt-5.4"]},
                {"slug": "custom-vllm", "name": "vLLM", "models": ["m1", "m2"]},
            ],
            "current_model": "gpt-5.4",
            "current_provider": "openai-codex",
            "on_model_selected": callback,
            "page": 0,
            "page_size": 9,
            "selected_provider": None,
            "resolved": False,
        }
        adapter._event_roles["$picker"] = {"role": "interactive_control", "control_type": "model_picker"}

        await adapter._on_reaction(_reaction_event("$r1", "@chris:example.org", "$picker", "2️⃣"))
        assert adapter._model_picker_state["$picker"]["stage"] == "model"
        assert adapter._model_picker_state["$picker"]["selected_provider"]["slug"] == "custom-vllm"

        await adapter._on_reaction(_reaction_event("$r2", "@chris:example.org", "$picker", "1️⃣"))
        callback.assert_awaited_once_with("!room:example.org", "m1", "custom-vllm")
        assert "$picker" not in adapter._model_picker_state

    async def test_model_picker_wrong_user_ignored(self):
        adapter = _make_adapter()
        callback = AsyncMock(return_value="Model switched")
        adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="$edit"))
        adapter._model_picker_state["$picker"] = {
            "control_type": "model_picker",
            "stage": "provider",
            "room_id": "!room:example.org",
            "thread_id": None,
            "session_key": "session-1",
            "authorized_actor": "@chris:example.org",
            "providers": [{"slug": "openai-codex", "name": "OpenAI Codex", "models": ["gpt-5.4"]}],
            "current_model": "gpt-5.4",
            "current_provider": "openai-codex",
            "on_model_selected": callback,
            "page": 0,
            "page_size": 9,
            "selected_provider": None,
            "resolved": False,
        }
        adapter._event_roles["$picker"] = {"role": "interactive_control", "control_type": "model_picker"}

        await adapter._on_reaction(_reaction_event("$r1", "@intruder:example.org", "$picker", "1️⃣"))
        callback.assert_not_called()
        assert adapter._model_picker_state["$picker"]["stage"] == "provider"

    async def test_model_picker_cancel_cleans_state(self):
        adapter = _make_adapter()
        callback = AsyncMock(return_value="Model switched")
        adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="$edit"))
        adapter._model_picker_state["$picker"] = {
            "control_type": "model_picker",
            "stage": "provider",
            "room_id": "!room:example.org",
            "thread_id": None,
            "session_key": "session-1",
            "authorized_actor": None,
            "providers": [{"slug": "openai-codex", "name": "OpenAI Codex", "models": ["gpt-5.4"]}],
            "current_model": "gpt-5.4",
            "current_provider": "openai-codex",
            "on_model_selected": callback,
            "page": 0,
            "page_size": 9,
            "selected_provider": None,
            "resolved": False,
        }
        adapter._event_roles["$picker"] = {"role": "interactive_control", "control_type": "model_picker"}

        await adapter._on_reaction(_reaction_event("$r1", "@chris:example.org", "$picker", "❌"))
        assert "$picker" not in adapter._model_picker_state
        callback.assert_not_called()
