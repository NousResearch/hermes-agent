"""Telegram forwards Kanban approvals only to the exact subscribed audience."""

import socket
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.kanban_approval_broker import FORWARDING_METADATA_KEY, ForwardedApproval
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter(monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_bootstrap",
                        SimpleNamespace(_happy_eyeballs_create_connection=socket.create_connection))
    from agent import secret_scope as ss
    monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", True)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test", extra={}))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    adapter._running = True
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    from hermes_constants import get_process_hermes_home
    home = get_process_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text("TELEGRAM_ALLOWED_USERS=*\nGATEWAY_ALLOW_ALL_USERS=false\n", encoding="utf-8")
    return adapter


def _binding(authority=None, request_id="request-1"):
    from hermes_constants import get_process_hermes_home

    return ForwardedApproval(
        authority=authority or SimpleNamespace(), session_id="worker-session", generation=3,
        route="worker-route", request_id=request_id, user_id="42", chat_id="100", thread_id="7",
        task_id="t_worker", run_id=9, notifier_profile="default",
        notifier_home=str(get_process_hermes_home()))


def _query(approval_id, *, choice="once", user="42", chat="100", thread="7"):
    query = AsyncMock()
    query.data = f"ea:{choice}:{approval_id}"
    query.from_user = SimpleNamespace(id=user, first_name="John")
    query.message = MagicMock()
    query.message.chat_id = int(chat)
    query.message.message_thread_id = int(thread) if thread is not None else None
    query.message.chat.type = "supergroup"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return query


async def _send(adapter, binding, *, command="edit AGENTS.md"):
    return await adapter.send_exec_approval(
        chat_id="100", command=command, description="protected write", session_key="worker-route",
        request_id=binding.request_id, allow_session=False, allow_permanent=False,
        metadata={FORWARDING_METADATA_KEY: binding, "thread_id": "7"})


@pytest.mark.asyncio
async def test_callback_rejects_wrong_audience_and_binds_exact_response(monkeypatch):
    adapter = _adapter(monkeypatch)
    binding = _binding()
    respond = AsyncMock(return_value=1)
    monkeypatch.setattr("gateway.kanban_approval_broker.respond_forwarded", respond)
    native = MagicMock(return_value=1)
    monkeypatch.setattr("tools.approval.resolve_gateway_approval_card", native)

    assert (await _send(adapter, binding)).success is True
    approval_id = next(iter(adapter._approval_forwarded))
    for wrong in ({"user": "43"}, {"chat": "101"}, {"thread": "8"}):
        query = _query(approval_id, **wrong)
        await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())
        respond.assert_not_called()
        native.assert_not_called()
        assert approval_id in adapter._approval_state

    query = _query(approval_id)
    await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())
    respond.assert_awaited_once_with(binding, "once", user_id="42", chat_id="100", thread_id="7")
    native.assert_not_called()

    await adapter._handle_callback_query(MagicMock(callback_query=_query(approval_id)), MagicMock())
    assert respond.await_count == 1  # duplicate/stale tap cannot consume a later approval


@pytest.mark.asyncio
async def test_forwarded_refusals_log_sanitized_branch_and_notifier(monkeypatch, caplog):
    adapter = _adapter(monkeypatch)
    binding = _binding()
    respond = AsyncMock(return_value=1)
    monkeypatch.setattr("gateway.kanban_approval_broker.respond_forwarded", respond)
    assert (await _send(adapter, binding, command="SECRET_TOOL_PAYLOAD")).success is True
    approval_id = next(iter(adapter._approval_forwarded))

    with caplog.at_level("WARNING"):
        await adapter._handle_callback_query(
            MagicMock(callback_query=_query(approval_id, user="43")), MagicMock())
        from pathlib import Path
        (Path(binding.notifier_home) / ".env").write_text(
            "TELEGRAM_ALLOWED_USERS=someone-else\nGATEWAY_ALLOW_ALL_USERS=false\n", encoding="utf-8")
        await adapter._handle_callback_query(MagicMock(callback_query=_query(approval_id)), MagicMock())

    messages = [record.getMessage() for record in caplog.records
                if "forwarded Telegram approval callback refused" in record.getMessage()]
    assert len(messages) == 2
    assert "audience_mismatch" in messages[0]
    assert "callback_auth_denied" in messages[1]
    assert all(binding.request_id in message for message in messages)
    assert all("notifier_profile=default" in message for message in messages)
    assert all(binding.notifier_home not in message for message in messages)
    assert all("expected_user=42" not in message and "expected_chat=100" not in message for message in messages)
    assert "SECRET_TOOL_PAYLOAD" not in caplog.text
    respond.assert_not_called()
    assert approval_id in adapter._approval_state


@pytest.mark.asyncio
async def test_concurrent_worker_cards_cannot_cross_resolve(monkeypatch):
    adapter = _adapter(monkeypatch)
    first, second = _binding(request_id="first"), _binding(request_id="second")
    respond = AsyncMock(return_value=1)
    monkeypatch.setattr("gateway.kanban_approval_broker.respond_forwarded", respond)
    await _send(adapter, first)
    await _send(adapter, second)
    approval_ids = {binding.request_id: approval_id
                    for approval_id, binding in adapter._approval_forwarded.items()}

    await adapter._handle_callback_query(
        MagicMock(callback_query=_query(approval_ids["second"], choice="deny")), MagicMock())
    await adapter._handle_callback_query(
        MagicMock(callback_query=_query(approval_ids["first"])), MagicMock())

    assert [call.args[:2] for call in respond.await_args_list] == [(second, "deny"), (first, "once")]


@pytest.mark.asyncio
async def test_disconnect_withdraws_forwarded_prompts_only(monkeypatch):
    adapter = _adapter(monkeypatch)
    binding = _binding(authority=SimpleNamespace())
    binding.authority.sessions = {
        "worker-session": SimpleNamespace(controls=SimpleNamespace(withdraw=MagicMock(return_value=True)))
    }
    native_id = 2
    adapter._approval_state[native_id] = "ordinary-session"
    adapter._approval_request_ids[native_id] = "ordinary-request"
    await _send(adapter, binding)

    adapter._mark_disconnected()

    controls = binding.authority.sessions["worker-session"].controls
    controls.withdraw.assert_called_once_with(
        "worker-session", 3, "request-1", "Telegram approval transport disconnected")
    assert native_id in adapter._approval_state
    assert adapter._approval_request_ids[native_id] == "ordinary-request"


@pytest.mark.asyncio
async def test_send_result_is_not_changed_by_forwarding_metadata(monkeypatch):
    adapter = _adapter(monkeypatch)
    result = await _send(adapter, _binding())
    assert isinstance(result, SendResult)
    assert result.success is True and result.message_id == "42"


class FakeBadRequest(Exception):
    pass


@pytest.mark.asyncio
async def test_forwarded_approval_fails_instead_of_leaving_its_bound_topic(monkeypatch):
    adapter = _adapter(monkeypatch)
    send_message = AsyncMock(side_effect=[
        FakeBadRequest("Message thread not found"), SimpleNamespace(message_id=43)])
    bot = adapter._bot
    assert bot is not None
    bot.send_message = send_message

    result = await _send(adapter, _binding())

    assert result.success is False
    assert "thread not found" in result.error.lower()
    assert send_message.await_count == 1
    assert adapter._approval_state == {}
    assert adapter._approval_forwarded == {}


@pytest.mark.asyncio
async def test_ordinary_approval_keeps_stale_thread_fallback(monkeypatch):
    adapter = _adapter(monkeypatch)
    send_message = AsyncMock(side_effect=[
        FakeBadRequest("Message thread not found"), SimpleNamespace(message_id=43)])
    bot = adapter._bot
    assert bot is not None
    bot.send_message = send_message

    result = await adapter.send_exec_approval(
        chat_id="100", command="edit fixture", session_key="ordinary-route",
        request_id="ordinary-request", metadata={"thread_id": "7"})

    assert result.success is True
    assert [call.kwargs.get("message_thread_id")
            for call in send_message.await_args_list] == [7, None]
