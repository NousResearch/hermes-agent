"""Prompt delivery keeps raw card data and scopes transport failure diagnostics."""

import asyncio
import concurrent.futures
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner


class RecordingAdapter:
    def __init__(self, platform, outcome):
        self.platform = platform
        self.outcome = outcome
        self.native_calls = []
        self.text_calls = []

    def pause_typing_for_chat(self, chat_id):
        pass

    def resume_typing_for_chat(self, chat_id):
        pass

    async def send_exec_approval(self, **kwargs):
        self.native_calls.append(kwargs)
        return self._result()

    async def send_clarify(self, **kwargs):
        self.native_calls.append(kwargs)
        return self._result()

    def _result(self):
        if self.outcome == "timeout":
            raise concurrent.futures.TimeoutError()
        if self.outcome == "exception":
            raise RuntimeError("private provider payload")
        return SendResult(success=False, error="private provider payload")

    async def send(self, chat_id, text, **kwargs):
        self.text_calls.append((chat_id, text, kwargs))
        return SendResult(success=True, message_id="fallback-result")


def make_runner(platform, outcome):
    adapter = RecordingAdapter(platform, outcome)
    runner = TurnRunner.__new__(TurnRunner)
    key = f"agent:main:{platform.value}:dm:15551234567"
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter,
        _status_chat_id="15551234567",
        _status_thread_metadata={"thread_id": "original-thread"},
        session_key=key,
    )
    runner._close_native_stream_boundary = Mock()
    runner._stream_consumer = lambda: None

    def schedule(coro, label):
        future = concurrent.futures.Future()
        try:
            future.set_result(asyncio.run(coro))
        except Exception as exc:
            future.set_exception(exc)
        return future

    runner._schedule = schedule
    return runner, adapter, key


@pytest.mark.parametrize("platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, Platform.TELEGRAM])
@pytest.mark.parametrize("outcome", ["error", "exception", "timeout"])
def test_approval_transport_logs_do_not_change_card_or_fallback(platform, outcome, caplog):
    runner, adapter, key = make_runner(platform, outcome)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        runner._approval_notify_sync({"command": "echo exact-operation", "description": "original description"})
    assert len(adapter.native_calls) == 1
    native = adapter.native_calls[0]
    assert native["chat_id"] == "15551234567"
    assert native["session_key"] == key
    assert native["command"] == "echo exact-operation"
    assert native["description"] == "original description"
    assert native["metadata"] == {"thread_id": "original-thread"}
    if outcome == "timeout":
        assert not adapter.text_calls
    else:
        assert len(adapter.text_calls) == 1
        assert adapter.text_calls[0][0] == "15551234567"
        assert "echo exact-operation" in adapter.text_calls[0][1]
        if platform == Platform.TELEGRAM:
            assert "private provider payload" in caplog.text
        else:
            assert "private provider payload" not in caplog.text


@pytest.mark.parametrize("platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, Platform.TELEGRAM])
@pytest.mark.parametrize("outcome", ["error", "exception", "timeout"])
def test_clarify_logs_preserve_registration_disposition_and_raw_question(platform, outcome, monkeypatch, caplog):
    from tools import clarify_gateway

    runner, adapter, key = make_runner(platform, outcome)
    register = Mock()
    clear = Mock()
    wait = Mock(return_value="original late answer")
    monkeypatch.setattr(clarify_gateway, "register", register)
    monkeypatch.setattr(clarify_gateway, "clear_session", clear)
    monkeypatch.setattr(clarify_gateway, "get_clarify_timeout", lambda: 600)
    monkeypatch.setattr(clarify_gateway, "wait_for_response", wait)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        response = runner._clarify_callback_sync("original private question", ["choice A", "choice B"])
    assert register.call_args.kwargs["session_key"] == key
    assert register.call_args.kwargs["question"] == "original private question"
    assert adapter.native_calls[0]["question"] == "original private question"
    assert adapter.native_calls[0]["choices"] == ["choice A", "choice B"]
    assert adapter.native_calls[0]["session_key"] == key
    if outcome == "timeout":
        clear.assert_not_called()
        wait.assert_called_once()
        assert response == "original late answer"
    else:
        clear.assert_called_once_with(key)
        wait.assert_not_called()
        assert response == "[clarify prompt could not be delivered]"
        if platform == Platform.TELEGRAM:
            assert "private provider payload" in caplog.text
        else:
            assert "private provider payload" not in caplog.text
