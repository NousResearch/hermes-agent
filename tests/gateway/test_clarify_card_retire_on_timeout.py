"""A clarify that ends without a click retires the adapter's native card (#110821, #111019).

Drives the real ``TurnRunner._clarify_callback_sync`` against a duck-typed adapter with a
persistent card: on timeout the gateway schedules ``retire_clarify_card`` with the expired
notice; an adapter without that method (text-prompt platforms) gets nothing scheduled.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from gateway.platforms.base import SendResult


class _CardAdapter:
    def __init__(self):
        self.retired: list[tuple[str, str]] = []

    def pause_typing_for_chat(self, chat_id):
        return None

    def resume_typing_for_chat(self, chat_id):
        return None

    async def send_clarify(self, **kwargs):
        return SendResult(success=True, message_id="1.2")

    async def retire_clarify_card(self, clarify_id, notice):
        self.retired.append((clarify_id, notice))


class _TextAdapter(_CardAdapter):
    retire_clarify_card = None  # type: ignore[assignment]


def _run_clarify(adapter, answer=None):
    """Returns (clarify response, labels of every coroutine the runner scheduled).
    ``answer`` resolves the pending clarify with that text instead of letting it time out."""
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="C1", _status_thread_metadata={},
        session_key="sk1", stream_consumer_holder=[None])
    labels: list[str] = []

    class _Fut:
        def __init__(self, r): self._r = r
        def result(self, timeout=None): return self._r

    def _schedule(coro, label):
        labels.append(label)
        return _Fut(asyncio.run(coro))

    runner._schedule = _schedule
    runner._close_native_stream_boundary = lambda *a, **k: None
    if answer is not None:
        from tools import clarify_gateway as cm
        real_register = cm.register

        def _register_and_answer(**kwargs):
            entry = real_register(**kwargs)
            cm.resolve_gateway_clarify(kwargs["clarify_id"], answer)
            return entry
        register_patch = patch.object(cm, "register", _register_and_answer)
    else:
        register_patch = patch("tools.clarify_gateway.get_clarify_timeout", return_value=1)
    with register_patch:
        return runner._clarify_callback_sync("Pick one", ["a", "b"]), labels


def test_timeout_retires_the_native_card_with_the_expired_notice():
    adapter = _CardAdapter()
    response, _labels = _run_clarify(adapter)
    assert response.startswith("[user did not respond")
    assert len(adapter.retired) == 1
    assert "expired" in adapter.retired[0][1].lower()


def test_timeout_schedules_nothing_for_adapters_without_a_card():
    _response, labels = _run_clarify(_TextAdapter())
    assert labels == ["Clarify send failed to schedule"]


def test_real_answer_starting_with_a_bracket_is_not_mistaken_for_a_sentinel():
    """'[A] staging' is a user answer, not a timeout: no card retirement, typing re-armed."""
    adapter = _CardAdapter()
    response, labels = _run_clarify(adapter, answer="[A] staging")
    assert response == "[A] staging"
    assert adapter.retired == []
    assert labels == ["Clarify send failed to schedule"]
