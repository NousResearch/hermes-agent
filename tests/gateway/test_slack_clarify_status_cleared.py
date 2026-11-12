"""A clarify prompt clears a durable status indicator before it is posted (#102704).

Pausing the typing refresh loop is enough where the indicator is ephemeral (Telegram's
``sendChatAction`` and Discord's typing expire within seconds). Slack's
``assistant.threads.setStatus`` stays until explicitly cleared, so without a clear the last
"still working…" freezes above the question and the turn hangs until ``clarify_timeout``.

Drives the real ``TurnRunner._clarify_callback_sync`` against a ``BasePlatformAdapter``
subclass so the clear goes through the real ``_stop_typing_with_metadata`` chokepoint.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult

_THREAD = {"thread_id": "1700000000.000100"}
_ONE_QUESTION = [{"qid": "q0", "question": "Which environment?", "choices": ["staging", "production"],
                  "multi_select": False}]


class _StatusCaptureAdapter(BasePlatformAdapter):
    """Records the typing/status lifecycle in call order."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.SLACK)
        self.calls = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="m-1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def stop_typing(self, chat_id, metadata=None) -> None:
        self.calls.append(("stop_typing", chat_id, metadata))

    def pause_typing_for_chat(self, chat_id) -> None:
        self.calls.append(("pause_typing_for_chat", chat_id))
        return super().pause_typing_for_chat(chat_id)

    async def send_clarify(self, **kwargs) -> SendResult:
        self.calls.append(("send_clarify", kwargs["question"]))
        return SendResult(success=True, message_id="clarify-1")


def _run_clarify(adapter, answer="staging"):
    from gateway.run_turn_runner import TurnRunner
    from tools import clarify_gateway as cm

    runner = object.__new__(TurnRunner)
    runner._approval_owner = None
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="C1", _status_thread_metadata=dict(_THREAD),
        session_key="sk1", stream_consumer_holder=[None])

    class _Fut:
        def __init__(self, r): self._r = r
        def result(self, timeout=None): return self._r

    runner._schedule = lambda coro, label: _Fut(asyncio.run(coro))
    runner._close_native_stream_boundary = lambda *a, **k: None
    real_register = cm.register

    def _register(**kwargs):
        entry = real_register(**kwargs)
        cm.resolve_gateway_clarify(kwargs["clarify_id"], answer)
        return entry

    with patch.object(cm, "register", _register), \
            patch("tools.clarify_gateway.get_clarify_timeout", return_value=0.05):
        return runner._clarify_callback_sync(_ONE_QUESTION)


def test_clarify_clears_status_after_pause_and_before_posting():
    """Ordering matters: a clear issued after the post can land behind the answer-time
    resume and blank an indicator that is legitimately active again (cf. #92201)."""
    adapter = _StatusCaptureAdapter()
    reply = _run_clarify(adapter)

    names = [c[0] for c in adapter.calls]
    assert "stop_typing" in names, f"status never cleared; calls={adapter.calls}"
    assert names.index("pause_typing_for_chat") < names.index("stop_typing") < names.index("send_clarify")
    assert reply["answers"] == {"q0": "staging"}


def test_clear_is_thread_scoped():
    """A metadata-less clear would blank a concurrent turn in a sibling thread of the channel."""
    adapter = _StatusCaptureAdapter()
    _run_clarify(adapter)

    clears = [c for c in adapter.calls if c[0] == "stop_typing"]
    assert clears == [("stop_typing", "C1", _THREAD)]

