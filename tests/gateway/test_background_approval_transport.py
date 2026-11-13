"""Off-turn approvals use the captured chat, never a completed native stream."""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import hermes_agent.gateway.run  # load bootstrap before the per-test home I/O guard
from hermes_agent.gateway.config import Platform
from hermes_agent.gateway.session import SessionSource
from hermes_agent.gateway.turn_context import TurnContext

_CHAT = SessionSource(platform=Platform.TELEGRAM, chat_id="a", chat_type="dm", user_id="u")
from hermes_agent.gateway.run_turn_runner import TurnRunner
from hermes_agent.gateway.run_turn_runner_approval_transport import approval_transport


def test_background_prompt_ignores_completed_stream(monkeypatch):
    sent = []

    class Adapter:
        async def send(self, chat_id, text, metadata=None):
            sent.append((chat_id, metadata))
            return SimpleNamespace(success=True)

        def pause_typing_for_chat(self, chat_id):
            raise AssertionError("completed turn typing state was reused")

    ctx = TurnContext(session_key="chat-a", source=_CHAT, _status_adapter_pinned=Adapter(),
                      _status_chat_id="a", _status_thread_metadata={"thread_id": "t"},
                      _run_still_current=lambda: False)
    turn = TurnRunner(None, ctx)

    def schedule(self, coro, *args):
        import asyncio
        fut = Future()
        fut.set_result(asyncio.run(coro))
        return fut

    monkeypatch.setattr(TurnRunner, "_schedule", schedule)
    transport = approval_transport(turn)
    ctx._status_chat_id = "new-chat"
    transport.background_notify({"command": "print(1)", "description": "test"})
    assert sent[0][0] == "a"
    assert sent[0][1]["thread_id"] == "t"
    assert sent[0][1]["is_approval_prompt"] is True


@pytest.mark.parametrize("buttons", [False, True])
def test_background_transport_refusal_fails_closed(monkeypatch, buttons):
    import asyncio

    sent = []

    class Adapter:
        async def send(self, *args, **kwargs):
            sent.append("text")
            return SimpleNamespace(success=False)

    class ButtonAdapter(Adapter):
        async def send_exec_approval(self, **kwargs):
            sent.append((kwargs["chat_id"], kwargs["session_key"]))
            return SimpleNamespace(success=False)

    ctx = TurnContext(session_key="origin-session", _status_chat_id="origin-chat", source=_CHAT,
                      _status_adapter_pinned=ButtonAdapter() if buttons else Adapter())
    transport = approval_transport(TurnRunner(None, ctx))

    def schedule(self, coro, *args):
        fut = Future()
        fut.set_result(asyncio.run(coro))
        return fut

    monkeypatch.setattr(TurnRunner, "_schedule", schedule)
    monkeypatch.setattr(hermes_agent.gateway.run, "_approval_send_outcome", lambda *args, **kw: "declined")
    with pytest.raises(RuntimeError):
        transport.background_notify({"command": "print(1)"})
    assert sent == [("origin-chat", "origin-session")] if buttons else sent == ["text"]


@pytest.mark.parametrize("buttons", [False, True], ids=["text-only", "card-fallback"])
def test_background_text_prompt_whose_receipt_times_out_stays_pending(monkeypatch, buttons):
    """The text prompt was sent but its receipt timed out: it may be on screen, so it is not withdrawn."""
    import asyncio
    from concurrent.futures import TimeoutError as FutureTimeout

    from hermes_agent.tools import approval
    from hermes_agent.tools.approval_gateway_wait import _await_gateway_decision

    sent = []

    class Adapter:
        async def send(self, chat_id, text, metadata=None):
            sent.append(text)
            return SimpleNamespace(success=True, message_id="landed")

    class ButtonAdapter(Adapter):
        async def send_exec_approval(self, **_kwargs):
            return SimpleNamespace(success=False)

    class LateReceipt:
        def done(self):
            return False

        def result(self, timeout=None):
            raise FutureTimeout("receipt late; the message already went out")

    def schedule(self, coro, *args):
        result = asyncio.run(coro)
        if getattr(result, "success", False):
            return LateReceipt()
        fut = Future()
        fut.set_result(result)
        return fut

    monkeypatch.setattr(TurnRunner, "_schedule", schedule)
    monkeypatch.setattr(approval.approval_context, "_get_approval_timeout", lambda: 0.02)
    ctx = TurnContext(session_key="late-receipt", source=_CHAT, _status_chat_id="a",
                      _status_adapter_pinned=ButtonAdapter() if buttons else Adapter())
    notify = approval_transport(TurnRunner(None, ctx)).background_notify
    result = _await_gateway_decision(
        "late-receipt", notify,
        {"command": "print(1)", "description": "test", "pattern_key": "k", "pattern_keys": ["k"]})

    # Sent once, never re-sent; it stayed pending until its own timeout (whose notice follows).
    assert sum("print(1)" in text for text in sent) == 1, sent
    assert not result.get("notify_failed"), result
