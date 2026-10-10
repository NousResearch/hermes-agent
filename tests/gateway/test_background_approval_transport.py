"""Off-turn approvals use the captured chat, never a completed native stream."""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import gateway.run  # load bootstrap before the per-test home I/O guard
from gateway.turn_context import TurnContext
from gateway.run_turn_runner import TurnRunner
from gateway.run_turn_runner_approval_transport import approval_transport


def test_background_prompt_ignores_completed_stream(monkeypatch):
    sent = []

    class Adapter:
        async def send(self, chat_id, text, metadata=None):
            sent.append((chat_id, metadata))
            return SimpleNamespace(success=True)

        def pause_typing_for_chat(self, chat_id):
            raise AssertionError("completed turn typing state was reused")

    ctx = TurnContext(session_key="chat-a", _status_adapter=Adapter(),
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

    ctx = TurnContext(session_key="origin-session", _status_chat_id="origin-chat",
                      _status_adapter=ButtonAdapter() if buttons else Adapter())
    transport = approval_transport(TurnRunner(None, ctx))

    def schedule(self, coro, *args):
        fut = Future()
        fut.set_result(asyncio.run(coro))
        return fut

    monkeypatch.setattr(TurnRunner, "_schedule", schedule)
    monkeypatch.setattr(gateway.run, "_approval_send_outcome", lambda *args, **kw: "declined")
    with pytest.raises(RuntimeError):
        transport.background_notify({"command": "print(1)"})
    assert sent == [("origin-chat", "origin-session")] if buttons else sent == ["text"]
