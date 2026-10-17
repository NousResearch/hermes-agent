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


@pytest.mark.parametrize("generation,present", [(1, True), (2, True), (1, False)])
def test_authority_background_never_borrows_parent_generation(monkeypatch, generation, present):
    authority = SimpleNamespace(
        sessions={"s": object()},
        db=SimpleNamespace(get_session=lambda sid: {"runtime_generation": 1}),
        logical_owner=lambda sid: "s",
    )
    monkeypatch.setattr("gateway.session_authorities.active_authority", lambda runner: authority)
    turn = TurnRunner(None, TurnContext(session_key="route", session_id="s"))
    transport = approval_transport(turn)
    authority.db.get_session = lambda sid: {"runtime_generation": generation}
    if not present:
        authority.sessions.clear()
    monkeypatch.setattr(TurnRunner, "_render_approval_sync", lambda *a, **kw: pytest.fail("native send"))
    with pytest.raises(RuntimeError, match="dispatch-owned approval lifecycle"):
        transport.background_notify({"command": "example"})


@pytest.mark.parametrize("declined", [False, True])
def test_foreground_authority_checks_renders_then_registers(monkeypatch, declined):
    events = []
    authority = SimpleNamespace(
        check_approval_generation=lambda *a: events.append("check"),
        register_approval=lambda *a, **kw: events.append("register"),
    )
    turn = TurnRunner(None, TurnContext(session_key="route"))
    turn._approval_owner = (authority, "s", 1)
    def render(*a, **kw):
        events.append("render")
        if declined:
            raise RuntimeError("connector declined")
    monkeypatch.setattr(turn, "_render_approval_sync", render)
    monkeypatch.setattr("gateway.local_viewer_events.prompt_timeout_s", lambda *a: 30)
    if declined:
        with pytest.raises(RuntimeError, match="connector declined"):
            turn._approval_notify_sync({"command": "example"})
    else:
        turn._approval_notify_sync({"command": "example"})
    assert events == (["check", "render"] if declined else ["check", "render", "register"])
