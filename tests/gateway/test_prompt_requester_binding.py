"""A slash-command confirmation or a clarify question in a shared chat answers to its requester.

With ``approvals.approvers`` narrowed (the same opt-in as exec approvals), another member's tap on
the prompt gets "Only <requester> can answer this approval." and the prompt stays pending; the
requester's own tap resolves it.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_agent.gateway.config import Platform, PlatformConfig
from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
from hermes_agent.gateway.session import SessionSource
from tests.gateway.test_approval_requester_binding import write_approvers

SESSION = "agent:main:telegram:group:-100:9"


def _raise_as_alice(kind):
    """Register the prompt inside Alice's turn, as the gateway does; returns a 'was it answered' probe."""
    from hermes_agent.gateway.session_context import clear_session_vars, set_session_vars
    from hermes_agent.tools import clarify_gateway, slash_confirm

    ran = []

    async def handler(choice):
        ran.append(choice)
        return "done"

    tokens = set_session_vars(platform="telegram", user_id="alice", user_name="Alice", chat_type="group",
                              session_key=SESSION)
    try:
        if kind == "slash_confirm":
            slash_confirm.register(SESSION, "cf1", "reload-mcp", handler)
            return lambda: bool(ran)
        entry = clarify_gateway.register("cl1", SESSION, "Which one?", ["red", "blue"])
        return lambda: entry.event.is_set() or entry.awaiting_text
    finally:
        clear_session_vars(tokens)


@pytest.fixture(autouse=True)
def _clean():
    from hermes_agent.tools import clarify_gateway, slash_confirm

    write_approvers([])
    yield
    slash_confirm.clear(SESSION)
    clarify_gateway.clear_session(SESSION)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["slash_confirm", "clarify"])
@pytest.mark.parametrize(("clicker", "answered"), [("bob", False), ("alice", True)])
async def test_relay_prompt_answer_bound_to_requester(kind, clicker, answered, monkeypatch):
    import hermes_agent.gateway.run as gateway_run
    from hermes_agent.gateway.config import GatewayConfig
    from hermes_agent.gateway.relay.adapter import RelayAdapter
    from tests.gateway.relay.stub_connector import StubConnector
    from tests.gateway.relay.test_relay_interactive import make_desc

    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: SimpleNamespace(
        config=GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})))
    was_answered = _raise_as_alice(kind)
    stub = StubConnector(make_desc())
    adapter = RelayAdapter(PlatformConfig(), make_desc(), transport=stub)
    if kind == "slash_confirm":
        await adapter.send_slash_confirm("c1", "Confirm", "reload?", SESSION, "cf1")
        option = "always"
    else:
        await adapter.send_clarify("c1", "Which one?", ["red", "blue"], "cl1", SESSION)
        option = "c0"
    press = MessageEvent(text=f"/{option}", message_type=MessageType.COMMAND, source=SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", chat_type="group", user_id=clicker),
        prompt_response={"prompt_id": stub.sent[-1]["prompt_id"], "option_id": option})

    assert await adapter._consume_prompt_response(press) is True
    assert was_answered() is answered


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["slash_confirm", "clarify"])
@pytest.mark.parametrize(("tapper", "answered"), [("bob", False), ("alice", True)])
async def test_telegram_prompt_tap_bound_to_requester(kind, tapper, answered, monkeypatch):
    from tests.gateway.test_telegram_approval_buttons import _make_adapter

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    was_answered = _raise_as_alice(kind)
    adapter = _make_adapter()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(chat_id=-100, message_id=42, message_thread_id=None))
    if kind == "slash_confirm":
        await adapter.send_slash_confirm("-100", "Confirm", "reload?", SESSION, "cf1")
        data = "sc:always:cf1"
    else:
        await adapter.send_clarify("-100", "Which one?", ["red", "blue"], "cl1", SESSION)
        data = "cl:cl1:0"
    query = AsyncMock(data=data, message=MagicMock(chat_id=-100, message_id=42, message_thread_id=None, text="x"),
                      from_user=MagicMock(first_name=tapper.title(), id=tapper))

    await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())
    assert was_answered() is answered


@pytest.mark.asyncio
async def test_voice_reply_settles_only_the_clarify_it_was_checked_against():
    """Alice's question expires while her voice note transcribes and Bob's turn asks its own: her
    transcript is her own message, never the answer to Bob's question."""
    from hermes_agent.gateway.config import GatewayConfig
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.session_context import clear_session_vars, set_session_vars
    from hermes_agent.tools import clarify_gateway

    _raise_as_alice("clarify")
    asked = []

    async def transcribe(_event):
        clarify_gateway.clear_session(SESSION)  # Alice's question timed out
        tokens = set_session_vars(platform="telegram", user_id="bob", user_name="Bob", chat_type="group",
                                  session_key=SESSION)
        try:
            asked.append(clarify_gateway.register("cl2", SESSION, "Bob's question?", None))
        finally:
            clear_session_vars(tokens)
        return "alice says yes"

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})
    runner._delivery_adapter_for = lambda _source: None
    runner._pending_event_audio_paths = lambda _event: ["voice.ogg"]
    runner._prepare_clarify_reply_text = transcribe
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", user_id="alice")

    assert await runner._hm_clarify_reply(MessageEvent(text="", source=source), source, SESSION) is None
    assert not asked[0].event.is_set() and clarify_gateway.get_pending("cl2") is asked[0]
