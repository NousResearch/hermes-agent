"""Prose written before a clarify call reaches the chat even with interim messages off.

With ``display.interim_assistant_messages: false`` and streaming off, ordinary commentary between
tool calls is suppressed on purpose. Prose in the same assistant message as a ``clarify`` call is
different: it is what the user is asked to decide on, and without it the chat shows bare buttons.
The real ``StreamDeliveryMixin._emit_interim_assistant_message`` runs inside a real gateway turn
(``GatewayRunner._run_agent``), so these tests cover both the agent's routing and the gateway's wiring.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import strip_think_blocks
from agent.stream_delivery import StreamDeliveryMixin
from tests.gateway.test_run_progress_topics import _run_with_agent

QUIET = {"display": {"tool_progress": "off", "interim_assistant_messages": False}, "streaming": {"enabled": False}}


def _assistant(text: str, tool: str) -> dict:
    return {"role": "assistant", "content": text,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": tool, "arguments": "{}"}}]}


class _EmittingAgent(StreamDeliveryMixin):
    """Runs the agent's real interim emit for one tool-calling message, then answers."""

    message: dict = {}
    _strip_think_blocks = strip_think_blocks

    def __init__(self, **kwargs):
        self.tools = []
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self._emit_interim_assistant_message(type(self).message)
        return {"final_response": "done", "messages": [], "api_calls": 1}


@pytest.mark.asyncio
async def test_prose_before_clarify_is_sent_when_interim_messages_are_off(monkeypatch, tmp_path):
    monkeypatch.setattr(_EmittingAgent, "message", _assistant("Draft: Hello Bob, lunch at noon?", "clarify"))
    adapter, result = await _run_with_agent(
        monkeypatch, tmp_path, _EmittingAgent, session_id="sess-clarify-ctx", config_data=QUIET)

    assert result["final_response"] == "done"
    assert [call["content"] for call in adapter.sent] == ["Draft: Hello Bob, lunch at noon?"]
    # Not the turn's final: unmarked, it would seal a live stream or be cached as the answer.
    assert adapter.sent[0]["metadata"].get("_interim_send") is True


@pytest.mark.asyncio
async def test_other_commentary_stays_suppressed_when_interim_messages_are_off(monkeypatch, tmp_path):
    monkeypatch.setattr(_EmittingAgent, "message", _assistant("Checking the disk now.", "terminal"))
    adapter, result = await _run_with_agent(
        monkeypatch, tmp_path, _EmittingAgent, session_id="sess-quiet-ctx", config_data=QUIET)

    assert result["final_response"] == "done"
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_clarify_card_waits_for_an_explanation_still_sending():
    """The agent thread stops waiting for the explanation after its ack window; the card must not
    overtake it anyway (the card path does not take Telegram's text-send lock)."""
    from hermes_agent.gateway.run_turn_runner import TurnRunner

    order: list[str] = []
    explanation = concurrent.futures.Future()

    async def send_clarify(**_kwargs):
        order.append("card")
        return SimpleNamespace(success=True)

    runner = SimpleNamespace(_ctx=SimpleNamespace(_status_adapter=SimpleNamespace(send_clarify=send_clarify)),
                             _approval_owner=None, _clarify_context_send=explanation)
    card = asyncio.ensure_future(TurnRunner._send_shared_clarify(runner, entry=None))
    for _ in range(20):  # let the card task run as far as it can
        await asyncio.sleep(0)
    assert order == [], "the card went out while its explanation was still sending"
    order.append("explanation")
    explanation.set_result(True)
    await asyncio.wait_for(card, 5)
    assert order == ["explanation", "card"]


@pytest.mark.asyncio
async def test_codex_clarify_sends_draft_not_commentary_when_interim_messages_are_off(monkeypatch, tmp_path):
    draft = "Draft: Hello Bob, lunch at noon?"
    msg = _assistant(draft, "clarify")
    # Codex Responses may include both a draft and a phase=commentary item.
    msg["codex_message_items"] = [
        {"type": "message", "role": "assistant", "phase": "commentary",
         "content": [{"type": "output_text", "text": "Checking what to send."}]},
        {"type": "message", "role": "assistant", "phase": "final_answer",
         "content": [{"type": "output_text", "text": draft}]},
    ]
    monkeypatch.setattr(_EmittingAgent, "message", msg)
    adapter, result = await _run_with_agent(
        monkeypatch, tmp_path, _EmittingAgent, session_id="sess-codex-clarify-ctx", config_data=QUIET)

    assert result["final_response"] == "done"
    assert [call["content"] for call in adapter.sent] == [draft]


@pytest.mark.asyncio
async def test_clarify_card_posts_when_its_explanation_send_never_returns(monkeypatch):
    """An adapter send that hangs must not hold back the question the turn is waiting on."""
    import hermes_agent.gateway.run_turn_runner_clarify_delivery as delivery
    from hermes_agent.gateway.run_turn_runner import TurnRunner

    monkeypatch.setattr(delivery, "SEND_ACK_WINDOW", 0.2)
    order: list[str] = []

    async def send_clarify(**_kwargs):
        order.append("card")
        return SimpleNamespace(success=True)

    runner = SimpleNamespace(_ctx=SimpleNamespace(_status_adapter=SimpleNamespace(send_clarify=send_clarify)),
                             _approval_owner=None, _clarify_context_send=concurrent.futures.Future())
    await asyncio.wait_for(TurnRunner._send_shared_clarify(runner, entry=None), 5)
    assert order == ["card"]


@pytest.mark.asyncio
async def test_a_question_cancelled_while_its_explanation_sends_posts_no_card():
    """/new or a cancel lands while the card waits for its explanation: the withdrawn card stays unsent."""
    from hermes_agent.gateway.run_turn_runner import TurnRunner
    from tools import clarify_gateway

    cards: list[dict] = []

    async def send_clarify(**kwargs):
        cards.append(kwargs)
        return SimpleNamespace(success=True)

    explanation = concurrent.futures.Future()
    runner = SimpleNamespace(_ctx=SimpleNamespace(_status_adapter=SimpleNamespace(send_clarify=send_clarify),
                                                  _run_still_current=lambda: True),
                             _approval_owner=None, _clarify_context_send=explanation)
    entry = clarify_gateway.register("cid-withdrawn", "sess-withdrawn", "Send it?", ["yes", "no"])
    card = asyncio.ensure_future(TurnRunner._send_shared_clarify(runner, entry, session_key="sess-withdrawn"))
    for _ in range(5):
        await asyncio.sleep(0)
    clarify_gateway.clear_session("sess-withdrawn")
    explanation.set_result(True)
    result = await asyncio.wait_for(card, 5)

    assert cards == []
    assert result.success is False


@pytest.mark.asyncio
async def test_the_card_waits_only_for_what_is_left_of_the_explanation_budget(monkeypatch):
    """The agent thread and the card share one explanation budget, not 15 s plus another wait."""
    import hermes_agent.gateway.run_turn_runner_clarify_delivery as delivery
    from hermes_agent.gateway.run_turn_runner import TurnRunner

    monkeypatch.setattr(delivery, "SEND_ACK_WINDOW", 0.3)
    hung = concurrent.futures.Future()

    async def send_clarify(**_kwargs):
        return SimpleNamespace(success=True)

    async def send(*_a, **_kw):
        return await asyncio.wrap_future(hung)

    loop = asyncio.get_running_loop()
    adapter = SimpleNamespace(send=send, send_clarify=send_clarify)
    runner = SimpleNamespace(
        _approval_owner=None,
        _ctx=SimpleNamespace(_status_adapter=adapter, _status_chat_id="1", _status_thread_metadata={},
                             _run_still_current=lambda: True,
                             source=SimpleNamespace(platform="telegram", is_bot=False)),
        _schedule=lambda coro, _msg: asyncio.run_coroutine_threadsafe(coro, loop))
    monkeypatch.setattr("hermes_agent.gateway.run_turn_progress.local_viewer_turn", lambda _r: False)
    monkeypatch.setattr("hermes_agent.gateway.bot_peer_notices.automatic_notices_allowed", lambda _s: True)
    sink = delivery.clarify_context_sink(runner)
    started = loop.time()
    await asyncio.to_thread(sink, "Draft: hello")
    await asyncio.wait_for(TurnRunner._send_shared_clarify(runner, None), 5)
    hung.set_result(SimpleNamespace(success=True))

    assert loop.time() - started < 0.5
