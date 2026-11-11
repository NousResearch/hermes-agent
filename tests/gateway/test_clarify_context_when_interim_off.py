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
    from gateway.run_turn_runner import TurnRunner

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
