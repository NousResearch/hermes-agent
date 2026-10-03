"""A reasoning-only clean stop promotes the turn's chain-of-thought to ``final_response``
(``agent/turn_final_response.py``): the provider ends the turn with ``finish_reason="stop"``, no
tool call and no visible text, and the loop returns the reasoning as the answer. That text is a
diagnostic for the operator — it is not an authored reply — so the gateway must not hand it to the
other participants of a multi-party chat.

Observed live on Hermes 0.21.5 (Signal, ``deepseek-flash`` through an OpenAI-compatible proxy): a
group chat received 2051 characters of English planning text ("…jokes: his oven has to leave for
choir practice…") as the reply. The log line was
``Reasoning-only clean stop (2051 chars) — returning the reasoning as the final response`` and the
delivered length matched the reasoning length byte for byte. The assistant row keeps ``content``
empty, so ``last_reasoning`` carries exactly the delivered text — that identity is what tells a
promotion apart from a real answer.

A private chat keeps the current behaviour: the owner still sees what the model was thinking.
"""

from types import SimpleNamespace

import pytest

from gateway.run_turn import GatewayTurnMixin


class _Runner(GatewayTurnMixin):
    def __init__(self):
        self.async_session_store = SimpleNamespace(clear_resume_pending=self._noop)

    async def _noop(self, *_a, **_k):
        return None

    async def _clear_restart_failure_count(self, *_a, **_k):
        return None


def _promoted_result(reasoning: str) -> dict:
    """What the loop returns for a promoted reasoning-only stop: final_response IS the reasoning."""
    return {"final_response": reasoning, "last_reasoning": reasoning, "messages": [], "api_calls": 22}


def _source(chat_type: str):
    return SimpleNamespace(
        chat_id="room-1", chat_type=chat_type, platform=SimpleNamespace(value="signal"),
    )


async def _shape(runner, agent_result, source):
    return await runner._hmwa_shape_agent_response(
        agent_result, source, history=[], session_entry=SimpleNamespace(session_id="s"), session_key=None,
        _quick_key=None, run_generation=0, _run_start_session_id="s", _platform_name="signal",
        _msg_start_time=0.0,
    )


_LEAKED_REASONING = (
    "The user jokes: his oven has to leave for choir practice. So he has ~7 min. Humor needed, "
    "short and funny - the task: the cake needs an hour in the oven."
)


@pytest.mark.asyncio
async def test_group_chat_never_receives_promoted_reasoning():
    runner = _Runner()
    response, silent, _messages = await _shape(runner, _promoted_result(_LEAKED_REASONING), _source("group"))

    assert silent is False
    assert "choir practice" not in response
    assert response  # something is said instead of nothing


@pytest.mark.asyncio
async def test_channel_and_forum_chats_are_multi_party_too():
    runner = _Runner()
    for chat_type in ("channel", "forum"):
        response, _silent, _messages = await _shape(runner, _promoted_result(_LEAKED_REASONING), _source(chat_type))
        assert "choir practice" not in response


@pytest.mark.asyncio
async def test_private_chat_still_shows_the_promoted_reasoning():
    reasoning = "The answer is 42. Let me double-check the units before answering."
    runner = _Runner()
    response, _silent, _messages = await _shape(runner, _promoted_result(reasoning), _source("dm"))

    assert response == reasoning


@pytest.mark.asyncio
async def test_a_real_answer_is_never_touched():
    runner = _Runner()
    agent_result = {
        "final_response": "Der Kuchen braucht 60 Minuten im Ofen.",
        "last_reasoning": "Plan: Zutaten abwiegen, Teig ruhen lassen, dann backen.",
        "messages": [],
    }
    response, _silent, _messages = await _shape(runner, agent_result, _source("group"))

    assert response == "Der Kuchen braucht 60 Minuten im Ofen."


@pytest.mark.asyncio
async def test_a_group_reply_that_quotes_its_own_reasoning_is_not_withheld():
    """Only the byte-identical promotion is withheld: a genuine answer that happens to embed the
    reasoning is still delivered (no substring matching)."""
    runner = _Runner()
    shared = "Kurzfassung: Fertigteig in die Form, 60 Minuten backen."
    agent_result = {
        "final_response": f"{shared}\n\n(Notiz an mich: {shared})", "last_reasoning": shared, "messages": [],
    }
    response, _silent, _messages = await _shape(runner, agent_result, _source("group"))

    assert response == agent_result["final_response"]
