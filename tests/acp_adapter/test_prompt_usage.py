"""``PromptResponse.usage`` reports the tokens of the prompt it answers, not the session's running total.

A client that bills each prompt stores every turn's usage and sums them; cumulative counters there
charge a resumed session for all of its earlier turns again (the total grows quadratically).
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import acp
import pytest
from acp.schema import TextContentBlock

from acp_adapter.gateway_server import GatewayACPAgent
from acp_adapter.prompt_usage import counters_of, prompt_usage, turn_delta
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, SessionState


def _turn_result(agent, *, input_tokens, output_tokens, cache_read, cache_write, reasoning=0):
    """What one turn does to the agent's counters (agent/turn_usage.py) and what its result carries
    (agent/turn_finalizer.py: ``result[key] = agent.session_<key>``)."""
    agent.session_input_tokens += input_tokens
    agent.session_output_tokens += output_tokens
    agent.session_cache_read_tokens += cache_read
    agent.session_cache_write_tokens += cache_write
    agent.session_reasoning_tokens += reasoning
    agent.session_prompt_tokens += input_tokens + cache_read + cache_write
    agent.session_completion_tokens += output_tokens
    agent.session_total_tokens += input_tokens + cache_read + cache_write + output_tokens
    keys = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
            "prompt_tokens", "completion_tokens", "total_tokens")
    return {"final_response": "ok", "messages": [], **{k: getattr(agent, f"session_{k}") for k in keys}}


def _fresh_counters(agent):
    for key in ("input", "output", "cache_read", "cache_write", "reasoning", "prompt", "completion", "total"):
        setattr(agent, f"session_{key}_tokens", 0)


@pytest.mark.asyncio
async def test_in_process_prompt_usage_is_the_turns_own_tokens(tmp_path):
    manager = SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))
    server = HermesACPAgent(session_manager=manager)
    sid = (await server.new_session(cwd=str(tmp_path))).session_id
    state = manager.get_session(sid)
    agent = state.agent
    agent.model, agent.provider = "test-model", "openrouter"
    _fresh_counters(agent)
    turns = iter([
        dict(input_tokens=20, output_tokens=300, cache_read=0, cache_write=40_000, reasoning=50),
        dict(input_tokens=5, output_tokens=200, cache_read=40_000, cache_write=1_000),
    ])
    agent.run_conversation = lambda **kwargs: _turn_result(agent, **next(turns))
    server._conn = MagicMock(spec=acp.Client)
    server._conn.session_update = AsyncMock()

    first = await server.prompt(prompt=[TextContentBlock(type="text", text="one")], session_id=sid)
    second = await server.prompt(prompt=[TextContentBlock(type="text", text="two")], session_id=sid)

    assert first.usage.model_dump(by_alias=True, exclude_none=True) == {
        "inputTokens": 20, "outputTokens": 300, "thoughtTokens": 50, "cachedWriteTokens": 40_000,
        "totalTokens": 40_320}
    # The second turn reports only its own tokens, and its cache writes are not dropped.
    assert second.usage.model_dump(by_alias=True, exclude_none=True) == {
        "inputTokens": 5, "outputTokens": 200, "cachedReadTokens": 40_000, "cachedWriteTokens": 1_000,
        "totalTokens": 41_205}
    # Summing what each prompt reported gives the session's real totals.
    assert first.usage.total_tokens + second.usage.total_tokens == agent.session_total_tokens


class _Wire:
    """The gateway client: a submit runs the turn on the owner's counters."""

    def __init__(self, server, sid, stats, turn):
        self.server, self.sid, self.stats, self.turn, self.methods = server, sid, stats, turn, []

    async def rpc(self, method, **params):
        self.methods.append(method)
        assert method == "prompt.submit", method
        for key, value in self.turn.items():
            self.stats[key] += value
        # The owner's turn-end ``session.usage`` frame, then its terminal ``message.complete``.
        self.server._usage[self.sid] = dict(self.stats)
        return {"admission_id": "a1"}


@pytest.mark.asyncio
async def test_gateway_prompt_usage_subtracts_the_counters_the_turn_started_with(monkeypatch):
    server = GatewayACPAgent()
    sid = "s1"
    server._snapshots[sid] = {}
    before = {"input_tokens": 400, "output_tokens": 9_000, "cache_read_tokens": 3_000_000,
              "cache_write_tokens": 200_000, "reasoning_tokens": 0, "total_tokens": 3_209_400}
    wire = _Wire(server, sid, before, {"input_tokens": 7, "output_tokens": 150, "cache_read_tokens": 90_000,
                                       "cache_write_tokens": 2_000, "total_tokens": 92_157})

    async def client():
        return wire

    async def no_slash(held_id, session_id, text):
        return False, text

    async def prepared(session_id, prompt):
        return "key", "input", ("hello", [])

    async def admitted(client, held_id, session_id, receipt):
        server._terminals[receipt["admission_id"]] = {"outcome": "completed"}
        return receipt["admission_id"]

    async def nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "_client", client)
    monkeypatch.setattr(server, "_run_slash", no_slash)
    monkeypatch.setattr(server, "_prepare_submit", prepared)
    monkeypatch.setattr(server, "_admission_from_receipt", admitted)
    monkeypatch.setattr(server, "_after_ack", nothing)

    stats_answered = asyncio.Event()

    async def attach_read():
        # The attach's ``session.stats`` (``_after_attach``) still in flight when the prompt is sent.
        await stats_answered.wait()
        server._usage[sid] = dict(before)

    server._attach_reads[sid] = asyncio.create_task(attach_read())
    prompt = asyncio.create_task(server.prompt([acp.text_block("hello")], sid))
    for _ in range(5):
        await asyncio.sleep(0)  # the prompt reaches the turn-start wait while the read is pending
    assert not prompt.done() and wire.methods == []
    stats_answered.set()
    response = await asyncio.wait_for(prompt, 5)

    assert response.usage.model_dump(by_alias=True, exclude_none=True) == {
        "inputTokens": 7, "outputTokens": 150, "cachedReadTokens": 90_000, "cachedWriteTokens": 2_000,
        "totalTokens": 92_157}
    assert wire.methods == ["prompt.submit"]


def test_counters_restarting_mid_turn_report_what_accumulated_since():
    before = {"input_tokens": 10, "output_tokens": 10, "cache_read_tokens": 500, "cache_write_tokens": 50,
              "reasoning_tokens": 0}
    after = {"input_tokens": 3, "output_tokens": 4, "cache_read_tokens": 0, "cache_write_tokens": 90,
             "reasoning_tokens": 0}
    assert turn_delta(before, after) == after


def test_missing_counters_report_no_usage():
    assert prompt_usage(None, counters_of({"final_response": "ok"})) is None
    assert prompt_usage(None, counters_of(None)) is None
    assert counters_of(SimpleNamespace()) is None
