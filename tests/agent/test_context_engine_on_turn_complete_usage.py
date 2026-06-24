"""Integration test: ``finalize_turn`` forwards the turn's real usage to the
``ContextEngine.on_turn_complete()`` observation hook.

The hook is the engine's post-turn observation point, so it must receive the
completed turn's canonical token usage (prompt/completion/total + the canonical
``input_tokens`` / ``output_tokens`` / ``cache_read_tokens`` /
``cache_write_tokens`` / ``reasoning_tokens`` buckets) when the host has it —
not a hardcoded ``None`` — so the engine can weigh how large/expensive the
selected context was before the next ``select_context()``.

The conversation loop stashes the most recent provider response's usage on the
agent as ``_last_turn_usage`` (the same dict shape fed to
``update_from_response``); ``finalize_turn`` forwards it. On turns that never
reach a provider response (early failure / interrupt) it stays ``None`` and the
hook receives ``None``. These tests pin both ends of that contract through the
real ``finalize_turn`` call site (the path that previously passed
``usage=None``).
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent.context_engine import ContextEngine

# Reuse the minimal agent harness that exercises the real finalize_turn path.
from tests.agent.test_turn_finalizer_cleanup_guard import _StubAgent, _run


class _CapturingEngine(ContextEngine):
    """Engine that records what on_turn_complete() receives."""

    last_prompt_tokens = 0

    def __init__(self) -> None:
        self.captured: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "capturing"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        pass

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return False

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        return messages

    def on_turn_complete(self, messages, usage=None, **kwargs):
        self.captured["seen"] = True
        self.captured["usage"] = usage
        self.captured["kwargs"] = kwargs


CANONICAL_USAGE = {
    "prompt_tokens": 1200,
    "completion_tokens": 80,
    "total_tokens": 1280,
    "input_tokens": 1200,
    "output_tokens": 80,
    "cache_read_tokens": 1024,
    "cache_write_tokens": 0,
    "reasoning_tokens": 16,
}


def _agent_with_engine() -> _StubAgent:
    agent = _StubAgent(raise_in=())
    agent.context_compressor = _CapturingEngine()
    return agent


def test_finalize_turn_forwards_canonical_usage_when_available():
    """A completed turn forwards the stashed canonical usage dict intact."""
    agent = _agent_with_engine()
    agent._last_turn_usage = dict(CANONICAL_USAGE)

    _run(agent, final_response="done")

    captured = agent.context_compressor.captured
    assert captured.get("seen") is True
    # The full canonical bucket set is forwarded unchanged — the engine relies
    # on cache_read/write + reasoning, not just the legacy aggregate keys.
    assert captured["usage"] == CANONICAL_USAGE
    # Turn metadata still rides alongside usage.
    assert captured["kwargs"]["turn_id"] == "turn-1"


def test_finalize_turn_forwards_none_when_no_response_usage():
    """An early-failure/interrupt turn (no stashed usage) forwards None."""
    agent = _agent_with_engine()
    # _last_turn_usage left unset, mirroring a turn that never reached a
    # provider response.
    if hasattr(agent, "_last_turn_usage"):
        delattr(agent, "_last_turn_usage")

    _run(agent, final_response="done")

    captured = agent.context_compressor.captured
    assert captured.get("seen") is True
    assert captured["usage"] is None
