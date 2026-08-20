"""Uncompressed context overflow guardrail (#89297).

When compression is explicitly disabled (``compression.enabled: false``),
sessions can grow past the model context window with nothing to shrink them.
The conversation loop's pre-API site warns (deduped, actionable); the
turn-context preflight re-arms the dedup once the session is back under the
window so a later re-overflow warns again.

The fake binds the PRODUCTION ``_warn_uncompressed_context_overflow`` /
``_clear_context_overflow_warn`` methods so their dedup logic is actually
under test (not a reimplementation).
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

from agent.turn_context import TurnContext, build_turn_context  # noqa: F401
from run_agent import AIAgent
from tests.agent.test_turn_context import _FakeAgent, _build


class _FakeUncompressedAgent(_FakeAgent):
    """Agent stub with compression disabled, bound to the REAL warn methods."""

    # Production methods under test — bound from AIAgent so the dedup key
    # handling and message text cannot silently drift from what ships.
    _warn_uncompressed_context_overflow = (
        AIAgent._warn_uncompressed_context_overflow
    )
    _clear_context_overflow_warn = AIAgent._clear_context_overflow_warn

    def __init__(self, model="deepseek-v4-flash", context_length=10_000):
        super().__init__()
        self.model = model
        self.provider = "deepseek"
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(
            protect_first_n=2,
            protect_last_n=2,
            context_length=context_length,
            threshold_tokens=int(context_length * 0.75),
            last_prompt_tokens=-1,
        )


def _oversized_history(n_turns: int = 10) -> list:
    large_turn = "Large context content " * 500  # ~2,500 tokens each
    history = []
    for i in range(n_turns):
        history.append({"role": "user", "content": f"Turn {i}: {large_turn}"})
        history.append({"role": "assistant", "content": f"Reply {i}: {large_turn}"})
    return history


def test_production_warn_emits_once_and_dedups():
    """The real method warns once, then dedups identical overflows."""
    agent = _FakeUncompressedAgent(context_length=10_000)
    agent._emit_warning = MagicMock()

    agent._warn_uncompressed_context_overflow(15_000, 10_000)
    agent._warn_uncompressed_context_overflow(16_000, 10_000)

    agent._emit_warning.assert_called_once()
    msg = agent._emit_warning.call_args[0][0]
    assert "exceeds the model context window" in msg
    assert "compression.enabled: false" in msg
    assert "10,000 tokens" in msg


def test_clear_rearms_the_warning():
    """After _clear_context_overflow_warn (session back under the window),
    a later re-overflow warns again."""
    agent = _FakeUncompressedAgent(context_length=10_000)
    agent._emit_warning = MagicMock()

    agent._warn_uncompressed_context_overflow(15_000, 10_000)
    agent._clear_context_overflow_warn()
    agent._warn_uncompressed_context_overflow(15_500, 10_000)

    assert agent._emit_warning.call_count == 2


def test_uncompressed_session_within_limits_emits_no_warning():
    agent = _FakeUncompressedAgent(context_length=128_000)
    agent._emit_warning = MagicMock()
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    tctx = _build(agent, conversation_history=history)
    assert isinstance(tctx, TurnContext)
    agent._emit_warning.assert_not_called()


def test_preflight_rearm_clears_dedup_when_back_under_window():
    """The turn-context preflight re-arms the dedup once the session fits
    again (e.g. after a manual /compress), so growth past the window later
    warns a second time."""
    agent = _FakeUncompressedAgent(context_length=128_000)
    agent._emit_warning = MagicMock()
    # Simulate a previously fired warning.
    agent._last_ctx_overflow_warn = ("uncompressed_ctx_overflow", 128_000)

    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    tctx = _build(agent, conversation_history=history)
    assert isinstance(tctx, TurnContext)

    # Preflight cleared the dedup — the next overflow warns again.
    assert agent._last_ctx_overflow_warn is None
    agent._warn_uncompressed_context_overflow(200_000, 128_000)
    agent._emit_warning.assert_called_once()


def test_preflight_does_not_rearm_while_still_over_window():
    """While the session is still over the window, the dedup must survive
    the preflight (no per-turn warn spam)."""
    agent = _FakeUncompressedAgent(context_length=10_000)
    agent._emit_warning = MagicMock()
    agent._last_ctx_overflow_warn = ("uncompressed_ctx_overflow", 10_000)

    tctx = _build(agent, conversation_history=_oversized_history())
    assert isinstance(tctx, TurnContext)

    assert agent._last_ctx_overflow_warn == ("uncompressed_ctx_overflow", 10_000)
    agent._emit_warning.assert_not_called()


def test_multimodal_content_forces_real_estimate_in_rearm_gate():
    """List (multimodal) content defeats a char count; the pre-check must
    treat it as over-gate so the real estimator decides. A tiny multimodal
    session is still under the window, so the dedup is re-armed."""
    agent = _FakeUncompressedAgent(context_length=128_000)
    agent._emit_warning = MagicMock()
    agent._last_ctx_overflow_warn = ("uncompressed_ctx_overflow", 128_000)

    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
        {"role": "assistant", "content": "looking"},
    ]
    tctx = _build(agent, conversation_history=history)
    assert isinstance(tctx, TurnContext)
    assert agent._last_ctx_overflow_warn is None


def test_none_content_tool_call_rows_do_not_defeat_cheap_gate():
    """Assistant tool-call rows routinely carry content=None; they must
    count as zero chars (NOT force the estimator) so the cheap gate keeps
    its value in ordinary tool-using sessions. Regression for the salvage
    follow-up's first draft, where `None` hit the over-gate branch."""
    from unittest.mock import patch as _patch

    agent = _FakeUncompressedAgent(context_length=128_000)
    agent._emit_warning = MagicMock()
    agent._last_ctx_overflow_warn = ("uncompressed_ctx_overflow", 128_000)

    history = [
        {"role": "user", "content": "run the tool"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "small result"},
        {"role": "assistant", "content": "done"},
    ]
    with _patch(
        "agent.turn_context.estimate_request_tokens_rough"
    ) as mock_est:
        tctx = _build(agent, conversation_history=history)

    assert isinstance(tctx, TurnContext)
    # Cheap gate decided (tiny session, under window): estimator never ran,
    # and the dedup was still re-armed via the raw-chars branch.
    mock_est.assert_not_called()
    assert agent._last_ctx_overflow_warn is None
