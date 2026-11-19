"""One-shot request overrides must survive a retry of the same call (#120030).

The agent sets two per-call overrides:

* ``_ephemeral_reasoning_off`` — set after a thinking-only ``length`` truncation, so the
  continuation answers without re-burning the thinking budget;
* ``_ephemeral_max_output_tokens`` — set by the continuation boost, the truncated-tool-call
  boost, and the output-cap clamp after a context-length 400.

Both used to be *consumed on read* inside the request builders, while
``_run_api_retry_loop`` rebuilds the request on every attempt of one call. So the override
applied to exactly one *attempt*: after a single transient 429 / 5xx / stream drop the
retry went out at full effort and burned one of the continuation attempts the override
exists to protect, and a clamped ``max_tokens`` re-hit the same context-length 400 and
clamped a second time.

The contract these tests pin: a retry of the same logical call carries the same overrides,
and the overrides never leak into the next turn (they are cleared at the top of the turn).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import (
    _consume_ephemeral_max_output,
    _consume_ephemeral_reasoning_off,
)


def _agent(**overrides) -> SimpleNamespace:
    base = {"_ephemeral_reasoning_off": False, "_ephemeral_max_output_tokens": None}
    base.update(overrides)
    return SimpleNamespace(**base)


class TestOverridesSurviveARetry:
    def test_reasoning_off_is_still_set_on_the_second_build(self):
        agent = _agent(_ephemeral_reasoning_off=True)
        assert _consume_ephemeral_reasoning_off(agent) is True
        # The retry rebuilds the request; the override must still be in force.
        assert _consume_ephemeral_reasoning_off(agent) is True

    def test_output_cap_is_still_set_on_the_second_build(self):
        agent = _agent(_ephemeral_max_output_tokens=49936)
        assert _consume_ephemeral_max_output(agent) == 49936
        # Without this the retry resends max_tokens=100000 and re-hits the same 400.
        assert _consume_ephemeral_max_output(agent) == 49936

    def test_reading_does_not_mutate_the_agent(self):
        """A read must be side-effect free; clearing is the turn boundary's job."""
        agent = _agent(_ephemeral_reasoning_off=True, _ephemeral_max_output_tokens=1234)
        _consume_ephemeral_reasoning_off(agent)
        _consume_ephemeral_max_output(agent)
        assert agent._ephemeral_reasoning_off is True
        assert agent._ephemeral_max_output_tokens == 1234

    def test_unset_overrides_stay_unset(self):
        agent = _agent()
        assert _consume_ephemeral_reasoning_off(agent) is False
        assert _consume_ephemeral_max_output(agent) is None


class TestTurnBoundaryClearsThem:
    def test_turn_start_resets_both_overrides(self):
        """A stale override must not survive into the next turn."""
        from agent.conversation_loop import _reset_ephemeral_overrides

        agent = _agent(_ephemeral_reasoning_off=True, _ephemeral_max_output_tokens=49936)
        _reset_ephemeral_overrides(agent)
        assert agent._ephemeral_reasoning_off is False
        assert agent._ephemeral_max_output_tokens is None

    def test_reset_is_idempotent(self):
        from agent.conversation_loop import _reset_ephemeral_overrides

        agent = _agent()
        _reset_ephemeral_overrides(agent)
        _reset_ephemeral_overrides(agent)
        assert agent._ephemeral_reasoning_off is False
        assert agent._ephemeral_max_output_tokens is None
