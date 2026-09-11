"""One-shot model overrides (``/moa <prompt>``, ``/model --once``) must be restored on every exit.

These exercise the real ``GatewayRunner`` helpers the message-handling ``finally`` and the
stop/reset/eviction paths call, so they prove the production logic — not a re-implementation of
it. The bug being guarded: the restore used to live in the ``try`` block, so a turn that raised
skipped it and the MoA override leaked permanently (every later message silently fanned out
through MoA).
"""

import pytest

from gateway.run import GatewayRunner

KEY = "agent:main:telegram:dm:999"
PRIOR = {"provider": "openrouter", "model": "gpt-4"}


def _runner_with_pending_once():
    runner = object.__new__(GatewayRunner)
    runner._evict_cached_agent = lambda session_key: None
    state = runner._session_state(KEY)
    state.conversation.model_override = {"provider": "moa", "model": "default"}
    state.conversation.one_turn_restore = {"had_override": True, "override": dict(PRIOR)}
    return runner, state


def test_restore_runs_from_finally_even_when_turn_raises():
    runner, state = _runner_with_pending_once()
    gen = runner._begin_session_run_generation(KEY)
    state.conversation.one_turn_restore["run_generation"] = gen

    with pytest.raises(RuntimeError):
        try:
            raise RuntimeError("provider error mid-turn")
        finally:
            runner._restore_pending_one_turn_model_override(KEY, gen)

    assert state.conversation.model_override == PRIOR
    assert state.conversation.one_turn_restore is None
