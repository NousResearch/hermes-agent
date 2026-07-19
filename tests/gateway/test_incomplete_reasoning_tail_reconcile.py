"""Regression: persisted-transcript lag must not resurrect a hidden-reasoning-
only incomplete Codex tail and loop provider continuation forever.

Verified incident (session 20260719_133329_0f161a77): the gateway logged
"Persisted transcript lagged live cached history ... disk=3, memory=4"
immediately before "Codex response remained incomplete after 3 continuation
attempts". Root cause: a retry-exhausted Codex turn is deliberately kept OUT of
the persisted transcript (only the user message is written), but the cached
agent keeps the poisoned reasoning-only assistant tail in its live
_session_messages. On the next turn disk<memory triggers the FTS write-
corruption guard, which resurrected the poisoned tail and re-seeded the
incomplete loop.

These tests reproduce the reconciliation the guard performs in
gateway/run.py (``_select_cached_agent_history`` then
``_strip_incomplete_reasoning_tail`` then ``_strip_stale_dangerous_confirmations``)
and assert the poisoned tail can no longer survive into provider continuation,
while genuine FTS-corruption recovery is preserved.
"""

import time

from agent.replay_cleanup import (
    strip_incomplete_reasoning_tail as _strip_incomplete_reasoning_tail,
    strip_stale_dangerous_confirmations as _strip_stale_dangerous_confirmations,
)
from gateway.run import _select_cached_agent_history


def _reconcile(persisted, live):
    """Mirror the gateway guard's live-vs-persisted reconciliation exactly."""
    selected = _select_cached_agent_history(persisted, live)
    if selected is persisted:
        return persisted
    selected = _strip_incomplete_reasoning_tail(selected)
    return _strip_stale_dangerous_confirmations(selected, now=time.time())


def _poisoned_tail():
    return {
        "role": "assistant",
        "content": "",
        "reasoning": "thinking but never answering",
        "finish_reason": "incomplete",
    }


def test_lagged_disk_does_not_resurrect_incomplete_tail():
    # Disk (transcript) intentionally ends at the user turn; the cached agent's
    # live memory still holds the poisoned reasoning-only tail (disk=3,memory=4).
    persisted = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    live = persisted + [_poisoned_tail()]

    out = _reconcile(persisted, live)

    # The poisoned tail must be gone: replay ends on the real user turn, so the
    # provider does not resume from a hidden-reasoning-only incomplete state.
    assert out[-1] == {"role": "user", "content": "q2"}
    assert not any(
        m.get("role") == "assistant" and m.get("finish_reason") == "incomplete"
        for m in out
    )


def test_genuine_fts_corruption_recovery_is_preserved():
    # True FTS write corruption: disk lost real, COMPLETED turns that live
    # memory still holds. Those must survive reconciliation untouched.
    persisted = [{"role": "user", "content": "only one"}]
    live = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "real answer 1", "finish_reason": "stop"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "real answer 2", "finish_reason": "stop"},
    ]

    out = _reconcile(persisted, live)

    assert out == live


def test_fts_corruption_recovery_still_strips_only_poisoned_tail():
    # Both conditions at once: memory recovered real content AND ends with a
    # poisoned tail. Keep the recovered content, drop only the tail.
    persisted = [{"role": "user", "content": "only one"}]
    live = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "real answer 1", "finish_reason": "stop"},
        {"role": "user", "content": "q2"},
        _poisoned_tail(),
    ]

    out = _reconcile(persisted, live)

    assert out == live[:3]


def test_no_lag_leaves_persisted_untouched():
    # Disk is current (>= memory): the guard is a no-op and never rewrites the
    # already-cleaned persisted history.
    persisted = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "finish_reason": "stop"},
    ]
    live = [{"role": "user", "content": "q1"}]

    assert _reconcile(persisted, live) is persisted
