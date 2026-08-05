"""Session-hygiene compression must escalate its cooldown for repeat failures.

Issue #79624: a gateway session whose summary model always times out retried
compaction on a flat ``hygiene_failure_cooldown_seconds`` interval forever.

The in-agent compressor already escalates repeat timeouts 60 -> 300 -> 900s via
``ContextCompressor.record_timeout_failure``, but that ladder reads the
in-memory ``_consecutive_timeout_failures`` counter, and:

  * session hygiene constructs a FRESH ``AIAgent`` for every run
    (``gateway/run.py`` ~16820), and
  * ``ContextCompressor.bind_session_state`` zeroes that counter.

so the in-agent ladder is *structurally unreachable* from the gateway — the
streak is always 0 there. These tests pin the streak to ``PersistentState``
(which outlives the per-run agent) and assert the ladder actually climbs.
"""

from __future__ import annotations

import pytest

from gateway.run import (
    _HYGIENE_COOLDOWN_LADDER_MULTIPLIERS,
    _hygiene_cooldown_for_failure,
    _record_hygiene_cooldown,
    _reset_hygiene_failure_streak,
)
from gateway.session_state import PersistentState, SessionState


class _Runner:
    """Minimal gateway stand-in exposing just ``_session_state``."""

    def __init__(self):
        self._sessions = {}

    def _session_state(self, session_key):
        state = self._sessions.get(session_key)
        if state is None:
            state = SessionState()
            self._sessions[session_key] = state
        return state

    def _peek_session_state(self, session_key):
        return self._sessions.get(session_key)


BASE = 300.0
KEY = "agent:main:telegram:private:123"


# ---------------------------------------------------------------------------
# The state field
# ---------------------------------------------------------------------------

def test_persistent_state_tracks_hygiene_failure_streak():
    """The streak must live on PersistentState, not the per-run agent."""
    assert PersistentState().hygiene_failure_streak == 0


def test_streak_survives_turn_and_conversation_resets():
    """PersistentState is not cleared wholesale by turn/boundary resets, which is
    exactly why the streak lives there rather than on the hygiene agent."""
    runner = _Runner()
    _hygiene_cooldown_for_failure(runner, KEY, BASE)
    state = runner._session_state(KEY)
    # Simulate what a turn/boundary reset does: replace the turn + conversation
    # scopes, leaving `persistent` alone.
    state.turn = type(state.turn)()
    state.conversation = type(state.conversation)()
    assert state.persistent.hygiene_failure_streak == 1


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

class TestCooldownLadder:
    def test_first_failure_uses_the_configured_base(self):
        """Operators who tuned hygiene_failure_cooldown_seconds keep rung 1."""
        runner = _Runner()
        assert _hygiene_cooldown_for_failure(runner, KEY, BASE) == BASE

    def test_consecutive_failures_escalate(self):
        runner = _Runner()
        seen = [
            _hygiene_cooldown_for_failure(runner, KEY, BASE) for _ in range(3)
        ]
        assert seen == [BASE * m for m in _HYGIENE_COOLDOWN_LADDER_MULTIPLIERS]
        assert seen == [300.0, 900.0, 2700.0]

    def test_ladder_saturates_at_the_top_rung(self):
        """A permanently un-compactable session must not grow without bound."""
        runner = _Runner()
        for _ in range(3):
            _hygiene_cooldown_for_failure(runner, KEY, BASE)
        top = BASE * _HYGIENE_COOLDOWN_LADDER_MULTIPLIERS[-1]
        for _ in range(10):
            assert _hygiene_cooldown_for_failure(runner, KEY, BASE) == top

    def test_streak_is_monotonic_across_calls(self):
        runner = _Runner()
        for expected in (1, 2, 3, 4):
            _hygiene_cooldown_for_failure(runner, KEY, BASE)
            assert (
                runner._session_state(KEY).persistent.hygiene_failure_streak
                == expected
            )

    def test_reset_returns_to_the_first_rung(self):
        """A session that recovers must start over, not stay pinned at the top."""
        runner = _Runner()
        for _ in range(3):
            _hygiene_cooldown_for_failure(runner, KEY, BASE)
        _reset_hygiene_failure_streak(runner, KEY)
        assert runner._session_state(KEY).persistent.hygiene_failure_streak == 0
        assert _hygiene_cooldown_for_failure(runner, KEY, BASE) == BASE

    def test_streaks_are_per_session(self):
        """One wedged session must not penalize every other chat."""
        runner = _Runner()
        other = "agent:main:telegram:private:999"
        for _ in range(3):
            _hygiene_cooldown_for_failure(runner, KEY, BASE)
        assert _hygiene_cooldown_for_failure(runner, other, BASE) == BASE

    def test_respects_a_custom_base(self):
        runner = _Runner()
        assert _hygiene_cooldown_for_failure(runner, KEY, 30.0) == 30.0
        assert _hygiene_cooldown_for_failure(runner, KEY, 30.0) == 90.0

    def test_absolute_cap_bounds_a_large_operator_base(self):
        """The multiplier ladder alone would reach 9h at base=3600, which is
        indistinguishable from 'compaction silently switched off'."""
        from gateway.run import _HYGIENE_COOLDOWN_MAX_SECONDS

        runner = _Runner()
        seen = [
            _hygiene_cooldown_for_failure(runner, KEY, 3600.0) for _ in range(4)
        ]
        assert max(seen) == _HYGIENE_COOLDOWN_MAX_SECONDS
        assert all(v <= _HYGIENE_COOLDOWN_MAX_SECONDS for v in seen)

    def test_cap_does_not_shrink_the_configured_base(self):
        """A base already above the cap must still be honoured on rung 1 —
        clamping must never hand back a SHORTER cooldown than configured."""
        from gateway.run import _HYGIENE_COOLDOWN_MAX_SECONDS

        runner = _Runner()
        big = _HYGIENE_COOLDOWN_MAX_SECONDS * 2
        assert _hygiene_cooldown_for_failure(runner, KEY, big) == pytest.approx(
            _HYGIENE_COOLDOWN_MAX_SECONDS
        )

    def test_zero_base_stays_zero(self):
        """A 0 base is 'cool down for no time'; escalation must not invent one."""
        runner = _Runner()
        assert _hygiene_cooldown_for_failure(runner, KEY, 0.0) == 0.0
        assert _hygiene_cooldown_for_failure(runner, KEY, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Degraded runners (the gateway test-double pitfall)
# ---------------------------------------------------------------------------

class TestDegradedRunners:
    def test_bare_runner_without_sessions_map_still_cools_down(self):
        """Many gateway tests build runners via object.__new__ with no _sessions.
        ``_sessions_map()`` self-heals, so this exercises the happy path on a
        bare runner rather than the except branch — pinned because the
        object.__new__ pattern is pervasive in gateway tests and must not raise.
        """
        from gateway.run import GatewayRunner

        bare = object.__new__(GatewayRunner)
        assert _hygiene_cooldown_for_failure(bare, KEY, BASE) == BASE

    def test_reset_on_bare_runner_is_a_noop(self):
        from gateway.run import GatewayRunner

        bare = object.__new__(GatewayRunner)
        _reset_hygiene_failure_streak(bare, KEY)  # must not raise

    def test_runner_whose_session_state_raises_still_cools_down(self):
        """The real degraded case: a stand-in whose _session_state blows up.

        A missing streak must degrade to 'no escalation'. It must NEVER let the
        exception escape, because the caller uses the return value to record the
        cooldown — losing it would mean no cooldown at all and a hot retry loop.
        """
        class _Exploding:
            def _session_state(self, session_key):
                raise RuntimeError("no sessions map")

        gw = _Exploding()
        assert _hygiene_cooldown_for_failure(gw, KEY, BASE) == BASE
        _reset_hygiene_failure_streak(gw, KEY)  # must not raise

    def test_absent_session_reset_is_a_noop(self):
        """Reset peeks rather than get-or-creates: a session with no state entry
        must not materialise one just to write a 0 that is already 0
        (_sessions entries are never evicted)."""
        runner = _Runner()
        _reset_hygiene_failure_streak(runner, "never-seen")
        assert "never-seen" not in runner._sessions


# ---------------------------------------------------------------------------
# The reset gate in _handle_message_with_agent
# ---------------------------------------------------------------------------

class TestResetGate:
    """The reset must require ACTUAL context reduction, not merely 'not aborted'.

    gateway/run.py has a degenerate branch ("did not rotate or compact in
    place ... no session_db on the hygiene agent", #21301) that sets
    ``_new_tokens = _approx_tokens`` and is NOT aborted. Gating the reset on
    'not aborted' alone cleared the streak on every such run, so a session
    wedged there could never escalate — silently defeating the whole fix.
    """

    @staticmethod
    def _gate_source():
        """The `if not _hyg_aborted:` / `if _hyg_aborted:` pair and their bodies.

        Sliced by AST node span rather than a fixed character count: a fixed
        slice silently truncates when the block grows and the assertions then
        pass or fail for the wrong reason.
        """
        import ast
        import inspect
        import textwrap

        import gateway.run as run_mod

        src = textwrap.dedent(
            inspect.getsource(run_mod.GatewayRunner._handle_message_with_agent)
        )
        tree = ast.parse(src)
        lines = src.splitlines()
        spans = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_hyg_aborted"
                for t in node.targets
            ):
                spans.append((node.lineno, node.end_lineno))
            if isinstance(node, ast.If) and "_hyg_aborted" in ast.unparse(node.test):
                spans.append((node.lineno, node.end_lineno))
        assert spans, "could not locate the _hyg_aborted gate"
        return "\n".join(lines[min(s[0] for s in spans) - 1:max(s[1] for s in spans)])

    def test_reset_is_gated_on_the_canonical_progress_predicate(self):
        gate = self._gate_source()
        assert "_reset_hygiene_failure_streak" in gate
        # Must reuse the shared predicate, not a hand-rolled comparison. A bare
        # `_new_tokens < _approx_tokens` gets three cases wrong: it misses a
        # row-count win when the summary keeps tokens flat, misses one where the
        # summary is slightly more verbose, and counts a sub-5% wobble as
        # recovery (#39548).
        assert "_compression_made_progress(" in gate, (
            "reset must use the canonical progress predicate"
        )
        assert "_new_tokens < _approx_tokens" not in gate, (
            "hand-rolled token comparison disagrees with the canonical predicate"
        )

    def test_progress_predicate_semantics_the_gate_depends_on(self):
        """Pin the behaviour the gate is now relying on.

        If these ever change, the hygiene recovery gate's meaning changes with
        them — so bind them here rather than assuming.
        """
        from agent.turn_context import compression_made_progress as prog

        # Rows dropped is progress even when the token estimate stays flat
        # (or rises slightly because the summary text is verbose).
        assert prog(220, 100, 50_000, 50_000) is True
        assert prog(220, 100, 50_000, 50_100) is True
        # Size-only win with equal row counts is progress (#39548).
        assert prog(220, 220, 288_000, 183_000) is True
        # A sub-5% wobble is noise, not recovery.
        assert prog(220, 220, 50_000, 49_900) is False
        # The degenerate no-rotate branch: nothing moved (the #79624 wedge).
        assert prog(220, 220, 50_000, 50_000) is False

    def test_abort_probe_is_computed_once(self):
        """Mutual exclusion between reset and the failure record must be
        explicit. Two separate getattr probes could disagree if a future edit
        inserts an await between them."""
        gate = self._gate_source()
        assert gate.count("_last_compress_aborted") == 1, (
            "compute the abort verdict once into _hyg_aborted and branch on it"
        )
        assert "if not _hyg_aborted:" in gate
        assert "if _hyg_aborted:" in gate


# ---------------------------------------------------------------------------
# Integration with the persist helper
# ---------------------------------------------------------------------------

class TestRecordedCooldownEscalates:
    """The escalated value must be what actually lands in the state DB."""

    class _DB:
        def __init__(self):
            self.calls = []

        def record_compression_failure_cooldown(self, sid, until, error=None):
            self.calls.append((sid, until))

    class _GW:
        def __init__(self, db):
            self._session_db = db
            self._sessions = {}

        def _session_state(self, session_key):
            state = self._sessions.get(session_key)
            if state is None:
                state = SessionState()
                self._sessions[session_key] = state
            return state

    def test_persisted_deadlines_grow(self, monkeypatch):
        import time as real_time

        db = self._DB()
        gw = self._GW(db)
        monkeypatch.setattr(
            "gateway.run.logger", __import__("logging").getLogger("test")
        )

        now = real_time.time()
        for _ in range(3):
            _record_hygiene_cooldown(
                gw, "sess-1", _hygiene_cooldown_for_failure(gw, KEY, BASE)
            )

        assert len(db.calls) == 3
        waits = [until - now for _, until in db.calls]
        # Strictly increasing, and each close to its ladder rung.
        assert waits[0] < waits[1] < waits[2]
        for wait, mult in zip(waits, _HYGIENE_COOLDOWN_LADDER_MULTIPLIERS):
            assert wait == pytest.approx(BASE * mult, abs=5.0)
