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
    hygiene_compaction_recovered,
    hygiene_wait_should_extend,
)
from gateway.run import GatewayRunner
from gateway.session_state import PersistentState, SessionState


def _Runner():
    """A real ``GatewayRunner`` with no ``__init__`` side effects.

    Deliberately NOT a hand-written stub: an earlier version reimplemented
    ``_session_state`` and ``_peek_session_state``, which meant the tests
    exercised the copies rather than the production accessors and could drift
    from them silently (the real ``_peek_session_state`` returns ``None`` on a
    falsy ``_sessions``, and ``_session_state`` goes through ``_sessions_map()``
    self-healing).  ``object.__new__`` is already the idiom elsewhere in this
    file, and the self-healing map means no attribute setup is needed.
    """
    return object.__new__(GatewayRunner)


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

    def test_consecutive_failures_escalate_across_gateway_restart(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session("before-rotation", "telegram", session_key=KEY)
            first_runner = _Runner()
            first_runner._session_db = db
            assert _hygiene_cooldown_for_failure(first_runner, KEY, BASE) == BASE

            db.create_session(
                "after-rotation",
                "telegram",
                session_key=KEY,
                parent_session_id="before-rotation",
            )
            restarted_runner = _Runner()
            restarted_runner._session_db = db
            assert _hygiene_cooldown_for_failure(
                restarted_runner, KEY, BASE
            ) == BASE * 3

            other_chat_runner = _Runner()
            other_chat_runner._session_db = db
            assert _hygiene_cooldown_for_failure(
                other_chat_runner, "agent:main:telegram:private:999", BASE
            ) == BASE
        finally:
            db.close()

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

    def test_reset_returns_restarted_gateway_to_first_rung(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            runner = _Runner()
            runner._session_db = db
            assert _hygiene_cooldown_for_failure(runner, KEY, BASE) == BASE
            _reset_hygiene_failure_streak(runner, KEY)

            restarted_runner = _Runner()
            restarted_runner._session_db = db
            assert _hygiene_cooldown_for_failure(
                restarted_runner, KEY, BASE
            ) == BASE
        finally:
            db.close()

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
        # Read through the production accessor: on a fresh runner `_sessions`
        # does not exist at all until something materialises it, which is a
        # stronger statement than "the key is absent" — the reset did not even
        # create the map.
        assert runner._peek_session_state("never-seen") is None
        assert not runner.__dict__.get("_sessions")


# ---------------------------------------------------------------------------
# The failure reason reaches the state DB
# ---------------------------------------------------------------------------

class TestFailureReasonForwarded:
    """`record_compression_failure_cooldown` writes compression_failure_error
    UNCONDITIONALLY, so omitting the reason clobbers to NULL whatever the
    in-conversation path recorded — and readers then show the user
    "unknown error". Matters more now that a cooldown can last an hour."""

    def _capture(self, *args):
        seen = {}

        class _DB:
            def record_compression_failure_cooldown(self, sid, until, error=None):
                seen.update(sid=sid, until=until, error=error)

        class _GW:
            _session_db = _DB()

        _record_hygiene_cooldown(_GW(), "sid-1", 300.0, *args)
        return seen

    def test_reason_is_forwarded_when_supplied(self):
        seen = self._capture("summary model timed out")
        assert seen["error"] == "summary model timed out"

    def test_absent_reason_is_passed_explicitly_as_none(self):
        """Still forwarded positionally, so the call shape stays uniform."""
        seen = self._capture()
        assert seen["error"] is None

    def test_deadline_is_still_absolute_epoch_seconds(self):
        import time

        seen = self._capture("x")
        assert seen["until"] > time.time() + 200


# ---------------------------------------------------------------------------
# The recovery predicate (extracted from _handle_message_with_agent)
# ---------------------------------------------------------------------------

class TestHygieneCompactionRecovered:
    """Direct unit tests for the recovery decision.

    This replaces three source-reading tests that asserted on
    ``inspect.getsource`` text. AGENTS.md bans reading source in tests outright
    and names this file's module as the case where the right answer is to
    extract the logic — which is what ``hygiene_compaction_recovered`` is. The
    old tests were also actively wrong: one asserted the buggy
    ``_new_tokens < _approx_tokens`` substring was present, so it passed while
    the gate was broken and had to be edited when the gate was fixed.
    """

    BASE = dict(
        aborted=False, rotated=True, in_place=False,
        msg_count=220, new_count=100,
        approx_tokens=50_000, new_tokens=30_000,
    )

    def _call(self, **over):
        return hygiene_compaction_recovered(**{**self.BASE, **over})

    def test_real_rotation_with_reduction_is_recovery(self):
        assert self._call() is True

    def test_abort_is_never_recovery(self):
        assert self._call(aborted=True) is False

    def test_no_rewrite_is_never_recovery_even_when_counts_look_good(self):
        """The degenerate #21301 path: not aborted, but nothing was rewritten.

        Deliberately passes counts that WOULD read as progress, so this binds
        the rotated/in_place guard specifically. Using equal counts here would
        pass vacuously — the progress predicate already rejects those, so the
        guard could be deleted and the test would still pass.
        """
        # Sanity: these counts do read as progress on their own.
        from agent.turn_context import compression_made_progress

        assert compression_made_progress(220, 100, 50_000, 30_000) is True
        # ...but with nothing rewritten it must still not count as recovery.
        assert self._call(rotated=False, in_place=False) is False

    def test_in_place_compaction_counts(self):
        assert self._call(rotated=False, in_place=True) is True

    def test_row_drop_with_flat_tokens_is_recovery(self):
        """Rows dropping is progress even when the summary keeps tokens flat.

        A bare token comparison misses this and would keep a recovered session
        escalating to the cap forever.
        """
        assert self._call(new_count=100, new_tokens=50_000) is True

    def test_row_drop_with_slightly_worse_tokens_is_recovery(self):
        """Same, when the summary text is marginally more verbose."""
        assert self._call(new_count=100, new_tokens=50_100) is True

    def test_size_only_win_is_recovery(self):
        """Equal rows, large token reduction (#39548)."""
        assert self._call(
            msg_count=220, new_count=220,
            approx_tokens=288_000, new_tokens=183_000,
        ) is True

    def test_sub_five_percent_wobble_is_not_recovery(self):
        """Noise must not clear the streak, or escalation is defeated again."""
        assert self._call(
            msg_count=220, new_count=220,
            approx_tokens=50_000, new_tokens=49_900,
        ) is False


class TestHygieneWaitShouldExtend:
    """Host must not keep waiting after the commit fence is already cancelled."""

    def test_extends_while_idle_and_under_ceiling(self):
        assert hygiene_wait_should_extend(
            idle=1.0, timeout=30.0, waited=10.0, ceiling=600.0,
        ) is True

    def test_stops_when_idle_budget_exhausted(self):
        assert hygiene_wait_should_extend(
            idle=30.0, timeout=30.0, waited=10.0, ceiling=600.0,
        ) is False

    def test_stops_at_ceiling(self):
        assert hygiene_wait_should_extend(
            idle=1.0, timeout=30.0, waited=600.0, ceiling=600.0,
        ) is False

    def test_fence_cancel_stops_even_with_fresh_progress(self):
        assert hygiene_wait_should_extend(
            idle=0.0, timeout=30.0, waited=0.1, ceiling=600.0,
            fence_cancelled=True,
        ) is False


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
