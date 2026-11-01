"""Sustained summary-provider overload: the abort budget, its escalation and its resets."""

from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor


class StubProviderError(Exception):
    def __init__(self, message, *, status_code=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class TestSustainedOverloadEscalation:
    """#123167: a sustained summary-provider overload must not guarantee a session wipe.

    One overload aborts so a retry can still win (#115906). But when every attempt keeps
    aborting, the transcript only grows until the session exits compression_exhausted and
    the gateway auto-resets — total loss, deferred. After 3 consecutive overload aborts the
    overload stops being terminal and compress() commits the deterministic fallback instead
    (the same bounded degrade the repeated-stall ladder takes, #112420).
    """

    def _err(self):
        return StubProviderError(
            "Our servers are currently overloaded. Please try again later.",
            status_code=503,
        )

    def _compressor(self, **kwargs):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=2,
                **kwargs,
            )
        c.summary_model = "test/auxiliary"
        return c

    @pytest.mark.parametrize("access_error", [False, True], ids=["overload", "403_overloaded"])
    def test_first_overloads_still_abort_then_third_commits_fallback(self, access_error):
        c = self._compressor(abort_on_summary_failure=False)
        # A stale terminal flag from one earlier network failure (only a success clears it) must
        # not pin the long-lived compressor in abort mode: the latest failure class decides.
        c._last_summary_network_failure = True
        msgs = self._msgs(12)
        err = StubProviderError("Error code: 403 - provider overloaded", status_code=403) if access_error else self._err()
        with patch("agent.context_compressor.call_llm", side_effect=err):
            first = c.compress(msgs, current_tokens=999999, force=True)
            second = c.compress(msgs, current_tokens=999999, force=True)
            # First two: preserve the transcript unchanged (#115906 semantics).
            assert first == msgs and second == msgs
            assert c._last_compress_aborted is True
            third = c.compress(msgs, current_tokens=999999, force=True)

        if access_error:
            # An auth/quota error that also says "overloaded" never escalates (#29559).
            assert third == msgs
            assert c._last_compress_aborted is True
            assert c._last_compression_telemetry["failure_class"] == "summary_auth_failure"
            # ...and never bumps the overload budget, so the next real 503 keeps its grace (#115906).
            assert c._consecutive_overload_aborts == 0
            return
        # Third: sustained overload escalates — bounded fallback beats a deferred total wipe.
        assert third != msgs
        assert c._last_compress_aborted is False
        assert c._last_summary_fallback_used is True
        assert c._last_summary_dropped_count > 0
        assert c._last_compression_telemetry["failure_class"] == "summary_overload_degraded"

    def test_a_healthy_summary_restarts_the_overload_budget(self):
        """A summary that lands resets the strikes even when its compaction is later refused, so one
        isolated overload afterwards aborts instead of escalating to the destructive fallback."""
        c = self._compressor(abort_on_summary_failure=False)
        msgs = self._msgs(12)
        with patch("agent.context_compressor.call_llm", side_effect=self._err()):
            for _ in range(2):
                assert c.compress(msgs, current_tokens=999999, force=True) == msgs
        assert c._consecutive_overload_aborts == 2
        ok = MagicMock()
        ok.choices[0].message.content = "Healthy summary."
        with patch("agent.context_compressor.call_llm", return_value=ok):
            assert c._generate_summary(msgs[1:6], bypass_cooldown=True)
        assert c._consecutive_overload_aborts == 0

    def test_overload_budget_survives_a_fresh_compressor_bound_to_the_same_session(self, tmp_path):
        """Review P1: the N=3 budget must be session-scoped, not object-local.

        The gateway binds a fresh compressor to the existing session on every turn /
        cache eviction, and restart/resume constructs one too. Each of those used to
        restart the budget at zero, so a sustained outage never escalated (#123167;
        same contract as the durable fallback streak, #100185).
        """
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "OVERLOAD_FRESH_BIND"
        db.create_session(session_id, source="telegram")
        msgs = self._msgs(12)

        first = self._compressor(abort_on_summary_failure=False)
        first.bind_session_state(db, session_id)
        with patch("agent.context_compressor.call_llm", side_effect=self._err()):
            for _ in range(2):  # exactly two aborts — one below the escalation threshold
                assert first.compress(msgs, current_tokens=999999, force=True) == msgs
        assert first._consecutive_overload_aborts == 2
        assert db.get_compression_overload_streak(session_id) == 2

        # Fresh agent, same session (eviction / restart / API-server request construction).
        second = self._compressor(abort_on_summary_failure=False)
        second.bind_session_state(db, session_id)
        assert second._consecutive_overload_aborts == 2  # inherited, not restarted
        with patch("agent.context_compressor.call_llm", side_effect=self._err()):
            third = second.compress(msgs, current_tokens=999999, force=True)
        # Third consecutive abort IN THE SESSION escalates even though this object saw one.
        # A fresh object's first overload burns the aux→main one-shot retry, which labels the
        # attempt aux_model_fallback first; the escalated commit must still say what happened.
        assert third != msgs
        assert second._last_summary_fallback_used is True
        assert second._last_summary_overload_degraded is True
        assert second._last_compression_telemetry["failure_class"] == "summary_overload_degraded"
        # The committed degraded fallback settles the budget so recovery gets a fresh run
        # (the boundary caller records the completed compaction, as compress_context does).
        second.record_completed_compaction(used_fallback=True)
        assert db.get_compression_overload_streak(session_id) == 0

    def _msgs(self, n=10):
        return [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(n)
        ]
