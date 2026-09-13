"""Tests for the temporal-claim guard (agent/temporal_guard.py).

The per-turn timestamp feature stamps ``Current time: ...`` into every
request; this guard verifies the model's FINAL response does not contradict
that stamp. Coverage: now-claims, anchored recency arithmetic, the
documented blind spots (unanchored recency, event references, future
claims, code blocks, location-qualified times), and the wiring in
``finalize_turn`` (log-only, never mutates, never raises).
"""

import pytest

from agent.temporal_guard import check_temporal_claims
from agent.turn_finalizer import finalize_turn

STAMP = "Current time: Monday 2026-08-03 07:56 BST"


class _StubBudget:
    used = 1
    max_total = 90
    remaining = 89


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self, stamp=None):
        self.max_iterations = 90
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"
        self.persisted_messages = None
        if stamp is not None:
            self._current_turn_timestamp = stamp

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = [dict(m) for m in messages]

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


# ---------------------------------------------------------------------------
# Unit: check_temporal_claims
# ---------------------------------------------------------------------------


class TestCheckTemporalClaims:
    @pytest.mark.parametrize(
        "text",
        [
            # Correct claims — silent
            "it's currently 07:56",
            "Current time is 07:56 BST",
            "It is 07:56 right now",
            # Event references, not now-claims
            "The reply was posted at 06:02, 23 seconds after I posted it",
            "gateway started 23:46:45 last night",
            "the 14:42:07 log line shows the error",
            "log shows 2026-08-03T05:02:23Z",
            "Gateway PID 660728, started Aug 2 23:46:45, up 6h19m",
            # Future/schedule claims
            "The watcher fires at 06:45",
            # Versions and durations without clocks
            "pytest-asyncio==1.3.0 and python-telegram-bot 22.6",
            "the session ran for 2 hours 15 minutes",
            # Location/timezone-qualified
            "it's 9am in Tokyo",
            "it's 5pm at the office",
            # Unanchored recency — documented blind spot
            "it's about 15 minutes ago since the reply",
            # Code blocks and inline code are not claims
            "```\nit's 07:45 now\n```",
            "the docs say `it's 07:45 now`",
            # Anchored recency that is CORRECT arithmetic
            "posted at 06:02, about 114 minutes ago",
            "posted at 06:02, about 2 hours ago",
            "at 05:02Z, 3h ago",
        ],
    )
    def test_silent_cases(self, text):
        assert check_temporal_claims(text, STAMP) == []

    @pytest.mark.parametrize(
        "text, needle",
        [
            # Wrong now-claims
            ("it's 07:45 now", "now-claim"),
            ("It's 06:45 now", "now-claim"),
            ("the time is 07:00", "now-claim"),
            ("it's 7:45 pm", "now-claim"),
            # Wrong anchored recency arithmetic
            (
                "The reply was posted at 06:02, so about 15 minutes ago",
                "recency",
            ),
            (
                "The reply was posted at 06:02 BST, about 15 minutes ago",
                "recency",
            ),
            ("posted at 06:02, about 45 minutes ago", "recency"),
        ],
    )
    def test_flag_cases(self, text, needle):
        flags = check_temporal_claims(text, STAMP)
        assert flags, f"expected a flag for: {text}"
        assert any(needle in f for f in flags), flags

    def test_empty_text_or_stamp(self):
        assert check_temporal_claims("", STAMP) == []
        assert check_temporal_claims("it's 07:45", "") == []

    def test_malformed_stamp(self):
        assert check_temporal_claims("it's 07:45", "Current time: ???") == []

    def test_tolerance_boundary(self):
        # exactly tolerance_min away: silent; one minute more: flagged
        assert check_temporal_claims("it's 07:51", STAMP) == []
        assert check_temporal_claims("it's 07:50", STAMP) != []

    def test_am_pm_matches_stamp(self):
        # stamp is 07:56 AM; an explicit am claim matches
        assert check_temporal_claims("it's 7:56 am", STAMP) == []

    def test_clock_wrap_distance(self):
        night_stamp = "Current time: Monday 2026-08-03 00:05 BST"
        assert check_temporal_claims("it's 00:02", night_stamp) == []
        assert check_temporal_claims("it's 23:59", night_stamp) != []


# ---------------------------------------------------------------------------
# Integration: finalize_turn wiring
# ---------------------------------------------------------------------------


def _finalize(agent, final_response, *, interrupted=False):
    messages = [
        {"role": "user", "content": "what time is it?"},
        {"role": "assistant", "content": final_response},
    ]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=interrupted,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="what time is it?",
        original_user_message="what time is it?",
        _should_review_memory=False,
        _turn_exit_reason="ok",
    )


class TestFinalizeTurnGuardWiring:
    def test_wrong_now_claim_logged(self, monkeypatch):
        from agent import conversation_loop

        warnings = []
        monkeypatch.setattr(conversation_loop.logger, "warning", lambda *a: warnings.append(a))
        agent = _StubAgent(stamp=STAMP)
        _finalize(agent, "it's 07:45 now, so about 11 minutes have passed.")
        assert any("Temporal-claim check" in str(a) for a in warnings), warnings

    def test_clean_response_silent(self, monkeypatch):
        from agent import conversation_loop

        warnings = []
        monkeypatch.setattr(conversation_loop.logger, "warning", lambda *a: warnings.append(a))
        agent = _StubAgent(stamp=STAMP)
        _finalize(agent, "Current time is 07:56 BST.")
        assert not any("Temporal-claim check" in str(a) for a in warnings)

    def test_no_stamp_silent(self, monkeypatch):
        from agent import conversation_loop

        warnings = []
        monkeypatch.setattr(conversation_loop.logger, "warning", lambda *a: warnings.append(a))
        agent = _StubAgent(stamp=None)
        _finalize(agent, "it's 07:45 now")
        assert not any("Temporal-claim check" in str(a) for a in warnings)

    def test_result_unchanged(self, monkeypatch):
        from agent import conversation_loop

        monkeypatch.setattr(conversation_loop.logger, "warning", lambda *a: None)
        agent = _StubAgent(stamp=STAMP)
        result = _finalize(agent, "it's 07:45 now")
        # Guard must not touch the returned transcript or response text.
        assert result["final_response"] == "it's 07:45 now"
        assert result["messages"][-1]["content"] == "it's 07:45 now"
