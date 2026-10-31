"""Tests for the turn-clock plugin: per-turn time + elapsed injection.

Covers the ``pre_llm_call`` hook contract (returns ``{"context": ...}``
merged by ``agent/turn_context.py``), the elapsed-time formatting, and the
elapsed sources: conversation-history timestamps first, in-process previous-
turn record as fallback.
"""

import re

from plugins.turn_clock import _format_elapsed, _pre_llm_call, register


# ---------------------------------------------------------------------------
# _format_elapsed
# ---------------------------------------------------------------------------


class TestFormatElapsed:
    def test_under_a_minute_is_just_now(self):
        assert _format_elapsed(30) == "just now"
        assert _format_elapsed(59.9) == "just now"

    def test_minutes(self):
        assert _format_elapsed(60) == "1 minute ago"
        assert _format_elapsed(125) == "2 minutes ago"

    def test_hours_with_minute_remainder(self):
        assert _format_elapsed(3600) == "1 hour 0 minutes ago"
        assert _format_elapsed(3725) == "1 hour 2 minutes ago"

    def test_days_with_hour_remainder(self):
        assert _format_elapsed(86400) == "1 day 0 hours ago"
        assert _format_elapsed(90000) == "1 day 1 hour ago"


# ---------------------------------------------------------------------------
# _pre_llm_call
# ---------------------------------------------------------------------------


class TestPreLlmCall:
    def test_returns_context_dict(self):
        result = _pre_llm_call(session_id="s1")
        assert isinstance(result, dict)
        assert "context" in result
        assert result["context"].startswith("[Current time ")

    def test_stamp_is_iso_like_with_timezone(self):
        result = _pre_llm_call(session_id="s2")
        assert re.match(r"^\[Current time \d{4}-\d{2}-\d{2} \d{2}:\d{2} [A-Z]+", result["context"])

    def test_elapsed_from_conversation_history(self):
        # [-1] is the current message, [-2] the previous one (prologue appends first)
        hist = [
            {"timestamp": 1_700_000_000.0},  # previous message (old)
            {"timestamp": 1_700_000_100.0},  # current message
        ]
        result = _pre_llm_call(session_id="s3", conversation_history=hist)
        assert "last message" in result["context"]

    def test_no_elapsed_when_previous_message_is_fresh(self):
        import time

        now = time.time()
        hist = [
            {"timestamp": now - 10},  # 10 seconds ago -> under 60s threshold
            {"timestamp": now},
        ]
        result = _pre_llm_call(session_id="s4", conversation_history=hist)
        assert "last message" not in result["context"]

    def test_no_elapsed_without_history(self):
        result = _pre_llm_call(session_id="s5", conversation_history=[])
        assert "last message" not in result["context"]

    def test_in_process_fallback_record(self):
        # With no usable history, the in-process record of the previous turn
        # serves as the elapsed source.
        from datetime import datetime, timedelta

        from plugins.turn_clock import _last_ts, _lock

        with _lock:
            _last_ts["s6"] = datetime.now().astimezone() - timedelta(minutes=5)
        result = _pre_llm_call(session_id="s6", conversation_history=[])
        assert "last message" in result["context"]
        assert "5 minutes ago" in result["context"]

    def test_session_isolation(self):
        from plugins.turn_clock import _last_ts, _lock

        with _lock:
            _last_ts.pop("s7", None)
        result = _pre_llm_call(session_id="s7", conversation_history=[])
        assert "last message" not in result["context"]


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


class TestRegister:
    def test_registers_pre_llm_call_hook(self):
        class FakeCtx:
            def __init__(self):
                self.hooks = []

            def register_hook(self, hook_name, callback):
                self.hooks.append((hook_name, callback))

        ctx = FakeCtx()
        register(ctx)
        assert ("pre_llm_call", _pre_llm_call) in ctx.hooks
