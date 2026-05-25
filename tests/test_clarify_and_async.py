"""Tests for tools/clarify_tool.py and agent/async_utils.py."""

import json
import asyncio
import pytest
from unittest import mock

from tools.clarify_tool import clarify_tool, check_clarify_requirements, MAX_CHOICES, CLARIFY_SCHEMA
from agent.async_utils import safe_schedule_threadsafe


# ── tools/clarify_tool.py ─────────────────────────────────────────────────────

class TestClarifyTool:
    """clarify_tool() — question validation, choices handling, callback dispatch."""

    def test_empty_question_returns_error(self):
        """Empty or whitespace-only question returns a JSON error."""
        result = clarify_tool("")
        data = json.loads(result)
        assert "error" in data
        assert "required" in data["error"].lower()

    def test_whitespace_only_question(self):
        """Whitespace-only question is treated as empty."""
        result = clarify_tool("   \t\n  ")
        data = json.loads(result)
        assert "error" in data

    def test_choices_not_a_list(self):
        """Non-list choices parameter returns an error."""
        result = clarify_tool("What?", choices="not-a-list")
        data = json.loads(result)
        assert "error" in data
        assert "list" in data["error"].lower()

    def test_choices_exceeding_max_are_truncated(self):
        """More than MAX_CHOICES are silently truncated to MAX_CHOICES."""
        result = clarify_tool(
            "Pick one:",
            choices=["a", "b", "c", "d", "e", "f"],
            callback=lambda q, c: "ok",
        )
        data = json.loads(result)
        assert len(data["choices_offered"]) == MAX_CHOICES
        assert data["choices_offered"] == ["a", "b", "c", "d"]

    def test_choices_whitespace_stripped(self):
        """Choices with leading/trailing whitespace are cleaned."""
        result = clarify_tool(
            "Pick:",
            choices=["  yes  ", " no "],
            callback=lambda q, c: "ok",
        )
        data = json.loads(result)
        assert data["choices_offered"] == ["yes", "no"]

    def test_empty_string_choices_filtered_out(self):
        """Empty or whitespace-only choice strings are removed."""
        result = clarify_tool(
            "Pick:",
            choices=["valid", "", "  ", "\t"],
            callback=lambda q, c: "ok",
        )
        data = json.loads(result)
        assert data["choices_offered"] == ["valid"]

    def test_all_empty_choices_becomes_none(self):
        """When all choices are empty after filtering, choices_offered is None."""
        result = clarify_tool(
            "What?",
            choices=["", "  "],
            callback=lambda q, c: "ok",
        )
        data = json.loads(result)
        assert data["choices_offered"] is None

    def test_no_choices_is_open_ended(self):
        """When choices is None, it stays None (open-ended question)."""
        result = clarify_tool(
            "How are you?",
            choices=None,
            callback=lambda q, c: "fine",
        )
        data = json.loads(result)
        assert data["choices_offered"] is None
        assert data["user_response"] == "fine"

    def test_no_callback_returns_error(self):
        """When no callback is provided, returns a context error."""
        result = clarify_tool("What?", choices=["a"])
        data = json.loads(result)
        assert "error" in data
        assert "not available" in data["error"].lower()

    def test_callback_exception_returns_error(self):
        """If the callback raises, the error is caught and returned as JSON."""
        def bad_callback(q, c):
            raise RuntimeError("connection lost")

        result = clarify_tool("What?", callback=bad_callback)
        data = json.loads(result)
        assert "error" in data
        assert "connection lost" in data["error"]

    def test_successful_callback_returns_full_response(self):
        """A successful callback returns question, choices, and response."""
        def my_callback(q, c):
            return "42"

        result = clarify_tool("Answer:", choices=["40", "41", "42"], callback=my_callback)
        data = json.loads(result)
        assert data["question"] == "Answer:"
        assert data["choices_offered"] == ["40", "41", "42"]
        assert data["user_response"] == "42"

    def test_user_response_is_stripped(self):
        """The user's response string is stripped of whitespace."""
        def my_callback(q, c):
            return "  hello world  \n"

        result = clarify_tool("Say hi:", callback=my_callback)
        data = json.loads(result)
        assert data["user_response"] == "hello world"

    def test_question_is_stripped(self):
        """The question text is stripped of surrounding whitespace."""
        result = clarify_tool(
            "  What is your name?  ",
            callback=lambda q, c: "Hermes",
        )
        data = json.loads(result)
        assert data["question"] == "What is your name?"


class TestClarifySchema:
    """CLARIFY_SCHEMA and related constants."""

    def test_max_choices_is_4(self):
        """MAX_CHOICES is 4 — the UI appends a 5th 'Other' option."""
        assert MAX_CHOICES == 4

    def test_schema_has_required_fields(self):
        """Schema has name, description, and parameters."""
        assert CLARIFY_SCHEMA["name"] == "clarify"
        assert "description" in CLARIFY_SCHEMA
        assert "parameters" in CLARIFY_SCHEMA

    def test_schema_question_is_required(self):
        """'question' is in the required list."""
        assert "question" in CLARIFY_SCHEMA["parameters"]["required"]

    def test_schema_choices_max_items(self):
        """choices has maxItems matching MAX_CHOICES."""
        props = CLARIFY_SCHEMA["parameters"]["properties"]
        assert props["choices"]["maxItems"] == MAX_CHOICES

    def test_check_requirements_always_true(self):
        """check_clarify_requirements always returns True."""
        assert check_clarify_requirements() is True


# ── agent/async_utils.py ──────────────────────────────────────────────────────

class TestSafeScheduleThreadsafe:
    """safe_schedule_threadsafe() — leak-safe coroutine scheduling onto event loops."""

    def test_none_loop_returns_none(self):
        """When loop is None, returns None and closes the coroutine."""
        async def my_coro():
            pass

        coro = my_coro()
        result = safe_schedule_threadsafe(coro, None)
        assert result is None
        # Coroutine should be closed (no "was never awaited" warning)
        with pytest.raises(RuntimeError, match="coroutine.*closed|cannot reuse"):
            coro.send(None)

    def test_none_loop_non_coroutine(self):
        """When loop is None and obj is not a coroutine, still returns None."""
        result = safe_schedule_threadsafe("not-a-coroutine", None)
        assert result is None

    def test_returns_future_on_success(self):
        """With a valid running loop, returns a Future."""
        async def my_coro():
            return "done"

        loop = asyncio.new_event_loop()
        try:
            coro = my_coro()
            result = safe_schedule_threadsafe(coro, loop)
            assert result is not None
            assert hasattr(result, "result")
        finally:
            loop.close()

    def test_exception_closes_coroutine(self):
        """When run_coroutine_threadsafe raises, the coroutine is closed."""
        async def my_coro():
            pass

        coro = my_coro()
        loop = asyncio.new_event_loop()
        try:
            with mock.patch("asyncio.run_coroutine_threadsafe",
                            side_effect=RuntimeError("loop closed")):
                result = safe_schedule_threadsafe(coro, loop)
                assert result is None
        finally:
            loop.close()
        # Coroutine should be closed
        with pytest.raises(RuntimeError, match="coroutine.*closed|cannot reuse"):
            coro.send(None)

    def test_custom_logger(self):
        """Custom logger receives the log message on failure."""
        custom_log = mock.Mock(spec=["log", "debug", "info", "warning", "error"])
        async def my_coro():
            pass

        coro = my_coro()
        result = safe_schedule_threadsafe(
            coro, None,
            logger=custom_log,
            log_message="Custom failure",
            log_level=30,  # WARNING
        )
        assert result is None
        custom_log.log.assert_called_once()
        args, kwargs = custom_log.log.call_args
        assert args[0] == 30  # log_level
        # Format is log.log(level, "%s: loop is None", log_message)
        assert args[2] == "Custom failure"
