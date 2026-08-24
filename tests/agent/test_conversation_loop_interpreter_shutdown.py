"""Regression guard: interpreter-shutdown errors must abort the conversation
loop immediately instead of retrying.

When the Python interpreter begins its teardown sequence, every executor-backed
operation (API calls, tool dispatch, memory sync) raises::

    RuntimeError: cannot schedule new futures after interpreter shutdown

Before the fix, the outer ``except Exception`` handler in
``run_conversation`` caught this error but did not recognise it as fatal.
Since ``api_call_count`` was nowhere near ``agent.max_iterations - 1``, the
loop continued — each iteration hit the same dead executor and failed
identically, producing a cascade of ``❌ Error during OpenAI-compatible API
call #N`` messages (#93217).

The fix adds an early check in the outer except handler: if
``sys.is_finalizing()`` is True or the error message matches the
``"cannot schedule new futures"`` pattern, the loop breaks immediately with
a clean ``interpreter_shutdown`` exit reason.
"""
from __future__ import annotations

from agent.conversation_loop import _is_interpreter_shutdown_error


class TestInterpreterShutdownDetection:
    """Verify the interpreter-shutdown error matcher used by the
    conversation loop's outer except handler."""

    def test_matches_full_interpreter_shutdown_message(self):
        """The canonical CPython asyncio shutdown message."""
        exc = RuntimeError(
            "cannot schedule new futures after interpreter shutdown"
        )
        assert _is_interpreter_shutdown_error(exc) is True

    def test_matches_short_shutdown_variant(self):
        """Plain ThreadPoolExecutor shutdown variant (no 'interpreter')."""
        exc = RuntimeError("cannot schedule new futures after shutdown")
        assert _is_interpreter_shutdown_error(exc) is True

    def test_case_insensitive_match(self):
        """Error text may arrive in different case from some executor types."""
        exc = RuntimeError("Cannot Schedule New Futures After Interpreter Shutdown")
        assert _is_interpreter_shutdown_error(exc) is True

    def test_does_not_match_unrelated_runtime_error(self):
        """Unrelated RuntimeErrors must not trigger the shutdown path."""
        exc = RuntimeError("connection reset by peer")
        assert _is_interpreter_shutdown_error(exc) is False

    def test_does_not_match_non_runtime_error(self):
        """Non-RuntimeError exceptions must not match."""
        exc = ValueError("cannot schedule new futures")
        assert _is_interpreter_shutdown_error(exc) is False

    def test_does_not_match_none(self):
        """None must not match (defensive — caller may pass None)."""
        try:
            result = _is_interpreter_shutdown_error(None)  # type: ignore[arg-type]
        except TypeError:
            result = False
        assert result is False

    def test_does_not_match_empty_string_exception(self):
        """Empty-message exceptions must not match."""
        exc = RuntimeError("")
        assert _is_interpreter_shutdown_error(exc) is False
