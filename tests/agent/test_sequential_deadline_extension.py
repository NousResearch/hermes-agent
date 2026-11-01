"""Sequential executor deadline must respect a tool's own bounded wait.

Regression tests (#“terminal 420s cut”): the sequential runner armed a single
generic deadline (``timeouts.tools.sequential_call``, default = the 420s
concurrent batch bound) and ignored the per-call timeout a tool itself
validates and enforces. ``terminal(timeout=600)`` (a legal foreground request
under the tool's own 600s ``FOREGROUND_MAX_TIMEOUT`` cap) was therefore killed
by the executor at 420s with ``tool_timeout`` — a mid-execution lie: the
command was still inside ITS OWN valid wait window. The model retried with
progressively shorter sleeps, all still above 420s, an infinite retry loop.

Contract fixed here:
* A tool call that declares a bounded, valid per-call timeout must be allowed
  to run for that whole window plus a small unwind margin — the executor
  deadline extends to it instead of cutting it short.
* The extension is an *extension only*: when the generic deadline is already
  at least as long as requested timeout + margin, nothing changes.
* A disabled (``None``) generic deadline stays disabled; boundedness is
  preserved for every other tool (no blanket raise of the global deadline).
* Invalid junk in the ``timeout`` arg (wrong type, NaN, negative, zero,
  non-numeric strings) must NOT corrupt the deadline math: the call keeps the
  generic deadline. The tool itself owns rejecting/normalizing bad values.
* ``process`` — a terminal tool that manages background jobs with its own
  supervision — gets no extension from a timeout-like arg it doesn't wait on.

No test here performs a real long wait: the margin and resolvers are
monkeypatched to tiny values; only the arithmetic of the deadline is exercised.
"""

import math

from agent import tool_executor


# ---------------------------------------------------------------------------
# Pure deadline arithmetic (no timing, no threads)
# ---------------------------------------------------------------------------


class TestSequentialDeadlineArithmetic:
    def _deadline(self, generic, tool_timeout):
        return tool_executor._sequential_deadline_seconds(generic, "terminal", {"timeout": tool_timeout})

    def test_requested_timeout_extends_short_generic_deadline(self):
        # The bug: 420 generic vs a legal 600s terminal request -> 420 cut it.
        assert self._deadline(420.0, 600) == 600.0 + tool_executor._SEQUENTIAL_UNWIND_MARGIN_S

    def test_oversized_timeout_does_not_fund_background_promotion(self):
        # Over-cap foreground calls are promoted to background by terminal.
        assert self._deadline(420.0, 99999) == 420.0

    def test_background_timeout_does_not_extend_foreground_executor(self):
        assert tool_executor._sequential_deadline_seconds(
            420.0, "terminal", {"timeout": 99999, "background": True}
        ) == 420.0

    def test_extension_is_extension_only(self):
        # A generic deadline already covering the request is untouched.
        assert self._deadline(900.0, 600) == 900.0

    def test_no_requested_timeout_keeps_generic_deadline(self):
        assert self._deadline(420.0, None) == 420.0

    def test_disabled_generic_deadline_stays_disabled(self):
        # None (unbounded) must not silently become bounded, and a tool's
        # request must not re-introduce a bound the operator disabled.
        assert self._deadline(None, 600) is None

    def test_junk_timeout_type_keeps_generic_deadline(self):
        for junk in ("600", True, [600], {"s": 600}, 600.5j, float("nan")):
            assert self._deadline(420.0, junk) == 420.0, f"junk {junk!r} corrupted the deadline"

    def test_nonpositive_timeout_keeps_generic_deadline(self):
        # terminal rejects <=0 itself; the executor must not treat 0 as the
        # default or negative as an immediate cut — that's the tool's job.
        assert self._deadline(420.0, 0) == 420.0
        assert self._deadline(420.0, -5) == 420.0

    def test_infinite_timeout_keeps_generic_deadline(self):
        assert self._deadline(420.0, math.inf) == 420.0

    def test_fractional_timeout_extends_deadline(self):
        assert self._deadline(10.0, 12.5) == 12.5 + tool_executor._SEQUENTIAL_UNWIND_MARGIN_S

    def test_extension_only_applies_to_terminal(self):
        # boundedness preserved for other tools: same args, different tool.
        assert (
            tool_executor._sequential_deadline_seconds(420.0, "web_search", {"timeout": 600})
            == 420.0
        )

    def test_process_tool_gets_no_extension(self):
        # process manages background jobs under its own supervision; its
        # timeout-ish args are not a foreground wait the executor must fund.
        assert (
            tool_executor._sequential_deadline_seconds(420.0, "process", {"timeout": 600})
            == 420.0
        )

    def test_intlike_float_string_not_coerced(self):
        # Coercing "600" would let a model-emitted string smuggle an unbounded
        # extension past the terminal tool's own validation (which sees the
        # raw value). Only real numbers count.
        assert self._deadline(420.0, "600") == 420.0


# ---------------------------------------------------------------------------
# Runner integration: the extension actually funds a legal longer wait
# ---------------------------------------------------------------------------


class TestSequentialRunnerFundsRequestedTimeout:
    def test_terminal_call_runs_past_generic_deadline(self, monkeypatch):
        """A terminal call whose own requested timeout exceeds the generic
        sequential deadline is NOT abandoned at the generic deadline: the
        wait is funded up to requested + margin (real wait here is tiny)."""
        import threading
        from agent.tool_executor import (
            _ManagedToolResult,
            _ToolTimeoutResult,
            _run_sequential_tool_execution_middleware,
        )

        emitted = []
        monkeypatch.setattr(tool_executor, "_SEQUENTIAL_INTERRUPT_POLL_SECONDS", 0.05)
        monkeypatch.setattr(tool_executor, "_SEQUENTIAL_UNWIND_MARGIN_S", 0.5)
        # Generic deadline would fire at 0.3s; the tool asks for 1.0s and
        # finishes at ~0.6s: past the generic deadline, inside its own window.
        monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 0.3)
        monkeypatch.setattr(
            tool_executor,
            "_emit_terminal_post_tool_call",
            lambda agent, **kw: emitted.append(kw),
        )

        released = threading.Event()

        def _fake_middleware(agent_arg, **kwargs):
            released.wait(5)  # released by the tool "finishing"
            return _ManagedToolResult(
                result="command output", args={}, middleware_trace=[], blocked=False, dispatched=True,
            )

        def _finish_on_time():
            # The tool finishes at 0.6s: after the 0.3s generic deadline
            # (old bug: tool_timeout here) but inside requested(1.0)+margin.
            import time as _t
            _t.sleep(0.6)
            released.set()

        monkeypatch.setattr(
            tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware
        )
        threading.Thread(target=_finish_on_time, daemon=True).start()

        class _Agent:
            _interrupt_requested = False
            _tool_interrupt_reason = None

            def __init__(self):
                self._tool_worker_threads = set()
                self._tool_worker_threads_lock = threading.Lock()

            def _touch_activity(self, desc):
                pass

            def interrupt(self, *a, **k):
                pass

        managed = _run_sequential_tool_execution_middleware(
            _Agent(),
            function_name="terminal",
            function_args={"command": "sleep 1", "timeout": 1},
            effective_task_id="t",
            tool_call_id="tc_ext",
            execute=lambda a: "unused",
        )

        assert not isinstance(managed.result, _ToolTimeoutResult), (
            "executor cut a terminal call that was still inside its own "
            "requested timeout window"
        )
        assert managed.result == "command output"

    def test_terminal_call_still_cut_past_requested_window(self, monkeypatch):
        """Boundedness is preserved: once requested + margin elapses with the
        tool still running, the executor abandons it with tool_timeout —
        the extension funds the request, it never removes the bound."""
        import threading
        import time
        from agent.tool_executor import (
            _ManagedToolResult,
            _ToolTimeoutResult,
            _run_sequential_tool_execution_middleware,
        )

        monkeypatch.setattr(tool_executor, "_SEQUENTIAL_INTERRUPT_POLL_SECONDS", 0.05)
        monkeypatch.setattr(tool_executor, "_SEQUENTIAL_UNWIND_MARGIN_S", 0.3)
        # Generic deadline 0.2s; requested timeout 0.5s + 0.3 margin funds a
        # 0.8s window -> the tool (sleeps 5s) is abandoned at ~0.8s: the
        # extension itself remains a hard bound.
        monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 0.2)
        monkeypatch.setattr(
            tool_executor, "_emit_terminal_post_tool_call", lambda agent, **kw: None
        )

        def _fake_middleware(agent_arg, **kwargs):
            time.sleep(5)
            return _ManagedToolResult(
                result="late", args={}, middleware_trace=[], blocked=False, dispatched=True,
            )

        monkeypatch.setattr(
            tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware
        )

        class _Agent:
            _interrupt_requested = False
            _tool_interrupt_reason = None

            def __init__(self):
                self._tool_worker_threads = set()
                self._tool_worker_threads_lock = threading.Lock()

            def _touch_activity(self, desc):
                pass

            def interrupt(self, *a, **k):
                pass

        started = time.monotonic()
        managed = _run_sequential_tool_execution_middleware(
            _Agent(),
            function_name="terminal",
            function_args={"command": "sleep 5", "timeout": 0.5},
            effective_task_id="t",
            tool_call_id="tc_cut",
            execute=lambda a: "unused",
        )
        elapsed = time.monotonic() - started

        assert isinstance(managed.result, _ToolTimeoutResult)
        assert elapsed < 4.0, f"abandonment waited too long: {elapsed:.1f}s"

    def test_non_terminal_call_cut_at_generic_deadline_unchanged(self, monkeypatch):
        """The generic deadline still bounds other tools exactly as before."""
        import threading
        import time
        from agent.tool_executor import (
            _ManagedToolResult,
            _ToolTimeoutResult,
            _run_sequential_tool_execution_middleware,
        )

        monkeypatch.setattr(tool_executor, "_SEQUENTIAL_INTERRUPT_POLL_SECONDS", 0.05)
        monkeypatch.setattr(
            tool_executor, "_emit_terminal_post_tool_call", lambda agent, **kw: None
        )

        def _fake_middleware(agent_arg, **kwargs):
            time.sleep(5)
            return _ManagedToolResult(
                result="late", args={}, middleware_trace=[], blocked=False, dispatched=True,
            )

        monkeypatch.setattr(
            tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware
        )

        class _Agent:
            _interrupt_requested = False
            _tool_interrupt_reason = None

            def __init__(self):
                self._tool_worker_threads = set()
                self._tool_worker_threads_lock = threading.Lock()

            def _touch_activity(self, desc):
                pass

            def interrupt(self, *a, **k):
                pass

        monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 0.4)
        # web_search is not terminal: even with a timeout arg, the generic
        # 0.4s deadline applies.
        managed = _run_sequential_tool_execution_middleware(
            _Agent(),
            function_name="web_search",
            function_args={"query": "x", "timeout": 600},
            effective_task_id="t",
            tool_call_id="tc_plain",
            execute=lambda a: "unused",
        )
        assert isinstance(managed.result, _ToolTimeoutResult)
