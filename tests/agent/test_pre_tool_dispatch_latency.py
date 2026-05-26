"""Regression tests for pre-tool dispatch latency observability (#32460).

The agent loop calls ``get_pre_tool_call_block_message()`` for every
tool invocation before the tool starts measuring its own duration, so a
slow plugin or shell hook here produces a silent wall-clock gap with no
indication of where time was spent.  The
:func:`agent.tool_dispatch_helpers.pre_tool_call_block_message_with_latency`
wrapper measures that window and logs a WARNING when it exceeds
``_PRE_TOOL_DISPATCH_SLOW_THRESHOLD_SECONDS`` — turning the previously-
silent stall into an actionable log line that names the offending tool.

These tests pin three invariants:

* Fast dispatch is silent — the warning must not fire on every tool
  call or the log noise would be intolerable.
* Slow dispatch logs a WARNING that names the tool and the elapsed
  time so the operator can correlate the latency with a specific tool
  invocation.
* The return value of the wrapper is the same as the underlying call —
  the wrapper is purely observational and must never mutate the block
  decision.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import patch

import pytest

from agent import tool_dispatch_helpers


@pytest.fixture
def low_threshold(monkeypatch):
    """Shrink the slow-dispatch threshold for fast, deterministic tests."""
    monkeypatch.setattr(
        tool_dispatch_helpers,
        "_PRE_TOOL_DISPATCH_SLOW_THRESHOLD_SECONDS",
        0.1,
    )


def _latency_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if "pre_tool_call dispatch took" in r.getMessage()
    ]


class TestPreToolDispatchLatency:
    def test_fast_dispatch_emits_no_warning(self, caplog, low_threshold):
        """Sub-threshold dispatch must stay silent.

        A naked pre-tool check on every single tool invocation would
        flood the log if the wrapper warned unconditionally, so we
        guard against accidental "always warn" regressions here.
        """
        with patch(
            "hermes_cli.plugins.get_pre_tool_call_block_message",
            return_value=None,
        ):
            with caplog.at_level("WARNING", logger="agent.tool_dispatch_helpers"):
                result = tool_dispatch_helpers.pre_tool_call_block_message_with_latency(
                    "terminal", {"command": "echo ok"},
                )

        assert result is None
        assert _latency_warnings(caplog) == []

    def test_slow_dispatch_warns_with_tool_name_and_elapsed(
        self, caplog, low_threshold
    ):
        """Above-threshold dispatch must log a WARNING that names the tool.

        The warning text is the operator-facing diagnostic for the
        symptom in #32460 — the line must include both the tool name
        (so the user can pinpoint which call stalled) and the elapsed
        time (so they can correlate with the magnitude of the gap).
        """
        def _slow_hook(*_args, **_kwargs):
            time.sleep(0.2)
            return None

        with patch(
            "hermes_cli.plugins.get_pre_tool_call_block_message",
            side_effect=_slow_hook,
        ):
            with caplog.at_level("WARNING", logger="agent.tool_dispatch_helpers"):
                tool_dispatch_helpers.pre_tool_call_block_message_with_latency(
                    "terminal", {"command": "echo slow"},
                )

        warnings = _latency_warnings(caplog)
        assert warnings, "wrapper must warn when dispatch exceeds threshold"
        msg = warnings[0]
        assert "terminal" in msg, "warning must name the offending tool"
        # Elapsed-time format is "%.1fs"; just check it's a plausible
        # decimal value near our sleep duration.  Allow generous slack
        # for slow CI hosts.
        assert "0.2s" in msg or "0.3s" in msg or "0.4s" in msg, (
            f"warning should report elapsed time ≈0.2s, got: {msg}"
        )

    def test_block_message_passes_through(self, caplog, low_threshold):
        """The wrapper must return the underlying block decision unchanged.

        Regression guard: an early implementation could mask block
        directives by accidentally swallowing the return value while
        timing the call.  We assert here that the wrapper is purely
        observational — when the upstream hook returns a block message,
        the wrapper returns the same string verbatim.
        """
        with patch(
            "hermes_cli.plugins.get_pre_tool_call_block_message",
            return_value="blocked by plugin policy",
        ):
            with caplog.at_level("WARNING", logger="agent.tool_dispatch_helpers"):
                result = tool_dispatch_helpers.pre_tool_call_block_message_with_latency(
                    "terminal", {"command": "rm -rf /"},
                )

        assert result == "blocked by plugin policy"

    def test_exception_propagates_and_still_logs_latency(
        self, caplog, low_threshold
    ):
        """Even if the underlying call raises, the latency check still runs.

        The wrapper uses ``try/finally`` so the timing path executes on
        every dispatch regardless of how the underlying call exits.
        A slow hook that raises after stalling must still produce the
        WARNING so the operator can locate the offending plugin.
        """
        def _slow_then_raise(*_args, **_kwargs):
            time.sleep(0.2)
            raise RuntimeError("plugin exploded")

        with patch(
            "hermes_cli.plugins.get_pre_tool_call_block_message",
            side_effect=_slow_then_raise,
        ):
            with caplog.at_level("WARNING", logger="agent.tool_dispatch_helpers"):
                with pytest.raises(RuntimeError, match="plugin exploded"):
                    tool_dispatch_helpers.pre_tool_call_block_message_with_latency(
                        "terminal", {"command": "echo boom"},
                    )

        # The warning still fires even though the call raised.
        assert _latency_warnings(caplog), (
            "latency warning must fire even when the hook raises"
        )

    def test_default_threshold_is_production_sane(self):
        """Production threshold should be high enough not to flood the log
        but low enough that a stuck hook surfaces well before the 60-second
        ``DEFAULT_TIMEOUT_SECONDS`` window in ``agent.shell_hooks``.

        This is a behavioural pin — the constant exists precisely
        because lowering it to satisfy tests (and forgetting to restore
        it) would silently re-introduce the #32460 symptom.
        """
        # Reference the un-patched module-level default explicitly.
        assert (
            0.5 <= tool_dispatch_helpers._PRE_TOOL_DISPATCH_SLOW_THRESHOLD_SECONDS <= 10.0
        ), (
            "production threshold must be a small-but-not-tiny value so "
            "real hangs surface fast while routine dispatch stays quiet"
        )
