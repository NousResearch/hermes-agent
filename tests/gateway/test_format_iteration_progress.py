"""Unit tests for gateway.run._format_iteration_progress (#102806).

``AIAgent.max_iterations`` defaults to ``sys.maxsize`` (unlimited
tool-calling iterations for a top-level session — see ``run_agent.py``).
Three user-facing gateway status lines render ``iteration N/M`` from
``AIAgent.get_activity_summary()``'s ``api_call_count`` / ``max_iterations``
pair: the long-running heartbeat, the busy-session acknowledgment, and the
gateway-timeout diagnostic message. All three share this helper; see
``test_busy_session_ack.py`` and ``test_gateway_timeout_iteration_progress.py``
for the call sites' own end-to-end regression tests.
"""

import sys

from gateway.run import _format_iteration_progress


class TestFormatIterationProgress:
    def test_unbounded_default_omits_denominator(self):
        """sys.maxsize (AIAgent's actual default) prints iteration count alone."""
        assert _format_iteration_progress(2, sys.maxsize) == "iteration 2"

    def test_above_sys_maxsize_also_omits_denominator(self):
        """Anything at or beyond the sentinel is treated as unbounded too."""
        assert _format_iteration_progress(5, sys.maxsize + 1) == "iteration 5"

    def test_finite_budget_still_shows_both_numbers(self):
        """A real, finite ceiling (e.g. a subagent's max_iterations: 250)
        keeps the existing N/M format — this is not a blanket format change."""
        assert _format_iteration_progress(7, 250) == "iteration 7/250"

    def test_small_finite_budget(self):
        assert _format_iteration_progress(0, 90) == "iteration 0/90"

    def test_non_numeric_max_iterations_falls_back_to_unbounded_rendering(self):
        """get_activity_summary() is a best-effort diagnostics snapshot; a
        malformed/missing max_iterations must not raise inside a status-line
        formatter (call sites wrap this in try/except, but the helper itself
        should degrade gracefully rather than propagate a TypeError)."""
        assert _format_iteration_progress(3, None) == "iteration 3"
        assert _format_iteration_progress(3, "not-a-number") == "iteration 3"

    def test_api_call_count_is_rendered_verbatim(self):
        """Only the denominator's unbounded-ness is special-cased; the
        numerator (api_call_count) always prints as given."""
        assert _format_iteration_progress(123, 250) == "iteration 123/250"
        assert _format_iteration_progress(123, sys.maxsize) == "iteration 123"
