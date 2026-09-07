"""Subagent failures surface as one clean user-facing notice.

Covers the Aug 2026 community report: a delegate_task child that dies
(provider 404, timeout, crash) previously vanished silently on platforms
with tool_progress off — the parent model saw the error but the human never
did. Now:

- ``tools.delegate_tool.format_subagent_failure_line`` renders one clean,
  human-readable line (no tracebacks/JSON walls).
- ``TurnRunner.progress_callback`` intercepts ``subagent.complete`` events
  with a terminal failure status FIRST (before every progress-queue gate)
  and delivers the line via ``_deliver_platform_notice``.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.turn_context import TurnContext
from tools.delegate_tool import (
    SUBAGENT_FAILURE_STATUSES,
    _clean_error_text,
    format_subagent_failure_line,
)


class TestCleanErrorText:
    def test_single_line_passthrough(self):
        assert _clean_error_text("Error code: 404 - model not found") == (
            "Error code: 404 - model not found"
        )

    def test_traceback_takes_last_line(self):
        tb = (
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in <module>\n'
            "    raise RuntimeError('boom')\n"
            "RuntimeError: boom"
        )
        assert _clean_error_text(tb) == "RuntimeError: boom"

    def test_multiline_non_traceback_takes_first_line(self):
        assert _clean_error_text("first line\nsecond line") == "first line"

    def test_caps_length(self):
        out = _clean_error_text("x" * 500, max_chars=100)
        assert len(out) == 100
        assert out.endswith("...")

    def test_empty_and_none(self):
        assert _clean_error_text("") == ""
        assert _clean_error_text(None) == ""
        assert _clean_error_text("   \n  ") == ""


class TestFormatSubagentFailureLine:
    def test_failed_with_goal_error_duration(self):
        line = format_subagent_failure_line(
            "research competitor pricing",
            "failed",
            error="Error code: 404 - model not found",
            duration_seconds=12.4,
        )
        assert line.startswith("⚠️ Subagent failed")
        assert '"research competitor pricing"' in line
        assert "404" in line
        assert "(after 12s)" in line

    def test_timeout_verb(self):
        line = format_subagent_failure_line("do a thing", "timeout")
        assert "timed out" in line

    def test_long_goal_truncated(self):
        line = format_subagent_failure_line("g" * 200, "failed")
        assert "g" * 57 + "..." in line
        assert "g" * 61 not in line

    def test_no_goal_no_error(self):
        line = format_subagent_failure_line(None, "error")
        assert line == "⚠️ Subagent failed"

    def test_multiline_goal_flattened(self):
        line = format_subagent_failure_line("a\nb", "failed")
        assert "\n" not in line

    def test_failure_statuses_frozen(self):
        assert SUBAGENT_FAILURE_STATUSES == {"failed", "error", "timeout"}


def _make_runner_and_captured(monkeypatch, run_still_current=True):
    """TurnRunner with a stub gateway runner; captures scheduled notices."""
    from gateway import run as run_mod

    captured: list[str] = []

    class _StubGatewayRunner:
        def _adapter_for_source(self, source):
            return None

        async def _deliver_platform_notice(self, source, content):
            captured.append(content)

    def _fake_schedule(coro, loop, logger=None, log_message=None):
        asyncio.run(coro)

    monkeypatch.setattr(run_mod, "safe_schedule_threadsafe", _fake_schedule)

    ctx = TurnContext(
        source=MagicMock(),
        _run_still_current=lambda: run_still_current,
        progress_queue=None,
        _loop_for_step=None,
    )
    return run_mod.TurnRunner(_StubGatewayRunner(), ctx), captured


class TestGatewayFailureNotice:
    @pytest.mark.parametrize("status", sorted(SUBAGENT_FAILURE_STATUSES))
    def test_failure_statuses_deliver_notice(self, monkeypatch, status):
        runner, captured = _make_runner_and_captured(monkeypatch)
        runner.progress_callback(
            "subagent.complete",
            preview="Error code: 404 - model not found",
            status=status,
            goal="scan the repo",
            duration_seconds=8.0,
        )
        assert len(captured) == 1
        assert "Subagent" in captured[0]
        assert "404" in captured[0]
        assert '"scan the repo"' in captured[0]

    @pytest.mark.parametrize("status", ["completed", "interrupted", None])
    def test_non_failure_statuses_stay_silent(self, monkeypatch, status):
        runner, captured = _make_runner_and_captured(monkeypatch)
        runner.progress_callback(
            "subagent.complete", preview="all done", status=status, goal="g"
        )
        assert captured == []

    def test_stale_run_stays_silent(self, monkeypatch):
        runner, captured = _make_runner_and_captured(
            monkeypatch, run_still_current=False
        )
        runner.progress_callback(
            "subagent.complete", preview="boom", status="failed", goal="g"
        )
        assert captured == []

    def test_fires_without_progress_queue(self, monkeypatch):
        """The notice must not depend on tool_progress being enabled —
        progress_queue=None is exactly the Telegram/Slack default where the
        silent-failure report came from."""
        runner, captured = _make_runner_and_captured(monkeypatch)
        assert runner._ctx.progress_queue is None
        runner.progress_callback(
            "subagent.complete", preview="err", status="error", goal="g"
        )
        assert len(captured) == 1

    def test_summary_preferred_over_preview(self, monkeypatch):
        runner, captured = _make_runner_and_captured(monkeypatch)
        runner.progress_callback(
            "subagent.complete",
            preview="short preview",
            status="failed",
            goal="g",
            summary="the real error detail",
        )
        assert "the real error detail" in captured[0]
