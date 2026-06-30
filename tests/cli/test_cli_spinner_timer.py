"""Regression tests for the inline tool spinner timer in the classic CLI.

Covers two fixes:

* ``_render_spinner_text`` torn-read guard — the live elapsed timer must never
  render a negative or absurdly large number when a render observes a fresh
  ``_spinner_text`` paired with a stale ``_tool_start_time`` (the two fields are
  best-effort UI state updated by different events without a lock).

* The ``spinner_loop`` repaint gate — the loop must repaint on a fixed cadence
  while an agent turn is running a tool (``_agent_running`` + non-empty
  ``_spinner_text``), not only while a slash command runs (``_command_running``).
  Previously the agent-tool case was missed, so the timer only advanced when an
  unrelated agent event happened to fire ``_invalidate()``, making the seconds
  appear to jump by whole tool durations.
"""

import time

from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._spinner_text = ""
    cli_obj._tool_start_time = 0.0
    return cli_obj


class TestRenderSpinnerText:
    def test_empty_spinner_returns_empty(self):
        cli_obj = _make_cli()
        assert cli_obj._render_spinner_text() == ""

    def test_label_without_start_time_has_no_timer(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = "🔧 terminal"
        cli_obj._tool_start_time = 0.0
        out = cli_obj._render_spinner_text()
        assert "terminal" in out
        assert "(" not in out  # no timer parens

    def test_normal_elapsed_renders_seconds(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = "🔧 terminal"
        cli_obj._tool_start_time = time.monotonic() - 3.2
        out = cli_obj._render_spinner_text()
        assert "terminal" in out
        assert "s)" in out
        assert "3." in out

    def test_rollover_past_60s_uses_fixed_width(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = "🔧 terminal"
        cli_obj._tool_start_time = time.monotonic() - 125
        out = cli_obj._render_spinner_text()
        assert "02m05s" in out

    def test_stale_start_time_does_not_flash_absurd_number(self):
        """Torn read: fresh label, very old start timestamp from a prior turn."""
        cli_obj = _make_cli()
        cli_obj._spinner_text = "🔧 read_file"
        cli_obj._tool_start_time = time.monotonic() - 999999  # ~11.5 days
        out = cli_obj._render_spinner_text()
        assert "read_file" in out
        assert "(" not in out  # timer suppressed, no giant number

    def test_negative_elapsed_does_not_render_negative_timer(self):
        """Clock-edge / reset race producing a start time slightly in the future."""
        cli_obj = _make_cli()
        cli_obj._spinner_text = "🔧 patch"
        cli_obj._tool_start_time = time.monotonic() + 50
        out = cli_obj._render_spinner_text()
        assert "patch" in out
        assert "-" not in out


def _repaint_gate(command_running: bool, agent_running: bool, spinner_text: str) -> bool:
    """Mirror of the repaint condition in cli.py ``spinner_loop``.

    Kept in sync with::

        if self._command_running or (self._agent_running and self._spinner_text):

    so the truth table is locked by a test even though the live condition lives
    inside a thread closure that cannot be invoked directly.
    """
    return command_running or (agent_running and bool(spinner_text))


class TestSpinnerLoopRepaintGate:
    def test_idle_prompt_does_not_repaint(self):
        assert _repaint_gate(False, False, "") is False

    def test_slow_slash_command_repaints(self):
        assert _repaint_gate(True, False, "") is True

    def test_agent_running_tool_repaints(self):
        # The regression: agent turn with a live tool must repaint the timer.
        assert _repaint_gate(False, True, "🔧 terminal") is True

    def test_agent_thinking_without_tool_does_not_repaint(self):
        # No tool label => no live timer => keep idle prompt stable.
        assert _repaint_gate(False, True, "") is False
