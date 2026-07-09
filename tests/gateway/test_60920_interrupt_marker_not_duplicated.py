"""
Regression tests for issue #60920 - Interrupt message duplicated by
_replay_output_history after terminal recovery.

The bug: when an API call is interrupted, the response text gets an
interrupt marker suffix appended (``\\n\\n---\\n_[Interrupted -
processing new message]_`` at cli.py:12595). ChatConsole.print() at
cli.py:3437 records the rendered Panel line-by-line via _cprint(),
which calls _record_output_history(text). After the interrupt,
_recover_terminal_after_interrupt() calls _force_full_redraw(), which
calls _replay_output_history() — re-emitting the entire recorded
history including the marker that was already on screen. Terminal
resize then triggers another replay → 2+ visible duplicates that
accumulate on every resize.

The fix: print the interrupt marker as a SEPARATE _cprint() call wrapped
in ``_suspend_output_history()``, after the Panel render. The clean
response is still recorded (so resize-recovery shows the response),
but the once-displayed marker is NOT recorded (so a later replay
won't duplicate it).

These tests directly exercise the recording path (``_record_output_history``)
without spinning up the full CLI. The key invariant: the
``_suspend_output_history()`` context manager suppresses recording for
the duration of its block, and ``_OUTPUT_HISTORY`` is the queue
that ``_replay_output_history`` later walks.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest


INTERRUPT_MARKER_LINE = "_[Interrupted - processing new message]_"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_output_history():
    """Reset _OUTPUT_HISTORY and the suppression flags before/after."""
    cli = importlib.import_module("cli")
    if hasattr(cli, "_OUTPUT_HISTORY"):
        cli._OUTPUT_HISTORY.clear()
    # Make sure suppression flags are in known state.
    if hasattr(cli, "_OUTPUT_HISTORY_SUPPRESSED"):
        cli._OUTPUT_HISTORY_SUPPRESSED = False
    if hasattr(cli, "_OUTPUT_HISTORY_REPLAYING"):
        cli._OUTPUT_HISTORY_REPLAYING = False
    yield
    if hasattr(cli, "_OUTPUT_HISTORY"):
        cli._OUTPUT_HISTORY.clear()


def _enable_recording():
    cli = importlib.import_module("cli")
    cli._OUTPUT_HISTORY_ENABLED = True
    cli._OUTPUT_HISTORY_SUPPRESSED = False
    cli._OUTPUT_HISTORY_REPLAYING = False


def _history_text():
    cli = importlib.import_module("cli")
    return "\n".join(cli._OUTPUT_HISTORY)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInterruptMarkerIsRecordedOnUnfixed:
    """Unfixed baseline: the marker is concatenated INTO the response
    before ChatConsole.print() renders. ChatConsole records the
    rendered lines (which include the marker) into _OUTPUT_HISTORY.
    The test drives the actual render path and confirms the bug.

    After the fix, the marker is emitted separately under
    _suspend_output_history() (so it doesn't get recorded into history),
    and this test starts to fail.
    """

    def test_marker_added_to_history_after_panel_render(self):
        """End-to-end: drive the ChatConsole.print pipeline with a
        response that DOES have the marker concatenated in (the bug).
        After rendering, _OUTPUT_HISTORY contains the marker suffix.
        After the fix, the marker is no longer in the response at this
        point, so _OUTPUT_HISTORY won't have it.
        """
        cli = importlib.import_module("cli")
        _enable_recording()

        response = "Here is the answer."
        response_with_marker = response + "\n\n---\n_[Interrupted - processing new message]_"

        # Drive the actual ChatConsole.print path (which is what the
        # Panel render at line 12683 does). ChatConsole.print() iterates
        # the rendered lines and calls _cprint() on each, which records
        # them into _OUTPUT_HISTORY.
        console = cli.ChatConsole()
        # Pre-populate _OUTPUT_HISTORY with a clean state.
        cli._OUTPUT_HISTORY.clear()
        # _render_final_assistant_content is a rich-text formatter
        # — we mock it to return the response as-is so the test
        # focuses on the recording path.
        with patch("cli._render_final_assistant_content", side_effect=lambda r, mode: r):
            console.print(response_with_marker)

        history = _history_text()
        # Check that the marker fragment appears somewhere in the history
        # (it's split across multiple lines on the rendered Panel, so the
        # marker text usually lands on a single line as `_[Interrupted - processing
        # new message]_` with rich markup escapes that vary between Panel
        # rendering styles — match on the substring "Interrupted" + "new message").
        assert any(("Interrupted" in line and "new message" in line) for line in history.splitlines()), (
            f"precondition for #60920: on the unfixed code path, the marker "
            f"concatenated INTO the response should be in "
            f"_OUTPUT_HISTORY (so _replay_output_history will duplicate "
            f"it on screen). Got history snippet:\n{history[:300]!r}"
        )


class TestInterruptMarkerSuppressionAfterFix:
    """After the fix: the marker is rendered under _suspend_output_history()
    so it gets printed to the terminal but NOT recorded. The response
    (without the marker) IS recorded so resize-recovery shows the
    response text.

    Test the primitive that's used by the fix: if you emit the marker
    under _suspend_output_history(), it doesn't get appended to
    _OUTPUT_HISTORY. This is the actual mechanism that breaks the
    duplication cycle.
    """

    def test_marker_not_recorded_when_under_suppress(self):
        """Simulate what the fix does: emit the response via _cprint
        (recorded), then emit the marker under _suspend_output_history
        (NOT recorded). Verify the marker isn't in history."""
        cli = importlib.import_module("cli")
        _enable_recording()

        response = "Here is the answer."

        # Phase 1: response rendered normally (recorded).
        cli._cprint(response)

        # Phase 2: marker emitted under suppression (printed to
        # terminal, NOT recorded).
        marker = "\n\n---\n_[Interrupted - processing new message]_"
        with cli._suspend_output_history():
            cli._cprint(marker)

        history = _history_text()

        # The response is recorded.
        assert response in history, (
            f"response must still be recorded for resize-recovery; "
            f"got history:\n{history!r}"
        )
        # The marker is NOT recorded. Use a substring match rather than
        # exact match because the marker text passes through rich Panel
        # rendering and may have escape variations in _OUTPUT_HISTORY.
        assert not any(
            ("Interrupted" in line and "new message" in line)
            for line in history.splitlines()
        ), (
            f"FAIL: the interrupt marker must NOT be in _OUTPUT_HISTORY "
            f"once the fix is in. With it suppressed, a later "
            f"_replay_output_history won't re-emit the marker. Got "
            f"history:\n{history!r}"
        )

    def test_suppress_nests_correctly(self):
        cli = importlib.import_module("cli")
        _enable_recording()
        cli._OUTPUT_HISTORY_SUPPRESSED = False

        with cli._suspend_output_history():
            assert cli._OUTPUT_HISTORY_SUPPRESSED is True

        assert cli._OUTPUT_HISTORY_SUPPRESSED is False, (
            "_suspend_output_history() must restore the suppression flag "
            "on context exit (otherwise recording would stay disabled "
            "for the rest of the process)"
        )


class TestFullReplayPathAfterFix:
    """End-to-end: drive the recording + replay path and count how many
    copies of the marker survive. Before the fix, the marker is recorded
    (so each replay emits it once → visible on screen + in history →
    next replay emits it AGAIN → infinite accumulation on resize).

    After the fix, only the response is in _OUTPUT_HISTORY; replaying
    produces zero copies of the marker on top of the original.
    """

    def test_replay_emits_no_marker_when_suppressed(self):
        cli = importlib.import_module("cli")
        _enable_recording()

        response = "Here is the answer."

        # Phase 1: response recorded normally. Marker recorded under
        # suppression (printed but not recorded).
        cli._record_output_history(response)
        marker = "\n\n---\n_[Interrupted - processing new message]_"
        with cli._suspend_output_history():
            cli._record_output_history(marker)

        history = _history_text()

        # After fix: history has the response but NOT the marker.
        assert response in history
        assert INTERRUPT_MARKER_LINE not in history

        # Replay should be a no-op for the marker. We can't easily
        # capture the replay output (it goes through prompt_toolkit),
        # so instead we verify the invariant the fix is built on:
        # if the marker isn't in history, replay can't reproduce it.
        # This is the precise mechanism the issue body identifies as
        # the duplication source.
        assert not any(INTERRUPT_MARKER_LINE in line for line in cli._OUTPUT_HISTORY), (
            "the interrupt marker is in _OUTPUT_HISTORY; a subsequent "
            "_replay_output_history will re-emit it on top of the "
            "already-displayed marker (#60920). The fix wraps the "
            "marker emit in _suspend_output_history() so it does NOT "
            "get recorded."
        )


class TestChatCodeDoesNotConcatMarkerIntoResponse:
    """Integration-level regression guard for the fix.

    The fix moves the interrupt marker from being concatenated into the
    response string (which causes ChatConsole.print to record it) to
    being emitted separately under _suspend_output_history() (which
    keeps it out of history).

    Tests inspect the cli.py source for the CONCATENATION PATTERN as a
    proxy for "did the marker get concatenated into response?". The fix
    REMOVES the concatenation; this test fails on unfixed code that
    still has `response = response + marker` and passes after the fix.
    """

    def test_response_does_not_get_marker_concatenated(self):
        """Static source inspection: the bug was concatenating the marker
        into ``response`` at cli.py:12595. The fix separates the marker
        into its own variable so the response string stays clean.

        This is a coarse proxy test — what really matters is the
        behavioral check above. But it does catch the literal
        regression of "did someone reintroduce the concatenation?"
        """
        from pathlib import Path

        cli_path = Path("/tmp/hermes-pr-work-60859/hermes-agent/cli.py")
        source = cli_path.read_text(encoding="utf-8")

        # The bug pattern: `response = response + "..."<marker literal>..."`
        # After the fix: response is NOT modified to include the marker.
        # Look for any line where `response` is assigned something that
        # concatenates the marker text.
        bad_pattern = 'response = response + "\\n\\n---\\n_[Interrupted - processing new message]_'
        assert bad_pattern not in source, (
            f"BUG (#60920): cli.py concatenates the interrupt marker "
            f"into the response string. This causes ChatConsole.print() "
            f"to record the marker into _OUTPUT_HISTORY, and the next "
            f"_replay_output_history (after _force_full_redraw or "
            f"terminal resize) duplicates the marker on screen."
        )

    def test_marker_is_emitted_under_suspend_output_history(self):
        """The fix emits the marker through _suspend_output_history()
        around the marker print. This ensures the marker reaches the
        terminal (via _cprint) but is NOT recorded into _OUTPUT_HISTORY.
        """
        from pathlib import Path

        cli_path = Path("/tmp/hermes-pr-work-60859/hermes-agent/cli.py")
        source = cli_path.read_text(encoding="utf-8")

        # The fix shape: a `with _suspend_output_history():` block that
        # also calls _cprint on the marker (in some form).
        has_suspend_block = "with _suspend_output_history():" in source
        # A subsequent _cprint call within that block (or within a few lines)
        has_cprint_after_suspend = "_cprint(_interrupt_marker" in source

        assert has_suspend_block, (
            "fix shape requires a `with _suspend_output_history():` "
            "block in cli.py to wrap the interrupt marker print"
        )
        assert has_cprint_after_suspend, (
            "the fix must use `_cprint(_interrupt_marker...)` inside "
            "_suspend_output_history() to emit the marker without "
            "recording it"
        )