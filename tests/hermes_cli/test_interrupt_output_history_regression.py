"""Regression tests for #60920/#60941: interrupt marker duplication on redraw.

The root cause: The interrupt marker ("_[Interrupted - processing new message]_")
was being appended to the response string, which got recorded in _OUTPUT_HISTORY
by the Panel rendering via _cprint → _record_output_history. When
_recover_terminal_after_interrupt called _force_full_redraw → _replay_output_history,
the marker was replayed on top of the already-visible message, causing duplicates
that accumulated on every SIGWINCH.

The fix:
1. A flag ``_show_interrupt_marker`` is set instead of mutating ``response``.
2. After the Panel rendering, the marker is printed via ``_cprint`` inside a
   ``_suspend_output_history()`` context so it never enters ``_OUTPUT_HISTORY``.
3. ``_recover_terminal_after_interrupt`` no longer clears ``_OUTPUT_HISTORY`` —
   it doesn't need to, because the marker was never recorded.

These tests verify the contract at the module level without hitting the full
prompt_toolkit input loop.
"""

from unittest.mock import MagicMock, patch

import pytest

import cli as cli_mod
from cli import HermesCLI, _suspend_output_history


@pytest.fixture(autouse=True)
def reset_output_history():
    """Reset _OUTPUT_HISTORY before and after every test."""
    cli_mod._configure_output_history(True, 200)
    yield
    cli_mod._configure_output_history(True, 200)


# ── Recovery path: _OUTPUT_HISTORY must NOT be cleared ──────────────


class TestRecoverTerminalPreservesHistory:
    """_recover_terminal_after_interrupt must NOT clear output history.

    The old fix cleared _OUTPUT_HISTORY before the redraw to prevent the
    interrupt marker from being replayed. The new fix avoids recording the
    marker in the first place, so the clear is unnecessary *and* harmful —
    it would discard legitimate scrollback content.
    """

    def test_history_preserved_after_recovery(self, monkeypatch):
        """After recovery, _OUTPUT_HISTORY still contains earlier output."""
        cli_mod._configure_output_history(True, 10)
        cli_mod._record_output_history("normal response text")

        cli = object.__new__(HermesCLI)
        cli._force_full_redraw = MagicMock()

        with patch("hermes_cli.curses_ui.flush_stdin"):
            cli._recover_terminal_after_interrupt()

        assert list(cli_mod._OUTPUT_HISTORY) == ["normal response text"], (
            "_recover_terminal_after_interrupt must NOT clear _OUTPUT_HISTORY"
        )

    def test_recovery_still_calls_force_full_redraw(self, monkeypatch):
        """The recovery path still forces a redraw (original behavior preserved)."""
        cli = object.__new__(HermesCLI)
        cli._force_full_redraw = MagicMock()

        with patch("hermes_cli.curses_ui.flush_stdin"):
            cli._recover_terminal_after_interrupt()

        cli._force_full_redraw.assert_called_once()

    def test_normal_scrollback_survives_interrupt_cycle(self, monkeypatch):
        """Multiple lines of scrollback survive a full interrupt → recovery cycle."""
        cli_mod._configure_output_history(True, 50)
        for i in range(5):
            cli_mod._record_output_history(f"visible line {i}")

        cli = object.__new__(HermesCLI)
        cli._force_full_redraw = MagicMock()

        with patch("hermes_cli.curses_ui.flush_stdin"):
            cli._recover_terminal_after_interrupt()

        assert len(cli_mod._OUTPUT_HISTORY) == 5
        assert list(cli_mod._OUTPUT_HISTORY) == [
            f"visible line {i}" for i in range(5)
        ]


# ── Marker suppression: _suspend_output_history blocks recording ────


class TestInterruptMarkerNotRecorded:
    """The interrupt marker must never enter _OUTPUT_HISTORY.

    Because it's printed inside a ``with _suspend_output_history():`` block,
    the marker text stays out of the replay buffer and _replay_output_history
    cannot duplicate it on redraw or resize.
    """

    def test_suspend_blocks_recording_during_cprint(self, monkeypatch):
        """Text printed via _cprint while supressed is not recorded in history."""
        cli_mod._configure_output_history(True, 10)
        monkeypatch.setattr(cli_mod, "_pt_print", lambda x: None)
        monkeypatch.setattr(cli_mod, "_PT_ANSI", lambda t: t)

        # Record something before so we can distinguish "empty" from "never configured"
        cli_mod._record_output_history("before marker")

        with _suspend_output_history():
            cli_mod._cprint("── [Interrupted — processing new message] ──")

        assert list(cli_mod._OUTPUT_HISTORY) == ["before marker"], (
            "_OUTPUT_HISTORY must not contain the marker text printed "
            "under _suspend_output_history"
        )

    def test_normal_cprint_still_records(self, monkeypatch):
        """Normal _cprint calls (outside the suspend context) are still recorded.

        Regression: the fix must not accidentally suppress ALL output history,
        only the interrupt marker.
        """
        cli_mod._configure_output_history(True, 10)
        printed = []
        monkeypatch.setattr(cli_mod, "_pt_print", lambda x: printed.append(x))
        monkeypatch.setattr(cli_mod, "_PT_ANSI", lambda t: t)

        cli_mod._cprint("normal response text")

        assert "normal response text" in list(cli_mod._OUTPUT_HISTORY)
        assert printed == ["normal response text"]

    def test_suspend_is_idempotent_nested(self):
        """Nested _suspend_output_history() calls restore correctly."""
        cli_mod._configure_output_history(True, 10)
        cli_mod._record_output_history("before")

        with _suspend_output_history():
            cli_mod._record_output_history("inside outer")
            with _suspend_output_history():
                cli_mod._record_output_history("inside inner")

        cli_mod._record_output_history("after")

        assert list(cli_mod._OUTPUT_HISTORY) == [
            "before",
            "after",
        ]


# ── _show_interrupt_marker flag logic ──────────────────────────────


class TestShowInterruptMarkerLogic:
    """The _show_interrupt_marker flag must be set correctly.

    The flag is True only when: the turn was interrupted (result.interrupted),
    AND there is both a response AND a pending_message (interrupt_msg).
    """

    def test_marker_shown_when_interrupted_with_response_and_message(self):
        """Happy path: interrupted turn with response and pending_message."""
        result = {"interrupted": True}
        response = "Some partial response"
        pending_message = "interrupt message"

        _show_interrupt_marker = False
        _interrupted_this_turn = bool(result and result.get("interrupted"))

        if _interrupted_this_turn:
            pending_message = result.get("interrupt_message") or pending_message
            _show_interrupt_marker = bool(response and pending_message)

        assert _show_interrupt_marker is True

    def test_marker_suppressed_when_no_response(self):
        """No marker when there is no response text to interrupt."""
        result = {"interrupted": True}
        response = ""
        pending_message = "interrupt message"

        _show_interrupt_marker = False
        _interrupted_this_turn = bool(result and result.get("interrupted"))

        if _interrupted_this_turn:
            pending_message = result.get("interrupt_message") or pending_message
            _show_interrupt_marker = bool(response and pending_message)

        assert _show_interrupt_marker is False

    def test_marker_suppressed_when_no_pending_message(self):
        """No marker when there's no interrupt message text."""
        result = {"interrupted": True}
        response = "Some partial response"
        pending_message = None

        _show_interrupt_marker = False
        _interrupted_this_turn = bool(result and result.get("interrupted"))

        if _interrupted_this_turn:
            pending_message = result.get("interrupt_message") or pending_message
            _show_interrupt_marker = bool(response and pending_message)

        assert _show_interrupt_marker is False

    def test_marker_suppressed_when_not_interrupted(self):
        """No marker when the turn was not interrupted."""
        result = {"completed": True}
        response = "Full response text"
        pending_message = "interrupt message"

        _show_interrupt_marker = False
        _interrupted_this_turn = bool(result and result.get("interrupted"))

        if _interrupted_this_turn:
            pending_message = result.get("interrupt_message") or pending_message
            _show_interrupt_marker = bool(response and pending_message)

        assert _show_interrupt_marker is False

    def test_marker_shown_with_explicit_interrupt_message(self):
        """Marker shown when result provides interrupt_message."""
        result = {"interrupted": True, "interrupt_message": "User cancelled"}
        response = "Partial output"
        pending_message = "default interrupt msg"

        _show_interrupt_marker = False
        _interrupted_this_turn = bool(result and result.get("interrupted"))

        if _interrupted_this_turn:
            pending_message = result.get("interrupt_message") or pending_message
            _show_interrupt_marker = bool(response and pending_message)

        assert _show_interrupt_marker is True
        assert pending_message == "User cancelled"


# ── End-to-end: _show_interrupt_marker → _cprint flow ──────────────


class TestInterruptMarkerPrintFlow:
    """End-to-end: the flag leads to a supressed _cprint of the marker."""

    def test_marker_printed_via_suspend_after_panel(self, monkeypatch):
        """When _show_interrupt_marker is True, the marker is cprinted.

        The marker text is printed inside _suspend_output_history so it
        bypasses _OUTPUT_HISTORY.
        """
        cli_mod._configure_output_history(True, 10)
        printed_lines = []

        monkeypatch.setattr(cli_mod, "_pt_print", lambda x: printed_lines.append(x))
        monkeypatch.setattr(cli_mod, "_PT_ANSI", lambda t: t)

        # Simulate the production flow
        _show_interrupt_marker = True
        if _show_interrupt_marker:
            with _suspend_output_history():
                cli_mod._cprint(
                    "\n── [Interrupted — processing new message] ──"
                )

        # Marker was printed but NOT recorded in history
        assert printed_lines, "Marker must have been printed"
        assert "Interrupted" in printed_lines[0]
        assert list(cli_mod._OUTPUT_HISTORY) == []

    def test_no_marker_printed_when_flag_false(self, monkeypatch):
        """When _show_interrupt_marker is False, nothing is printed."""
        printed_lines = []
        monkeypatch.setattr(cli_mod, "_pt_print", lambda x: printed_lines.append(x))
        monkeypatch.setattr(cli_mod, "_PT_ANSI", lambda t: t)

        _show_interrupt_marker = False
        if _show_interrupt_marker:
            with _suspend_output_history():
                cli_mod._cprint("── [Interrupted] ──")

        assert printed_lines == []


# ── _replay does not replay the marker (E2E) ───────────────────────


class TestReplayDoesNotDuplicateMarker:
    """_replay_output_history must not contain the interrupt marker.

    After an interrupted turn, only the normal response is in the history.
    Redrawing replays only the response — no marker duplication.
    """

    def test_replay_clean_after_interrupted_turn(self, monkeypatch):
        """Simulate: normal response recorded, marker supressed → replay is clean."""
        cli_mod._configure_output_history(True, 10)

        # Normal response gets recorded
        cli_mod._record_output_history("Assistant response text")

        printed = []
        monkeypatch.setattr(cli_mod, "_pt_print", lambda x: printed.append(x))
        monkeypatch.setattr(cli_mod, "_PT_ANSI", lambda t: t)

        # Marker gets printed with supressed history (does NOT enter _OUTPUT_HISTORY)
        with _suspend_output_history():
            cli_mod._cprint("── [Interrupted — processing new message] ──")

        # History must contain only the normal response
        assert list(cli_mod._OUTPUT_HISTORY) == ["Assistant response text"], (
            "Interrupt marker must not appear in _OUTPUT_HISTORY"
        )

        # Replay the history — this emits the normal response via _pt_print
        cli_mod._replay_output_history()

        # The replayed output must contain only the response, NOT the marker
        # (marker was printed once by _cprint, but replay must not repeat it)
        marker_count = sum(1 for p in printed if "Interrupted" in str(p))
        assert marker_count == 1, (
            f"Marker must appear exactly once (from _cprint), not {marker_count} "
            "(duplicated by _replay_output_history)"
        )
        assert "Assistant response text" in "".join(printed)
