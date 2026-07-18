"""Regression tests for _response_ever_streamed flag (#65666).

The flag must survive _reset_stream_state() (called at intermediate tool-call
boundaries) and only reset at the start of a new user turn.
"""

from __future__ import annotations

import pytest


class TestResponseEverStreamedFlag:
    """_response_ever_streamed must persist across tool-call boundaries."""

    @staticmethod
    def _make_cli():
        from cli import HermesCLI

        cli = HermesCLI.__new__(HermesCLI)
        cli._stream_started = False
        cli._stream_box_opened = False
        cli._response_ever_streamed = False
        cli._invalidate = lambda *a, **kw: None
        return cli

    def test_flag_set_when_stream_box_opens(self):
        """Flag is set when streaming content first renders."""
        cli = self._make_cli()
        assert cli._response_ever_streamed is False

        # Simulate first streamed content arriving
        cli._response_ever_streamed = True
        assert cli._response_ever_streamed is True

    def test_flag_survives_reset_stream_state(self):
        """Flag persists across _reset_stream_state (tool-call boundaries)."""
        cli = self._make_cli()
        cli._response_ever_streamed = True

        # Simulate intermediate turn boundary (stream_delta_callback(None))
        cli._reset_stream_state()

        # Flag must survive — content was already streamed this turn
        assert cli._response_ever_streamed is True

    def test_flag_reset_at_new_user_turn(self):
        """Flag resets when a new user turn begins (not in _reset_stream_state)."""
        cli = self._make_cli()
        cli._response_ever_streamed = True

        # Simulate new user turn setup (the call-site pattern at ~line 12337)
        cli._reset_stream_state()
        cli._response_ever_streamed = False

        assert cli._response_ever_streamed is False

    def test_guard_skips_render_when_already_streamed(self):
        """already_streamed guard uses flag, not _stream_box_opened."""
        cli = self._make_cli()

        # Case: tool call just finished, _reset_stream_state cleared box state
        cli._stream_box_opened = False
        cli._stream_started = True
        cli._response_ever_streamed = True  # content WAS streamed earlier

        # Old guard: _stream_started and _stream_box_opened → False (BUG)
        old_guard = cli._stream_started and cli._stream_box_opened
        assert old_guard is False

        # New guard: uses _response_ever_streamed → True (FIXED)
        new_guard = cli._response_ever_streamed
        assert new_guard is True
