"""Tests for Warp terminal detection used to throttle spinner repaints (#51039).

Warp re-renders the whole active block on each repaint, so the interactive
CLI throttles its work-time chrome invalidation under Warp. The detection
helper keys off ``TERM_PROGRAM=WarpTerminal``.
"""

import os
from unittest.mock import patch

from cli import _is_warp_terminal


def test_warp_terminal_detected():
    with patch.dict(os.environ, {"TERM_PROGRAM": "WarpTerminal"}, clear=False):
        assert _is_warp_terminal() is True


def test_warp_terminal_detected_case_insensitive():
    with patch.dict(os.environ, {"TERM_PROGRAM": "warpterminal"}, clear=False):
        assert _is_warp_terminal() is True


def test_warp_terminal_detected_with_whitespace():
    with patch.dict(os.environ, {"TERM_PROGRAM": "  WarpTerminal  "}, clear=False):
        assert _is_warp_terminal() is True


def test_other_terminal_not_warp():
    with patch.dict(os.environ, {"TERM_PROGRAM": "iTerm.app"}, clear=False):
        assert _is_warp_terminal() is False


def test_unset_term_program_not_warp():
    with patch.dict(os.environ, {}, clear=True):
        assert _is_warp_terminal() is False
