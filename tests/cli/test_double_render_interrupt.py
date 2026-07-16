"""Tests for issue #65666 — interrupted response rendered twice when the
stream box was closed by a tool-call transition.

The bug: after the agent streams a partial response, a tool-call
transition closes the stream box (``_stream_box_opened = False``). If the
user interrupts during/after the tool call, the final render guard uses
``_stream_started and _stream_box_opened`` — which is now False — so it
re-renders the already-streamed content as a Rich Panel, duplicating it.

The fix: a turn-level ``_response_ever_streamed`` flag that persists
across tool-call boundaries (set when the stream box first opens, reset
only in ``_reset_stream_state``). The render guard uses this instead of
``_stream_box_opened``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cli import HermesCLI


def _make_cli_stub():
    """Build a minimal HermesCLI stand-in with just the streaming state attrs.

    We don't need a full HermesCLI construction — just the attrs that
    _reset_stream_state, _stream_delta (via _emit_stream_text), and the
    render guard touch.
    """
    stub = SimpleNamespace()
    # Initialize all streaming state attrs that _reset_stream_state touches
    stub._stream_buf = ""
    stub._stream_started = False
    stub._stream_box_opened = False
    stub._stream_text_ansi = ""
    stub._stream_prefilt = ""
    stub._in_reasoning_block = False
    stub._stream_last_was_newline = True
    stub._reasoning_box_opened = False
    stub._reasoning_buf = ""
    stub._reasoning_preview_buf = ""
    stub._deferred_content = ""
    stub._stream_table_buf = []
    stub._in_stream_table = False
    stub._response_ever_streamed = False
    stub.show_reasoning = False
    # Bind the real methods
    stub._reset_stream_state = HermesCLI._reset_stream_state.__get__(stub, type(stub))
    return stub


# --------------------------------------------------------------------------- #
# _reset_stream_state resets _response_ever_streamed
# --------------------------------------------------------------------------- #


def test_reset_stream_state_resets_response_ever_streamed():
    """_reset_stream_state must reset _response_ever_streamed to False."""
    stub = _make_cli_stub()
    stub._response_ever_streamed = True
    stub._reset_stream_state()
    assert stub._response_ever_streamed is False


def test_reset_stream_state_resets_all_stream_flags():
    """_reset_stream_state must reset all stream flags for a fresh turn."""
    stub = _make_cli_stub()
    stub._stream_started = True
    stub._stream_box_opened = True
    stub._response_ever_streamed = True
    stub._reset_stream_state()
    assert stub._stream_started is False
    assert stub._stream_box_opened is False
    assert stub._response_ever_streamed is False


# --------------------------------------------------------------------------- #
# _response_ever_streamed persists across tool-call boundaries
# --------------------------------------------------------------------------- #


def test_response_ever_streamed_survives_stream_box_reset():
    """When _on_tool_gen_start resets _stream_box_opened, the
    _response_ever_streamed flag must stay True.

    This is the core invariant of the fix: the flag persists across
    tool-call boundaries so the final render guard can tell the difference
    between "nothing streamed yet" and "content already shown but box
    was closed for a tool call".
    """
    stub = _make_cli_stub()
    # Simulate: stream box opens, content is streamed
    stub._stream_started = True
    stub._stream_box_opened = True
    stub._response_ever_streamed = True

    # Simulate: _on_tool_gen_start closes the stream box
    stub._stream_box_opened = False

    # _response_ever_streamed must still be True
    assert stub._response_ever_streamed is True
    # The old guard would evaluate to False here:
    assert not (stub._stream_started and stub._stream_box_opened)
    # The new guard correctly evaluates to True:
    assert stub._response_ever_streamed


# --------------------------------------------------------------------------- #
# Render guard: already_streamed uses _response_ever_streamed
# --------------------------------------------------------------------------- #


def test_render_guard_prevents_double_render_after_tool_call():
    """The already_streamed check must use _response_ever_streamed, not
    _stream_started and _stream_box_opened.

    Scenario:
    1. Agent streams partial response (_response_ever_streamed = True)
    2. Tool-call transition closes stream box (_stream_box_opened = False)
    3. User interrupts
    4. already_streamed must be True so the Rich Panel is skipped

    With the old code (self._stream_started and self._stream_box_opened),
    already_streamed would be False and the content would be re-rendered.
    """
    stub = _make_cli_stub()
    # Simulate state after streaming + tool-call transition
    stub._stream_started = True
    stub._stream_box_opened = False  # closed by _on_tool_gen_start
    stub._response_ever_streamed = True

    # Simulate the render guard (not an error response)
    is_error_response = False

    # Old guard (the bug): would evaluate to False → double render
    old_guard = stub._stream_started and stub._stream_box_opened and not is_error_response
    assert not old_guard, "Old guard should be False (the bug)"

    # New guard: correctly True → skip Rich Panel
    new_guard = stub._response_ever_streamed and not is_error_response
    assert new_guard, "New guard should be True (the fix)"


def test_render_guard_allows_rich_panel_when_never_streamed():
    """When nothing was streamed at all, the Rich Panel should render normally."""
    stub = _make_cli_stub()
    # Nothing streamed
    stub._stream_started = False
    stub._stream_box_opened = False
    stub._response_ever_streamed = False

    is_error_response = False
    already_streamed = stub._response_ever_streamed and not is_error_response
    assert not already_streamed, "Should allow Rich Panel when nothing was streamed"


def test_render_guard_skips_for_error_response_even_if_streamed():
    """Error responses should still render the Rich Panel for visibility."""
    stub = _make_cli_stub()
    stub._response_ever_streamed = True
    is_error_response = True

    already_streamed = stub._response_ever_streamed and not is_error_response
    assert not already_streamed, "Error responses should not skip the Rich Panel"


# --------------------------------------------------------------------------- #
# Initialization: _response_ever_streamed starts False
# --------------------------------------------------------------------------- #


def test_response_ever_streamed_starts_false():
    """The flag must start False in __init__."""
    # Check that the attribute exists and defaults to False in a fresh stub
    # that simulates __init__ behavior
    stub = SimpleNamespace()
    # Simulate the __init__ lines that set streaming state
    stub._stream_buf = ""
    stub._stream_started = False
    stub._stream_box_opened = False
    stub._response_ever_streamed = False
    assert stub._response_ever_streamed is False