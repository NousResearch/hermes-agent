"""
Regression tests for issue #60543 - /steer leftover path races the
drain hook and lands in _pending_input as raw text instead of the
[OUT-OF-BAND USER MESSAGE] marker.

The bug: when a /steer arrives between the tool batch drain point
and the next API call, _apply_pending_steer_to_tool_results() drains
_pending_steer *before* the steer has actually been stored, so the
drain returns None and the steer remains in _pending_steer. The next
turn's leftover handler in cli.py:12736-12740 then puts the raw steer
text into _pending_input — bypassing the format_steer_marker() wrap
that turns it into a recognizable OUT-OF-BAND USER MESSAGE marker.

The fix: in the leftover handler (cli.py:12736-12740), wrap
_leftover_steer with format_steer_marker() before feeding
self._pending_input.put(). The handler is extracted into
``HermesCLI._deliver_leftover_steer`` so this regression test (and any
future ones) can exercise it directly without driving the entire
chat() pipeline.

These tests build a minimal HermesCLI via __new__ and exercise the
leftover-steer helper directly.
"""

from __future__ import annotations

import queue

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_voice_cli():
    """Build a minimal HermesCLI with only the attributes the leftover
    handler touches: _pending_input (queue.Queue). Mirrors the pattern
    in tests/tools/test_voice_cli_integration.py.
    """
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    return cli


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLeftoverSteer:
    """The leftover handler at cli.py:12736-12740 (extracted to
    ``HermesCLI._deliver_leftover_steer`` by the #60543 fix) must wrap
    the steer text with ``format_steer_marker()`` before queueing it,
    so the model receives it as OUT-OF-BAND USER MESSAGE on the next
    turn.
    """

    def test_leftover_ster_handler_exists(self):
        """After the #60543 fix, HermesCLI has a
        ``_deliver_leftover_steer`` method that handles the leftover
        queue path. Before the fix, this method does not exist.
        """
        from cli import HermesCLI
        assert hasattr(HermesCLI, "_deliver_leftover_steer"), (
            "HermesCLI._deliver_leftover_steer is missing; the #60543 "
            "fix extracts the leftover-steer logic into a testable helper."
        )
        assert callable(getattr(HermesCLI, "_deliver_leftover_steer"))

    def test_leftover_steer_is_wrapped_with_format_steer_marker(self):
        """The regression: raw steer text was put into _pending_input
        without the format_steer_marker() wrap, so the next turn sees
        raw steer text instead of the OUT-OF-BAND USER MESSAGE marker.

        After the fix: the queued value equals format_steer_marker(text).
        """
        from agent.prompt_builder import format_steer_marker

        cli = _make_voice_cli()
        steer_text = "check the logs again please"
        result = {"pending_steer": steer_text}

        cli._deliver_leftover_steer(result)

        queued = cli._pending_input.get_nowait()
        expected = format_steer_marker(steer_text)
        assert queued == expected, (
            f"leftover /steer was queued without format_steer_marker wrap; "
            f"got {queued!r}, expected {expected!r}. Issue #60543: the next "
            f"turn receives raw steer text instead of the OUT-OF-BAND USER "
            f"MESSAGE marker."
        )

    def test_leftover_steer_with_multiline_text_is_wrapped(self):
        """The wrap must work for multiline steer text too — the format
        function preserves newlines (see format_steer_marker in
        agent/prompt_builder.py).
        """
        from agent.prompt_builder import format_steer_marker

        cli = _make_voice_cli()
        steer_text = "first line\nsecond line\nthird line"
        result = {"pending_steer": steer_text}

        cli._deliver_leftover_steer(result)

        queued = cli._pending_input.get_nowait()
        assert queued == format_steer_marker(steer_text)
        # The wrap must surround the steer text with markers (length
        # strictly greater than the bare text plus reasonable slack).
        assert len(queued) > len(steer_text) + 10, (
            f"queued value {queued!r} does not appear to be wrapped; "
            f"expected format_steer_marker output with surrounding markers."
        )

    def test_leftover_steer_only_fires_when_pending_steer_set(self):
        """Regression guard: when result has no pending_steer (the
        common case), the leftover handler must NOT put anything into
        the queue. A None result must also be safe.
        """
        cli = _make_voice_cli()

        for r in (None, {}, {"pending_steer": None}, {"pending_steer": ""}):
            cli._deliver_leftover_steer(r)
            assert cli._pending_input.empty(), (
                f"leftover handler put something when it shouldn't have; "
                f"input was {r!r}"
            )