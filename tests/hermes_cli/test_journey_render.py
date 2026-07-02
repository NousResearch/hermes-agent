"""Behavior contracts for /journey output routing.

The interactive CLI captures Rich output and re-renders it through
prompt_toolkit, so it needs forced ANSI (``--force-color``); chat surfaces
render plain text, so the default captured path must stay escape-free.
"""

from __future__ import annotations

import argparse
import contextlib
import io


def _capture(argv: list[str], *, force: bool) -> str:
    from hermes_cli.journey import register_cli

    parser = argparse.ArgumentParser(add_help=False)
    register_cli(parser)
    args = parser.parse_args(argv)
    if force:
        args.force_color = True

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        args.func(args)
    return buf.getvalue()


def test_force_color_emits_ansi_for_reemission():
    assert "\x1b[" in _capture([], force=True)
    assert "\x1b[" in _capture(["list"], force=True)


def test_default_capture_is_plain_for_chat_bubbles():
    # Rich auto-detects the StringIO as non-tty → no color, no raw escapes.
    assert "\x1b[" not in _capture([], force=False)
    assert "\x1b[" not in _capture(["list"], force=False)
