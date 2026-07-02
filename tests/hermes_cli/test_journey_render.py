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


# ---------------------------------------------------------------------------
# --reveal parsing contract
# ---------------------------------------------------------------------------

def _reveal_reaching_renderer(argv: list[str], monkeypatch) -> float:
    """Run the real ``args.func`` path and report the ``reveal`` value that
    reaches the frame renderer."""
    import hermes_cli.journey as journey
    from rich.text import Text

    seen: dict[str, float] = {}

    def fake_frame(payload, *, cols, rows, reveal, color):
        seen["reveal"] = reveal
        return Text("frame")

    monkeypatch.setattr(journey, "_frame_renderable", fake_frame)
    # A non-empty payload so _cmd_show reaches the render call.
    monkeypatch.setattr(journey, "_build_payload", lambda: {"nodes": [{"kind": "skill"}]})

    parser = argparse.ArgumentParser(add_help=False)
    journey.register_cli(parser)
    args = parser.parse_args(argv)

    with contextlib.redirect_stdout(io.StringIO()):
        args.func(args)
    return seen["reveal"]


def test_reveal_zero_renders_oldest_frame(monkeypatch):
    # "0=oldest" per the --reveal help text; 0.0 is falsy and must not be
    # swallowed into the fully-revealed default.
    assert _reveal_reaching_renderer(["--reveal", "0"], monkeypatch) == 0.0


def test_reveal_defaults_to_fully_revealed(monkeypatch):
    assert _reveal_reaching_renderer([], monkeypatch) == 1.0


def test_reveal_fraction_passes_through(monkeypatch):
    assert _reveal_reaching_renderer(["--reveal", "0.25"], monkeypatch) == 0.25
