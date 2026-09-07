"""Behavior contracts for /journey output routing.

The interactive CLI captures Rich output and re-renders it through
prompt_toolkit, so it needs forced ANSI (``--force-color``); chat surfaces
render plain text, so the default captured path must stay escape-free.
"""

from __future__ import annotations

import argparse
import contextlib
import io


def _relative_luminance(color: str) -> float:
    value = color.lstrip("#")
    channels = [int(value[index : index + 2], 16) / 255 for index in (0, 2, 4)]
    red, green, blue = (
        channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    )
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(foreground: str, background: str) -> float:
    high, low = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)), reverse=True
    )
    return (high + 0.05) / (low + 0.05)


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


def test_charted_signal_labels_clear_readable_contrast_floor(monkeypatch):
    from rich.text import Text

    from hermes_cli import journey

    palette = {
        "bg": "#08080C",
        "dim": "#525255",
        "label": "#A9A9AA",
        "memory": "#FFDC1F",
        "skill": "#043D92",
    }
    monkeypatch.setattr(journey, "_palette", lambda: palette)
    payload = {
        "nodes": [
            {
                "id": "old-skill",
                "label": "old-skill",
                "kind": "skill",
                "timestamp": 1_700_000_000,
                "category": "research",
                "useCount": 0,
            },
            {
                "id": "new-skill",
                "label": "new-skill",
                "kind": "skill",
                "timestamp": 1_800_000_000,
                "category": "research",
                "useCount": 1,
            },
        ],
        "edges": [],
        "clusters": [{"category": "research", "count": 2}],
        "stats": {
            "learned_skills": 2,
            "memory_nodes": 0,
            "related_edges": 0,
            "memory_skill_edges": 0,
        },
    }

    output = journey._frame_renderable(
        payload, cols=72, rows=24, reveal=1.0, color=True
    )
    label_row = next(
        renderable
        for renderable in output.renderables
        if isinstance(renderable, Text) and "old-skill" in renderable.plain
    )
    label_offset = label_row.plain.index("old-skill")
    label_span = next(
        span for span in label_row.spans if span.start <= label_offset < span.end
    )

    assert isinstance(label_span.style, str)
    assert (
        _contrast_ratio(label_span.style, palette["bg"])
        >= journey._CHARTED_SIGNAL_MIN_CONTRAST
    )
