"""``hermes journey`` — what Hermes has learned, on a timeline.

A terminal-native rendition of the desktop Star Map / Memory Graph: a horizontal
timeline bar chart of learned skills and memories over time (oldest at top,
newest at bottom) plus the playable constellation scrubber. Graph assembly,
layout, and the (ported-from-desktop) palette all live in
``agent.learning_graph`` / ``agent.learning_graph_render`` so the CLI, the TUI
``/journey`` overlay, and the desktop panel draw the same data.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from functools import lru_cache
from typing import Any, Optional

_TITLE_COLOR = "#E8C463"


def _build_payload() -> dict[str, Any]:
    from agent.learning_graph import build_learning_graph

    return build_learning_graph()


@lru_cache(maxsize=1)
def _primary_hex() -> str:
    """The active skin's primary color (mirrors the TUI theme primary)."""
    try:
        from hermes_cli.skin_engine import get_active_skin

        skin = get_active_skin()
        return skin.get_color("ui_primary", "") or skin.get_color("banner_title", "#FFD700")
    except Exception:
        return "#FFD700"


@lru_cache(maxsize=1)
def _palette() -> dict[str, str]:
    from agent.learning_graph_render import derive_palette

    return derive_palette(_primary_hex(), dark=True)


def _fade(base: Optional[str], alpha: float) -> Optional[str]:
    from agent.learning_graph_render import hex_to_rgb, mix_rgb, rgb_to_hex

    if not base:
        return None
    if alpha >= 0.999:
        return base
    return rgb_to_hex(mix_rgb(hex_to_rgb(_palette()["bg"]), hex_to_rgb(base), alpha))


def _resolve(style: str, alpha: float) -> Optional[str]:
    """Fade the style's base ink toward the background by ``alpha`` (rgba-over-bg)."""
    return _fade(_palette().get(style), alpha)


def _row_to_text(row: list, color: bool):
    from rich.text import Text

    text = Text()
    for run in row:
        chunk = run[0]
        style = run[1]
        alpha = run[2] if len(run) > 2 else 1.0
        override = run[3] if len(run) > 3 else None
        if not color:
            text.append(chunk)
        elif override:
            text.append(chunk, style=_fade(override, alpha))
        else:
            text.append(chunk, style=_resolve(style, alpha))
    return text


def _term_size(width: Optional[int], height: Optional[int]) -> tuple[int, int]:
    size = shutil.get_terminal_size((90, 30))
    return max(40, width or size.columns), max(10, height or size.lines)


def _frame_renderable(payload, *, cols, rows, reveal, color):
    from rich.console import Group
    from rich.text import Text

    from agent import learning_graph_render as render

    legend = render.build_legend(payload)
    categories = render.category_legend(payload)
    summary = render.build_summary(payload)
    axis = render.axis_labels(payload)
    # Lines are pad_left(2), so content must fit in cols-2.
    inner = max(24, cols - 2)
    # Reserve rows for title/legend/blank/axis/footer/labels + summary; field gets rest.
    field_rows = max(6, rows - 10 - len(summary))
    frame = render.render_graph(payload, cols=inner, rows=field_rows, reveal=reveal)
    count = len(payload.get("nodes", []))

    parts: list[Any] = []

    title = Text()
    title.append("✦ Journey ", style=f"bold {_TITLE_COLOR}" if color else None)
    title.append("· learned skills & memories over time", style="grey62" if color else None)
    parts.append(title)

    legend_line = Text("  ")
    for i, item in enumerate(legend):
        if i:
            legend_line.append("   ")
        legend_line.append(item["glyph"] + " ", style=_resolve(item["style"], 1.0) if color else None)
        legend_line.append(item["label"], style="grey62" if color else None)
    parts.append(legend_line)

    if categories:
        cat_line = Text("  ")
        for i, item in enumerate(categories):
            if i:
                cat_line.append("  ")
            cat_line.append(item["glyph"] + " ", style=_fade(item.get("color"), 1.0) if color else None)
            cat_line.append(item["label"], style="grey54" if color else None)
        parts.append(cat_line)

    parts.append(Text(""))

    for grow in frame["grid"]:
        line = _row_to_text(grow, color)
        line.pad_left(2)
        parts.append(line)

    # Date axis under the field (oldest → now), with the playhead date centered.
    axis_line = Text("  ")
    axis_line.append(axis["start"], style="grey54" if color else None)
    gap = max(1, inner - len(axis["start"]) - len(axis["end"]))
    axis_line.append(" " * gap)
    axis_line.append(axis["end"], style="grey54" if color else None)
    parts.append(axis_line)

    pct = int(round(reveal * 100))
    foot = Text("  ")
    foot.append("◷ ", style="grey54" if color else None)
    foot.append(frame["date"] or "—", style=_TITLE_COLOR if color else None)
    foot.append(f"   {frame['visible']}/{count} revealed · {pct}%", style="grey54" if color else None)
    parts.append(foot)

    labels = frame.get("labels", [])
    if labels:
        parts.append(Text(""))
        heading = Text("  charted signals", style="grey62" if color else None)
        parts.append(heading)

        def label_row(item) -> Text:
            row = Text("  ")
            row.append(f"{item['key']} ", style="grey70" if color else None)
            row.append(f"{item['glyph']} ", style=_resolve(item["style"], float(item.get("alpha", 1.0))) if color else None)
            row.append(str(item["label"]), style=_resolve(item["style"], float(item.get("alpha", 1.0))) if color else None)
            meta = str(item["meta"])
            row.append(f"  {meta if len(meta) <= 32 else meta[:29] + '…'}", style="grey54" if color else None)
            return row

        for item in labels[:6]:
            row = label_row(item)
            parts.append(row)

    for line_text in summary:
        parts.append(Text("  " + line_text, style="grey62" if color else None))

    return Group(*parts)


def _cmd_show(args: argparse.Namespace) -> int:
    from rich.console import Console

    if getattr(args, "json", False):
        import json

        Console(no_color=bool(getattr(args, "no_color", False))).print_json(json.dumps(_build_payload()))
        return 0

    payload = _build_payload()
    color = not bool(getattr(args, "no_color", False))
    cols, rows = _term_size(getattr(args, "width", None), getattr(args, "height", None))
    console = Console(no_color=not color, width=cols)

    if not payload.get("nodes"):
        console.print(
            "[grey62]No learning yet — use Hermes a while and your learned skills and "
            "memories will start mapping out here.[/grey62]"
        )
        return 0

    if getattr(args, "play", False):
        return _play(console, payload, cols=cols, rows=rows, color=color, fps=getattr(args, "fps", 12))

    reveal = _clamp(float(getattr(args, "reveal", 1.0) or 1.0), 0.0, 1.0)
    console.print(_frame_renderable(payload, cols=cols, rows=rows, reveal=reveal, color=color))
    return 0


def _play(console, payload, *, cols, rows, color, fps: int) -> int:
    from rich.live import Live

    frames = 42
    delay = 1.0 / max(1, min(60, fps))
    try:
        with Live(console=console, refresh_per_second=max(1, fps), screen=False) as live:
            for i in range(frames):
                reveal = i / (frames - 1)
                live.update(_frame_renderable(payload, cols=cols, rows=rows, reveal=reveal, color=color))
                time.sleep(delay)
            live.update(_frame_renderable(payload, cols=cols, rows=rows, reveal=1.0, color=color))
    except KeyboardInterrupt:
        console.print("[grey54]interrupted[/grey54]")
        return 130
    return 0


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def register_cli(parent: argparse.ArgumentParser) -> None:
    parent.add_argument(
        "--reveal",
        type=float,
        default=1.0,
        metavar="0..1",
        help="Render the timeline built up to this point (0=oldest, 1=now).",
    )
    parent.add_argument("--play", action="store_true", help="Animate the build-up over time (Ctrl-C to stop).")
    parent.add_argument("--fps", type=int, default=12, help="Animation frames per second for --play (default 12).")
    parent.add_argument("--width", type=int, default=None, help="Override render width in columns.")
    parent.add_argument("--height", type=int, default=None, help="Override render height in rows.")
    parent.add_argument("--no-color", action="store_true", help="Disable color output.")
    parent.add_argument("--json", action="store_true", help="Print the raw graph payload as JSON and exit.")
    parent.set_defaults(func=_cmd_show)


def cmd_journey(args: argparse.Namespace) -> int:
    return _cmd_show(args)


if __name__ == "__main__":
    _p = argparse.ArgumentParser(prog="hermes journey")
    register_cli(_p)
    _a = _p.parse_args()
    sys.exit(_a.func(_a))
