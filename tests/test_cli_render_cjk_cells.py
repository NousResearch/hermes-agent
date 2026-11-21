"""T2.1: banner/preview padding must be measured in terminal cells, not chars.

RED premise: ``_build_compact_banner`` pads rows with ``line[:w].ljust(w)`` (char-based),
so a CJK tagline pads 45 cells to 80 chars and pushes box borders 11 cells past the
terminal edge (probe-render-baseline.json: max_row_cells 95 at width 84).
"""
import unicodedata

from hermes_cli.cli_render import _clip_cells, _pad_cells


def _cells(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def test_ja_tagline_pad_hits_exact_width():
    tag = "☤ NOUS HERMES - AI エージェントフレームワーク"
    assert _cells(tag) == 45
    assert _cells(_pad_cells(tag, 84)) == 84


def test_ja_tagline_clip_is_exact():
    tag = "☤ NOUS HERMES - AI エージェントフレームワーク"
    assert _cells(_clip_cells(tag, 20)) == 20


def test_emoji_pad_hits_exact_width():
    label = "🚀 実行中 — writing 45KB"
    assert _cells(_pad_cells(label, 40)) == 40


def test_stream_tool_label_clip_never_exceeds_budget():
    # A CJK preview longer than the budget must land on exactly the budget, char-slicing would overshoot.
    from hermes_cli.cli_render import _clip_cells
    label = "実行: git push origin main — デバイス変更 書き込み 45KB 🚀 長いう long tail suffix"
    assert _cells(_clip_cells(label, 30)) == 30
