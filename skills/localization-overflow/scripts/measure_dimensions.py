#!/usr/bin/env python3
"""Measure per-project text-box budgets for safe localization.

The localization plugin counts ROWS, not pixels, and the default window does
not wrap long lines. So the two hard limits are:

  * WIDTH  = the widest ORIGINAL (source) row in a file, in characters.
            The game's author already fitted that many chars on screen, so a
            translated row may not be wider than this. English glyphs are
            narrower than CJK, so this char budget is generous for English but
            is the safe, proven value. Measure it PER FILE (per message group),
            because different windows have different widths.
  * HEIGHT = the number of rows the window shows (the message-box row budget).
            A message is ONE sentence the author split across its rows; a
            translated message must stay within that same row count and never
            repeat a sentence across two rows.

Usage:
  python3 measure_dimensions.py <draft.json>

Output: per-file WIDTH (chars) and HEIGHT (rows), plus global maxes.
"""
import json
import sys
from collections import defaultdict, Counter

NL = chr(10)      # real newline inside a source string (author's line breaks)


def measure(draft_path):
    dr = json.load(open(draft_path, encoding="utf-8"))
    units = dr["units"]

    rows_by_file = defaultdict(list)   # file -> list of row char-lengths
    rows_by_unit = {}                  # unit_id -> source row count
    for u in units:
        src = u.get("source", "")
        if not isinstance(src, str) or not src:
            continue
        rows_by_unit[u["id"]] = src.count(NL) + 1
        f = u.get("source_file") or u.get("kind") or "unknown"
        for line in src.split(NL):
            rows_by_file[f].append(len(line))

    width = {f: max(lengths) for f, lengths in rows_by_file.items()}
    height = max(rows_by_unit.values()) if rows_by_unit else 0

    print("=== Per-file WIDTH (widest original row, chars) ===")
    for f in sorted(width, key=lambda k: -width[k]):
        print(f"  {width[f]:>4} chars  {f}")
    print()
    print(f"Global WIDTH limit (chars): {max(width.values()) if width else 0}")
    print(f"Global HEIGHT limit (rows): {height}")
    print()

    width_vals = sorted(width.values())
    hvals = sorted(rows_by_unit.values())
    print(f"WIDTH distribution: min={width_vals[0]} median={width_vals[len(width_vals)//2]} max={width_vals[-1]}")
    print(f"HEIGHT distribution: min={hvals[0]} median={hvals[len(hvals)//2]} max={hvals[-1]}")
    print(f"units by source row count: {dict(sorted(Counter(rows_by_unit.values()).items()))}")


if __name__ == "__main__":
    measure(sys.argv[1] if len(sys.argv) > 1 else "draft.json")
