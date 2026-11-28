#!/usr/bin/env python3
"""Structural QA for a .pptx (any language; tuned for Persian RTL decks).

Checks: Latin digits, shapes outside margins, text-shape overlaps,
estimated text overflow (PIL font metrics). Map-label sized boxes are allowed
to sit close; thresholds below are conservative on purpose so false positives
get eyeballed. Prints JSON-style findings and exits non-zero on problems.

  python verify_deck.py deck.pptx [--margin 0.45] [--font-regular R.ttf]
"""
import argparse
import re
import sys
from pathlib import Path

from pptx import Presentation

EMU_IN = 914400.0
DECOR_NAMES = {"pattern", "frame", "frame2", "footline", "spine", "picture"}

FONT_SEARCH = [
    Path.home() / ".local/share/fonts",
    Path.home() / ".fonts",
    Path("/usr/share/fonts"),
    Path("/Library/Fonts"),
]


def _find_font(filename):
    for base in FONT_SEARCH:
        hit = next(base.rglob(filename), None) if base.exists() else None
        if hit:
            return str(hit)
    return None


def main(argv=None):
    if hasattr(sys.stdout, "re.configure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pptx")
    ap.add_argument("--margin", type=float, default=0.45)
    ap.add_argument("--font-regular", default=None,
                    help="TTF for plain-run width estimates; auto-discovered (Vazirmatn) if omitted")
    ap.add_argument("--font-bold", default=None)
    args = ap.parse_args(argv)

    args.font_regular = args.font_regular or _find_font("Vazirmatn-Regular.ttf")
    args.font_bold = args.font_bold or args.font_regular or _find_font("Vazirmatn-Bold.ttf")
    overflow_ok = False
    ImageFont = None
    try:
        from PIL import ImageFont
        overflow_ok = True
    except ImportError:
        overflow_ok = False

    prs = Presentation(args.pptx)
    W = prs.slide_width / EMU_IN
    H = prs.slide_height / EMU_IN
    problems = []
    latin = re.compile(r"[0-9]")

    for si, slide in enumerate(prs.slides, start=1):
        # digits + margins
        for sh in slide.shapes:
            if sh.has_text_frame:
                for p in sh.text_frame.paragraphs:
                    txt = "".join(r.text for r in p.runs)
                    if txt.strip() and latin.search(txt):
                        problems.append(f"S{si}: Latin digits: {txt[:50]!r}")
            if getattr(sh, "has_table", False) and sh.has_table:
                for row in sh.table.rows:
                    for cell in row.cells:
                        if latin.search(cell.text):
                            problems.append(f"S{si}: Latin digits in table: {cell.text[:40]!r}")
            if sh.left is None:
                continue
            l, t = sh.left / EMU_IN, sh.top / EMU_IN
            r, b = l + sh.width / EMU_IN, t + sh.height / EMU_IN
            name = sh.name or ""
            if name in DECOR_NAMES:
                continue
            if l < -0.001 or t < -0.001 or r > W + 0.001 or b > H + 0.001:
                problems.append(f"S{si}: '{name}' off-canvas ({l:.2f},{t:.2f})-({r:.2f},{b:.2f})")
            elif l < args.margin or t < 0.25 or r > W - args.margin or b > H - 0.05:
                if getattr(sh, "has_table", False) and sh.has_table and l >= 0.6 and r <= W - 0.6:
                    continue
                problems.append(f"S{si}: '{name}' over margin ({l:.2f},{t:.2f})-({r:.2f},{b:.2f})")

        # estimated overflow (needs Pillow + a font file)
        if overflow_ok and args.font_regular:
            for sh in slide.shapes:
                if not sh.has_text_frame or sh.name in DECOR_NAMES:
                    continue
                bw = sh.width / EMU_IN - 0.1
                bh = sh.height / EMU_IN
                need = 0.0
                for p in sh.text_frame.paragraphs:
                    txt = "".join(r.text for r in p.runs)
                    if not txt.strip():
                        continue
                    size, bold = None, False
                    for rr in p.runs:
                        if rr.font.size:
                            size, bold = rr.font.size.pt, bool(rr.font.bold)
                        break
                    if not size:
                        continue
                    font = ImageFont.truetype(args.font_bold if bold else args.font_regular,
                                              int(size * 4))
                    ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.0
                    lines, cur = 1, ""
                    for wd in txt.split():
                        trial = (cur + " " + wd).strip()
                        if font.getlength(trial) / 4 / 72.0 > bw:
                            lines += 1
                            cur = wd
                        else:
                            cur = trial
                    need += lines * size / 72.0 * (ls or 1.0)
                    need += (p.space_after.pt if p.space_after else 0) / 72.0
                if need > bh * 1.12 + 0.02:
                    problems.append(f"S{si}: overflow? '{sh.name}': "
                                    f"need {need:.2f} > box {bh:.2f}in")

        # text box overlaps
        boxes = []
        for sh in slide.shapes:
            if not sh.has_text_frame or sh.name in DECOR_NAMES:
                continue
            if not sh.text_frame.text.strip():
                continue
            l, t = sh.left / EMU_IN, sh.top / EMU_IN
            r, b = l + sh.width / EMU_IN, t + sh.height / EMU_IN
            boxes.append((l, t, r, b, sh.name,
                          sh.text_frame.text[:20].replace("\n", "/")))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, c = boxes[i], boxes[j]
                ox = min(a[2], c[2]) - max(a[0], c[0])
                oy = min(a[3], c[3]) - max(a[1], c[1])
                if ox > 0.35 and oy > 0.18:
                    problems.append(f"S{si}: overlap '{a[4]}'/{a[5]!r} vs '{c[4]}'/{c[5]!r}")

    print("PROBLEMS:" if problems else "NO PROBLEMS FOUND")
    for pr in problems:
        print(" -", pr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
