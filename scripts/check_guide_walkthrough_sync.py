#!/usr/bin/env python3
"""Sync gate: the in-dashboard Persian guide's walkthrough captions must match guide.html.

The five-step first-run walkthrough exists in two surfaces:

  * guide.html — the standalone Persian guide (``<figcaption>`` elements in
    ``figure.shot`` blocks, ``guide-images/0N-*.png``),
  * web/src/components/PersianGuide.tsx — the same walkthrough rendered on
    the dashboard's Docs page (a ``WALKTHROUGH`` array with
    ``src``/``caption`` fields, served from ``web/public/guide-images/``).

They were added together (1:1 Persian captions), but nothing kept them
synchronized: an editor who rewords one caption — or reorders/retakes a
screenshot — would silently leave the other surface stale. This checker
extracts both caption sequences, normalizes the HTML one (strip tags,
collapse whitespace, unescape entities, drop the Vazirmatn-only BOM-ish
zero-width joiners), and fails when the sequences drift.

Dependency-free (stdlib only, like scripts/check_compat_pointers.py) so a
CI job can run it before any setup step.

Usage:
    python scripts/check_guide_walkthrough_sync.py            # gate mode
    python scripts/check_guide_walkthrough_sync.py --diff     # show unified diff
"""

from __future__ import annotations

import argparse
import difflib
import html as html_mod
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GUIDE_HTML = REPO_ROOT / "guide.html"
PERSIAN_GUIDE = REPO_ROOT / "web" / "src" / "components" / "PersianGuide.tsx"

# ``figure.shot`` blocks pair a guide-images/0N-*.png <img> with the
# <figcaption> we mirror in PersianGuide. The img src is the stable join
# key: both surfaces reference the same file (guide.html from the repo
# root, PersianGuide from web/public/ — the basename is identical).
FIGURE_RE = re.compile(
    r"""<figure[^>]*class="shot"[^>]*>.*?
        <img[^>]*src="(?P<src>[^"]+)".*?
        <figcaption>(?P<caption>.*?)</figcaption>""",
    re.VERBOSE | re.DOTALL,
)

# The WALKTHROUGH entries in PersianGuide.tsx:
#   { src: "guide-images/01-first-launch.png", caption: "۱ — ..." },
ENTRY_RE = re.compile(
    r"""\{\s*src:\s*"(?P<src>[^"]+)",\s*caption:\s*"(?P<caption>[^"]*)",?\s*\}"""
)

# Normalization: the HTML caption carries markup the TSX copy stores as
# plain text (<strong>فارسی</strong> -> فارسی). Collapse runs of whitespace,
# unescape entities, and drop invisible junk (ZWSP, BOM). U+200C (ZWNJ,
# نیم‌فاصله) is deliberately KEPT: it is meaningful Persian orthography, so
# a half-space edit on one side only is real drift the gate must catch.
TAG_RE = re.compile(r"<[^>]+>")
ZW_RE = re.compile(r"[\u200b\ufeff]")


def _normalize_caption(raw: str) -> str:
    """HTML caption -> plain text, whitespace-collapsed, zero-width-stripped."""
    text = TAG_RE.sub("", raw)
    text = html_mod.unescape(text)
    text = ZW_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_tsx_caption(raw: str) -> str:
    """TSX caption -> the same canonical form (zero-width stripped, collapsed)."""
    text = ZW_RE.sub("", raw)
    return re.sub(r"\s+", " ", text).strip()


def extract_html_figures() -> list[tuple[str, str]]:
    """``[(img src, normalized caption)]`` in document order from guide.html."""
    try:
        text = GUIDE_HTML.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"✗ cannot read {GUIDE_HTML}: {exc}")
    figures: list[tuple[str, str]] = []
    for m in FIGURE_RE.finditer(text):
        src = Path(m.group("src")).name
        figures.append((src, _normalize_caption(m.group("caption"))))
    return figures


def extract_tsx_walkthrough() -> list[tuple[str, str]]:
    """``[(img basename, caption)]`` from PersianGuide.tsx's WALKTHROUGH array."""
    try:
        text = PERSIAN_GUIDE.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"✗ cannot read {PERSIAN_GUIDE}: {exc}")
    # Only the const WALKTHROUGH array — not other object literals that
    # happen to carry src/caption keys.
    m = re.search(
        r"const WALKTHROUGH[^=]*=\s*\[(?P<body>.*?)\n\];",
        text,
        re.DOTALL,
    )
    if not m:
        raise SystemExit(f"✗ no WALKTHROUGH array found in {PERSIAN_GUIDE}")
    entries: list[tuple[str, str]] = []
    for e in ENTRY_RE.finditer(m.group("body")):
        src = Path(e.group("src")).name
        entries.append((src, _normalize_tsx_caption(e.group("caption"))))
    return entries


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Fail when PersianGuide.tsx walkthrough captions drift from guide.html figcaptions."
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Print a unified diff of the caption sequences on mismatch.",
    )
    args = parser.parse_args(argv)

    html_figs = extract_html_figures()
    tsx_figs = extract_tsx_walkthrough()

    errors: list[str] = []
    if not html_figs:
        errors.append(f"no <figure class=\"shot\"> captions found in {GUIDE_HTML.name}")
    if not tsx_figs:
        errors.append(f"no WALKTHROUGH entries found in {PERSIAN_GUIDE.name}")
    if errors:
        for err in errors:
            print(f"✗ {err}", file=sys.stderr)
        return 1

    if len(html_figs) != len(tsx_figs):
        errors.append(
            f"figure count drifted: guide.html has {len(html_figs)}, "
            f"PersianGuide.tsx has {len(tsx_figs)}"
        )

    html_map = dict(html_figs)
    tsx_map = dict(tsx_figs)
    only_html = sorted(set(html_map) - set(tsx_map))
    only_tsx = sorted(set(tsx_map) - set(html_map))
    for src in only_html:
        errors.append(f"{src}: figure present in guide.html but missing from PersianGuide.tsx")
    for src in only_tsx:
        errors.append(f"{src}: figure present in PersianGuide.tsx but missing from guide.html")

    order_html = [src for src, _ in html_figs]
    order_tsx = [src for src, _ in tsx_figs]
    if order_html != order_tsx:
        errors.append(
            "figure order drifted: guide.html="
            f"{order_html} vs PersianGuide.tsx={order_tsx}"
        )

    for src in order_html:
        if src not in html_map or src not in tsx_map:
            continue
        if html_map[src] != tsx_map[src]:
            errors.append(
                f"{src}: caption drift\n  guide.html:      {html_map[src]}\n  PersianGuide.tsx: {tsx_map[src]}"
            )

    if errors:
        print(
            f"✗ guide.html and PersianGuide.tsx walkthroughs have drifted "
            f"({len(errors)} problem(s)). Update both surfaces together, "
            f"or run with --diff to see the caption diff.",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        if args.diff:
            html_seq = [f"{src}\n    {cap}" for src, cap in html_figs]
            tsx_seq = [f"{src}\n    {cap}" for src, cap in tsx_figs]
            for line in difflib.unified_diff(
                html_seq, tsx_seq, fromfile="guide.html", tofile="PersianGuide.tsx", lineterm=""
            ):
                print(line)
        return 1

    print(
        f"✓ walkthrough in sync: {len(html_figs)} figures match between "
        f"guide.html and PersianGuide.tsx"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
