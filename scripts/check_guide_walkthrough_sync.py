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

It also verifies the «راه‌اندازی سریع در ۶ گام» quickstart section
(guide.html ``<section id="quickstart">`` vs PersianGuide's ``QUICKSTART``
array): the step count must match and every TSX ``cmd`` must appear
verbatim, in order, inside its matching guide.html step. Desc prose may be
reworded per surface; commands may not — they are the exact strings a user
copies, so only verbatim presence counts.

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

# Quickstart: guide.html's numbered <li> steps inside <section id="quickstart">
# vs PersianGuide.tsx's QUICKSTART array ({ cmd: "...", desc: "..." },
# one per step — desc may contain escaped quotes, cmd never does).
QUICKSTART_SECTION_RE = re.compile(
    r'<section id="quickstart">(?P<body>.*?)</section>', re.DOTALL
)
QUICKSTART_LI_RE = re.compile(r"<li>(?P<li>.*?)</li>", re.DOTALL)
QUICKSTART_TSX_ARRAY_RE = re.compile(
    r"const QUICKSTART[^=]*=\s*\[(?P<body>.*?)\n\];", re.DOTALL
)
QUICKSTART_TSX_ENTRY_RE = re.compile(
    r"""\{\s*cmd:\s*"(?P<cmd>[^"]+)",\s*desc:\s*"(?P<desc>(?:[^"\\]|\\.)*)",?\s*\}""",
    re.VERBOSE,
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


def extract_html_quickstart_steps() -> list[str]:
    """Normalized text of each numbered step in guide.html's quickstart section."""
    try:
        text = GUIDE_HTML.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"✗ cannot read {GUIDE_HTML}: {exc}")
    section = QUICKSTART_SECTION_RE.search(text)
    if not section:
        return []
    return [_normalize_caption(m.group("li")) for m in QUICKSTART_LI_RE.finditer(section.group("body"))]


def extract_tsx_quickstart_cmds() -> list[str]:
    """The ``cmd`` values of PersianGuide.tsx's QUICKSTART array, in order."""
    try:
        text = PERSIAN_GUIDE.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"✗ cannot read {PERSIAN_GUIDE}: {exc}")
    array = QUICKSTART_TSX_ARRAY_RE.search(text)
    if not array:
        return []
    return [m.group("cmd") for m in QUICKSTART_TSX_ENTRY_RE.finditer(array.group("body"))]


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Fail when PersianGuide.tsx walkthrough captions or quickstart commands drift from guide.html."
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

    # Quickstart commands: TSX cmd must appear verbatim, in order, inside its
    # matching HTML step (HTML steps carry extra <code> probes around the
    # command, so containment — not equality — is the invariant). Desc prose
    # may drift; commands may not.
    html_steps = extract_html_quickstart_steps()
    tsx_cmds = extract_tsx_quickstart_cmds()
    if not html_steps:
        errors.append(f'no <li> steps found in <section id="quickstart"> of {GUIDE_HTML.name}')
    if not tsx_cmds:
        errors.append(f"no QUICKSTART array entries found in {PERSIAN_GUIDE.name}")
    if html_steps and tsx_cmds:
        if len(html_steps) != len(tsx_cmds):
            errors.append(
                f"quickstart step count drifted: guide.html has {len(html_steps)}, "
                f"PersianGuide.tsx has {len(tsx_cmds)}"
            )
        for idx, cmd in enumerate(tsx_cmds):
            if idx < len(html_steps):
                if cmd not in html_steps[idx]:
                    errors.append(
                        f"quickstart step {idx + 1} command drift\n"
                        f"  PersianGuide.tsx cmd: {cmd}\n"
                        f"  guide.html step:      {html_steps[idx]}"
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
        f"✓ in sync: {len(html_figs)} walkthrough figures match between "
        f"guide.html and PersianGuide.tsx"
        + (
            f", quickstart commands match across {len(html_steps)} steps"
            if html_steps and tsx_cmds
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
