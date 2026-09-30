#!/usr/bin/env python3
"""md2html — deterministic Markdown -> HTML with inlined CSS.

Standing convention: HTML from Markdown is deterministic; an LLM must not
write it. This script renders one or more .md files to self-contained HTML
pages (CSS inlined) using python-markdown + Pygments.

Usage:
    python3 md2html.py FILE.md [FILE2.md ...] [--out PATH] [--css PATH] [--stdout]

Writes a sibling .html next to each input (unless --out / --stdout).
The default stylesheet ships at templates/style.css beside this script.
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

HERE = Path(__file__).resolve().parent
DEFAULT_CSS = HERE.parent / "templates" / "style.css"

MD_EXTENSIONS = ["extra", "toc", "sane_lists", "codehilite", "admonition"]
MD_CONFIG = {
    "codehilite": {"guess_lang": False, "noclasses": False},
    "toc": {"permalink": False},
}

# Indicator convention: bold status words become colored spans, so a table
# cell like **green** renders as a colored indicator instead of plain bold.
STATUS_WORDS = {
    "green": "st-ok",
    "amber": "st-warn",
    "yellow": "st-warn",
    "red": "st-bad",
    "gray": "st-neutral",
    "grey": "st-neutral",
    "silver": "st-neutral",
}


def status_postpass(html_text: str) -> str:
    for word, cls in STATUS_WORDS.items():
        html_text = html_text.replace(
            f"<strong>{word.capitalize()}</strong>",
            f'<span class="{cls}">{word.capitalize()}</span>',
        )
        html_text = html_text.replace(
            f"<strong>{word.upper()}</strong>",
            f'<span class="{cls}">{word.upper()}</span>',
        )
    return html_text


def extract_title(md_text: str, fallback: str) -> str:
    for line in md_text.splitlines():
        m = re.match(r"^#\s+(.+?)\s*#*\s*$", line.strip())
        if m:
            return m.group(1)
    return fallback


def render(md_text: str, title: str, css_text: str, py_css: str, date: str) -> str:
    body = markdown.markdown(md_text, extensions=MD_EXTENSIONS, extension_configs=MD_CONFIG)
    body = status_postpass(body)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{py_css}
{css_text}
</style>
</head>
<body>
<header class="doc-head">
<h1>{title}</h1>
<div class="meta">Generated from Markdown &middot; {date}</div>
</header>
{body}
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Markdown -> HTML with inlined CSS")
    ap.add_argument("files", nargs="+", help=".md input file(s)")
    ap.add_argument("--out", help="output path (single input only)")
    ap.add_argument("--css", help="alternate CSS file (default: templates/style.css)")
    ap.add_argument("--stdout", action="store_true", help="print HTML instead of writing files")
    args = ap.parse_args()

    if args.out and len(args.files) != 1:
        ap.error("--out requires exactly one input file")

    css_path = Path(args.css) if args.css else DEFAULT_CSS
    css_text = css_path.read_text(encoding="utf-8")
    py_css = HtmlFormatter(style="default").get_style_defs(".codehilite")

    rc = 0
    for src in args.files:
        src = Path(src).expanduser()
        if not src.is_file():
            print(f"md2html: not a file: {src}", file=sys.stderr)
            rc = 1
            continue
        md_text = src.read_text(encoding="utf-8")
        title = extract_title(md_text, src.stem.replace("-", " ").replace("_", " ").title())
        page = render(md_text, title, css_text, py_css, datetime.date.today().isoformat())
        if args.stdout:
            sys.stdout.write(page)
        else:
            out = Path(args.out) if args.out else src.with_suffix(".html")
            out.write_text(page, encoding="utf-8")
            print(f"md2html: {src} -> {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())