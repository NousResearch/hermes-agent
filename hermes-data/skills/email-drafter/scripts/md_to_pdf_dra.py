#!/usr/bin/env python3
"""Convert a markdown doc to a DRA-branded PDF (for email attachments / sharing).

Usage:  /opt/hermes/.venv/bin/python3 md_to_pdf_dra.py <input.md> <output.pdf>

Uses the `markdown` lib + `weasyprint` (both present in the Hermes venv) — no pandoc or
LaTeX needed. Style: navy #1F3864 headings, gold #C99A2E rules, cream #F5EFDF zebra
table rows, A4 with page numbers. Handles markdown tables, fenced code, lists, blockquotes.

Verified 2026-09-09: DRA Sales Head requirements + JD analysis (md -> PDF attachments
for the People Fractions reply-all draft).
"""
import sys
import markdown
from weasyprint import HTML

CSS = """
@page { size: A4; margin: 18mm 16mm; @bottom-right { content: counter(page) " / " counter(pages); font-size: 8pt; color: #888; } }
body { font-family: Arial, Helvetica, sans-serif; font-size: 10.5pt; color: #212121; line-height: 1.5; }
h1 { color: #1F3864; font-size: 17pt; border-bottom: 2.5px solid #C99A2E; padding-bottom: 5px; }
h2 { color: #1F3864; font-size: 13.5pt; margin-top: 18px; border-bottom: 1px solid #C99A2E; padding-bottom: 3px; }
h3 { color: #1F3864; font-size: 11.5pt; margin-top: 12px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9.5pt; }
th { background: #1F3864; color: #fff; padding: 5px 7px; text-align: left; }
td { border: 1px solid #bbb; padding: 5px 7px; vertical-align: top; }
tr:nth-child(even) td { background: #F5EFDF; }
li { margin-bottom: 2px; }
strong { color: #1F3864; }
code { font-family: monospace; background: #f3f3f3; padding: 1px 3px; }
blockquote { border-left: 3px solid #C99A2E; margin: 8px 0; padding: 4px 12px; color: #555; }
hr { border: none; border-top: 1px solid #ccc; margin: 14px 0; }
"""

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        md = f.read()
    html = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    full = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
    HTML(string=full).write_pdf(dst)
    print("OK", dst)

if __name__ == "__main__":
    main()