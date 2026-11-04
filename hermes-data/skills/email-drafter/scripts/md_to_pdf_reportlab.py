#!/usr/bin/env python3
"""Convert a markdown doc to a clean PDF for email attachments / sharing — reportlab fallback.

Why this exists: the older md_to_pdf_dra.py uses weasyprint, which is NOT installed in the
Hermes venv on this VPS (uv pip install works if you want it). This version needs only
reportlab — install once with:
    uv pip install --python /opt/hermes/.venv/bin/python3 reportlab

Usage:  /opt/hermes/.venv/bin/python3 md_to_pdf_reportlab.py <input.md> <output.pdf>

Handles: ATX headings (#..####), pipe tables, "-/* / 1." lists, blockquotes-as-body, hr.
Style mirrors md_to_pdf_dra.py: navy #1F3864 headings, gold #C99A2E rules, cream #F5EFDF
zebra table rows, A4 with page numbers, Helvetica (no font installs needed).

Validated 2026-09-14: Ranka Udaya interior execution spec (1251-line md, 46 sections,
many tables) -> 39 KB PDF attached to the Moduli RFP email draft.
"""
import re
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

NAVY = colors.HexColor("#1F3864")
GOLD = colors.HexColor("#C99A2E")
CREAM = colors.HexColor("#F5EFDF")

styles = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=NAVY, spaceAfter=2),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=NAVY, spaceBefore=12, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=NAVY, spaceBefore=10, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=NAVY, spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=4),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.5, leading=13, leftIndent=14, bulletIndent=4, spaceAfter=2),
    "table_hdr": ParagraphStyle("table_hdr", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.white),
    "table_cell": ParagraphStyle("table_cell", fontName="Helvetica", fontSize=8.5, leading=11),
}


def inline(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    lines = open(src, encoding="utf-8").read().splitlines()
    story, first_title, i, n = [], True, 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("# ") and first_title:
            story.append(Paragraph(inline(s[2:]), styles["title"]))
            story.append(Spacer(1, 4))
            first_title = False
            i += 1
            continue
        if s.startswith("## "):
            story.append(Paragraph(inline(s[3:]), styles["h1"]))
            i += 1
            continue
        if s.startswith("### "):
            story.append(Paragraph(inline(s[4:]), styles["h2"]))
            i += 1
            continue
        if s.startswith("#### "):
            story.append(Paragraph(inline(s[5:]), styles["h3"]))
            i += 1
            continue
        if s.startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|", lines[i + 1]):
            header = [Paragraph(inline(c), styles["table_hdr"]) for c in split_row(s)]
            rows, i = [], i + 2
            while i < n and lines[i].strip().startswith("|"):
                rows.append([Paragraph(inline(c), styles["table_cell"]) for c in split_row(lines[i])])
                i += 1
            t = Table([header] + rows, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CREAM]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(t)
            story.append(Spacer(1, 6))
            continue
        m = re.match(r"^([-*]|\d+\.)\s+(.*)$", s)
        if m:
            story.append(Paragraph(inline(m.group(2)), styles["bullet"], bulletText="•"))
            i += 1
            continue
        if re.match(r"^-{3,}$", s):
            story.append(Spacer(1, 4))
            i += 1
            continue
        story.append(Paragraph(inline(s), styles["body"]))
        i += 1

    doc = SimpleDocTemplate(
        dst, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
    )
    doc.build(story)
    print("OK", dst)


if __name__ == "__main__":
    main()