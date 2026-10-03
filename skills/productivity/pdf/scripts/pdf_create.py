#!/usr/bin/env python3
"""Create a PDF from a JSON spec using reportlab platypus.

Spec format (UTF-8 JSON):
{
  "title": "Example Report",
  "author": "example-author",
  "page_size": "A4",            // or "letter" (default: A4)
  "page_numbers": true,          // default true
  "elements": [
    {"type": "heading", "text": "Section 1", "level": 1},
    {"type": "paragraph", "text": "Body text..."},
    {"type": "table", "rows": [["H1", "H2"], ["a", "b"]], "header": true},
    {"type": "image", "path": "chart.png", "width": 400},
    {"type": "pagebreak"}
  ]
}
"""
from __future__ import annotations

import argparse
import json
import sys


def _reconfigure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def build_pdf(spec: dict, out_path: str) -> int:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        print("Missing dependency: install with 'python3 -m pip install reportlab'", file=sys.stderr)
        return 2

    page_size = letter if str(spec.get("page_size", "A4")).lower() == "letter" else A4
    styles = getSampleStyleSheet()
    story = []
    for el in spec.get("elements", []):
        etype = el.get("type")
        if etype == "heading":
            level = min(max(int(el.get("level", 1)), 1), 3)
            story.append(Paragraph(el.get("text", ""), styles[f"Heading{level}"]))
        elif etype == "paragraph":
            story.append(Paragraph(el.get("text", ""), styles["BodyText"]))
            story.append(Spacer(1, 6))
        elif etype == "table":
            rows = el.get("rows", [])
            if not rows:
                continue
            table = Table(rows, repeatRows=1 if el.get("header", True) else 0)
            style = [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
            if el.get("header", True):
                style += [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            table.setStyle(TableStyle(style))
            story.append(table)
            story.append(Spacer(1, 10))
        elif etype == "image":
            kwargs = {}
            if el.get("width"):
                kwargs["width"] = float(el["width"])
            if el.get("height"):
                kwargs["height"] = float(el["height"])
            img = Image(el["path"], **kwargs)
            if "width" in kwargs and "height" not in kwargs:
                # keep aspect ratio
                ratio = img.imageHeight / img.imageWidth
                img.drawWidth = kwargs["width"]
                img.drawHeight = kwargs["width"] * ratio
            story.append(img)
            story.append(Spacer(1, 10))
        elif etype == "pagebreak":
            story.append(PageBreak())
        else:
            print(f"Warning: unknown element type {etype!r}, skipped", file=sys.stderr)

    def draw_page_number(canvas, doc):
        if spec.get("page_numbers", True):
            canvas.saveState()
            canvas.setFont("Helvetica", 9)
            canvas.drawCentredString(page_size[0] / 2.0, 0.5 * inch, f"Page {doc.page}")
            canvas.restoreState()

    doc = SimpleDocTemplate(
        out_path,
        pagesize=page_size,
        title=spec.get("title", ""),
        author=spec.get("author", ""),
    )
    doc.build(story, onFirstPage=draw_page_number, onLaterPages=draw_page_number)
    print(json.dumps({"output": out_path, "elements": len(spec.get("elements", []))}))
    return 0


def main() -> int:
    _reconfigure_stdio()
    parser = argparse.ArgumentParser(description="Create a PDF from a JSON spec (reportlab).")
    parser.add_argument("spec", help="Path to UTF-8 JSON spec file")
    parser.add_argument("-o", "--output", required=True, help="Output PDF path")
    args = parser.parse_args()
    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    return build_pdf(spec, args.output)


if __name__ == "__main__":
    sys.exit(main())
