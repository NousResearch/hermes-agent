#!/usr/bin/env python3
# MIT License. Part of the Hermes docx skill.
"""Read a .docx: text, structure outline, styles, images, revision detection.

Usage:
  docx_read.py file.docx --text        # full text incl. tables + headers/footers
  docx_read.py file.docx --structure   # JSON outline (headings, tables, counts)
  docx_read.py file.docx --styles      # JSON list of styles actually used
  docx_read.py file.docx --images DIR  # extract embedded images into DIR
  docx_read.py file.docx --revisions   # JSON: tracked changes / comments present?

Text output is JSON: {"body": [...], "tables": [[...rows]], "headers": [...],
"footers": [...]}. Body text is the accepted/as-is text (python-docx ignores
deleted-in-revision text and shows inserted text).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

from docx import Document


def cell_text(cell) -> str:
    """Flatten a cell's own paragraphs and any nested table text."""
    parts = [p.text for p in cell.paragraphs if p.text]
    for nested in cell.tables:
        for row in table_to_rows(nested):
            parts.extend(text for text in row if text)
    return "\n".join(parts)


def table_to_rows(table) -> list:
    return [[cell_text(cell) for cell in row.cells] for row in table.rows]


def append_part_text(target: list[str], part, seen: set[int]) -> None:
    """Append one header/footer variant without duplicating linked parts."""
    marker = id(part._element)
    if marker in seen:
        return
    seen.add(marker)
    target.extend(p.text for p in part.paragraphs)
    for table in part.tables:
        target.append(json.dumps(table_to_rows(table), ensure_ascii=False))


def extract_text(doc) -> dict:
    out = {"body": [p.text for p in doc.paragraphs],
           "tables": [table_to_rows(t) for t in doc.tables],
           "headers": [], "footers": []}
    seen_headers: set[int] = set()
    seen_footers: set[int] = set()
    for section in doc.sections:
        for part in (section.header, section.first_page_header,
                     section.even_page_header):
            append_part_text(out["headers"], part, seen_headers)
        for part in (section.footer, section.first_page_footer,
                     section.even_page_footer):
            append_part_text(out["footers"], part, seen_footers)
    return out


def extract_structure(doc) -> dict:
    outline = []
    for i, para in enumerate(doc.paragraphs):
        style = para.style.name if para.style else ""
        if style.startswith("Heading"):
            try:
                level = int(style.split()[-1])
            except ValueError:
                level = 1
            outline.append({"index": i, "level": level, "text": para.text})
    return {
        "outline": outline,
        "paragraph_count": len(doc.paragraphs),
        "table_count": len(doc.tables),
        "tables": [{"rows": len(t.rows), "cols": len(t.columns)}
                   for t in doc.tables],
        "section_count": len(doc.sections),
    }


def styles_used(doc) -> list:
    used = set()
    for para in doc.paragraphs:
        if para.style:
            used.add(para.style.name)
        for run in para.runs:
            if run.style:
                used.add(run.style.name)
    for table in doc.tables:
        if table.style:
            used.add(table.style.name)
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    if para.style:
                        used.add(para.style.name)
    return sorted(used)


def extract_images(path: str, outdir: str) -> list:
    os.makedirs(outdir, exist_ok=True)
    written = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.startswith("word/media/"):
                target = os.path.join(outdir, os.path.basename(name))
                with open(target, "wb") as f:
                    f.write(zf.read(name))
                written.append(target)
    return written


def detect_revisions(path: str) -> dict:
    """Detect tracked changes and comments by scanning the raw XML parts."""
    markers = {"insertions": b"<w:ins ", "deletions": b"<w:del ",
               "format_changes": b"<w:rPrChange"}
    result = {k: False for k in markers}
    result["comments"] = False
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        result["comments"] = any(n.startswith("word/comments") for n in names)
        for name in names:
            if name.startswith("word/") and name.endswith(".xml"):
                data = zf.read(name)
                for key, marker in markers.items():
                    if marker in data:
                        result[key] = True
    result["has_tracked_changes"] = any(
        result[k] for k in ("insertions", "deletions", "format_changes"))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Read/inspect a .docx file.")
    ap.add_argument("path", help=".docx file to read")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", action="store_true", help="extract all text as JSON")
    g.add_argument("--structure", action="store_true", help="outline JSON")
    g.add_argument("--styles", action="store_true", help="styles used, JSON")
    g.add_argument("--images", metavar="DIR", help="extract images to DIR")
    g.add_argument("--revisions", action="store_true",
                   help="detect tracked changes / comments")
    args = ap.parse_args()

    if args.images:
        print(json.dumps({"images": extract_images(args.path, args.images)},
                         ensure_ascii=False))
        return 0
    if args.revisions:
        print(json.dumps(detect_revisions(args.path), ensure_ascii=False))
        return 0

    doc = Document(args.path)
    if args.text:
        out = extract_text(doc)
    elif args.structure:
        out = extract_structure(doc)
    else:
        out = {"styles": styles_used(doc)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
