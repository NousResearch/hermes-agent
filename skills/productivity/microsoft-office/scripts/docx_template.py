#!/usr/bin/env python3
# MIT License. Part of the Hermes docx skill.
"""Fill {{placeholder}} tokens in a .docx from a JSON mapping.

Tokens are replaced everywhere: body paragraphs, tables (including nested
tables), headers and footers. Run formatting is preserved; tokens split
across runs are handled.

Usage:
  docx_template.py template.docx values.json output.docx
  docx_template.py template.docx values.json output.docx --strict

values.json: {"name": "Ada", "date": "2026-01-01"}  fills {{name}}, {{date}}.
With --strict, exits 1 if any {{token}} remains unfilled after processing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from docx import Document

from docx_common import iter_all_paragraphs, replace_in_paragraph

TOKEN_RE = re.compile(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fill {{token}} placeholders in a .docx from JSON.")
    ap.add_argument("template", help="input .docx with {{tokens}}")
    ap.add_argument("values", help="JSON file of token -> value")
    ap.add_argument("output", help="output .docx path")
    ap.add_argument("--strict", action="store_true",
                    help="fail if any token remains unfilled")
    args = ap.parse_args()

    with open(args.values, encoding="utf-8") as f:
        values = json.load(f)

    doc = Document(args.template)
    filled = {}
    for para in iter_all_paragraphs(doc):
        # Normalize whitespace variants like {{ name }} first.
        for m in set(TOKEN_RE.findall(para.text)):
            if m in values:
                # Replace any spacing variant with canonical token, then fill.
                for variant in set(
                        t.group(0) for t in TOKEN_RE.finditer(para.text)
                        if t.group(1) == m):
                    n = replace_in_paragraph(para, variant, str(values[m]))
                    filled[m] = filled.get(m, 0) + n

    remaining = sorted({m for para in iter_all_paragraphs(doc)
                        for m in TOKEN_RE.findall(para.text)})
    doc.save(args.output)
    result = {"ok": True, "output": args.output, "filled": filled,
              "unfilled_tokens": remaining}
    if args.strict and remaining:
        result["ok"] = False
        print(json.dumps(result, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
