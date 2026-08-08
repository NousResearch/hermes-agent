#!/usr/bin/env python3
# MIT License. Part of the Hermes docx skill.
"""Edit an existing .docx in place (or to a new file).

Subcommands:
  replace   find-and-replace text, preserving run formatting
  set-cell  set the text of a table cell
  insert    insert a paragraph before a given body paragraph index
  delete    delete a body paragraph by index
  style     apply a paragraph style to a body paragraph by index

Examples:
  docx_edit.py replace in.docx --find old --replace new -o out.docx
  docx_edit.py set-cell in.docx --table 0 --row 1 --col 2 --text "42"
  docx_edit.py insert in.docx --index 3 --text "New para" --style Normal
  docx_edit.py delete in.docx --index 3
  docx_edit.py style in.docx --index 0 --style "Heading 1"
"""
from __future__ import annotations

import argparse
import json
import sys

from docx import Document

from docx_common import iter_all_paragraphs, replace_in_paragraph


def cmd_replace(doc, args) -> dict:
    n = 0
    for para in iter_all_paragraphs(doc):
        n += replace_in_paragraph(para, args.find, args.replace)
    return {"replacements": n}


def cmd_set_cell(doc, args) -> dict:
    cell = doc.tables[args.table].cell(args.row, args.col)
    cell.text = args.text
    return {"table": args.table, "row": args.row, "col": args.col}


def cmd_insert(doc, args) -> dict:
    paras = doc.paragraphs
    if args.index < len(paras):
        anchor = paras[args.index]
        new_para = anchor.insert_paragraph_before(args.text, style=args.style)
    else:
        new_para = doc.add_paragraph(args.text, style=args.style)
    return {"inserted_at": args.index, "text": new_para.text}


def cmd_delete(doc, args) -> dict:
    para = doc.paragraphs[args.index]
    el = para._element
    el.getparent().remove(el)
    return {"deleted_index": args.index}


def cmd_style(doc, args) -> dict:
    doc.paragraphs[args.index].style = doc.styles[args.style]
    return {"index": args.index, "style": args.style}


def main() -> int:
    ap = argparse.ArgumentParser(description="Edit a .docx file.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("path", help="input .docx")
        p.add_argument("-o", "--output",
                       help="output path (default: overwrite input)")

    p = sub.add_parser("replace", help="find-and-replace text")
    common(p)
    p.add_argument("--find", required=True)
    p.add_argument("--replace", required=True)
    p.add_argument("--body-only", action="store_true",
                   help="skip headers/footers")

    p = sub.add_parser("set-cell", help="set table cell text")
    common(p)
    p.add_argument("--table", type=int, required=True, help="table index")
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--col", type=int, required=True)
    p.add_argument("--text", required=True)

    p = sub.add_parser("insert", help="insert paragraph at body index")
    common(p)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--style", default=None)

    p = sub.add_parser("delete", help="delete body paragraph by index")
    common(p)
    p.add_argument("--index", type=int, required=True)

    p = sub.add_parser("style", help="apply style to body paragraph")
    common(p)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--style", required=True)

    args = ap.parse_args()
    doc = Document(args.path)

    if args.cmd == "replace":
        if args.body_only:
            n = 0
            for para in iter_all_paragraphs(doc, include_headers_footers=False):
                n += replace_in_paragraph(para, args.find, args.replace)
            result = {"replacements": n}
        else:
            result = cmd_replace(doc, args)
    elif args.cmd == "set-cell":
        result = cmd_set_cell(doc, args)
    elif args.cmd == "insert":
        result = cmd_insert(doc, args)
    elif args.cmd == "delete":
        result = cmd_delete(doc, args)
    else:
        result = cmd_style(doc, args)

    out = args.output or args.path
    doc.save(out)
    result.update({"ok": True, "output": out})
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
