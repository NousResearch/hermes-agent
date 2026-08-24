#!/usr/bin/env python3
# MIT License. Part of the Hermes docx skill.
"""Inspect and resolve tracked changes (w:ins / w:del) in a .docx.

Subcommands:
  list        JSON list of revisions: id, author, date, type, text
  accept-all  accept every insertion and deletion
  reject-all  reject every insertion and deletion
  accept      accept one revision by --id
  reject      reject one revision by --id

Examples:
  docx_revisions.py list report.docx
  docx_revisions.py accept-all report.docx -o accepted.docx
  docx_revisions.py reject report.docx --id 3 -o out.docx

Semantics (direct XML manipulation, python-docx oxml layer):
  accept w:ins -> unwrap (keep inserted runs)   reject w:ins -> remove
  accept w:del -> remove                        reject w:del -> restore
  (restore = w:delText tags renamed to w:t, wrapper unwrapped)

Covers run-level insertions/deletions anywhere in body, tables (nested
included), headers and footers. Row/paragraph-mark revisions and format
changes (w:rPrChange etc.) are reported by docx_read.py --revisions but
not resolved here.
"""
from __future__ import annotations

import argparse
import json
import sys

from docx import Document

from docx_common import iter_part_roots

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def q(tag: str) -> str:
    return f"{{{W}}}{tag}"


INS, DEL = q("ins"), q("del")


def _iter_revision_elements(doc):
    """Yield every w:ins / w:del element across body, headers, footers."""
    for root in iter_part_roots(doc):
        for el in root.iter(INS, DEL):
            yield el


def _rev_text(el) -> str:
    tag = q("delText") if el.tag == DEL else q("t")
    return "".join(t.text or "" for t in el.iter(tag))


def _rev_record(el) -> dict:
    return {
        "id": el.get(q("id")),
        "author": el.get(q("author")),
        "date": el.get(q("date")),
        "type": "insertion" if el.tag == INS else "deletion",
        "text": _rev_text(el),
    }


def _unwrap(el) -> None:
    """Replace `el` with its children, keeping document order."""
    parent = el.getparent()
    idx = list(parent).index(el)
    for child in list(el):
        parent.insert(idx, child)
        idx += 1
    parent.remove(el)


def _apply(el, accept: bool) -> None:
    if el.getparent() is None:  # already detached via an outer wrapper
        return
    if el.tag == INS:
        if accept:
            _unwrap(el)
        else:
            el.getparent().remove(el)
    else:  # w:del
        if accept:
            el.getparent().remove(el)
        else:
            for dt in list(el.iter(q("delText"))):
                dt.tag = q("t")
            _unwrap(el)


def resolve(doc, accept: bool, rev_id: str | None = None) -> int:
    targets = [el for el in _iter_revision_elements(doc)
               if rev_id is None or el.get(q("id")) == rev_id]
    for el in targets:
        _apply(el, accept)
    return len(targets)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="List, accept, or reject tracked changes in a .docx.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, out=True):
        p.add_argument("path", help="input .docx")
        if out:
            p.add_argument("-o", "--output",
                           help="output path (default: overwrite input)")

    common(sub.add_parser("list", help="list revisions as JSON"), out=False)
    common(sub.add_parser("accept-all", help="accept every revision"))
    common(sub.add_parser("reject-all", help="reject every revision"))
    for name in ("accept", "reject"):
        p = sub.add_parser(name, help=f"{name} one revision by id")
        common(p)
        p.add_argument("--id", required=True, help="revision id (w:id)")

    args = ap.parse_args()
    doc = Document(args.path)

    if args.cmd == "list":
        revs = [_rev_record(el) for el in _iter_revision_elements(doc)]
        print(json.dumps({"ok": True, "revisions": revs}, ensure_ascii=False))
        return 0

    accept = args.cmd in ("accept-all", "accept")
    rev_id = getattr(args, "id", None)
    n = resolve(doc, accept, rev_id)
    if rev_id is not None and n == 0:
        print(json.dumps({"ok": False,
                          "error": f"no revision with id {rev_id}"}))
        return 1
    out = args.output or args.path
    doc.save(out)
    print(json.dumps({"ok": True, "output": out, "resolved": n,
                      "action": "accept" if accept else "reject"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
