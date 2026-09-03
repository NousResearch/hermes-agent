#!/usr/bin/env python3
# MIT License. Part of the Hermes docx skill.
"""List, add, and delete comments in a .docx.

Subcommands:
  list    JSON per comment: id, author, initials, date, text, anchored_text
  add     add a comment anchored to the first occurrence of --target
  delete  remove a comment (and its range markers) by --id

Examples:
  docx_comments.py list report.docx
  docx_comments.py add report.docx --target "Q3 revenue" \
      --text "Needs a source" --author "Reviewer" -o out.docx
  docx_comments.py delete report.docx --id 0 -o out.docx

Uses the native python-docx comments API (>= 1.2) when available; falls
back to building word/comments.xml and the range markers directly for
older versions (or when --xml is passed). Listing and deletion always
work at the XML level so they handle documents from any producer.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from copy import deepcopy

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from lxml import etree

from docx_common import iter_part_roots

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
COMMENTS_CT = ("application/vnd.openxmlformats-officedocument"
               ".wordprocessingml.comments+xml")


def q(tag: str) -> str:
    return f"{{{W}}}{tag}"


# ---------------------------------------------------------------- reading

def _comments_root(doc):
    """Return the XML root of the comments part, or None."""
    for rel in doc.part.rels.values():
        if rel.reltype == RT.COMMENTS:
            part = rel.target_part
            el = getattr(part, "_element", None)
            if el is not None:
                return el
            return etree.fromstring(part.blob)
    return None


def _anchored_texts(doc) -> dict:
    """Map comment id -> document text between its range markers."""
    anchored: dict[str, list[str]] = {}
    for root in iter_part_roots(doc):
        active: set[str] = set()
        for el in root.iter():
            if el.tag == q("commentRangeStart"):
                cid = el.get(q("id"))
                active.add(cid)
                anchored.setdefault(cid, [])
            elif el.tag == q("commentRangeEnd"):
                active.discard(el.get(q("id")))
            elif el.tag == q("t") and active:
                for cid in active:
                    anchored[cid].append(el.text or "")
    return {cid: "".join(parts) for cid, parts in anchored.items()}


def list_comments(doc) -> list:
    root = _comments_root(doc)
    if root is None:
        return []
    anchored = _anchored_texts(doc)
    out = []
    for c in root.iter(q("comment")):
        cid = c.get(q("id"))
        text = "\n".join(
            "".join(t.text or "" for t in p.iter(q("t")))
            for p in c.iter(q("p")))
        out.append({"id": cid, "author": c.get(q("author")),
                    "initials": c.get(q("initials")),
                    "date": c.get(q("date")), "text": text,
                    "anchored_text": anchored.get(cid, "")})
    return out


# ---------------------------------------------------------------- anchoring

def _split_run(para, run_el, offset: int):
    """Split a run element at text offset; return the new right-hand run."""
    text = "".join(t.text or "" for t in run_el.iter(q("t")))
    right = deepcopy(run_el)
    run_el.addnext(right)
    for el, s in ((run_el, text[:offset]), (right, text[offset:])):
        for t in list(el.iter(q("t"))):
            el.remove(t)
        t = etree.SubElement(el, q("t"))
        t.text = s
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return right


def find_anchor_runs(doc, target: str):
    """Isolate `target`'s first occurrence into whole runs; return them."""
    from docx_common import iter_all_paragraphs
    for para in iter_all_paragraphs(doc):
        full = para.text
        start = full.find(target)
        if start < 0:
            continue
        end = start + len(target)
        pos = 0
        covered = []
        for run_el in para._p.iter(q("r")):
            rtext = "".join(t.text or "" for t in run_el.iter(q("t")))
            r_start, r_end = pos, pos + len(rtext)
            pos = r_end
            if r_end <= start or r_start >= end:
                continue
            if r_start < start:  # split off the left part
                run_el = _split_run(para, run_el, start - r_start)
                r_start = start
            if r_end > end:      # split off the right part
                _split_run(para, run_el, end - r_start)
            covered.append(run_el)
        return para, covered
    return None, []


# ---------------------------------------------------------------- adding

def _next_id(doc) -> int:
    root = _comments_root(doc)
    if root is None:
        return 0
    ids = [int(c.get(q("id"), "0")) for c in root.iter(q("comment"))
           if c.get(q("id"), "").isdigit()]
    return max(ids) + 1 if ids else 0


def add_comment_native(doc, runs, text, author, initials):
    from docx.text.run import Run
    run_objs = [Run(r, None) for r in runs]
    comment = doc.add_comment(run_objs, text=text, author=author,
                              initials=initials or "")
    return str(comment.comment_id)


def add_comment_xml(doc, runs, text, author, initials) -> str:
    cid = str(_next_id(doc))
    root = _comments_root(doc)
    if root is None:
        root = etree.fromstring(
            f'<w:comments xmlns:w="{W}"/>'.encode("utf-8"))
        from docx.opc.packuri import PackURI
        from docx.opc.part import Part
        blob = etree.tostring(root, xml_declaration=True,
                              encoding="UTF-8", standalone=True)
        part = Part(PackURI("/word/comments.xml"), COMMENTS_CT, blob,
                    doc.part.package)
        doc.part.relate_to(part, RT.COMMENTS)
        # keep a live element on the part so edits reach save()
        part._element = root
        part.blob_ = None

        def _blob(self=part):
            return etree.tostring(self._element, xml_declaration=True,
                                  encoding="UTF-8", standalone=True)
        part.__class__ = type("CommentsXmlPart", (Part,),
                              {"blob": property(lambda self: _blob(self))})
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    comment = etree.SubElement(root, q("comment"))
    comment.set(q("id"), cid)
    comment.set(q("author"), author)
    if initials:
        comment.set(q("initials"), initials)
    comment.set(q("date"), now)
    p = etree.SubElement(comment, q("p"))
    r = etree.SubElement(p, q("r"))
    t = etree.SubElement(r, q("t"))
    t.text = text
    # range markers around the anchor runs + reference run after them
    first, last = runs[0], runs[-1]
    start = first.makeelement(q("commentRangeStart"), {q("id"): cid})
    first.addprevious(start)
    end = last.makeelement(q("commentRangeEnd"), {q("id"): cid})
    last.addnext(end)
    ref_run = last.makeelement(q("r"), {})
    ref = etree.SubElement(ref_run, q("commentReference"))
    ref.set(q("id"), cid)
    end.addnext(ref_run)
    return cid


# ---------------------------------------------------------------- deleting

def delete_comment(doc, cid: str) -> bool:
    root = _comments_root(doc)
    found = False
    if root is not None:
        for c in list(root.iter(q("comment"))):
            if c.get(q("id")) == cid:
                c.getparent().remove(c)
                found = True
    for part_root in iter_part_roots(doc):
        for tag in ("commentRangeStart", "commentRangeEnd",
                    "commentReference"):
            for el in list(part_root.iter(q(tag))):
                if el.get(q("id")) == cid:
                    parent = el.getparent()
                    # remove the wrapping run for reference marks
                    if tag == "commentReference" and parent.tag == q("r"):
                        parent.getparent().remove(parent)
                    else:
                        parent.remove(el)
                    found = True
    return found


def main() -> int:
    ap = argparse.ArgumentParser(
        description="List, add, or delete comments in a .docx.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list comments as JSON")
    p.add_argument("path", help="input .docx")

    p = sub.add_parser("add", help="add a comment anchored to text")
    p.add_argument("path", help="input .docx")
    p.add_argument("-o", "--output", help="output path (default: in place)")
    p.add_argument("--target", required=True,
                   help="anchor: first occurrence of this text")
    p.add_argument("--text", required=True, help="comment body")
    p.add_argument("--author", default="Hermes")
    p.add_argument("--initials", default="")
    p.add_argument("--xml", action="store_true",
                   help="force the XML fallback (skip native API)")

    p = sub.add_parser("delete", help="delete a comment by id")
    p.add_argument("path", help="input .docx")
    p.add_argument("-o", "--output", help="output path (default: in place)")
    p.add_argument("--id", required=True, help="comment id")

    args = ap.parse_args()
    doc = Document(args.path)

    if args.cmd == "list":
        print(json.dumps({"ok": True, "comments": list_comments(doc)},
                         ensure_ascii=False))
        return 0

    if args.cmd == "add":
        para, runs = find_anchor_runs(doc, args.target)
        if not runs:
            print(json.dumps({"ok": False,
                              "error": f"target not found: {args.target}"}))
            return 1
        native = hasattr(doc, "add_comment") and not args.xml
        if native:
            cid = add_comment_native(doc, runs, args.text, args.author,
                                     args.initials)
        else:
            cid = add_comment_xml(doc, runs, args.text, args.author,
                                  args.initials)
        result = {"ok": True, "comment_id": cid,
                  "native_api": native, "anchored_to": args.target}
    else:  # delete
        if not delete_comment(doc, args.id):
            print(json.dumps({"ok": False,
                              "error": f"no comment with id {args.id}"}))
            return 1
        result = {"ok": True, "deleted_id": args.id}

    out = args.output or args.path
    doc.save(out)
    result["output"] = out
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
