#!/usr/bin/env python3
# MIT License. Part of the Hermes docx skill.
"""Health-check a .docx package and report issues as JSON.

Usage: docx_validate.py file.docx

Checks (health-check tier, NOT full XSD schema validation):
  - the file is a readable zip and python-docx can open it
  - required package parts exist ([Content_Types].xml, document.xml)
  - every relationship in every .rels file resolves to a part in the
    package (dangling image/hyperlink/etc. rels are reported; external
    targets such as hyperlinks are skipped)
  - r:embed / r:id references in document.xml resolve to relationships
  - embedded images are non-empty and start with known magic bytes
    (PNG/JPEG/GIF/BMP/TIFF/EMF/WMF/SVG); no PIL required
  - paragraph, run, and table style references in document, comments,
    headers, footers, footnotes, and endnotes resolve to explicit styles
    or the implicit CommentText/CommentReference built-ins; malformed
    optional stories are reported without stopping other style checks

Output: {"ok": bool, "issues": [{"severity": "error"|"warning", ...}]}
Exit code 1 when any error-severity issue is found (warnings exit 0).
"""
from __future__ import annotations

import argparse
import json
import posixpath
import sys
import zipfile

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"

IMAGE_MAGIC = (
    b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a",
    b"BM", b"II*\x00", b"MM\x00*",
    b"\x01\x00\x00\x00",              # EMF
    b"\xd7\xcd\xc6\x9a", b"\x01\x00\x09\x00",  # WMF variants
    b"<?xml", b"<svg",
)


def _issue(issues, severity, code, detail):
    issues.append({"severity": severity, "code": code, "detail": detail})


def _rel_target(base_part: str, target: str) -> str:
    base_dir = posixpath.dirname(base_part)
    return posixpath.normpath(posixpath.join(base_dir, target)).lstrip("/")


def _check_style_references(root, part_name, defined, issues):
    # python-docx's native comment producer deliberately emits these without
    # definitions (CT_Comments.add_comment / CT_R._new_comment_reference).
    # latentStyles holds UI metadata, not a generic style-name -> styleId map.
    implicit = {("pStyle", "CommentText"), ("rStyle", "CommentReference")}
    for tag in ("pStyle", "rStyle", "tblStyle"):
        for el in root.iter(f"{{{W}}}{tag}"):
            sid = el.get(f"{{{W}}}val")
            if sid and sid not in defined and (tag, sid) not in implicit:
                detail = f"{part_name}: style id referenced but not defined: {sid}"
                _issue(issues, "error", "missing-style", detail)


def validate(path: str) -> dict:
    issues: list[dict] = []

    try:
        zf = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        _issue(issues, "error", "not-a-zip", str(exc))
        return {"ok": False, "issues": issues}

    names = set(zf.namelist())
    bad = zf.testzip()
    if bad is not None:
        _issue(issues, "error", "corrupt-member", f"CRC check failed: {bad}")

    for required in ("[Content_Types].xml", "word/document.xml"):
        if required not in names:
            _issue(issues, "error", "missing-part",
                   f"required part absent: {required}")
    if issues and any(i["severity"] == "error" for i in issues):
        return {"ok": False, "issues": issues}

    # --- relationships resolve ------------------------------------------
    rel_ids_by_source: dict[str, dict] = {}
    for rels_name in [n for n in names if n.endswith(".rels")]:
        try:
            root = etree.fromstring(zf.read(rels_name))
        except etree.XMLSyntaxError as exc:
            _issue(issues, "error", "bad-rels-xml", f"{rels_name}: {exc}")
            continue
        source_part = posixpath.normpath(
            posixpath.join(posixpath.dirname(rels_name), ".."))
        source_part = "" if source_part == "." else source_part
        ids = {}
        for rel in root.iter(f"{{{PR}}}Relationship"):
            rid, target = rel.get("Id"), rel.get("Target", "")
            mode = rel.get("TargetMode", "Internal")
            ids[rid] = target
            if mode == "External":
                continue
            resolved = _rel_target(source_part + "/x" if source_part
                                   else "x", target)
            if resolved not in names:
                _issue(issues, "error", "dangling-rel",
                       f"{rels_name}: {rid} -> {target} (missing part)")
        rel_ids_by_source[source_part or "_package"] = ids

    # --- r:id / r:embed references in document.xml -----------------------
    doc_root = etree.fromstring(zf.read("word/document.xml"))
    doc_rels = rel_ids_by_source.get("word", {})
    for el in doc_root.iter():
        for attr in (f"{{{R}}}id", f"{{{R}}}embed", f"{{{R}}}link"):
            rid = el.get(attr)
            if rid and rid not in doc_rels:
                _issue(issues, "error", "unresolved-reference",
                       f"document.xml references {rid} with no relationship")

    # --- embedded images decode ------------------------------------------
    for name in [n for n in names if n.startswith("word/media/")]:
        data = zf.read(name)
        if not data:
            _issue(issues, "error", "empty-image", name)
        elif not any(data.startswith(m) for m in IMAGE_MAGIC):
            _issue(issues, "warning", "unknown-image-format",
                   f"{name}: unrecognized magic bytes")

    # --- styles referenced exist ------------------------------------------
    defined = set()
    if "word/styles.xml" in names:
        styles_root = etree.fromstring(zf.read("word/styles.xml"))
        defined = {s.get(f"{{{W}}}styleId")
                   for s in styles_root.iter(f"{{{W}}}style")}
    _check_style_references(doc_root, "word/document.xml", defined, issues)
    optional_stories = {"word/comments.xml", "word/footnotes.xml", "word/endnotes.xml"}
    optional_stories.update(
        name for name in names
        if posixpath.dirname(name) == "word"
        and posixpath.basename(name).startswith(("header", "footer"))
        and name.endswith(".xml")
    )
    for part_name in sorted(optional_stories & names):
        try:
            root = etree.fromstring(zf.read(part_name))
        except etree.XMLSyntaxError as exc:
            code = "bad-comments-xml" if part_name == "word/comments.xml" else "bad-story-xml"
            _issue(issues, "error", code, f"{part_name}: {exc}")
            continue
        _check_style_references(root, part_name, defined, issues)

    # --- python-docx can open it ------------------------------------------
    try:
        from docx import Document
        Document(path)
    except Exception as exc:  # noqa: BLE001 - triage tool, report anything
        _issue(issues, "error", "python-docx-open-failed", str(exc))

    ok = not any(i["severity"] == "error" for i in issues)
    return {"ok": ok, "issues": issues}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Health-check a .docx (not XSD schema validation).")
    ap.add_argument("path", help="the .docx file to check")
    args = ap.parse_args()
    report = validate(args.path)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
