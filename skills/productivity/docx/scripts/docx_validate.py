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
  - paragraph and run style ids referenced by the document exist in
    styles.xml

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
    for tag, attr in ((f"{{{W}}}pStyle", f"{{{W}}}val"),
                      (f"{{{W}}}rStyle", f"{{{W}}}val"),
                      (f"{{{W}}}tblStyle", f"{{{W}}}val")):
        for el in doc_root.iter(tag):
            sid = el.get(attr)
            if sid and sid not in defined:
                _issue(issues, "error", "missing-style",
                       f"style id referenced but not defined: {sid}")

    # --- footnote IDs: large random IDs break Word (issue #102228) ---------
    if "word/footnotes.xml" in names:
        try:
            fn_root = etree.fromstring(zf.read("word/footnotes.xml"))
            fn_ids = []
            for fn in fn_root.iter(f"{{{W}}}footnote"):
                v = fn.get(f"{{{W}}}id")
                if v is None:
                    continue
                try:
                    fn_ids.append(int(v))
                except ValueError:
                    _issue(issues, "error", "bad-footnote-id",
                           f"footnotes.xml: non-integer footnote id {v!r}")
            # Reserved 0/1 are separator types; real footnotes should be 2..N
            real_ids = sorted(i for i in fn_ids if i not in (0, 1))
            if real_ids:
                expected = list(range(2, 2 + len(real_ids)))
                if real_ids != expected:
                    # Large gaps or random IDs
                    if any(i > 10000 for i in real_ids) or max(real_ids) - min(real_ids) > len(real_ids) * 10:
                        _issue(issues, "error", "footnote-id-range",
                               f"footnotes.xml has large/sparse footnote IDs {real_ids[:10]}{'...' if len(real_ids) > 10 else ''} — "
                               "Word may refuse to open the file; run docx_footnote_fix.py to renumber to 2..N (#102228)")
                    else:
                        _issue(issues, "warning", "footnote-id-nonsequential",
                               f"footnotes.xml footnote IDs are non-sequential {real_ids[:10]}; consider renumbering to 2..N")
            # Also check document.xml references resolve
            doc_ids = set()
            for ref in doc_root.iter(f"{{{W}}}footnoteReference"):
                v = ref.get(f"{{{W}}}id")
                if v is not None:
                    try:
                        doc_ids.add(int(v))
                    except ValueError:
                        pass
            footnote_set = set(fn_ids)
            for did in doc_ids:
                if did not in footnote_set:
                    _issue(issues, "error", "footnote-reference-missing",
                           f"document.xml references footnote id {did} with no matching w:footnote in footnotes.xml")
        except etree.XMLSyntaxError as exc:
            _issue(issues, "error", "bad-footnotes-xml", f"word/footnotes.xml: {exc}")

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
