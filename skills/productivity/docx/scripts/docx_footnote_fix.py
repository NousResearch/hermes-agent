#!/usr/bin/env python3
# MIT License. Part of the Hermes docx skill.
"""Repair large random footnote IDs that make Word refuse to open the file.

Issue #102228: docx-plus (and similar tooling) can emit footnote IDs that are
very large random integers. Word's schema validation rejects them, surfacing
"The file ... cannot be opened because there is something wrong with the
content". Renumbering IDs to sequential small integers in both
word/footnotes.xml and word/document.xml fixes the file while keeping
separator (id=0) and continuationSeparator (id=1) intact.

Usage:
  python docx_footnote_fix.py in.docx [-o out.docx] [--dry-run]

* Without -o, the file is fixed in place.
* --dry-run reports what would change without writing.
* Exit 0 on success, 1 on error, 2 when --dry-run finds issues.
* Prints JSON: {"fixed": bool, "renamed": int, "mapping": {...}, "issues": []}
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
import posixpath
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WORD_NS = {"w": W}

# Styles for footnote IDs that must stay at 0/1 (separator types)
_RESERVED_FOOTNOTE_IDS = {0, 1}


def _parse_args():
    ap = argparse.ArgumentParser(description="Renumber footnote IDs to sequential small integers.")
    ap.add_argument("path", help="input .docx")
    ap.add_argument("-o", "--output", help="output .docx (default: overwrite input)")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    return ap.parse_args()


def collect_footnote_ids(zf: zipfile.ZipFile):
    """Return (footnotes_root or None, list of footnote elements, mapping of id->element)."""
    if "word/footnotes.xml" not in zf.namelist():
        return None, [], {}
    root = etree.fromstring(zf.read("word/footnotes.xml"))
    footnotes = list(root.iter(f"{{{W}}}footnote"))
    by_id = {}
    for fn in footnotes:
        v = fn.get(f"{{{W}}}id")
        if v is not None:
            try:
                by_id[int(v)] = fn
            except ValueError:
                pass
    return root, footnotes, by_id


def build_renumber_map(by_id: dict) -> dict:
    """Map old non-reserved IDs to new sequential 2,3,4,..."""
    non_reserved = sorted(k for k in by_id.keys() if k not in _RESERVED_FOOTNOTE_IDS)
    if not non_reserved:
        return {}
    # Detect whether renumbering is needed: large IDs or gaps > 1000 or non-sequential sparse
    # Simple policy: always renumber non-reserved to 2..N if any ID differs from expected sequential.
    expected = list(range(2, 2 + len(non_reserved)))
    if non_reserved == expected:
        # Already sequential 2..N - check for excessively large max (should not happen if sequential)
        if max(non_reserved) < 5000:
            return {}
    mapping = {old: new for old, new in zip(non_reserved, expected)}
    # Filter no-ops (should be none except reserved)
    mapping = {k: v for k, v in mapping.items() if k != v}
    return mapping


def renumber_footnotes(docx_path: str, output_path: str | None = None, dry_run: bool = False) -> dict:
    path = Path(docx_path)
    if not path.exists():
        return {"fixed": False, "renamed": 0, "mapping": {}, "issues": [f"file not found: {docx_path}"]}

    try:
        zf = zipfile.ZipFile(str(path), "r")
    except Exception as exc:
        return {"fixed": False, "renamed": 0, "mapping": {}, "issues": [f"not a zip: {exc}"]}

    try:
        names = zf.namelist()
        if "word/footnotes.xml" not in names:
            return {"fixed": False, "renamed": 0, "mapping": {}, "issues": [], "note": "no footnotes.xml - nothing to fix"}
        foot_root, footnotes, by_id = collect_footnote_ids(zf)
        mapping = build_renumber_map(by_id)
        if not mapping:
            return {"fixed": False, "renamed": 0, "mapping": {}, "issues": []}

        # Also need to update document.xml + headers/footers that reference footnotes
        # footnoteReference elements have w:id matching footnote ids
        # Prepare to rewrite affected parts
        parts_to_update = []
        # footnotes.xml rewrite
        for old, new in mapping.items():
            el = by_id.get(old)
            if el is not None:
                el.set(f"{{{W}}}id", str(new))
        parts_to_update.append(("word/footnotes.xml", etree.tostring(foot_root, xml_declaration=True, encoding="UTF-8", standalone=True)))

        # document.xml and other parts containing w:footnoteReference
        for part_name in [n for n in names if n.startswith("word/") and n.endswith(".xml")]:
            if part_name == "word/footnotes.xml":
                continue
            try:
                data = zf.read(part_name)
            except KeyError:
                continue
            try:
                root = etree.fromstring(data)
            except etree.XMLSyntaxError:
                continue
            changed = False
            for ref in root.iter(f"{{{W}}}footnoteReference"):
                v = ref.get(f"{{{W}}}id")
                if v is None:
                    continue
                try:
                    iv = int(v)
                except ValueError:
                    continue
                if iv in mapping:
                    ref.set(f"{{{W}}}id", str(mapping[iv]))
                    changed = True
            # Also footnoteRef inside footnote continuations? keep as is for reserved
            if changed:
                parts_to_update.append((part_name, etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)))

        if dry_run:
            return {"fixed": False, "renamed": len(mapping), "mapping": {str(k): v for k, v in mapping.items()}, "issues": [], "dry_run": True}

        out = Path(output_path) if output_path else path
        # Rebuild zip
        tmp_out = out.with_suffix(out.suffix + ".tmp") if out == path else out
        tmp_out.parent.mkdir(parents=True, exist_ok=True)
        update_map = dict(parts_to_update)
        with zipfile.ZipFile(str(tmp_out), "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                if name in update_map:
                    zout.writestr(name, update_map[name])
                else:
                    zout.writestr(name, zf.read(name))
        if out == path:
            tmp_out.replace(out)
        return {"fixed": True, "renamed": len(mapping), "mapping": {str(k): v for k, v in mapping.items()}, "issues": []}
    finally:
        try:
            zf.close()
        except Exception:
            pass


def main() -> int:
    args = _parse_args()
    result = renumber_footnotes(args.path, args.output, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False))
    if result.get("issues"):
        # real errors
        if any("not a zip" in s or "not found" in s for s in result["issues"]):
            return 1
    if args.dry_run and result.get("renamed", 0) > 0:
        return 2
    return 0 if result.get("fixed") or result.get("renamed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
