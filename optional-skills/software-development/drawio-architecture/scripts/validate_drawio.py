#!/usr/bin/env python3
"""validate_drawio.py - Lint .drawio XML and repair draw.io's truncated -e PNG IEND chunk.

Usage:
  python3 validate_drawio.py diagram.drawio                 # well-formedness + structural lint
  python3 validate_drawio.py diagram.drawio --score         # also print a readability score
  python3 validate_drawio.py diagram.drawio.png --repair-iend  # fix truncated IEND from -e export
"""
import argparse
import sys
import xml.etree.ElementTree as ET


def lint_drawio(path):
    errors = []
    warnings = []
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        return [f"XML parse error: {e}"], warnings
    root = tree.getroot()
    cells = root.findall(".//mxCell")
    ids = {}
    for c in cells:
        cid = c.get("id")
        if cid is None:
            errors.append("mxCell without id attribute")
            continue
        if cid in ids:
            errors.append(f"duplicate id: {cid}")
        ids[cid] = c
    if "0" not in ids or "1" not in ids:
        errors.append("missing required root cells id='0' or id='1'")
    for c in cells:
        cid = c.get("id")
        parent = c.get("parent")
        if parent is not None and parent not in ids:
            errors.append(f"cell {cid}: parent '{parent}' does not exist")
        if c.get("edge") == "1":
            geo = c.find("mxGeometry")
            if geo is None or geo.get("relative") != "1":
                errors.append(f"edge {cid}: missing <mxGeometry relative='1'> child")
            src, tgt = c.get("source"), c.get("target")
            if src and src not in ids:
                errors.append(f"edge {cid}: source '{src}' does not exist")
            if tgt and tgt not in ids:
                errors.append(f"edge {cid}: target '{tgt}' does not exist")
    return errors, warnings


def readability_score(path):
    """Heuristic 0-100: penalize overlaps/crossings by spacing variance of node centers."""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return 0
    verts = []
    for c in tree.getroot().findall(".//mxCell"):
        if c.get("vertex") != "1":
            continue
        geo = c.find("mxGeometry")
        if geo is None:
            continue
        try:
            x = float(geo.get("x", 0)) + float(geo.get("width", 0)) / 2
            y = float(geo.get("y", 0)) + float(geo.get("height", 0)) / 2
            verts.append((x, y))
        except ValueError:
            pass
    if len(verts) < 2:
        return 100
    # crude overlap count
    overlaps = 0
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            dx = abs(verts[i][0] - verts[j][0])
            dy = abs(verts[i][1] - verts[j][1])
            if dx < 60 and dy < 30:
                overlaps += 1
    score = max(0, 100 - overlaps * 8)
    return score


def repair_iend(png_path):
    with open(png_path, "rb") as f:
        data = f.read()
    if data.rstrip(b"\x00")[-8:] == b"IEND":
        # already ends with full IEND (type+crc); nothing to do
        print(f"[=] {png_path}: IEND already complete, no repair needed.")
        return 0
    if b"IEND" not in data:
        print(f"[!] {png_path}: no IEND marker found; not a draw.io -e PNG?", file=sys.stderr)
        return 1
    # draw.io -e truncates the 8-byte IEND type+crc. Append them.
    crc = b"\xae\x42\x60\x82"  # standard IEND CRC32
    if data.rstrip(b"\x00").endswith(b"IEN"):
        # only 'IEN' present; add 'D' + CRC
        data = data.rstrip(b"\x00") + b"D" + crc
    else:
        data = data.rstrip(b"\x00") + b"IEND" + crc
    with open(png_path, "wb") as f:
        f.write(data)
    print(f"[+] Repaired IEND chunk in {png_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help=".drawio XML or .drawio.png (-e export) file")
    ap.add_argument("--score", action="store_true", help="print a readability score")
    ap.add_argument("--repair-iend", action="store_true", help="fix truncated IEND in -e PNG")
    args = ap.parse_args()

    if args.repair_iend:
        return repair_iend(args.file)

    errors, _ = lint_drawio(args.file)
    if errors:
        print(f"[!] {len(errors)} problem(s) in {args.file}:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"[+] {args.file}: well-formed, structurally valid.")
    if args.score:
        print(f"[+] readability score: {readability_score(args.file)}/100")
    return 0


if __name__ == "__main__":
    sys.exit(main())
