#!/usr/bin/env python3
"""Embed TTF fonts into a .pptx using OOXML embeddedFontLst.

Example:
  python embed_fonts.py deck.pptx --typeface Vazirmatn \
      --regular Vazirmatn-Regular.ttf --bold Vazirmatn-Bold.ttf

PowerPoint honors embedded fonts; LibreOffice ignores them (install the font
locally too). Modifies the file in place (rewrites the zip cleanly).
"""
import argparse
import re
import shutil
import sys
import zipfile

R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pptx")
    ap.add_argument("--typeface", required=True, help="font family name as used in runs")
    ap.add_argument("--regular")
    ap.add_argument("--bold")
    ap.add_argument("--italic")
    ap.add_argument("--bolditalic")
    args = ap.parse_args(argv)

    styles = [("regular", args.regular), ("bold", args.bold),
              ("italic", args.italic), ("bolditalic", args.bolditalic)]
    styles = [(n, p) for n, p in styles if p]
    if not styles:
        print("no font files given", file=sys.stderr)
        return 2

    with zipfile.ZipFile(args.pptx) as zin:
        data = {i.filename: zin.read(i.filename) for i in zin.infolist()}

    parts = []
    embed = f'<p:embeddedFontLst xmlns:p="{P}" xmlns:r="{R}"><p:embeddedFont typeface="{args.typeface}">'
    for i, (style, path) in enumerate(styles, start=1):
        rid, part = f"rIdFont{i}", f"fonts/font{i}.fntdata"
        embed += f'<p:{style} r:id="{rid}"/>'
        parts.append((rid, part, path))
    embed += "</p:embeddedFont></p:embeddedFontLst>"

    # 1. presentation.xml: insert after <p:notesSz> (schema order)
    pres = data["ppt/presentation.xml"].decode("utf-8")
    m = re.search(r"<p:notesSz[^>]*/>", pres)
    assert m, "no <p:notesSz> in presentation.xml"
    data["ppt/presentation.xml"] = (pres[:m.end()] + embed + pres[m.end():]).encode("utf-8")

    # 2. relationships
    rels = data["ppt/_rels/presentation.xml.rels"].decode("utf-8")
    add = "".join(f'<Relationship Id="{rid}" Type="{R}/embeddedFont" Target="{part}"/>'
                  for rid, part, _ in parts)
    data["ppt/_rels/presentation.xml.rels"] = rels.replace(
        "</Relationships>", add + "</Relationships>").encode("utf-8")

    # 3. content types
    ct = data["[Content_Types].xml"].decode("utf-8")
    data["[Content_Types].xml"] = ct.replace(
        "</Types>", '<Default Extension="fntdata" ContentType="application/x-fontdata"/></Types>'
    ).encode("utf-8")

    # 4. font parts
    for _, part, path in parts:
        with open(path, "rb") as fh:
            data[part] = fh.read()

    tmp = args.pptx + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for name, blob in data.items():
            zo.writestr(name, blob)
    shutil.move(tmp, args.pptx)
    print(f"embedded {len(parts)} font style(s) into {args.pptx}: "
          f"{[p for _, p, _ in parts]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
