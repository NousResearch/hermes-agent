#!/usr/bin/env python3
"""Search the bundled reference library of 1,400+ real-world SVG logos.

Use it to study how existing marks solve a problem (a technique, a subject, a mark type), to see what a
category already looks like (so you can avoid it), and to pull a handful of files to read as SVG examples.
The logos are trademarks of their owners: study them, never copy them.

Examples:
  python3 scripts/search_library.py --technique negative-space --exemplary
  python3 scripts/search_library.py --type letterform --geometry circle --max-colors 1 --limit 12
  python3 scripts/search_library.py --subject "bird" --format paths
  python3 scripts/search_library.py --industry payments-fintech --summary
  python3 scripts/search_library.py --query cloud --type pictorial --format json
  python3 scripts/search_library.py --list-values          # show every allowed filter value

Filters combine with AND. Multi-value filters (--technique, --geometry) accept a comma list and require all.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.dont_write_bytecode = True  # keep the skill folder clean (no __pycache__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import svglib  # noqa: E402


def load_catalog():
    if not os.path.exists(svglib.CATALOG_PATH):
        sys.exit("catalog.json not found - run scripts/build_catalog.py first")
    with open(svglib.CATALOG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def matches(r, a):
    if a.type and r.get("mark_type") != a.type:
        return False
    if a.symbol_type and r.get("symbol_type") != a.symbol_type:
        return False
    if a.technique and not all(t in r.get("techniques", []) for t in a.technique.split(",")):
        return False
    if a.geometry and not all(g in r.get("geometry", []) for g in a.geometry.split(",")):
        return False
    if a.industry and r.get("industry") != a.industry:
        return False
    if a.color and a.color not in (r.get("color_families", []) + [r.get("primary_family")]):
        return False
    if a.primary_color and r.get("primary_family") != a.primary_color:
        return False
    if a.max_colors is not None and r.get("n_colors", 99) > a.max_colors:
        return False
    if a.min_colors is not None and r.get("n_colors", 0) < a.min_colors:
        return False
    if a.aspect and r.get("aspect_class") != a.aspect:
        return False
    if a.variant and r.get("variant") != a.variant:
        return False
    if a.type_style and r.get("type_style") != a.type_style:
        return False
    if a.case and r.get("case") != a.case:
        return False
    if a.mood and a.mood.lower() not in [m.lower() for m in r.get("mood", [])]:
        return False
    if a.exemplary and not r.get("exemplary"):
        return False
    if a.no_gradient and r.get("gradients"):
        return False
    if a.subject and a.subject.lower() not in (r.get("subject") or "").lower():
        return False
    if a.query:
        hay = " ".join([r.get("file", ""), r.get("subject") or "", r.get("note") or "", " ".join(r.get("mood", []))]).lower()
        if not all(w in hay for w in a.query.lower().split()):
            return False
    return True


def summary(rows, total):
    n = len(rows)
    print(f"{n} logos match (of {total}).\n")
    if not n:
        return

    def block(title, counter, k=10):
        print(title)
        for key, cnt in counter.most_common(k):
            print(f"  {str(key):24s} {cnt:4d}  {cnt / n * 100:5.1f}%")
        print()
    block("Mark types", Counter(r.get("mark_type") for r in rows))
    block("Symbol type (combination marks)", Counter(r.get("symbol_type") for r in rows if r.get("symbol_type")))
    block("Primary colour family", Counter(r.get("primary_family") for r in rows))
    block("Techniques", Counter(t for r in rows for t in r.get("techniques", [])), 12)
    block("Geometry", Counter(g for r in rows for g in r.get("geometry", [])))
    block("Type style (when type present)", Counter(r.get("type_style") for r in rows if r.get("type_style")))
    cols = Counter(r.get("n_colors") for r in rows)
    few = sum(v for k, v in cols.items() if k is not None and k <= 2)
    print(f"Colour count: {few / n * 100:.0f}% use 1-2 colours; gradients in "
          f"{sum(1 for r in rows if r.get('gradients')) / n * 100:.0f}%.")
    ex = [r["file"] for r in rows if r.get("exemplary")][:12]
    if ex:
        print("Exemplary examples:", ", ".join(ex))
    print("\nReading: the most frequent types/colours/techniques are the category's conventions. "
          "Repeating them signals belonging; departing from them creates distinction.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", help="mark type: wordmark|lettermark|letterform|pictorial|abstract|mascot|emblem|combination")
    ap.add_argument("--symbol-type", help="symbol type inside combination marks")
    ap.add_argument("--technique", help="e.g. negative-space, geometric-construction, letter-substitution (comma = AND)")
    ap.add_argument("--geometry", help="e.g. circle, hexagon, triangle, organic (comma = AND)")
    ap.add_argument("--industry", help="e.g. developer-tools, payments-fintech, database-data")
    ap.add_argument("--color", help="colour family present anywhere: red|orange|yellow|green|cyan|blue|purple|pink|black|white|gray")
    ap.add_argument("--primary-color", help="dominant family: a hue, 'multi' or 'mono'")
    ap.add_argument("--max-colors", type=int)
    ap.add_argument("--min-colors", type=int)
    ap.add_argument("--aspect", help="square|wide|horizontal|extra-wide|tall")
    ap.add_argument("--variant", help="main|icon (icon = standalone symbol of a brand that also has a lockup)")
    ap.add_argument("--type-style", help="geometric-sans|grotesque-sans|humanist-sans|rounded-sans|serif|slab|script|display-custom|monospace")
    ap.add_argument("--case", help="lowercase|uppercase|titlecase|mixed")
    ap.add_argument("--mood", help="single mood adjective, e.g. friendly, technical, playful")
    ap.add_argument("--subject", help="substring of the visual subject, e.g. bird, shield, cloud, letter a")
    ap.add_argument("--query", help="free-text words matched against file name, subject, note, mood")
    ap.add_argument("--exemplary", action="store_true", help="only strong teaching examples")
    ap.add_argument("--no-gradient", action="store_true")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--format", choices=("table", "paths", "json"), default="table")
    ap.add_argument("--summary", action="store_true", help="aggregate view: conventions of the matching set")
    ap.add_argument("--list-values", action="store_true", help="print allowed values for every filter")
    a = ap.parse_args()

    cat = load_catalog()
    if a.list_values:
        for key in ("mark_type", "symbol_type", "industry", "primary_family", "aspect_class", "type_style", "case"):
            print(f"{key}: {', '.join(sorted({str(r.get(key)) for r in cat if r.get(key)}))}")
        for key in ("techniques", "geometry"):
            vals = Counter(v for r in cat for v in r.get(key, []))
            print(f"{key}: {', '.join(f'{k}({c})' for k, c in vals.most_common())}")
        moods = Counter(m.lower() for r in cat for m in r.get("mood", []))
        print("mood (top 40):", ", ".join(k for k, _ in moods.most_common(40)))
        return

    # forgive near-miss filter values: "security" -> "security-identity", "fintech" -> "payments-fintech"
    for attr, field in (("industry", "industry"), ("type", "mark_type"), ("symbol_type", "symbol_type"),
                        ("type_style", "type_style"), ("aspect", "aspect_class"), ("primary_color", "primary_family")):
        val = getattr(a, attr)
        if not val:
            continue
        known = sorted({str(r.get(field)) for r in cat if r.get(field)})
        if val in known:
            continue
        close = [k for k in known if val.lower() in k.lower() or k.lower() in val.lower()]
        if len(close) == 1:
            print(f"(--{attr.replace('_', '-')} '{val}' → using '{close[0]}')")
            setattr(a, attr, close[0])
        else:
            hint = ", ".join(close) if close else ", ".join(known)
            sys.exit(f"unknown --{attr.replace('_', '-')} '{val}'. Did you mean: {hint}")
    for attr, field in (("technique", "techniques"), ("geometry", "geometry")):
        val = getattr(a, attr)
        if not val:
            continue
        known = sorted({v for r in cat for v in r.get(field, [])})
        fixed = []
        for part in val.split(","):
            if part in known:
                fixed.append(part)
                continue
            close = [k for k in known if part.lower() in k.lower()]
            if len(close) == 1:
                print(f"(--{attr} '{part}' → using '{close[0]}')")
                fixed.append(close[0])
            else:
                sys.exit(f"unknown --{attr} '{part}'. Did you mean: {', '.join(close) if close else ', '.join(known)}")
        setattr(a, attr, ",".join(fixed))

    rows = [r for r in cat if matches(r, a)]
    rows.sort(key=lambda r: (not r.get("exemplary"), r.get("anchors", 0)))
    if a.summary:
        summary(rows, len(cat))
        return
    shown = rows[: a.limit]
    if a.format == "json":
        print(json.dumps(shown, ensure_ascii=False, indent=1))
    elif a.format == "paths":
        for r in shown:
            print(os.path.join(svglib.LIBRARY_SVG_DIR, r["file"]))
    else:
        print(f"{len(rows)} match, showing {len(shown)}  (★ = exemplary; files in {svglib.LIBRARY_SVG_DIR})")
        for r in shown:
            star = "★" if r.get("exemplary") else " "
            kind = r.get("mark_type") or "?"
            if r.get("symbol_type"):
                kind += "/" + r["symbol_type"]
            print(f"{star} {r['file']:34s} {kind:22s} {r.get('n_colors', 0):2d}c  {(r.get('subject') or '')[:48]}")
            if r.get("note") and r.get("exemplary"):
                print(f"      ↳ {r['note']}")


if __name__ == "__main__":
    main()
