#!/usr/bin/env python3
"""Audit a logo SVG against logo-design principles and production rules.

Checks structure (viewBox, title, text/raster/filters), colour discipline, complexity versus the
reference library, near-miss angles (lines 0.3–3° off a clean angle), tiny details that vanish when small,
padding/centring inside the viewBox, strokes that should be expanded, and white "fake knockouts".

Usage:
  python3 scripts/svg_audit.py logo.svg [more.svg ...]
  python3 scripts/svg_audit.py logo.svg --json            # machine-readable
  python3 scripts/svg_audit.py logo.svg --bg "#0F7C80"    # also report contrast of colours on a background

Exit code is 0 unless a file cannot be parsed. Findings are advice, not law: a deliberate choice can
override a warning — but you should be able to say why.
"""
import argparse
import json
import math
import os
import re
import sys

sys.dont_write_bytecode = True  # keep the skill folder clean (no __pycache__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import svglib  # noqa: E402

CLEAN_ANGLES = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]


def load_stats():
    try:
        with open(svglib.STATS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except OSError:
        return None


def audit(path, bg=None):
    findings = []  # (level, code, message)

    def add(level, code, msg):
        findings.append({"level": level, "code": code, "message": msg})

    raw, root = svglib.load_svg(path)
    counts = svglib.element_counts(root)
    vb = svglib.view_box(root)
    info = {"file": os.path.basename(path), "bytes": os.path.getsize(path)}

    # --- structure -------------------------------------------------------------------------
    if not root.get("viewBox"):
        add("WARN", "no-viewbox", "No viewBox: the logo will not scale predictably. Add viewBox=\"0 0 W H\".")
    if vb:
        info["viewBox"] = vb
        w, h = vb[2], vb[3]
        info["aspect"] = round(w / h, 3) if h else None
        info["aspect_class"] = svglib.aspect_class(info["aspect"])
        if any(abs(v - round(v)) > 1e-6 for v in vb):
            add("INFO", "fractional-viewbox", "viewBox has fractional values; integer canvases are easier to grid.")
    if counts.get("text", 0) or counts.get("tspan", 0):
        add("FAIL", "live-text", "Contains <text>: final logos must use outlined paths (fonts differ per machine).")
    if counts.get("image", 0):
        add("FAIL", "raster-image", "Contains <image> (embedded raster). A logo master must be pure vector.")
    if counts.get("filter", 0):
        add("WARN", "filter", "Uses filters (blur/shadow/glow). They render inconsistently and cannot be printed; use flat shapes.")
    if counts.get("mask", 0):
        add("INFO", "mask", "Uses <mask>. Fine for exploration; bake into paths for production (cutters/embroidery).")
    if counts.get("foreignObject", 0):
        add("FAIL", "foreign-object", "Contains <foreignObject> (HTML inside SVG). Not portable.")
    if counts.get("style", 0):
        add("INFO", "style-block", "Uses a <style> block; inline fill attributes are more portable across tools.")
    if not counts.get("title", 0):
        add("INFO", "no-title", "No <title>: add one for accessibility (e.g. <title>Brand logo</title>).")
    if re.search(r"\d+\.\d{4,}", raw):
        add("INFO", "precision", "Coordinates with 4+ decimals: round to 0–2 decimals to reduce size and reveal misalignments.")
    for attr in ("sodipodi", "inkscape:", "sketch:", "data-name", "xmlns:xlink"):
        if attr in raw and attr != "xmlns:xlink":
            add("INFO", "editor-metadata", f"Editor metadata ('{attr}') present; strip it from delivery files.")
            break

    # --- colour ------------------------------------------------------------------------------
    colors = svglib.collect_colors(root)
    info["colors"] = colors
    grads = counts.get("linearGradient", 0) + counts.get("radialGradient", 0)
    info["gradients"] = grads
    n = len(colors)
    if n > 4:
        add("WARN", "many-colors", f"{n} distinct colours. ~75% of reference logos use ≤3; each extra colour costs recall and reproduction.")
    elif n == 0:
        add("WARN", "no-paint", "No visible paint found.")
    if grads:
        add("WARN", "gradient", f"{grads} gradient(s). Keep a flat-colour master; gradients weaken in print and at small sizes (~19% of reference logos use them).")
    opac = re.findall(r'(?:fill-)?opacity\s*[:=]\s*"?\s*(0?\.\d+)', raw)
    if opac:
        add("INFO", "transparency", "Uses transparency/opacity. Overlap effects need a solid-colour fallback for print and embroidery.")
    # fake knockouts: a white shape painted on top of (inside the box of) an earlier coloured shape
    boxes = svglib.painted_shape_boxes(root)
    knock, on_tile = 0, 0
    canvas_area = (vb[2] * vb[3]) if vb else None
    for i, (_, bb_w, fill_w) in enumerate(boxes):
        if not fill_w or fill_w == "url" or svglib.lightness(fill_w) <= 0.97:
            continue
        for _, bb_c, fill_c in boxes[:i]:
            if fill_c and (fill_c == "url" or svglib.lightness(fill_c) <= 0.97):
                if bb_w[0] >= bb_c[0] - 0.5 and bb_w[1] >= bb_c[1] - 0.5 and bb_w[2] <= bb_c[2] + 0.5 and bb_w[3] <= bb_c[3] + 0.5:
                    area_c = (bb_c[2] - bb_c[0]) * (bb_c[3] - bb_c[1])
                    if canvas_area and area_c >= 0.8 * canvas_area:
                        on_tile += 1
                    else:
                        knock += 1
                    break
    if on_tile and not knock:
        add("INFO", "on-tile", "White artwork on a full-canvas tile/container — fine for app icons and avatars; make sure a "
            "version without the tile exists for other uses.")
    if knock:
        add("WARN", "white-knockout", f"{knock} white shape(s) are painted on top of coloured shapes. If they are meant as "
            "holes, they will show as white blobs on coloured/photo backgrounds and break one-colour versions — make real "
            "holes (fill-rule=\"evenodd\" or subtracted paths). Ignore if the white is deliberate (e.g. a symbol on a tile).")
    all_white = bool(colors) and all(svglib.lightness(c) > 0.9 for c in colors)
    if bg:
        bgc = svglib.normalize_color(bg)
        if bgc and all_white and svglib.lightness(bgc) > 0.6:
            add("INFO", "reversed-file", f"All paint is white/near-white: this looks like a reversed version — test it with a "
                "dark --bg instead of " + bgc + ".")
        elif bgc:
            low = [f"{c} ({svglib.contrast_ratio(c, bgc):.1f}:1)" for c in colors if svglib.contrast_ratio(c, bgc) < 3]
            if low:
                add("WARN", "low-contrast", f"Low contrast on {bgc}: " + ", ".join(low) + " — aim for ≥3:1 for logo parts, 4.5:1 for small text.")

    # --- strokes -----------------------------------------------------------------------------
    stroked = 0
    for el in root.iter():
        props = svglib.style_props(el)
        s = props.get("stroke", el.get("stroke"))
        if s and s.strip().lower() not in ("none", "transparent"):
            stroked += 1
    if stroked:
        add("INFO", "strokes", f"{stroked} element(s) with strokes. Expand strokes to filled outlines in the master so "
            "the mark scales and cuts predictably (or document that it is a stroke-based variant).")

    # --- geometry ----------------------------------------------------------------------------
    geo = svglib.document_geometry(root)
    info["anchors"] = geo["anchors"]
    size = max(vb[2], vb[3]) if vb else None
    stats = load_stats()
    if stats:
        key = "anchors_square" if info.get("aspect_class") == "square" else "anchors_all"
        d = stats[key]
        info["library_anchor_percentiles"] = d
        if geo["anchors"] > d["p95"]:
            add("WARN", "complex", f"{geo['anchors']} anchor points — more than 95% of comparable reference logos "
                f"(median {d['median']}). Simplify: fewer points, merged shapes, less detail.")
        elif geo["anchors"] > d["p75"]:
            add("INFO", "complexity", f"{geo['anchors']} anchor points (library median {d['median']}, p75 {d['p75']}). "
                "Check every point earns its place.")
    # near-miss angles
    if size:
        near = []
        for ang, length, a, b in geo["lines"]:
            if length < size * 0.04:
                continue
            closest = min(CLEAN_ANGLES, key=lambda c: abs(c - ang))
            dev = abs(closest - ang)
            if 0.3 < dev <= 3.0:
                near.append((round(ang, 1), closest % 180, round(length, 1), a))
        if near:
            sample = "; ".join(f"{x[0]}° (→{x[1]}°) len {x[2]} at ({x[3][0]:.0f},{x[3][1]:.0f})" for x in near[:6])
            add("WARN", "near-miss-angle", f"{len(near)} straight edge(s) are 0.3–3° off a clean angle — they read as "
                f"mistakes. Snap them: {sample}" + (" …" if len(near) > 6 else "") +
                ". (Expected — and fine — for type set on a curve or deliberately rotated elements.)")
        # tiny details — symbols must survive ~48 px, lockups ~32 px of height
        if info.get("aspect_class") in ("wide", "horizontal", "extra-wide"):
            ref, label = vb[3] / 32, "1/32 of the lockup height (≈1 px at 32 px tall)"
        else:
            ref, label = size / 48, "1/48 of the canvas (≈1 px at 48 px)"
        tiny = [bb for bb in geo["subpath_boxes"]
                if max(bb[2] - bb[0], bb[3] - bb[1]) < ref and (bb[2] - bb[0]) * (bb[3] - bb[1]) > 0]
        if tiny:
            ex = "; ".join(f"{bb[2] - bb[0]:.1f}×{bb[3] - bb[1]:.1f} at ({bb[0]:.0f},{bb[1]:.0f})" for bb in tiny[:4])
            add("WARN", "tiny-detail", f"{len(tiny)} sub-shape(s) smaller than {label}: {ex}{' …' if len(tiny) > 4 else ''}. "
                "They vanish when small — enlarge, merge or remove them, or provide a small-size version.")
    # padding / centring
    bb = geo["bbox"]
    if bb and vb:
        x0, y0, w, h = vb
        left, top = bb[0] - x0, bb[1] - y0
        right, bottom = x0 + w - bb[2], y0 + h - bb[3]
        info["content_bbox"] = [round(v, 2) for v in bb]
        info["margins"] = {"left": round(left, 1), "right": round(right, 1), "top": round(top, 1), "bottom": round(bottom, 1)}
        tol = max(w, h) * 0.01
        if min(left, right, top, bottom) < -tol:
            add("WARN", "overflow", "Artwork extends beyond the viewBox and will be clipped.")
        if abs(left - right) > max(w, h) * 0.03:
            add("INFO", "off-centre-x", f"Horizontal margins differ (left {left:.1f}, right {right:.1f}). Intentional optical "
                "centring? Otherwise centre it.")
        if abs(top - bottom) > max(w, h) * 0.03:
            add("INFO", "off-centre-y", f"Vertical margins differ (top {top:.1f}, bottom {bottom:.1f}). Remember the optical "
                "centre sits slightly above the geometric centre.")
        fill_ratio = ((bb[2] - bb[0]) * (bb[3] - bb[1])) / (w * h) if w and h else 0
        info["bbox_fill_ratio"] = round(fill_ratio, 3)
        if info.get("aspect_class") == "square" and fill_ratio < 0.35:
            add("INFO", "small-in-canvas", "The mark occupies little of its square canvas; tighten the viewBox or scale up.")
    # aspect advice
    ac = info.get("aspect_class")
    if ac == "tall":
        add("INFO", "tall", "Tall proportions (<0.8:1) are awkward in headers and app icons; consider a squarer symbol.")
    if ac == "extra-wide":
        add("INFO", "extra-wide", "Extra-wide lockup (>4.5:1) shrinks badly; provide a stacked lockup and a standalone symbol.")

    score = 100
    for f in findings:
        score -= {"FAIL": 25, "WARN": 8, "INFO": 1}[f["level"]]
    info["score"] = max(0, score)
    info["findings"] = findings
    return info


def print_report(info):
    print(f"\n=== {info['file']}  ({info['bytes']} bytes)")
    vb = info.get("viewBox")
    if vb:
        print(f"viewBox {vb[0]:g} {vb[1]:g} {vb[2]:g} {vb[3]:g} · aspect {info.get('aspect')} ({info.get('aspect_class')})")
    print(f"colours {len(info.get('colors', []))}: {' '.join(info.get('colors', []))} · gradients {info.get('gradients', 0)} · "
          f"anchors {info.get('anchors')}")
    if info.get("margins"):
        m = info["margins"]
        print(f"margins L{m['left']} R{m['right']} T{m['top']} B{m['bottom']} · bbox fill {info.get('bbox_fill_ratio')}")
    order = {"FAIL": 0, "WARN": 1, "INFO": 2}
    if not info["findings"]:
        print("✔ no issues found")
    for f in sorted(info["findings"], key=lambda f: order[f["level"]]):
        icon = {"FAIL": "✖", "WARN": "▲", "INFO": "·"}[f["level"]]
        print(f"{icon} {f['level']:4s} [{f['code']}] {f['message']}")
    print(f"production-readiness score: {info['score']}/100 (heuristic)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--bg", help="background colour to test contrast against, e.g. '#ffffff'")
    a = ap.parse_args()
    results, rc = [], 0
    for p in a.files:
        try:
            results.append(audit(p, a.bg))
        except Exception as exc:
            rc = 1
            results.append({"file": p, "error": str(exc)})
    if a.json:
        print(json.dumps(results, indent=1, ensure_ascii=False))
    else:
        for r in results:
            if "error" in r:
                print(f"\n=== {r['file']}\n✖ could not parse: {r['error']}")
            else:
                print_report(r)
    return rc


if __name__ == "__main__":
    sys.exit(main())
