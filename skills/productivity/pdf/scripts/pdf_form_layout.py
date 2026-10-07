#!/usr/bin/env python3
"""Validate a form-spec layout BEFORE building the PDF, with optional
visual overlay rendering for review with a vision model.

Input is the same JSON spec pdf_make_form.py consumes: each field has
"page", "label_box" and "entry_box" as [x0, y0, x1, y1] in PDF points
(origin bottom-left). Checks per field:
  - boxes lie within the page bounds
  - boxes are well-formed (x0 < x1, y0 < y1)
  - entry boxes meet minimum sizes (default 8x8 pt; 12 pt height for text)
  - no two entry boxes on the same page overlap
  - the label sits near its entry box (default within 150 pt gap)

Prints a JSON report {"ok": bool, "fields": [...], "errors": N};
exit 0 when clean, 1 when any check fails.

--render-overlay OUT.png rasterizes --overlay-page (default 1) of an
existing PDF (--pdf; a blank page of spec size if omitted) and draws
label boxes (blue) and entry boxes (red) with field names, for
review with `vision_analyze`. If no rasterizer (pypdfium2/pdftoppm) is
available the overlay is skipped with {"rendered": false, "missing": [...]}
and validation exit status is unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys

MIN_W = 8.0
MIN_H = 8.0
MIN_TEXT_H = 12.0
MAX_LABEL_GAP = 150.0


def _boxes_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _box_gap(a, b) -> float:
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return (dx ** 2 + dy ** 2) ** 0.5


def _page_size(spec: dict) -> tuple[float, float]:
    sizes = {"a4": (595.27, 841.89), "letter": (612.0, 792.0)}
    ps = spec.get("page_size", "A4")
    if isinstance(ps, (list, tuple)) and len(ps) == 2:
        return float(ps[0]), float(ps[1])
    return sizes.get(str(ps).lower(), sizes["a4"])


def _check_box(box, width, height, min_w, min_h, kind) -> list[str]:
    problems = []
    if box is None:
        return [f"{kind}_box missing"]
    x0, y0, x1, y1 = (float(v) for v in box)
    if x0 >= x1 or y0 >= y1:
        problems.append(f"{kind}_box malformed (need x0<x1 and y0<y1): {box}")
        return problems
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        problems.append(f"{kind}_box outside page bounds {width}x{height}: {box}")
    if x1 - x0 < min_w or y1 - y0 < min_h:
        problems.append(f"{kind}_box below minimum size {min_w}x{min_h}: {box}")
    return problems


def validate(spec: dict) -> dict:
    width, height = _page_size(spec)
    fields = spec.get("fields", [])
    report = []
    entry_boxes: dict[int, list[tuple[str, list[float]]]] = {}
    for f in fields:
        name = f.get("name", "?")
        page = int(f.get("page", 1))
        problems = []
        min_h = MIN_TEXT_H if f.get("type", "text") in ("text", "dropdown") else MIN_H
        entry = f.get("entry_box")
        problems += _check_box(entry, width, height, MIN_W, min_h, "entry")
        label = f.get("label_box")
        if f.get("label"):
            problems += _check_box(label, width, height, 4, 4, "label")
            if entry and label and len(problems) == 0:
                gap = _box_gap([float(v) for v in label], [float(v) for v in entry])
                if gap > MAX_LABEL_GAP:
                    problems.append(f"label is {gap:.0f}pt from its entry box (max {MAX_LABEL_GAP:.0f})")
                if _boxes_overlap([float(v) for v in label], [float(v) for v in entry]):
                    problems.append("label_box overlaps its own entry_box")
        if entry and not any("malformed" in p or "missing" in p for p in problems):
            ebox = [float(v) for v in entry]
            for other_name, other_box in entry_boxes.get(page, []):
                if _boxes_overlap(ebox, other_box):
                    problems.append(f"entry_box overlaps field {other_name!r}")
            entry_boxes.setdefault(page, []).append((name, ebox))
        report.append({"name": name, "page": page, "ok": not problems, "problems": problems})
    errors = sum(1 for r in report if not r["ok"])
    return {"ok": errors == 0, "page_size": [width, height],
            "field_count": len(report), "errors": errors, "fields": report}


def render_overlay(spec: dict, pdf_path: str | None, page: int, out_png: str,
                   dpi: int = 100) -> dict:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import _raster
    if not _raster.available_backends() and pdf_path:
        return {"rendered": False, "missing": _raster.missing_hints()}
    from PIL import Image, ImageDraw
    width, height = _page_size(spec)
    if pdf_path:
        img = _raster.rasterize_page(pdf_path, page, dpi=dpi)
        if img is None:
            return {"rendered": False, "missing": _raster.missing_hints()}
        scale = img.width / width
    else:
        scale = dpi / 72.0
        img = Image.new("RGB", (int(width * scale), int(height * scale)), "white")
    draw = ImageDraw.Draw(img)

    def to_px(box):
        x0, y0, x1, y1 = (float(v) for v in box)
        return [x0 * scale, img.height - y1 * scale, x1 * scale, img.height - y0 * scale]

    for f in spec.get("fields", []):
        if int(f.get("page", 1)) != page:
            continue
        if f.get("entry_box"):
            px = to_px(f["entry_box"])
            draw.rectangle(px, outline=(220, 30, 30), width=2)
            draw.text((px[0] + 2, px[1] + 2), str(f.get("name", "?")), fill=(220, 30, 30))
        if f.get("label_box"):
            draw.rectangle(to_px(f["label_box"]), outline=(30, 60, 220), width=2)
    img.save(out_png)
    return {"rendered": True, "overlay": out_png, "page": page,
            "legend": {"entry_box": "red", "label_box": "blue"}}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="Validate form-spec layout (boxes, overlaps, label pairing); "
                    "optionally render an annotated overlay image.")
    parser.add_argument("spec", help="Form spec JSON (same format as pdf_make_form.py)")
    parser.add_argument("--pdf", help="Existing PDF to rasterize under the overlay "
                                      "(blank page if omitted)")
    parser.add_argument("--render-overlay", metavar="OUT_PNG",
                        help="Write an annotated PNG for visual review")
    parser.add_argument("--overlay-page", type=int, default=1, help="1-based page (default 1)")
    parser.add_argument("--dpi", type=int, default=100, help="Overlay render DPI (default 100)")
    args = parser.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    result = validate(spec)
    if args.render_overlay:
        result["overlay"] = render_overlay(spec, args.pdf, args.overlay_page,
                                           args.render_overlay, args.dpi)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
