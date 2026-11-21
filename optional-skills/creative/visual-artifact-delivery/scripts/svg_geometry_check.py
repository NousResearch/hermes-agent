#!/usr/bin/env python3
"""Report geometric collisions in an SVG's box primitives (stdlib only, no network).

Authored diagrams fail in the same handful of ways: a label box overlapping the
panel next to it, a legend drawn under a toolbar, a callout that crosses out of
its container, a shape that extends past the canvas, a box duplicated at the same
coordinates. Those are all rectangle arithmetic — they do not need a rendered PNG
and they do not need a vision model, so they can be caught deterministically
before anyone looks at the picture.

This tool reads the *source* of an authored ``.svg`` (or the first ``<svg>`` block
inside an ``.html``), collects the box primitives it can measure exactly
(``rect``, ``circle``, ``ellipse``, including ancestor ``translate()`` offsets),
and reports:

  collision          two shapes partially overlap
  duplicate          two shapes share the same box
  contained          one shape sits inside a much larger one (normal nesting — reported, not a failure)
  stacked            one shape all but covers another of similar size (worth a look)
  boundary           a dashed-stroke frame overlaps a filled shape (deliberate in
                     diagrams that draw a boundary rule around a panel)
  outside-canvas     a shape extends past the viewBox / width-height
  notes              shapes this tool cannot measure (hidden, under a non-translate
                     transform, a path/text with no box) are listed, never silently skipped

Exit status: 0 clean, 1 findings that need a fix, 2 unusable input.

Usage:
    python3 svg_geometry_check.py diagram.svg
    python3 svg_geometry_check.py page.html --json
    python3 svg_geometry_check.py - < diagram.svg
    python3 svg_geometry_check.py diagram.svg --strict-boundaries --min-overlap 100
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

DEFAULT_TOLERANCE = 2.0
DEFAULT_MIN_OVERLAP = 64.0
CONTAINED_RATIO = 0.95
# A container this much larger than what it holds is ordinary nesting (a panel
# wrapping its contents, a chip under a label). Below it, two shapes of similar
# size stacked on each other is worth a second look.
CONTAINMENT_AREA_RATIO = 3.0

_SVG_SPAN = re.compile(r"<svg\b.*?</svg\s*>", re.IGNORECASE | re.DOTALL)
_SVG_SELF_CLOSING = re.compile(r"<svg\b[^>]*/>", re.IGNORECASE | re.DOTALL)
_TRANSLATE = re.compile(r"translate\(\s*([-+0-9.eE]+)(?:[,\s]+([-+0-9.eE]+))?\s*\)")
_OTHER_TRANSFORM = re.compile(r"(matrix|rotate|scale|skewX|skewY)\s*\(")
_NUMBER = re.compile(r"[-+]?(?:[0-9]*\.[0-9]+|[0-9]+)(?:[eE][-+]?[0-9]+)?")


@dataclass
class Box:
    """One measurable primitive, in canvas coordinates."""

    tag: str
    elem_id: str
    x: float
    y: float
    w: float
    h: float
    filled: bool
    dashed: bool
    path: str

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    def label(self) -> str:
        where = f" ({self.x:g},{self.y:g} {self.w:g}x{self.h:g})"
        return f"{self.elem_id}{where}"


@dataclass
class Report:
    canvas: dict | None = None
    shapes: list[Box] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, kind: str, severity: str, a: Box, b: Box | None = None,
            overlap: float | None = None, detail: str = "") -> None:
        self.findings.append({
            "kind": kind,
            "severity": severity,
            "a": a.label(),
            "b": b.label() if b is not None else None,
            "overlap_px": round(overlap, 2) if overlap is not None else None,
            "detail": detail,
        })


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _num(raw: str | None, default: float = 0.0) -> float:
    """SVG length as a float: '120', '120px', '12.5pt' → number, units dropped."""
    if raw is None:
        return default
    m = _NUMBER.search(raw)
    return float(m.group(0)) if m else default


def _elem_id(elem: ET.Element, path: str) -> str:
    for key in ("id", "data-testid"):
        value = elem.get(key)
        if value:
            return f"{local_name(elem.tag)}#{value}"
    href = elem.get("{http://www.w3.org/1999/xlink}href") or elem.get("href")
    if href:
        return f"{local_name(elem.tag)}->{href.lstrip('#')}"
    # No identifier: the ancestry is the identifier (a doc-order path).
    return f"{path}/{local_name(elem.tag)}" if path else local_name(elem.tag)


def extract_svg(text: str) -> str | None:
    """The SVG document inside *text*: the file itself, or the first block in HTML."""
    stripped = text.lstrip("\ufeff \t\r\n")
    if stripped[:400].lower().lstrip().startswith("<svg"):
        return stripped
    m = _SVG_SPAN.search(text)
    if m:
        return m.group(0)
    m = _SVG_SELF_CLOSING.search(text)
    return m.group(0) if m else None


def _canvas_size(root: ET.Element) -> dict:
    """The drawing surface: viewBox when present, else width/height, else None."""
    view_box = root.get("viewBox") or root.get("viewbox")
    if view_box:
        parts = [_num(p) for p in re.split(r"[,\s]+", view_box.strip()) if p]
        if len(parts) == 4:
            return {"x": parts[0], "y": parts[1], "w": parts[2], "h": parts[3],
                    "source": "viewBox"}
    w = root.get("width")
    h = root.get("height")
    if w and h:
        return {"x": 0.0, "y": 0.0, "w": _num(w), "h": _num(h), "source": "width/height"}
    return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0, "source": "unknown"}


def _is_hidden(elem: ET.Element) -> bool:
    if (elem.get("display") or "").strip().lower() == "none":
        return True
    if (elem.get("visibility") or "").strip().lower() in {"hidden", "collapse"}:
        return True
    if (elem.get("style") or "").replace(" ", "").lower().startswith("display:none"):
        return True
    return _num(elem.get("opacity"), 1.0) == 0.0


def _box_of(elem: ET.Element, dx: float, dy: float, tag: str, path: str) -> Box | None:
    fill = (elem.get("fill") or "").strip().lower()
    stroke = (elem.get("stroke") or "").strip().lower()
    style = (elem.get("style") or "").replace(" ", "").lower()
    filled = fill not in {"none", "transparent"} and "fill:none" not in style
    dashed = "dasharray" in style or bool((elem.get("stroke-dasharray") or "").strip())
    if fill in {"none", "transparent"} and not stroke:
        return None
    if tag == "rect":
        x, y = _num(elem.get("x")) + dx, _num(elem.get("y")) + dy
        w, h = _num(elem.get("width")), _num(elem.get("height"))
    elif tag == "circle":
        r = _num(elem.get("r"))
        x, y, w, h = _num(elem.get("cx")) + dx - r, _num(elem.get("cy")) + dy - r, 2 * r, 2 * r
    elif tag == "ellipse":
        rx, ry = _num(elem.get("rx")), _num(elem.get("ry"))
        x, y, w, h = _num(elem.get("cx")) + dx - rx, _num(elem.get("cy")) + dy - ry, 2 * rx, 2 * ry
    else:  # pragma: no cover - guarded by the caller's tag set
        return None
    if w <= 0 or h <= 0:
        return None
    return Box(tag=tag, elem_id=_elem_id(elem, path), x=x, y=y, w=w, h=h,
               filled=filled, dashed=dashed, path=path)


def collect(root: ET.Element) -> tuple[list[Box], list[str]]:
    """Every measurable box in document order, plus the paths this tool skipped."""
    boxes: list[Box] = []
    notes: list[str] = []
    shape_tags = {"rect", "circle", "ellipse"}
    other_tags = {"path", "polygon", "polyline", "line", "text", "tspan", "use", "image"}

    def walk(elem: ET.Element, dx: float, dy: float, path: str, opaque: bool,
             hidden: bool) -> None:
        tag = local_name(elem.tag)
        if tag in {"defs", "clippath", "mask", "symbol", "metadata", "title", "desc"}:
            return
        hidden = hidden or _is_hidden(elem)
        transform = (elem.get("transform") or "").strip()
        local_opaque = opaque
        if transform:
            if _OTHER_TRANSFORM.search(transform):
                # Rotate/scale/matrix/skew: the box below is no longer axis-aligned, so
                # the numbers this tool would report are wrong. Say so instead of guessing.
                local_opaque = True
                notes.append(f"unverifiable {_elem_id(elem, path)} "
                             f"(transform={transform!r}) — box arithmetic skipped")
            for m in _TRANSLATE.finditer(transform):
                dx += float(m.group(1))
                dy += float(m.group(2) or 0.0)
        here = f"{path}/{tag}#{elem.get('id')}" if elem.get("id") else f"{path}/{tag}"
        if tag in shape_tags:
            if hidden:
                notes.append(f"skipped {_elem_id(elem, path)} (hidden)")
            elif local_opaque:
                notes.append(f"unverifiable {_elem_id(elem, path)} "
                             "(inside a group this tool cannot resolve)")
            else:
                box = _box_of(elem, dx, dy, tag, here)
                if box is None:
                    notes.append(f"skipped {_elem_id(elem, path)} (no fill or stroke)")
                else:
                    boxes.append(box)
        elif tag in other_tags and not hidden:
            notes.append(f"skipped {_elem_id(elem, path)} — a {tag} has no box to measure")
        for child in list(elem):
            walk(child, dx, dy, here, local_opaque, hidden)

    walk(root, 0.0, 0.0, "", False, False)
    return boxes, notes


def _intersection(a: Box, b: Box) -> tuple[float, float]:
    return (min(a.x2, b.x2) - max(a.x, b.x), min(a.y2, b.y2) - max(a.y, b.y))


def _same_box(a: Box, b: Box, tol: float) -> bool:
    return (abs(a.x - b.x) <= tol and abs(a.y - b.y) <= tol
            and abs(a.w - b.w) <= tol and abs(a.h - b.h) <= tol)


def find_findings(report: Report, tolerance: float = DEFAULT_TOLERANCE,
                  min_overlap: float = DEFAULT_MIN_OVERLAP) -> None:
    boxes = report.shapes
    for a, b in ((boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
        iw, ih = _intersection(a, b)
        if iw <= tolerance or ih <= tolerance:
            continue
        overlap = iw * ih
        if overlap < min_overlap:
            continue
        if _same_box(a, b, tolerance):
            report.add("duplicate", "fail", a, b, overlap, "two shapes share one box")
            continue
        smaller = min(a.area, b.area)
        larger = max(a.area, b.area)
        if smaller > 0 and overlap >= CONTAINED_RATIO * smaller:
            if larger >= CONTAINMENT_AREA_RATIO * smaller:
                report.add("contained", "info", a, b, overlap, "normal nesting, not a defect")
            else:
                report.add("stacked", "warn", a, b, overlap,
                           "one shape all but covers another of similar size — deliberate?")
            continue
        if a.dashed or b.dashed:
            report.add("boundary", "warn", a, b, overlap,
                       "a dashed frame overlaps a filled shape (intentional boundary rule?)")
            continue
        report.add("collision", "fail", a, b, overlap, "shapes overlap")

    canvas = report.canvas
    if canvas and canvas["w"] > 0 and canvas["h"] > 0:
        cx, cy, cx2, cy2 = canvas["x"], canvas["y"], canvas["x"] + canvas["w"], canvas["y"] + canvas["h"]
        for box in report.shapes:
            if box.x < cx - tolerance or box.y < cy - tolerance \
                    or box.x2 > cx2 + tolerance or box.y2 > cy2 + tolerance:
                report.add("outside-canvas", "fail", box, None, None,
                           f"extends past the {canvas['source']} "
                           f"[{cx:g},{cy:g} {canvas['w']:g}x{canvas['h']:g}]")


def check_text(text: str, tolerance: float = DEFAULT_TOLERANCE,
               min_overlap: float = DEFAULT_MIN_OVERLAP) -> Report:
    svg = extract_svg(text)
    if not svg:
        raise ValueError("no <svg> element found")
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise ValueError(f"SVG is not well-formed XML: {exc}") from exc
    report = Report(canvas=_canvas_size(root))
    report.shapes, report.notes = collect(root)
    find_findings(report, tolerance=tolerance, min_overlap=min_overlap)
    return report


def _failures(report: Report, strict_boundaries: bool) -> list[dict]:
    wanted = {"fail"} | ({"warn"} if strict_boundaries else set())
    return [f for f in report.findings if f["severity"] in wanted]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report geometric collisions in an SVG's box primitives.")
    parser.add_argument("path", nargs="?", default="-",
                        help="an .svg file, an .html file containing <svg>, or - for stdin")
    parser.add_argument("--json", action="store_true", help="machine-readable report on stdout")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help=f"pixels of overlap/shift to ignore (default {DEFAULT_TOLERANCE:g})")
    parser.add_argument("--min-overlap", type=float, default=DEFAULT_MIN_OVERLAP,
                        help=f"ignore overlaps below this area in px^2 (default {DEFAULT_MIN_OVERLAP:g})")
    parser.add_argument("--strict-boundaries", action="store_true",
                        help="treat a dashed frame overlapping a filled shape as a failure")
    args = parser.parse_args(argv)

    try:
        text = sys.stdin.read() if args.path == "-" else open(args.path, encoding="utf-8",
                                                              errors="replace").read()
    except OSError as exc:
        print(f"cannot read {args.path}: {exc}", file=sys.stderr)
        return 2

    try:
        report = check_text(text, tolerance=args.tolerance, min_overlap=args.min_overlap)
    except ValueError as exc:
        print(f"{args.path}: {exc}", file=sys.stderr)
        return 2

    failures = _failures(report, args.strict_boundaries)
    if args.json:
        print(json.dumps({
            "input": args.path,
            "canvas": report.canvas,
            "shapes_checked": [b.label() for b in report.shapes],
            "notes": report.notes,
            "findings": report.findings,
            "failures": len(failures),
        }, indent=1))
    else:
        for finding in report.findings:
            pair = f"{finding['a']} ∩ {finding['b']}" if finding["b"] else finding["a"]
            area = f" overlap={finding['overlap_px']:g}px^2" if finding["overlap_px"] else ""
            print(f"{finding['severity']:4} {finding['kind']:14} {pair}{area} — {finding['detail']}")
        for note in report.notes:
            print(f"note {note}")
        canvas = report.canvas or {}
        surface = (f"{canvas.get('source')} {canvas.get('w'):g}x{canvas.get('h'):g}"
                   if canvas.get("w") else "no canvas size declared")
        print(f"{len(report.shapes)} shape(s) checked on {surface}; "
              f"{len(report.findings)} finding(s), {len(failures)} to fix")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
