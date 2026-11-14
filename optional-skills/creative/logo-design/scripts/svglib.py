"""Shared, dependency-free SVG helpers for the logo-design skill scripts.

Parses SVG structure, colours, transforms and path geometry well enough to audit logos:
anchor counts, bounding boxes, straight-segment angles and tiny sub-paths. Python 3.8+.
"""
import base64
import colorsys
import math
import os
import re
import sys
import xml.etree.ElementTree as ET


def _use_utf8_console():
    # Windows consoles default to cp1252, which cannot print the reports' arrows and symbols.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_use_utf8_console()

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY_DIR = os.path.join(SKILL_DIR, "assets", "library")
LIBRARY_SVG_DIR = os.path.join(LIBRARY_DIR, "svg")
CATALOG_PATH = os.path.join(LIBRARY_DIR, "catalog.json")
STATS_PATH = os.path.join(LIBRARY_DIR, "stats.json")

NAMED_COLORS = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#008000", "blue": "#0000ff",
    "yellow": "#ffff00", "orange": "#ffa500", "purple": "#800080", "gray": "#808080", "grey": "#808080",
    "silver": "#c0c0c0", "navy": "#000080", "teal": "#008080", "maroon": "#800000", "lime": "#00ff00",
    "aqua": "#00ffff", "cyan": "#00ffff", "fuchsia": "#ff00ff", "magenta": "#ff00ff", "olive": "#808000",
    "pink": "#ffc0cb", "brown": "#a52a2a", "gold": "#ffd700", "indigo": "#4b0082", "violet": "#ee82ee",
}
HEX_RE = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
RGB_RE = re.compile(r"rgba?\(\s*([\d.]+%?)\s*,\s*([\d.]+%?)\s*,\s*([\d.]+%?)")
SHAPE_TAGS = ("path", "circle", "ellipse", "rect", "polygon", "polyline", "line")


def local(tag):
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def load_svg(path):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()
    root = ET.fromstring(raw)
    return raw, root


def svg_data_uri(path):
    """Base64 data URI for <img> use. Adds width/height from the viewBox when missing, because browsers give
    viewBox-only SVGs a default 300×150 box (or none) and they can vanish inside flex/max-height layouts."""
    with open(path, "rb") as fh:
        data = fh.read()
    text = data.decode("utf-8", errors="ignore")
    m = re.search(r"<svg\b[^>]*>", text)
    if m:
        tag = m.group(0)
        has_w = re.search(r'\swidth\s*=\s*"', tag)
        has_h = re.search(r'\sheight\s*=\s*"', tag)
        vb = re.search(r'viewBox\s*=\s*"([^"]+)"', tag)
        if vb and not (has_w and has_h):
            parts = [p for p in re.split(r"[ ,]+", vb.group(1).strip()) if p]
            if len(parts) == 4:
                new_tag = re.sub(r'\s(width|height)\s*=\s*"[^"]*"', "", tag)
                new_tag = new_tag[:-1].rstrip("/") + f' width="{parts[2]}" height="{parts[3]}">'
                data = text.replace(tag, new_tag, 1).encode("utf-8")
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode("ascii")


# ----------------------------------------------------------------------------- colours

def normalize_color(value):
    """Return '#rrggbb' for hex / named / rgb() values, None for none/url/currentColor/unknown."""
    if not value:
        return None
    v = value.strip().lower()
    if v in ("none", "transparent", "currentcolor", "inherit") or v.startswith("url("):
        return None
    m = HEX_RE.fullmatch(v) if v.startswith("#") else None
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return "#" + h
    if v in NAMED_COLORS:
        return NAMED_COLORS[v]
    m = RGB_RE.match(v)
    if m:
        vals = []
        for comp in m.groups():
            vals.append(round(float(comp[:-1]) * 2.55) if comp.endswith("%") else round(float(comp)))
        return "#%02x%02x%02x" % tuple(max(0, min(255, c)) for c in vals)
    return None


def style_props(el):
    props = {}
    st = el.get("style")
    if st:
        for part in st.split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                props[k.strip().lower()] = v.strip()
    return props


def hex_to_rgb(hx):
    return tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))


def lightness(hx):
    r, g, b = [c / 255 for c in hex_to_rgb(hx)]
    return colorsys.rgb_to_hls(r, g, b)[1]


def hue_family(hx):
    r, g, b = [c / 255 for c in hex_to_rgb(hx)]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if s < 0.15 or l < 0.08 or l > 0.95:
        if l < 0.2:
            return "black"
        if l > 0.9:
            return "white"
        return "gray"
    d = h * 360
    for lim, name in ((15, "red"), (40, "orange"), (65, "yellow"), (165, "green"), (195, "cyan"),
                      (255, "blue"), (290, "purple"), (335, "pink"), (361, "red")):
        if d < lim:
            return name
    return "red"


def relative_luminance(hx):
    def chan(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = hex_to_rgb(hx)
    return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)


def contrast_ratio(a, b):
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _in_defs_helper(el, parents):
    """True if el lives inside clipPath/mask/defs-only helpers (not visible paint)."""
    p = parents.get(el)
    while p is not None:
        if local(p.tag) in ("clipPath", "mask", "pattern", "symbol", "marker"):
            return True
        p = parents.get(p)
    return False


def collect_colors(root):
    """Visible paint colours (fill, stroke, gradient stops). Defaults to black if nothing is painted."""
    parents = {c: p for p in root.iter() for c in p}
    found = []
    painted_shape = False
    for el in root.iter():
        tag = local(el.tag)
        if _in_defs_helper(el, parents):
            continue
        props = style_props(el)
        for attr in ("fill", "stroke", "stop-color"):
            val = props.get(attr, el.get(attr))
            col = normalize_color(val)
            if col:
                found.append(col)
        if tag in SHAPE_TAGS:
            painted_shape = True
    uniq = []
    for c in found:
        if c not in uniq:
            uniq.append(c)
    if not uniq and painted_shape:
        uniq = ["#000000"]
    return uniq


# ----------------------------------------------------------------------------- transforms

def mat_mul(a, b):
    return (a[0] * b[0] + a[2] * b[1], a[1] * b[0] + a[3] * b[1],
            a[0] * b[2] + a[2] * b[3], a[1] * b[2] + a[3] * b[3],
            a[0] * b[4] + a[2] * b[5] + a[4], a[1] * b[4] + a[3] * b[5] + a[5])


IDENTITY = (1, 0, 0, 1, 0, 0)


def parse_transform(s):
    m = IDENTITY
    if not s:
        return m
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s):
        v = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", args)]
        if name == "matrix" and len(v) == 6:
            t = tuple(v)
        elif name == "translate":
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale":
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate":
            a = math.radians(v[0])
            r = (math.cos(a), math.sin(a), -math.sin(a), math.cos(a), 0, 0)
            if len(v) == 3:
                t = mat_mul(mat_mul((1, 0, 0, 1, v[1], v[2]), r), (1, 0, 0, 1, -v[1], -v[2]))
            else:
                t = r
        elif name == "skewX":
            t = (1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
        elif name == "skewY":
            t = (1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
        else:
            continue
        m = mat_mul(m, t)
    return m


def apply(m, pt):
    x, y = pt
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


# ----------------------------------------------------------------------------- path parsing

NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
TOKEN_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
ARGC = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


_NUM_AT = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def tokenize_path(d):
    """Split path data into command letters and numbers.

    Handles compact arc flags such as 'a10 10 0 0110 10' (flags are single 0/1 characters).
    """
    d = d or ""
    out, i, n = [], 0, len(d)
    cmd, argi = None, 0
    while i < n:
        ch = d[i]
        if ch.isspace() or ch == ",":
            i += 1
            continue
        if ch.isalpha() and ch not in "eE":
            cmd, argi = ch.upper(), 0
            out.append(ch)
            i += 1
            continue
        if cmd == "A" and argi % 7 in (3, 4) and ch in "01":
            out.append(ch)
            argi += 1
            i += 1
            continue
        m = _NUM_AT.match(d, i)
        if not m:
            i += 1
            continue
        out.append(m.group(0))
        argi += 1
        i = m.end()
    return out


def parse_path(d):
    """Return list of subpaths; each subpath is a list of segments in absolute coordinates.

    Segment tuples: ('L', p0, p1) | ('C', p0, c1, c2, p1) | ('Q', p0, c, p1) | ('A', p0, p1, arcparams)
    """
    tokens = tokenize_path(d)
    subpaths, seg_list = [], []
    cur = start = (0.0, 0.0)
    last_ctrl = None
    last_cmd = None
    i = 0
    cmd = None

    def num():
        nonlocal i
        v = float(tokens[i])
        i += 1
        return v

    while i < len(tokens):
        t = tokens[i]
        if re.fullmatch(r"[A-Za-z]", t):
            cmd = t
            i += 1
            if cmd in "Zz":
                if seg_list:
                    if cur != start:
                        seg_list.append(("L", cur, start))
                    subpaths.append(seg_list)
                    seg_list = []
                cur = start
                last_ctrl, last_cmd = None, "Z"
                continue
        if cmd is None:
            i += 1
            continue
        up = cmd.upper()
        rel = cmd.islower()
        need = ARGC[up]
        if i + need > len(tokens) or any(re.fullmatch(r"[A-Za-z]", tokens[j]) for j in range(i, i + need)):
            # malformed: skip token
            i += 1
            continue
        ox, oy = cur if rel else (0.0, 0.0)
        if up == "M":
            x, y = num() + ox, num() + oy
            if seg_list:
                subpaths.append(seg_list)
                seg_list = []
            cur = start = (x, y)
            cmd = "l" if rel else "L"  # implicit lineto after moveto
            last_ctrl = None
        elif up == "L":
            p = (num() + ox, num() + oy)
            seg_list.append(("L", cur, p))
            cur, last_ctrl = p, None
        elif up == "H":
            x = num() + (cur[0] if rel else 0)
            p = (x, cur[1])
            seg_list.append(("L", cur, p))
            cur, last_ctrl = p, None
        elif up == "V":
            y = num() + (cur[1] if rel else 0)
            p = (cur[0], y)
            seg_list.append(("L", cur, p))
            cur, last_ctrl = p, None
        elif up == "C":
            c1 = (num() + ox, num() + oy)
            c2 = (num() + ox, num() + oy)
            p = (num() + ox, num() + oy)
            seg_list.append(("C", cur, c1, c2, p))
            cur, last_ctrl = p, c2
        elif up == "S":
            c1 = (2 * cur[0] - last_ctrl[0], 2 * cur[1] - last_ctrl[1]) if last_cmd in ("C", "S") and last_ctrl else cur
            c2 = (num() + ox, num() + oy)
            p = (num() + ox, num() + oy)
            seg_list.append(("C", cur, c1, c2, p))
            cur, last_ctrl = p, c2
        elif up == "Q":
            c = (num() + ox, num() + oy)
            p = (num() + ox, num() + oy)
            seg_list.append(("Q", cur, c, p))
            cur, last_ctrl = p, c
        elif up == "T":
            c = (2 * cur[0] - last_ctrl[0], 2 * cur[1] - last_ctrl[1]) if last_cmd in ("Q", "T") and last_ctrl else cur
            p = (num() + ox, num() + oy)
            seg_list.append(("Q", cur, c, p))
            cur, last_ctrl = p, c
        elif up == "A":
            rx, ry, rot = abs(num()), abs(num()), num()
            fa, fs = num(), num()
            p = (num() + ox, num() + oy)
            seg_list.append(("A", cur, p, (rx, ry, rot, int(fa) != 0, int(fs) != 0)))
            cur, last_ctrl = p, None
        last_cmd = up if up != "M" else "L"
    if seg_list:
        subpaths.append(seg_list)
    return subpaths


def _arc_points(p0, p1, params, steps=24):
    rx, ry, phi_deg, fa, fs = params
    if rx == 0 or ry == 0 or p0 == p1:
        return [p0, p1]
    phi = math.radians(phi_deg)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1p, y1p = cp * dx + sp * dy, -sp * dx + cp * dy
    lam = (x1p ** 2) / (rx ** 2) + (y1p ** 2) / (ry ** 2)
    if lam > 1:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num_ = rx ** 2 * ry ** 2 - rx ** 2 * y1p ** 2 - ry ** 2 * x1p ** 2
    den = rx ** 2 * y1p ** 2 + ry ** 2 * x1p ** 2
    coef = math.sqrt(max(0.0, num_ / den)) if den else 0.0
    if fa == fs:
        coef = -coef
    cxp, cyp = coef * rx * y1p / ry, -coef * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (p0[0] + p1[0]) / 2
    cy = sp * cxp + cp * cyp + (p0[1] + p1[1]) / 2

    def ang(u, v):
        a = math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])
        return a
    t1 = ang((1, 0), ((x1p - cxp) / rx, (y1p - cyp) / ry))
    dt = ang(((x1p - cxp) / rx, (y1p - cyp) / ry), ((-x1p - cxp) / rx, (-y1p - cyp) / ry))
    if not fs and dt > 0:
        dt -= 2 * math.pi
    elif fs and dt < 0:
        dt += 2 * math.pi
    pts = []
    for k in range(steps + 1):
        t = t1 + dt * k / steps
        x = cx + rx * math.cos(t) * cp - ry * math.sin(t) * sp
        y = cy + rx * math.cos(t) * sp + ry * math.sin(t) * cp
        pts.append((x, y))
    return pts


def segment_points(seg, steps=16):
    kind = seg[0]
    if kind == "L":
        return [seg[1], seg[2]]
    if kind == "C":
        p0, c1, c2, p1 = seg[1:]
        pts = []
        for k in range(steps + 1):
            t = k / steps
            mt = 1 - t
            pts.append((mt ** 3 * p0[0] + 3 * mt * mt * t * c1[0] + 3 * mt * t * t * c2[0] + t ** 3 * p1[0],
                        mt ** 3 * p0[1] + 3 * mt * mt * t * c1[1] + 3 * mt * t * t * c2[1] + t ** 3 * p1[1]))
        return pts
    if kind == "Q":
        p0, c, p1 = seg[1:]
        pts = []
        for k in range(steps + 1):
            t = k / steps
            mt = 1 - t
            pts.append((mt * mt * p0[0] + 2 * mt * t * c[0] + t * t * p1[0],
                        mt * mt * p0[1] + 2 * mt * t * c[1] + t * t * p1[1]))
        return pts
    if kind == "A":
        return _arc_points(seg[1], seg[2], seg[3])
    return []


def bbox_of(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def union_bbox(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


# ----------------------------------------------------------------------------- geometry of a whole document

def shape_to_subpaths(el):
    tag = local(el.tag)
    f = lambda k, d=0.0: float(re.sub(r"[^0-9.eE+-]", "", el.get(k, str(d))) or d)
    if tag == "path":
        return parse_path(el.get("d", ""))
    if tag == "rect":
        x, y, w, h = f("x"), f("y"), f("width"), f("height")
        rx = f("rx", 0) or f("ry", 0)
        if rx:
            rx = min(rx, w / 2, h / 2)
            k = rx
            d = (f"M{x + k},{y} H{x + w - k} A{k},{k} 0 0 1 {x + w},{y + k} V{y + h - k} "
                 f"A{k},{k} 0 0 1 {x + w - k},{y + h} H{x + k} A{k},{k} 0 0 1 {x},{y + h - k} "
                 f"V{y + k} A{k},{k} 0 0 1 {x + k},{y} Z")
            return parse_path(d)
        return parse_path(f"M{x},{y} H{x + w} V{y + h} H{x} Z")
    if tag in ("circle", "ellipse"):
        cx, cy = f("cx"), f("cy")
        rx = f("r") if tag == "circle" else f("rx")
        ry = f("r") if tag == "circle" else f("ry")
        return parse_path(f"M{cx - rx},{cy} A{rx},{ry} 0 1 0 {cx + rx},{cy} A{rx},{ry} 0 1 0 {cx - rx},{cy} Z")
    if tag in ("polygon", "polyline"):
        nums = [float(n) for n in NUM_RE.findall(el.get("points", ""))]
        pts = list(zip(nums[0::2], nums[1::2]))
        if not pts:
            return []
        d = "M" + " L".join(f"{x},{y}" for x, y in pts) + (" Z" if tag == "polygon" else "")
        return parse_path(d)
    if tag == "line":
        return parse_path(f"M{f('x1')},{f('y1')} L{f('x2')},{f('y2')}")
    return []


def iter_shapes(root):
    """Yield (element, matrix) for visible shapes, composing ancestor transforms; skips defs helpers."""
    def walk(el, m, hidden):
        tag = local(el.tag)
        if tag in ("defs", "clipPath", "mask", "pattern", "symbol", "marker", "title", "desc", "metadata", "style"):
            hidden = True
        mm = mat_mul(m, parse_transform(el.get("transform")))
        if tag in SHAPE_TAGS and not hidden:
            yield el, mm
        for ch in el:
            yield from walk(ch, mm, hidden)
    yield from walk(root, IDENTITY, False)


def view_box(root):
    vb = root.get("viewBox")
    if vb:
        parts = [float(x) for x in re.split(r"[ ,]+", vb.strip()) if x]
        if len(parts) == 4:
            return tuple(parts)
    try:
        w = float(re.sub(r"[^0-9.]", "", root.get("width", "")) or 0)
        h = float(re.sub(r"[^0-9.]", "", root.get("height", "")) or 0)
        if w and h:
            return (0.0, 0.0, w, h)
    except ValueError:
        pass
    return None


def document_geometry(root):
    """Return dict with anchors, bbox, straight segments (angle/length) and subpath boxes."""
    anchors = 0
    bbox = None
    lines = []
    sub_boxes = []
    for el, m in iter_shapes(root):
        for sp in shape_to_subpaths(el):
            anchors += len(sp) + 1
            pts_all = []
            for seg in sp:
                pts = [apply(m, p) for p in segment_points(seg)]
                pts_all.extend(pts)
                if seg[0] == "L":
                    a, b = apply(m, seg[1]), apply(m, seg[2])
                    length = math.hypot(b[0] - a[0], b[1] - a[1])
                    if length > 0:
                        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180
                        lines.append((ang, length, a, b))
            bb = bbox_of(pts_all)
            if bb:
                sub_boxes.append(bb)
                bbox = union_bbox(bbox, bb)
    return {"anchors": anchors, "bbox": bbox, "lines": lines, "subpath_boxes": sub_boxes}


def element_counts(root):
    counts = {}
    for el in root.iter():
        t = local(el.tag)
        counts[t] = counts.get(t, 0) + 1
    return counts


def aspect_class(aspect):
    if aspect is None:
        return "unknown"
    if aspect < 0.8:
        return "tall"
    if aspect <= 1.25:
        return "square"
    if aspect <= 2.5:
        return "wide"
    if aspect <= 4.5:
        return "horizontal"
    return "extra-wide"


def resolve_paint(el, parents, prop="fill"):
    """Effective paint of an element: '#rrggbb', 'url' (gradient/pattern) or None (no paint)."""
    node = el
    while node is not None:
        props = style_props(node)
        v = props.get(prop, node.get(prop))
        if v is not None and v.strip().lower() != "inherit":
            low = v.strip().lower()
            if low in ("none", "transparent"):
                return None
            if low.startswith("url("):
                return "url"
            return normalize_color(v) or "#000000"
        node = parents.get(node)
    return "#000000" if prop == "fill" else None


def painted_shape_boxes(root):
    """List of (element, bbox, fill) for visible shapes, in document (paint) order."""
    parents = {c: p for p in root.iter() for c in p}
    out = []
    for el, m in iter_shapes(root):
        pts = []
        for sp in shape_to_subpaths(el):
            for seg in sp:
                pts.extend(apply(m, p) for p in segment_points(seg))
        bb = bbox_of(pts)
        if bb:
            out.append((el, bb, resolve_paint(el, parents, "fill")))
    return out
