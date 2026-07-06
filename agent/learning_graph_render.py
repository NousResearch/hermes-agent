"""Terminal renderer for the learning timeline (learned skills + memories).

The desktop app (``apps/desktop/src/app/starmap``) paints a GPU radial
constellation; a terminal can't, so this is a *rendition* of the same data as a
timeline bar chart — date rows, proportional skill/memory bars colored by the
day's dominant category, and a cumulative trajectory sparkline — plus per-slice
bucket metadata the TUI walks as a tree. The age gradient and complementary
memory ink are ported from the desktop source, not guessed.

Grids are emitted as style runs — ``[text, style, alpha, hex?]`` — so each
consumer maps the semantic style + brightness onto its own palette; the
optional 4th element overrides the base color (category heatmap). Pure,
stdlib-only.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

# time-axis.ts LEAD_IN: the oldest node sits just off recency 0.
LEAD_IN = 0.06

# constants.ts AGE_GRADIENT — old quiet, recent bright.
AGE_OLD_INK = 0.42
AGE_MID_INK = 0.74
AGE_NEW_INK = 0.95
AGE_MID = 0.52

# Style keys consumers map to base colors (brightness = the run alpha).
STYLE_BG = "bg"
STYLE_SKILL = "skill"
STYLE_MEMORY = "memory"
STYLE_LABEL = "label"
STYLE_DIM = "dim"

# Legend glyphs mirror NODE_SHAPE (skill = circle, memory = diamond).
SKILL_GLYPH = "●"
MEMORY_GLYPH = "◆"
_LABEL_KEYS = tuple("123456789abc")

Run = list  # [text, style, alpha, hex?]
Row = list  # list[Run]
Grid = list  # list[Row]


def _to_ts(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _smoothstep(p: float) -> float:
    p = _clamp(p, 0.0, 1.0)
    return p * p * (3 - 2 * p)


def recency_ink(rec: float) -> float:
    """Port of geometry.ts ``recencyInk`` — smoothstep age → ink alpha."""
    t = _clamp(rec, 0.0, 1.0)
    if t <= AGE_MID:
        return AGE_OLD_INK + (AGE_MID_INK - AGE_OLD_INK) * _smoothstep(t / AGE_MID)
    return AGE_MID_INK + (AGE_NEW_INK - AGE_MID_INK) * _smoothstep((t - AGE_MID) / (1 - AGE_MID))


def format_date(ts: Optional[float]) -> str:
    if not ts:
        return "unknown"
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return f"{dt.day} {dt.strftime('%b %Y')}"
    except (ValueError, OSError, OverflowError):
        return "unknown"


def compute_recency(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Port of time-axis.ts ``computeRecency`` (id → recency ratio, timed flag)."""
    known = [t for t in (_to_ts(n.get("timestamp")) for n in nodes) if t is not None]
    min_ts = min(known) if known else None
    max_ts = max(known) if known else None
    timed = min_ts is not None and max_ts is not None and max_ts > min_ts

    ordered = sorted(
        nodes,
        key=lambda n: (
            _to_ts(n.get("timestamp")) if _to_ts(n.get("timestamp")) is not None else math.inf,
            str(n.get("id", "")),
        ),
    )
    last = max(len(ordered) - 1, 1)
    ord_ratio = {str(n.get("id", "")): (i / last if len(ordered) > 1 else 0.0) for i, n in enumerate(ordered)}

    rec: dict[str, float] = {}
    for n in nodes:
        nid = str(n.get("id", ""))
        ts = _to_ts(n.get("timestamp"))
        if timed and ts is not None and min_ts is not None and max_ts is not None:
            ratio = (ts - min_ts) / (max_ts - min_ts)
        else:
            ratio = ord_ratio.get(nid, 0.0)
        rec[nid] = LEAD_IN + (1 - LEAD_IN) * _clamp(ratio, 0.0, 1.0)

    return {"rec": rec, "timed": timed, "minTs": min_ts, "maxTs": max_ts}


def _date_at(rec: dict[str, Any], reveal: float) -> Optional[float]:
    if not rec.get("timed"):
        return None
    lo, hi = rec.get("minTs"), rec.get("maxTs")
    if lo is None or hi is None:
        return None
    return round(lo + _clamp(reveal, 0, 1) * (hi - lo))


# ── Color: ported from color.ts so memory ink + age fade match the desktop ──


def hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except (ValueError, IndexError):
        return 255, 215, 0


def rgb_to_hex(c: tuple) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(_clamp(v, 0, 255)) for v in c))


def mix_rgb(a: tuple, b: tuple, t: float) -> tuple[int, int, int]:
    p = _clamp(t, 0.0, 1.0)
    return tuple(round(a[i] + (b[i] - a[i]) * p) for i in range(3))  # type: ignore[return-value]


def _rgb_to_hsl(c: tuple) -> tuple[float, float, float]:
    r, g, b = (x / 255 for x in c)
    mx, mn = max(r, g, b), min(r, g, b)
    light = (mx + mn) / 2
    d = mx - mn
    if not d:
        return 0.0, 0.0, light
    s = d / (2 - mx - mn) if light > 0.5 else d / (mx + mn)
    if mx == r:
        h = (g - b) / d + (6 if g < b else 0)
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return h * 60, s, light


def _hsl_to_rgb(h: float, s: float, light: float) -> tuple[int, int, int]:
    hue = ((h % 360) + 360) % 360
    c = (1 - abs(2 * light - 1)) * s
    x = c * (1 - abs(((hue / 60) % 2) - 1))
    m = light - c / 2
    if hue < 60:
        r, g, b = c, x, 0.0
    elif hue < 120:
        r, g, b = x, c, 0.0
    elif hue < 180:
        r, g, b = 0.0, c, x
    elif hue < 240:
        r, g, b = 0.0, x, c
    elif hue < 300:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return round((r + m) * 255), round((g + m) * 255), round((b + m) * 255)


def _complementary_ink(c: tuple) -> tuple[int, int, int]:
    h, s, light = _rgb_to_hsl(c)
    return _hsl_to_rgb(h + 165, max(s, 0.5), _clamp(light, 0.5, 0.7))


def derive_palette(primary_hex: str, *, dark: bool = True) -> dict[str, str]:
    """Port of color.ts ``computePalette`` (the bits a terminal needs)."""
    primary = hex_to_rgb(primary_hex)
    base = (255, 255, 255) if dark else (0, 0, 0)
    bg = (8, 8, 12) if dark else (250, 250, 250)
    return {
        "primary": primary_hex,
        # Memories are drillable → primary "clickable" ink; skills are dead-ends
        # → muted complement.
        "memory": rgb_to_hex(mix_rgb(primary, base, 0.12 if dark else 0.18)),
        "skill": rgb_to_hex(mix_rgb(_complementary_ink(primary), bg, 0.45)),
        "label": rgb_to_hex(mix_rgb(base, bg, 0.35)),
        "dim": rgb_to_hex(mix_rgb(base, bg, 0.7)),
        "bg": rgb_to_hex(bg),
    }


def _node_score(node: dict[str, Any], rec: float) -> float:
    """Pick which visible objects deserve map markers + label rows."""
    if node.get("kind") == "memory":
        return 3.5 + rec
    use = float(node.get("useCount", 0) or 0)
    return rec * 2 + math.sqrt(max(0.0, use)) + (2.0 if node.get("pinned") else 0.0)


def _node_label(node: dict[str, Any]) -> str:
    text = str(node.get("label") or node.get("id") or "unknown").strip()
    return text if len(text) <= 26 else text[:23].rstrip() + "…"


def _node_meta(node: dict[str, Any]) -> str:
    if node.get("kind") == "memory":
        source = "profile memory" if node.get("memorySource") == "profile" else "memory"
        return f"{source} · {format_date(_to_ts(node.get('timestamp')))}"
    bits = [str(node.get("category") or "skill"), format_date(_to_ts(node.get("timestamp")))]
    count = int(node.get("useCount", 0) or 0)
    if count:
        bits.append(f"x{count}")
    if node.get("pinned"):
        bits.append("pinned")
    return " · ".join(bits)


# ── Timeline chart frame ─────────────────────────────────────────────────────


class _ChartBucket:
    __slots__ = ("label", "ts", "skills", "memories", "nodes", "rec")

    def __init__(self, label: str, ts: float):
        self.label = label
        self.ts = ts
        self.skills = 0
        self.memories = 0
        self.nodes: list[dict[str, Any]] = []
        self.rec = 1.0

    @property
    def total(self) -> int:
        return self.skills + self.memories


def _period_key(ts: float, granularity: str) -> tuple[int, ...]:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    if granularity == "day":
        return (dt.year, dt.month, dt.day)
    if granularity == "month":
        return (dt.year, dt.month)
    return (dt.year,)


def _period_label(ts: float, granularity: str) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    if granularity == "day":
        return f"{dt.day} {dt.strftime('%b')}"
    if granularity == "month":
        return dt.strftime("%b %Y")
    return dt.strftime("%Y")


def _build_chart_buckets(nodes: list[dict[str, Any]], rec: dict[str, Any], max_rows: int) -> list[_ChartBucket]:
    """Timeline rows: finest date granularity that fits, oldest → newest."""
    if not nodes:
        return []
    if not rec["timed"]:
        ordered = sorted(nodes, key=lambda n: rec["rec"].get(str(n.get("id", "")), 0.0))
        n_bins = min(max_rows, max(1, len(ordered)))
        buckets = [_ChartBucket(f"#{i + 1}", float(i)) for i in range(n_bins)]
        for node in ordered:
            idx = int(_clamp(math.floor(rec["rec"].get(str(node.get("id", "")), 0.0) * n_bins), 0, n_bins - 1))
            b = buckets[idx]
            b.nodes.append(node)
            if node.get("kind") == "memory":
                b.memories += 1
            else:
                b.skills += 1
        return buckets

    chosen: Optional[list[_ChartBucket]] = None
    for granularity in ("day", "month", "year"):
        groups: dict[tuple[int, ...], _ChartBucket] = {}
        for node in nodes:
            ts = _to_ts(node.get("timestamp"))
            if ts is None:
                continue
            key = _period_key(ts, granularity)
            bucket = groups.get(key)
            if bucket is None:
                bucket = _ChartBucket(_period_label(ts, granularity), ts)
                groups[key] = bucket
            bucket.nodes.append(node)
            if node.get("kind") == "memory":
                bucket.memories += 1
            else:
                bucket.skills += 1
        # For short spans, keep the useful day-by-day graph even when the caller
        # asked for fewer rows; terminal scrollback is better than collapsing a
        # month of activity into one unreadable bar.
        if len(groups) <= max_rows or (granularity == "day" and len(groups) <= 32):
            chosen = [groups[key] for key in sorted(groups)]
            break

    if chosen is None:
        # If even yearly buckets overflow, fall back to even time bins.
        min_ts, max_ts = rec.get("minTs"), rec.get("maxTs")
        n_bins = max(1, max_rows)
        chosen = []
        for i in range(n_bins):
            ts = min_ts + (i / max(1, n_bins - 1)) * (max_ts - min_ts) if min_ts and max_ts else float(i)
            chosen.append(_ChartBucket(format_date(ts), ts))
        for node in nodes:
            r = rec["rec"].get(str(node.get("id", "")), 0.0)
            idx = int(_clamp(math.floor(r * n_bins), 0, n_bins - 1))
            b = chosen[idx]
            b.nodes.append(node)
            if node.get("kind") == "memory":
                b.memories += 1
            else:
                b.skills += 1

    min_ts, max_ts = rec.get("minTs"), rec.get("maxTs")
    span = (max_ts - min_ts) if min_ts is not None and max_ts is not None and max_ts > min_ts else 0
    for bucket in chosen:
        bucket.rec = LEAD_IN + (1 - LEAD_IN) * ((bucket.ts - min_ts) / span) if span else 1.0
    return chosen


def _bucket_label_node(bucket: _ChartBucket) -> Optional[dict[str, Any]]:
    if not bucket.nodes:
        return None
    return max(bucket.nodes, key=lambda node: _node_score(node, _to_ts(node.get("timestamp")) or bucket.ts))


def _bucket_nodes(bucket: _ChartBucket, memory_lookup: Optional[dict[str, dict[str, Any]]] = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Chronological within the slice so the TUI tree reads oldest → newest.
    ordered = sorted(bucket.nodes, key=lambda n: _to_ts(n.get("timestamp")) or bucket.ts)
    for node in ordered:
        style = STYLE_MEMORY if node.get("kind") == "memory" else STYLE_SKILL
        raw_label = str(node.get("label") or node.get("id") or "unknown").strip()
        memory = (memory_lookup or {}).get(str(node.get("id", "")))
        out.append(
            {
                "id": str(node.get("id", "")),
                "glyph": MEMORY_GLYPH if node.get("kind") == "memory" else SKILL_GLYPH,
                "label": _node_label(node),
                "fullLabel": raw_label,
                "meta": _node_meta(node),
                "body": str(memory.get("body", "")) if memory else "",
                "style": style,
            }
        )
    return out


def _bucket_rows(buckets: list[_ChartBucket], payload: dict[str, Any]) -> list[dict[str, Any]]:
    cmap = category_color_map(payload)
    memory_lookup = {
        f"memory:{card.get('source')}:{idx}": card
        for idx, card in enumerate(payload.get("memory", []) or [])
        if isinstance(card, dict)
    }
    rows: list[dict[str, Any]] = []
    for idx, bucket in enumerate(buckets):
        cat = _bucket_category(bucket)
        rows.append(
            {
                "index": idx,
                "label": bucket.label,
                "date": format_date(bucket.ts),
                "skills": bucket.skills,
                "memories": bucket.memories,
                "total": bucket.total,
                "category": cat,
                "color": cmap.get(cat) if cat else None,
                "nodes": _bucket_nodes(bucket, memory_lookup),
            }
        )
    return rows


def _category_counts(payload: dict[str, Any]) -> list[tuple[str, int]]:
    clusters = [
        (str(c.get("category")), int(c.get("count", 0)))
        for c in payload.get("clusters", []) or []
        if c.get("category") and c.get("category") != "memory"
    ]
    if clusters:
        return clusters
    counts: dict[str, int] = {}
    for node in payload.get("nodes", []):
        if node.get("kind") == "memory":
            continue
        cat = str(node.get("category") or "skill")
        counts[cat] = counts.get(cat, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def category_color_map(payload: dict[str, Any]) -> dict[str, str]:
    """Deterministic, evenly-spread hue per skill category (theme-independent)."""
    clusters = _category_counts(payload)
    n = max(1, len(clusters))
    # Golden-angle hue spacing so adjacent categories never collide in color.
    return {cat: rgb_to_hex(_hsl_to_rgb((i * 137.508) % 360, 0.55, 0.62)) for i, (cat, _c) in enumerate(clusters)}


def category_legend(payload: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    cmap = category_color_map(payload)
    cats = _category_counts(payload)
    shown = cats[:limit]
    hidden = max(0, len(cats) - len(shown))
    return [
        {"glyph": "●", "color": cmap.get(cat, ""), "label": f"{cat} ({count})"}
        for cat, count in shown
    ] + ([{"glyph": "·", "color": "", "label": f"+{hidden}"}] if hidden else [])


def _bucket_category(bucket: _ChartBucket) -> Optional[str]:
    counts: dict[str, int] = {}
    for node in bucket.nodes:
        if node.get("kind") == "memory":
            continue
        cat = str(node.get("category") or "skill")
        counts[cat] = counts.get(cat, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else None


def _trajectory_row(buckets: list[_ChartBucket], width: int, reveal: float) -> Row:
    """Cumulative learning curve as a compact star-path sparkline."""
    if not buckets:
        return []
    total = sum(b.total for b in buckets) or 1
    visible = int(_clamp(math.ceil(reveal * len(buckets)), 0, len(buckets)))
    acc = 0
    points: list[int] = []
    for b in buckets[:visible]:
        acc += b.total
        points.append(round((acc / total) * (width - 1)))
    cells = [" "] * width
    last = 0
    for p in points:
        for x in range(min(last, p), max(last, p) + 1):
            if 0 <= x < width and cells[x] == " ":
                cells[x] = "·"
        if 0 <= p < width:
            cells[p] = "✦"
        last = p
    return [["trajectory ", STYLE_LABEL, 0.55], ["".join(cells), STYLE_SKILL, 0.48]]


def render_graph(payload: dict[str, Any], *, cols: int = 80, rows: int = 16, reveal: float = 1.0) -> dict[str, Any]:
    """Render one timeline frame at ``reveal`` (0→1).

    Date rows with proportional skill/memory bars colored by the day's dominant
    category, numbered markers tied to label rows, and a cumulative trajectory
    sparkline underneath.
    """
    reveal = _clamp(reveal, 0.0, 1.0)
    cols = max(44, cols)
    rows = max(14, rows)
    nodes = list(payload.get("nodes", []))
    if not nodes:
        placeholder = [["no learning yet — keep using Hermes and it maps out here", STYLE_DIM, 0.7]]
        return {"grid": [placeholder], "date": "", "reveal": reveal, "visible": 0}

    rec = compute_recency(nodes)
    cmap = category_color_map(payload)
    buckets = _build_chart_buckets(nodes, rec, max_rows=max(4, rows - 3))
    n_buckets = len(buckets)
    visible_bucket_count = int(_clamp(math.ceil(reveal * n_buckets), 0, n_buckets))
    max_total = max((b.total for b in buckets), default=1) or 1
    label_w = min(9, max(len(b.label) for b in buckets))
    bar_w = max(14, cols - label_w - 16)

    grid: Grid = []
    labels: list[dict[str, Any]] = []
    visible = 0
    for i, bucket in enumerate(buckets):
        if i >= visible_bucket_count:
            grid.append([])
            continue
        visible += bucket.total
        ink = recency_ink(bucket.rec)
        bar_len = max(1, round((bucket.total / max_total) * bar_w)) if bucket.total else 0
        skill_len = round((bucket.skills / bucket.total) * bar_len) if bucket.total else 0
        if bucket.skills and skill_len == 0:
            skill_len = 1
        memory_len = bar_len - skill_len
        if bucket.memories and memory_len == 0 and bar_len > 1:
            memory_len = 1
            skill_len = bar_len - 1

        node = _bucket_label_node(bucket)
        marker = ""
        if node and len(labels) < 6:
            marker = _LABEL_KEYS[len(labels)]
            style = STYLE_MEMORY if node.get("kind") == "memory" else STYLE_SKILL
            labels.append(
                {
                    "key": marker,
                    "glyph": MEMORY_GLYPH if node.get("kind") == "memory" else SKILL_GLYPH,
                    "label": _node_label(node),
                    "meta": _node_meta(node),
                    "style": style,
                    "alpha": round(ink, 3),
                }
            )

        cat = _bucket_category(bucket)
        cat_hex = cmap.get(cat) if cat else None

        row: Row = [[f"{bucket.label:>{label_w}} ", STYLE_LABEL, ink], ["│ ", STYLE_DIM, 0.55]]
        if marker:
            row.append([marker, STYLE_LABEL, 0.95])
        elif bucket.total:
            head_hex = cat_hex if bucket.skills else None
            row.append(["✦" if bucket.skills else "◆", STYLE_SKILL if bucket.skills else STYLE_MEMORY, ink, head_hex])
        if skill_len:
            # Bar colored by the day's dominant category — a learning heatmap.
            row.append(["━" * skill_len, STYLE_SKILL, ink, cat_hex])
        if memory_len:
            if memory_len == 1:
                mem_trail = "◆"
            else:
                mem_trail = "◆" + ("━" * (memory_len - 2)) + "◆"
            row.append([mem_trail, STYLE_MEMORY, max(0.65, ink)])
        if bar_len < bar_w:
            # Empty space keeps counts aligned; starmap texture lives in the
            # trajectory row below, where it reads as signal rather than noise.
            row.append([" " * (bar_w - bar_len), STYLE_BG, 1.0])
        row.append(["  ", STYLE_BG, 1.0])
        row.append([str(bucket.skills), STYLE_SKILL, max(0.72, ink)])
        if bucket.memories:
            row.append(["+", STYLE_DIM, 0.6])
            row.append([str(bucket.memories), STYLE_MEMORY, max(0.72, ink)])
        if i == visible_bucket_count - 1:
            row.append(["  ◀ now", STYLE_LABEL, 0.9])
        elif bucket.total == max_total and max_total > 1:
            row.append(["  ☄ peak", STYLE_LABEL, 0.75])
        grid.append(row)

    # Cumulative learning trajectory underneath the rows.
    grid.append([[(" " * (label_w + 2)), STYLE_BG, 1.0], *_trajectory_row(buckets, max(12, cols - label_w - 13), reveal)])

    return {
        "grid": grid,
        "date": format_date(_date_at(rec, reveal)),
        "reveal": reveal,
        "visible": visible,
        "labels": labels,
    }


# ── Trimmings ──────────────────────────────────────────────────────────────


def build_legend(payload: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = payload.get("nodes", [])
    skills = sum(1 for n in nodes if n.get("kind") != "memory")
    memories = sum(1 for n in nodes if n.get("kind") == "memory")
    return [
        {"glyph": SKILL_GLYPH, "style": STYLE_SKILL, "label": f"skills ({skills})"},
        {"glyph": MEMORY_GLYPH, "style": STYLE_MEMORY, "label": f"memories ({memories})"},
    ]


def axis_labels(payload: dict[str, Any]) -> dict[str, str]:
    rec = compute_recency(list(payload.get("nodes", [])))
    if not rec["timed"]:
        return {"start": "oldest", "end": "now"}
    return {"start": format_date(rec.get("minTs")), "end": format_date(rec.get("maxTs"))}


def _peak_day(payload: dict[str, Any]) -> Optional[str]:
    counts: dict[tuple[int, ...], int] = {}
    reps: dict[tuple[int, ...], float] = {}
    for node in payload.get("nodes", []):
        ts = _to_ts(node.get("timestamp"))
        if ts is None:
            continue
        key = _period_key(ts, "day")
        counts[key] = counts.get(key, 0) + 1
        reps[key] = ts
    if not counts:
        return None
    best = max(counts, key=lambda k: counts[k])
    return f"busiest day {_period_label(reps[best], 'day')} · {counts[best]} learned"


def build_summary(payload: dict[str, Any]) -> list[str]:
    stats = payload.get("stats", {}) or {}
    lines: list[str] = []
    learned = stats.get("learned_skills", stats.get("nodes", 0))
    mem = stats.get("memory_nodes", 0)
    edges = stats.get("related_edges", 0)
    lines.append(f"{learned} learned skills · {mem} memories · {edges} skill links")
    extra = []
    if stats.get("memory_skill_edges"):
        extra.append(f"{stats['memory_skill_edges']} memory↔skill links")
    peak = _peak_day(payload)
    if peak:
        extra.append(peak)
    if extra:
        lines.append(" · ".join(extra))
    return lines


def _merge_runs(cells: Iterable[Run]) -> Row:
    out: Row = []
    for run in cells:
        text, style, alpha = run[0], run[1], (run[2] if len(run) > 2 else 1.0)
        hex_override = run[3] if len(run) > 3 else None
        prev_hex = out[-1][3] if out and len(out[-1]) > 3 else None
        if out and out[-1][1] == style and abs(out[-1][2] - alpha) < 1e-6 and prev_hex == hex_override:
            out[-1][0] += text
        else:
            merged: Run = [text, style, alpha]
            if hex_override:
                merged.append(hex_override)
            out.append(merged)
    return out


def render_frames(payload: dict[str, Any], *, cols: int = 80, rows: int = 16, frames: int = 48) -> dict[str, Any]:
    """Pre-render a full play-through (reveal 0→1) plus static legend/summary."""
    frames = max(2, min(frames, 240))
    nodes = list(payload.get("nodes", []))
    rec = compute_recency(nodes)
    # Mirror render_graph's bucketing so the interactive row list lines up with
    # what the user sees.
    buckets = _build_chart_buckets(nodes, rec, max_rows=max(4, rows - 3)) if nodes else []
    out_frames = []
    for i in range(frames):
        reveal = i / (frames - 1)
        frame = render_graph(payload, cols=cols, rows=rows, reveal=reveal)
        out_frames.append(
            {
                "reveal": frame["reveal"],
                "date": frame["date"],
                "visible": frame["visible"],
                "grid": frame["grid"],
                "labels": frame.get("labels", []),
            }
        )
    return {
        "frames": out_frames,
        "legend": build_legend(payload),
        "categories": category_legend(payload),
        "buckets": _bucket_rows(buckets, payload),
        "summary": build_summary(payload),
        "axis": axis_labels(payload),
        "count": len(payload.get("nodes", [])),
        "cols": cols,
        "rows": rows,
    }
