"""Terminal renderer for the learning timeline (learned skills + memories): the desktop starmap's data
(``apps/desktop/src/app/starmap``) drawn as a timeline bar chart (date rows, skill/memory bars colored by dominant
category, cumulative trajectory sparkline) plus per-slice bucket metadata the TUI walks as a tree. Age gradient and
memory ink are ported from the desktop source. Grids are style runs ``[text, style, alpha, hex?]``: consumers map
style + brightness onto their palette; hex overrides the base color (category heatmap). Pure, stdlib-only."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

LEAD_IN = 0.06  # time-axis.ts LEAD_IN: the oldest node sits just off recency 0.
# constants.ts AGE_GRADIENT — old quiet, recent bright.
AGE_OLD_INK, AGE_MID_INK, AGE_NEW_INK, AGE_MID = 0.42, 0.74, 0.95, 0.52
# Style keys consumers map to base colors (brightness = the run alpha).
STYLE_BG, STYLE_SKILL, STYLE_MEMORY, STYLE_LABEL, STYLE_DIM = "bg", "skill", "memory", "label", "dim"
# Legend glyphs mirror NODE_SHAPE (skill = circle, memory = diamond).
SKILL_GLYPH, MEMORY_GLYPH = "●", "◆"
_LABEL_KEYS = tuple("123456789abc")

Row = list  # of runs ``[text, style, alpha, hex?]``; a grid is a list of rows


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(p: float) -> float:
    p = _clamp(p, 0.0, 1.0)
    return p * p * (3 - 2 * p)


def _is_memory(node: dict[str, Any]) -> bool:
    return node.get("kind") == "memory"


def _node_id(node: dict[str, Any]) -> str:
    return str(node.get("id", ""))


def _utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _lead_in(ratio: float) -> float:
    return LEAD_IN + (1 - LEAD_IN) * ratio


def _visible_count(reveal: float, n: int) -> int:
    return int(_clamp(math.ceil(reveal * n), 0, n))


def _node_raw_label(node: dict[str, Any]) -> str:
    return str(node.get("label") or node.get("id") or "unknown").strip()


def _node_ts(node: dict[str, Any]) -> Optional[float]:
    try:
        return None if node.get("timestamp") is None else float(node["timestamp"])
    except (TypeError, ValueError):
        return None


def recency_ink(rec: float) -> float:
    """Port of geometry.ts ``recencyInk`` — smoothstep age → ink alpha."""
    t = _clamp(rec, 0.0, 1.0)
    return _lerp(AGE_OLD_INK, AGE_MID_INK, _smoothstep(t / AGE_MID)) if t <= AGE_MID else _lerp(AGE_MID_INK, AGE_NEW_INK, _smoothstep((t - AGE_MID) / (1 - AGE_MID)))


def format_date(ts: Optional[float]) -> str:
    try:
        dt = _utc(float(ts)) if ts else None
    except (ValueError, OSError, OverflowError):
        dt = None
    return f"{dt.day} {dt.strftime('%b %Y')}" if dt else "unknown"


def compute_recency(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Port of time-axis.ts ``computeRecency`` (id → recency ratio, timed flag).
    Untimed graphs (no spread of timestamps) fall back to ordinal position so
    every node still gets a distinct recency."""
    known = [t for t in (_node_ts(n) for n in nodes) if t is not None]
    min_ts, max_ts = (min(known), max(known)) if known else (None, None)
    timed = bool(known) and max_ts > min_ts
    ordered = sorted(nodes, key=lambda n: (_node_ts(n) if _node_ts(n) is not None else math.inf, _node_id(n)))
    last = max(len(ordered) - 1, 1)
    ord_ratio = {_node_id(n): (i / last if len(ordered) > 1 else 0.0) for i, n in enumerate(ordered)}
    rec = {
        nid: _lead_in(_clamp((ts - min_ts) / (max_ts - min_ts) if timed and ts is not None else ord_ratio.get(nid, 0.0), 0.0, 1.0))
        for nid, ts in ((_node_id(n), _node_ts(n)) for n in nodes)
    }
    return {"rec": rec, "timed": timed, "minTs": min_ts, "maxTs": max_ts}


def _date_at(rec: dict[str, Any], reveal: float) -> Optional[float]:
    lo, hi = rec.get("minTs"), rec.get("maxTs")
    return None if not rec.get("timed") or lo is None or hi is None else round(lo + _clamp(reveal, 0, 1) * (hi - lo))


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
    return tuple(round(_lerp(a[i], b[i], _clamp(t, 0.0, 1.0))) for i in range(3))  # type: ignore[return-value]


def _rgb_to_hsl(c: tuple) -> tuple[float, float, float]:
    r, g, b = (x / 255 for x in c)
    mx, mn = max(r, g, b), min(r, g, b)
    light, d = (mx + mn) / 2, mx - mn
    if not d:
        return 0.0, 0.0, light
    s = d / (2 - mx - mn) if light > 0.5 else d / (mx + mn)
    h = (g - b) / d + (6 if g < b else 0) if mx == r else (b - r) / d + 2 if mx == g else (r - g) / d + 4
    return h * 60, s, light


# Hue sextant → (r, g, b) as a permutation of (c, x, 0).
_HUE_SEXTANTS = (
    lambda c, x: (c, x, 0.0), lambda c, x: (x, c, 0.0), lambda c, x: (0.0, c, x),
    lambda c, x: (0.0, x, c), lambda c, x: (x, 0.0, c), lambda c, x: (c, 0.0, x),
)


def _hsl_to_rgb(h: float, s: float, light: float) -> tuple[int, int, int]:
    hue = ((h % 360) + 360) % 360
    c = (1 - abs(2 * light - 1)) * s
    x, m = c * (1 - abs(((hue / 60) % 2) - 1)), light - c / 2
    return tuple(round((v + m) * 255) for v in _HUE_SEXTANTS[min(int(hue // 60), 5)](c, x))  # type: ignore[return-value]


def derive_palette(primary_hex: str, *, dark: bool = True) -> dict[str, str]:
    """Port of color.ts ``computePalette`` (the bits a terminal needs)."""
    primary = hex_to_rgb(primary_hex)
    base, bg = ((255, 255, 255), (8, 8, 12)) if dark else ((0, 0, 0), (250, 250, 250))
    h, s, light = _rgb_to_hsl(primary)
    return {
        "primary": primary_hex,
        # Memories are drillable → primary "clickable" ink; skills are dead-ends → muted complement.
        "memory": rgb_to_hex(mix_rgb(primary, base, 0.12 if dark else 0.18)),
        "skill": rgb_to_hex(mix_rgb(_hsl_to_rgb(h + 165, max(s, 0.5), _clamp(light, 0.5, 0.7)), bg, 0.45)),
        "label": rgb_to_hex(mix_rgb(base, bg, 0.35)), "dim": rgb_to_hex(mix_rgb(base, bg, 0.7)), "bg": rgb_to_hex(bg),
    }


def _node_score(node: dict[str, Any], rec: float) -> float:
    """Pick which visible objects deserve map markers + label rows."""
    return 3.5 + rec if _is_memory(node) else rec * 2 + math.sqrt(max(0.0, float(node.get("useCount", 0) or 0))) + (2.0 if node.get("pinned") else 0.0)


def _node_card(node: dict[str, Any]) -> dict[str, Any]:
    """Shared glyph/label/meta/style fields for label rows and bucket trees."""
    mem, text, date = _is_memory(node), _node_raw_label(node), format_date(_node_ts(node))
    if mem:
        meta = f"{'profile memory' if node.get('memorySource') == 'profile' else 'memory'} · {date}"
    else:
        count = int(node.get("useCount", 0) or 0)
        meta = " · ".join([str(node.get("category") or "skill"), date] + ([f"x{count}"] if count else []) + (["pinned"] if node.get("pinned") else []))
    return {
        "glyph": MEMORY_GLYPH if mem else SKILL_GLYPH, "label": text if len(text) <= 26 else text[:23].rstrip() + "…",
        "meta": meta, "style": STYLE_MEMORY if mem else STYLE_SKILL,
    }


def _skill_category_counts(nodes: Iterable[dict[str, Any]]) -> Counter:
    return Counter(str(n.get("category") or "skill") for n in nodes if not _is_memory(n))


# ── Timeline chart frame ─────────────────────────────────────────────────────

class _ChartBucket:
    __slots__ = ("label", "ts", "nodes", "rec")

    def __init__(self, label: str, ts: float):
        self.label, self.ts, self.rec = label, ts, 1.0
        self.nodes: list[dict[str, Any]] = []

    memories = property(lambda self: sum(1 for n in self.nodes if _is_memory(n)))
    skills = property(lambda self: len(self.nodes) - self.memories)
    total = property(lambda self: len(self.nodes))

    def category(self) -> Optional[str]:
        return max(counts, key=lambda k: counts[k]) if (counts := _skill_category_counts(self.nodes)) else None


# granularity → (period key, row label) from a UTC datetime.
_PERIODS: dict[str, tuple] = {
    "day": (lambda dt: (dt.year, dt.month, dt.day), lambda dt: f"{dt.day} {dt.strftime('%b')}"),
    "month": (lambda dt: (dt.year, dt.month), lambda dt: dt.strftime("%b %Y")),
    "year": (lambda dt: (dt.year,), lambda dt: dt.strftime("%Y")),
}


def _period(ts: float, granularity: str) -> tuple[tuple[int, ...], str]:
    return tuple(fn(_utc(ts)) for fn in _PERIODS.get(granularity, _PERIODS["year"]))  # type: ignore[return-value]


def _fill_even_bins(buckets: list[_ChartBucket], nodes: Iterable[dict[str, Any]], rec: dict[str, Any]) -> None:
    """Drop each node into the bin its recency ratio maps to (order preserved)."""
    for node in nodes:
        buckets[int(_clamp(math.floor(rec["rec"].get(_node_id(node), 0.0) * len(buckets)), 0, len(buckets) - 1))].nodes.append(node)


def _build_chart_buckets(nodes: list[dict[str, Any]], rec: dict[str, Any], max_rows: int) -> list[_ChartBucket]:
    """Timeline rows: finest date granularity that fits, oldest → newest."""
    if not nodes:
        return []
    if not rec["timed"]:
        buckets = [_ChartBucket(f"#{i + 1}", float(i)) for i in range(min(max_rows, len(nodes)))]
        _fill_even_bins(buckets, sorted(nodes, key=lambda n: rec["rec"].get(_node_id(n), 0.0)), rec)
        return buckets

    chosen: Optional[list[_ChartBucket]] = None
    for granularity in ("day", "month", "year"):
        groups: dict[tuple[int, ...], _ChartBucket] = {}
        for node in nodes:
            ts = _node_ts(node)
            if ts is not None:
                key, label = _period(ts, granularity)
                groups.setdefault(key, _ChartBucket(label, ts)).nodes.append(node)
        # For short spans, keep the useful day-by-day graph even when the caller
        # asked for fewer rows; scrollback beats collapsing a month into one bar.
        if len(groups) <= max_rows or (granularity == "day" and len(groups) <= 32):
            chosen = [groups[key] for key in sorted(groups)]
            break

    min_ts, max_ts = rec.get("minTs"), rec.get("maxTs")
    if chosen is None:  # even yearly buckets overflow → fall back to even time bins
        n_bins = max(1, max_rows)
        stops = (min_ts + (i / max(1, n_bins - 1)) * (max_ts - min_ts) if min_ts and max_ts else float(i) for i in range(n_bins))
        chosen = [_ChartBucket(format_date(ts), ts) for ts in stops]
        _fill_even_bins(chosen, nodes, rec)

    span = (max_ts - min_ts) if min_ts is not None and max_ts is not None and max_ts > min_ts else 0
    for bucket in chosen:
        bucket.rec = _lead_in((bucket.ts - min_ts) / span) if span else 1.0
    return chosen


def _bucket_rows(buckets: list[_ChartBucket], payload: dict[str, Any]) -> list[dict[str, Any]]:
    cmap = category_color_map(payload)
    memory_lookup = {f"memory:{card.get('source')}:{idx}": card for idx, card in enumerate(payload.get("memory", []) or []) if isinstance(card, dict)}

    def node_row(node: dict[str, Any]) -> dict[str, Any]:
        card, memory = _node_card(node), memory_lookup.get(_node_id(node))
        return {
            "id": _node_id(node), "glyph": card["glyph"], "label": card["label"], "fullLabel": _node_raw_label(node),
            "meta": card["meta"], "body": str(memory.get("body", "")) if memory else "", "style": card["style"],
        }

    def bucket_row(idx: int, bucket: _ChartBucket) -> dict[str, Any]:
        cat = bucket.category()
        return {
            "index": idx, "label": bucket.label, "date": format_date(bucket.ts),
            "skills": bucket.skills, "memories": bucket.memories, "total": bucket.total,
            "category": cat, "color": cmap.get(cat) if cat else None,
            # Chronological within the slice so the TUI tree reads oldest → newest.
            "nodes": [node_row(n) for n in sorted(bucket.nodes, key=lambda n: _node_ts(n) or bucket.ts)],
        }

    return [bucket_row(idx, bucket) for idx, bucket in enumerate(buckets)]


def _category_counts(payload: dict[str, Any]) -> list[tuple[str, int]]:
    clusters = [(str(c.get("category")), int(c.get("count", 0))) for c in payload.get("clusters", []) or [] if c.get("category") and c.get("category") != "memory"]
    return clusters or sorted(_skill_category_counts(payload.get("nodes", [])).items(), key=lambda kv: (-kv[1], kv[0]))


def category_color_map(payload: dict[str, Any]) -> dict[str, str]:
    """Deterministic, evenly-spread hue per skill category (theme-independent).
    Golden-angle spacing so adjacent categories never collide in color."""
    return {cat: rgb_to_hex(_hsl_to_rgb((i * 137.508) % 360, 0.55, 0.62)) for i, (cat, _c) in enumerate(_category_counts(payload))}


def category_legend(payload: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    cmap, cats = category_color_map(payload), _category_counts(payload)
    out = [{"glyph": "●", "color": cmap.get(cat, ""), "label": f"{cat} ({count})"} for cat, count in cats[:limit]]
    return out + ([{"glyph": "·", "color": "", "label": f"+{len(cats) - limit}"}] if len(cats) > limit else [])


def _trajectory_row(buckets: list[_ChartBucket], width: int, reveal: float) -> Row:
    """Cumulative learning curve as a compact star-path sparkline."""
    if not buckets:
        return []
    total, cells, acc, last = sum(b.total for b in buckets) or 1, [" "] * width, 0, 0
    for b in buckets[:_visible_count(reveal, len(buckets))]:
        acc += b.total
        p = round((acc / total) * (width - 1))
        for x in range(min(last, p), max(last, p) + 1):
            if 0 <= x < width and cells[x] == " ":
                cells[x] = "·"
        if 0 <= p < width:
            cells[p] = "✦"
        last = p
    return [["trajectory ", STYLE_LABEL, 0.55], ["".join(cells), STYLE_SKILL, 0.48]]


def _bar_lengths(bucket: _ChartBucket, max_total: int, bar_w: int) -> tuple[int, int, int]:
    """(bar, skill, memory) cell counts; a present kind never rounds to zero."""
    bar_len = max(1, round((bucket.total / max_total) * bar_w)) if bucket.total else 0
    skill_len = max(1, round((bucket.skills / bucket.total) * bar_len)) if bucket.skills else 0
    if bucket.memories and skill_len == bar_len > 1:
        skill_len = bar_len - 1
    return bar_len, skill_len, bar_len - skill_len


def render_graph(payload: dict[str, Any], *, cols: int = 80, rows: int = 16, reveal: float = 1.0) -> dict[str, Any]:
    """Render one timeline frame at ``reveal`` (0→1): date rows with proportional
    skill/memory bars colored by dominant category, numbered markers tied to
    label rows, and a cumulative trajectory sparkline underneath."""
    reveal, cols, rows = _clamp(reveal, 0.0, 1.0), max(44, cols), max(14, rows)
    nodes = list(payload.get("nodes", []))
    if not nodes:
        return {"grid": [[["no learning yet — keep using Hermes and it maps out here", STYLE_DIM, 0.7]]], "date": "", "reveal": reveal, "visible": 0}

    rec, cmap = compute_recency(nodes), category_color_map(payload)
    buckets = _build_chart_buckets(nodes, rec, max_rows=max(4, rows - 3))
    visible_bucket_count, max_total = _visible_count(reveal, len(buckets)), max((b.total for b in buckets), default=1) or 1
    label_w = min(9, max(len(b.label) for b in buckets))
    bar_w = max(14, cols - label_w - 16)

    grid: list[Row] = []
    labels: list[dict[str, Any]] = []
    visible = 0
    for i, bucket in enumerate(buckets[:visible_bucket_count]):
        visible += bucket.total
        ink, cat = recency_ink(bucket.rec), bucket.category()
        bar_len, skill_len, memory_len = _bar_lengths(bucket, max_total, bar_w)
        cat_hex = cmap.get(cat) if cat else None
        row: Row = [[f"{bucket.label:>{label_w}} ", STYLE_LABEL, ink], ["│ ", STYLE_DIM, 0.55]]
        if bucket.nodes and len(labels) < 6:
            node, marker = max(bucket.nodes, key=lambda n: _node_score(n, _node_ts(n) or bucket.ts)), _LABEL_KEYS[len(labels)]
            labels.append({"key": marker, **_node_card(node), "alpha": round(ink, 3)})
            row.append([marker, STYLE_LABEL, 0.95])
        elif bucket.total:
            row.append(["✦" if bucket.skills else "◆", STYLE_SKILL if bucket.skills else STYLE_MEMORY, ink, cat_hex if bucket.skills else None])
        # Skill bar colored by the day's dominant category (a learning heatmap); trailing empty space keeps
        # counts aligned — starmap texture lives in the trajectory row.
        row += ([["━" * skill_len, STYLE_SKILL, ink, cat_hex]] if skill_len else [])
        row += ([["◆" if memory_len == 1 else "◆" + ("━" * (memory_len - 2)) + "◆", STYLE_MEMORY, max(0.65, ink)]] if memory_len else [])
        row += ([[" " * (bar_w - bar_len), STYLE_BG, 1.0]] if bar_len < bar_w else []) + [["  ", STYLE_BG, 1.0], [str(bucket.skills), STYLE_SKILL, max(0.72, ink)]]
        row += ([["+", STYLE_DIM, 0.6], [str(bucket.memories), STYLE_MEMORY, max(0.72, ink)]] if bucket.memories else [])
        if i == visible_bucket_count - 1:
            row.append(["  ◀ now", STYLE_LABEL, 0.9])
        elif bucket.total == max_total and max_total > 1:
            row.append(["  ☄ peak", STYLE_LABEL, 0.75])
        grid.append(row)

    grid += [[] for _ in buckets[visible_bucket_count:]]  # not-yet-revealed rows stay blank
    grid.append([[(" " * (label_w + 2)), STYLE_BG, 1.0], *_trajectory_row(buckets, max(12, cols - label_w - 13), reveal)])
    return {"grid": grid, "date": format_date(_date_at(rec, reveal)), "reveal": reveal, "visible": visible, "labels": labels}


# ── Trimmings ──────────────────────────────────────────────────────────────

def build_legend(payload: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = payload.get("nodes", [])
    memories = sum(1 for n in nodes if _is_memory(n))
    return [
        {"glyph": SKILL_GLYPH, "style": STYLE_SKILL, "label": f"skills ({len(nodes) - memories})"},
        {"glyph": MEMORY_GLYPH, "style": STYLE_MEMORY, "label": f"memories ({memories})"},
    ]


def axis_labels(payload: dict[str, Any]) -> dict[str, str]:
    rec = compute_recency(list(payload.get("nodes", [])))
    return {"start": format_date(rec["minTs"]), "end": format_date(rec["maxTs"])} if rec["timed"] else {"start": "oldest", "end": "now"}


def _peak_day(payload: dict[str, Any]) -> Optional[str]:
    periods = [_period(ts, "day") for ts in (_node_ts(n) for n in payload.get("nodes", [])) if ts is not None]
    counts, labels = Counter(key for key, _label in periods), dict(periods)
    if not counts:
        return None
    best = max(counts, key=lambda k: counts[k])
    return f"busiest day {labels[best]} · {counts[best]} learned"


def build_summary(payload: dict[str, Any]) -> list[str]:
    stats = payload.get("stats", {}) or {}
    learned = stats.get("learned_skills", stats.get("nodes", 0))
    lines = [f"{learned} learned skills · {stats.get('memory_nodes', 0)} memories · {stats.get('related_edges', 0)} skill links"]
    extra = ([f"{stats['memory_skill_edges']} memory↔skill links"] if stats.get("memory_skill_edges") else []) + list(filter(None, [_peak_day(payload)]))
    return lines + ([" · ".join(extra)] if extra else [])


def render_frames(payload: dict[str, Any], *, cols: int = 80, rows: int = 16, frames: int = 48) -> dict[str, Any]:
    """Pre-render a full play-through (reveal 0→1) plus static legend/summary."""
    frames, nodes = max(2, min(frames, 240)), list(payload.get("nodes", []))
    # Mirror render_graph's bucketing so the interactive row list lines up with what the user sees.
    buckets = _build_chart_buckets(nodes, compute_recency(nodes), max_rows=max(4, rows - 3)) if nodes else []
    out_frames = [
        {k: frame[k] for k in ("reveal", "date", "visible", "grid")} | {"labels": frame.get("labels", [])}
        for frame in (render_graph(payload, cols=cols, rows=rows, reveal=i / (frames - 1)) for i in range(frames))
    ]
    return {
        "frames": out_frames, "legend": build_legend(payload), "categories": category_legend(payload),
        "buckets": _bucket_rows(buckets, payload), "summary": build_summary(payload), "axis": axis_labels(payload),
        "count": len(payload.get("nodes", [])), "cols": cols, "rows": rows,
    }


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

Grid = list  # list[Row]

Run = list  # [text, style, alpha, hex?]
# ---- END PLUGIN-COMPAT ----
