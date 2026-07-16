// Dependency-free SVG charting primitives, shared across markets/crypto/stocks
// and the detail windows. Everything returns an <svg> built with the same DOM
// idioms as utils.js (no canvas, no libraries). Colors come from theme tokens
// so charts track light/dark and the accent presets automatically.

const NS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    node.setAttribute(k, String(v));
  }
  return node;
}

function frame(width, height, cls) {
  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: "none",
    class: cls,
    "aria-hidden": "true",
  });
  return svg;
}

const cssVar = (name, fallback) => {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
};

/**
 * Multi-series line/area chart.
 *   series: [{ points:[y,…] | [{x,y},…], color?, area?, width?, dash? }]
 * opts: { width, height, pad, grid, minY, maxY }
 */
export function lineChart(series, opts = {}) {
  const { width = 320, height = 140, pad = 6, grid = true } = opts;
  const svg = frame(width, height, "chart chart-line");
  const norm = series
    .map((s) => ({
      ...s,
      pts: (s.points || []).map((p, i) => (typeof p === "number" ? { x: i, y: p } : p))
        .filter((p) => p && Number.isFinite(p.y)),
    }))
    .filter((s) => s.pts.length >= 2);
  if (!norm.length) return svg;

  const allY = norm.flatMap((s) => s.pts.map((p) => p.y));
  const allX = norm.flatMap((s) => s.pts.map((p) => p.x));
  const minY = opts.minY ?? Math.min(...allY);
  const maxY = opts.maxY ?? Math.max(...allY);
  const minX = Math.min(...allX);
  const maxX = Math.max(...allX);
  const spanY = maxY - minY || 1;
  const spanX = maxX - minX || 1;
  const sx = (x) => pad + ((x - minX) / spanX) * (width - pad * 2);
  const sy = (y) => pad + (1 - (y - minY) / spanY) * (height - pad * 2);

  if (grid) {
    const gridColor = cssVar("--chart-grid", "#1e2b32");
    for (let i = 0; i <= 3; i++) {
      const y = pad + (i / 3) * (height - pad * 2);
      svg.append(el("line", { x1: pad, y1: y, x2: width - pad, y2: y, stroke: gridColor, "stroke-width": 0.5 }));
    }
  }

  const accent = cssVar("--chart-line", cssVar("--accent", "#41d3ea"));
  for (const s of norm) {
    const d = s.pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
    const color = s.color || accent;
    if (s.area) {
      const area = `${d}L${sx(s.pts[s.pts.length - 1].x).toFixed(1)},${sy(minY).toFixed(1)}L${sx(s.pts[0].x).toFixed(1)},${sy(minY).toFixed(1)}Z`;
      svg.append(el("path", { d: area, fill: cssVar("--chart-area", "rgba(65,211,234,0.10)"), stroke: "none" }));
    }
    svg.append(el("path", {
      d, fill: "none", stroke: color, "stroke-width": s.width || 1.6,
      "stroke-linejoin": "round", "stroke-linecap": "round",
      "stroke-dasharray": s.dash || null,
    }));
  }
  return svg;
}

/**
 * Candlestick chart. candles: [{t,o,h,l,c},…]. overlays: same shape as
 * lineChart series, drawn on the same price scale (moving averages, bands).
 */
export function candleChart(candles, opts = {}) {
  const { width = 320, height = 160, pad = 6 } = opts;
  const svg = frame(width, height, "chart chart-candle");
  const cs = (candles || []).filter((c) => c && Number.isFinite(c.c));
  if (cs.length < 2) return svg;

  const overlays = (opts.overlays || []).map((s) => ({
    ...s, pts: (s.points || []).map((y, i) => ({ x: i, y })).filter((p) => Number.isFinite(p.y)),
  }));
  const overlayY = overlays.flatMap((s) => s.pts.map((p) => p.y));
  const hi = Math.max(...cs.map((c) => c.h), ...overlayY);
  const lo = Math.min(...cs.map((c) => c.l), ...overlayY);
  const span = hi - lo || 1;
  const sy = (v) => pad + (1 - (v - lo) / span) * (height - pad * 2);
  const slot = (width - pad * 2) / cs.length;
  const cw = Math.max(1, slot * 0.6);

  const up = cssVar("--delta-up", "#0ca30c");
  const down = cssVar("--delta-down", "#e66767");
  cs.forEach((c, i) => {
    const cx = pad + slot * (i + 0.5);
    const rising = c.c >= c.o;
    const color = rising ? up : down;
    svg.append(el("line", { x1: cx, y1: sy(c.h), x2: cx, y2: sy(c.l), stroke: color, "stroke-width": 0.8 }));
    const yO = sy(c.o); const yC = sy(c.c);
    svg.append(el("rect", {
      x: cx - cw / 2, y: Math.min(yO, yC), width: cw, height: Math.max(1, Math.abs(yC - yO)),
      fill: color, opacity: 0.85,
    }));
  });

  for (const s of overlays) {
    if (s.pts.length < 2) continue;
    const d = s.pts.map((p, i) => `${i ? "L" : "M"}${(pad + slot * (p.x + 0.5)).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
    svg.append(el("path", {
      d, fill: "none", stroke: s.color || cssVar("--accent", "#41d3ea"),
      "stroke-width": s.width || 1.1, "stroke-dasharray": s.dash || null, opacity: 0.9,
    }));
  }
  return svg;
}

/** Donut chart. slices: [{value, color?, label?}]. Returns an <svg>. */
export function donut(slices, opts = {}) {
  const { size = 120, thickness = 16 } = opts;
  const svg = frame(size, size, "chart chart-donut");
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const total = slices.reduce((a, s) => a + Math.max(0, s.value), 0);
  if (total <= 0) return svg;
  const r = (size - thickness) / 2;
  const cx = size / 2; const cy = size / 2;
  let angle = -Math.PI / 2;
  slices.forEach((s, i) => {
    const frac = Math.max(0, s.value) / total;
    const end = angle + frac * Math.PI * 2;
    const large = frac > 0.5 ? 1 : 0;
    const x1 = cx + r * Math.cos(angle); const y1 = cy + r * Math.sin(angle);
    const x2 = cx + r * Math.cos(end); const y2 = cy + r * Math.sin(end);
    svg.append(el("path", {
      d: `M${x1.toFixed(2)},${y1.toFixed(2)} A${r},${r} 0 ${large} 1 ${x2.toFixed(2)},${y2.toFixed(2)}`,
      fill: "none", stroke: s.color || SERIES[i % SERIES.length](), "stroke-width": thickness,
    }));
    angle = end;
  });
  return svg;
}

// Lazily read the theme series palette (utils exports arrays; keep chart.js
// self-contained by re-deriving from tokens where possible).
const SERIES = [
  () => cssVar("--accent", "#41d3ea"),
  () => "#199e70", () => "#c98500", () => "#9085e9",
  () => "#e66767", () => "#d55181", () => "#d95926", () => "#008300",
];
