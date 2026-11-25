import { iso, pathOf } from "./iso-kit.mjs";
import * as G from "./lathe.mjs";

const LIGHT = (() => {
  const length = Math.hypot(-0.55, -0.835);
  return [-0.55 / length, -0.835 / length];
})();

const poly = (points, closed = true) => pathOf(points, closed);

const area = (points) => points.reduce((sum, [x, y], index) => {
  const [nx, ny] = points[(index + 1) % points.length];
  return sum + x * ny - nx * y;
}, 0);

function sideOf(frames) {
  const out = { left: [], right: [], shineA: [], shineB: [], shadeA: [], shadeB: [] };
  for (const { p, nx, ny, half } of frames) {
    out.left.push([p[0] + nx * half, p[1] + ny * half]);
    out.right.push([p[0] - nx * half, p[1] - ny * half]);
    const toward = nx * LIGHT[0] + ny * LIGHT[1];
    const shineAt = half * 0.44 * toward;
    const shineHalf = half * 0.24;
    out.shineA.push([p[0] + nx * (shineAt + shineHalf), p[1] + ny * (shineAt + shineHalf)]);
    out.shineB.push([p[0] + nx * (shineAt - shineHalf), p[1] + ny * (shineAt - shineHalf)]);
    const shadeAt = -half * 0.62 * toward;
    const shadeHalf = half * 0.2;
    out.shadeA.push([p[0] + nx * (shadeAt + shadeHalf), p[1] + ny * (shadeAt + shadeHalf)]);
    out.shadeB.push([p[0] + nx * (shadeAt - shadeHalf), p[1] + ny * (shadeAt - shadeHalf)]);
  }
  return out;
}

function capOf(centre, tangent, radius, inward, P) {
  const V = G.viewOf(P);
  let n1 = G.cross3(tangent, V);
  if (G.len3(n1) < 1e-6) n1 = G.frameAlong([0, 0, 0], tangent).u;
  n1 = G.unit3(n1);
  const n2 = G.unit3(G.cross3(tangent, n1));
  const steps = 28;
  const ring = Array.from({ length: steps }, (_, index) => {
    const angle = (index / steps) * Math.PI * 2;
    return iso(G.add3(centre, G.add3(G.mul3(n1, radius * Math.cos(angle)), G.mul3(n2, radius * Math.sin(angle)))), P);
  });
  const c = iso(centre, P);
  const probe = iso(G.add3(centre, n2), P);
  const along = (probe[0] - c[0]) * inward[0] + (probe[1] - c[1]) * inward[1];
  const from = along > 0 ? steps / 2 : 0;
  const half = (start) => Array.from({ length: steps / 2 + 1 }, (_, index) => ring[(start + index) % steps]);
  return { ring, arc: half(from), inner: half((from + steps / 2) % steps) };
}

function stripeEnd(half, centre, nx, ny, from, to) {
  const offset = (p) => (p[0] - centre[0]) * nx + (p[1] - centre[1]) * ny;
  const values = half.map(offset);
  const lo = Math.min(from, to);
  const hi = Math.max(from, to);
  const points = [];
  const cross = (k, level) => {
    const t = (level - values[k]) / (values[k + 1] - values[k] || 1e-9);
    return [half[k][0] + (half[k + 1][0] - half[k][0]) * t, half[k][1] + (half[k + 1][1] - half[k][1]) * t];
  };
  for (let k = 0; k < half.length; k++) {
    if (values[k] >= lo && values[k] <= hi) points.push(half[k]);
    if (k < half.length - 1) for (const level of [lo, hi]) if ((values[k] - level) * (values[k + 1] - level) < 0) points.push(cross(k, level));
  }
  const order = points.map((p) => offset(p));
  const sorted = points.map((p, index) => [p, order[index]]).sort((x, y) => (from > to ? y[1] - x[1] : x[1] - y[1]));
  return sorted.map(([p]) => p);
}

function spansOf(total, gaps, breaks, maxLength, closed) {
  const cuts = gaps
    .map(([a, b]) => [Math.max(0, Math.min(a, b)), Math.min(total, Math.max(a, b))])
    .filter(([a, b]) => b - a > 1e-6)
    .sort((x, y) => x[0] - y[0]);
  const runs = [];
  let at = 0;
  for (const [a, b] of cuts) {
    if (a > at + 1e-6) runs.push([at, a, at > 0, true]);
    at = Math.max(at, b);
  }
  if (total > at + 1e-6) runs.push([at, total, at > 0, cuts.length > 0 && cuts[cuts.length - 1][1] >= total - 1e-6]);
  const pieces = [];
  for (const [from, to, gapStart, gapEnd] of runs) {
    const stops = [from, ...breaks.filter((b) => b > from + 1e-6 && b < to - 1e-6).sort((x, y) => x - y), to];
    for (let index = 0; index < stops.length - 1; index++) {
      const a = stops[index];
      const b = stops[index + 1];
      const count = Math.max(1, Math.ceil((b - a) / maxLength - 1e-9));
      for (let c = 0; c < count; c++) {
        pieces.push({
          from: a + ((b - a) * c) / count,
          to: a + ((b - a) * (c + 1)) / count,
          open0: !(index === 0 && c === 0),
          open1: !(index === stops.length - 2 && c === count - 1),
          gap0: index === 0 && c === 0 && gapStart,
          gap1: index === stops.length - 2 && c === count - 1 && gapEnd,
        });
      }
    }
  }
  if (closed && !cuts.length && pieces.length) {
    pieces[0].open0 = true;
    pieces[pieces.length - 1].open1 = true;
  }
  return pieces;
}

export function runsOf(total, gaps = []) {
  const cuts = gaps
    .map(([a, b]) => [Math.max(0, Math.min(a, b)), Math.min(total, Math.max(a, b))])
    .filter(([a, b]) => b - a > 1e-6)
    .sort((x, y) => x[0] - y[0]);
  const runs = [];
  let at = 0;
  for (const [a, b] of cuts) {
    if (a > at + 1e-6) runs.push([at, a]);
    at = Math.max(at, b);
  }
  if (total > at + 1e-6) runs.push([at, total]);
  return runs;
}

export function crossingsOf(points, centre, axis, radius) {
  const n = G.unit3(axis);
  const out = [];
  let walked = 0;
  for (let index = 1; index < points.length; index++) {
    const a = points[index - 1];
    const b = points[index];
    const length = G.len3(G.sub3(b, a));
    const da = G.dot3(G.sub3(a, centre), n);
    const db = G.dot3(G.sub3(b, centre), n);
    if ((da <= 0 && db > 0) || (da >= 0 && db < 0)) {
      const t = da / (da - db);
      const p = G.lerp3(a, b, t);
      if (G.len3(G.sub3(p, centre)) <= radius) out.push(walked + t * length);
    }
    walked += length;
  }
  return out;
}

function ringsOf(dense, total, radiusAt, P, span, { pitch = 3, twist = 0, cross = false, fade = [0.1, 0.55], phase = 0, steps = 16 } = {}) {
  const V = G.viewOf(P);
  const [lo, hi] = span;
  const margin = twist ? 0 : Math.min(pitch * 0.35, (hi - lo) / 2);
  const from = Math.max(lo, span.real0 ? lo + margin : lo);
  const to = Math.min(hi, span.real1 ? hi - margin : hi);
  if (to <= from) return [];
  const out = [];
  const hands = twist ? (cross ? [1, -1] : [1]) : [0];
  const reach = Math.abs(twist) * Math.max(...[0, 0.5, 1].map(radiusAt));
  const first = Math.ceil((from - reach - phase) / pitch);
  const last = Math.floor((to + reach - phase) / pitch);
  const surface = (t, theta) => {
    const at = Math.max(0, Math.min(total, t));
    const c = G.pointAlong(dense, at);
    const T = G.tangentAlong(dense, at);
    let n1 = G.cross3(T, V);
    if (G.len3(n1) < 1e-6) return null;
    n1 = G.unit3(n1);
    let n2 = G.unit3(G.cross3(T, n1));
    if (G.dot3(n2, V) < 0) n2 = G.mul3(n2, -1);
    const r = radiusAt(at / (total || 1));
    const normal = G.add3(G.mul3(n1, Math.cos(theta)), G.mul3(n2, Math.sin(theta)));
    return { p: iso(G.add3(c, G.mul3(normal, r)), P), facing: G.dot3(normal, V), r };
  };
  for (let index = first; index <= last; index++) {
    const station = phase + index * pitch;
    for (const hand of hands) {
      const samples = [];
      for (let step = 0; step <= steps; step++) {
        const theta = (step / steps) * Math.PI;
        const t = station + hand * twist * radiusAt(Math.max(0, Math.min(1, station / (total || 1)))) * (step / steps - 0.5);
        samples.push({ t, theta });
      }
      for (let step = 0; step < steps; step++) {
        let a = samples[step];
        let b = samples[step + 1];
        if ((a.t < from && b.t < from) || (a.t > to && b.t > to)) continue;
        const clipTo = (x, y, edge) => ({ t: edge, theta: x.theta + ((y.theta - x.theta) * (edge - x.t)) / (y.t - x.t || 1e-9) });
        if (a.t < from) a = clipTo(a, b, from);
        if (b.t < from) b = clipTo(b, a, from);
        if (a.t > to) a = clipTo(a, b, to);
        if (b.t > to) b = clipTo(b, a, to);
        const pa = surface(a.t, a.theta);
        const pb = surface(b.t, b.theta);
        if (!pa || !pb) continue;
        const alpha = fade ? G.fadeOf((pa.facing + pb.facing) / 2, fade) : 1;
        if (alpha <= 0.02) continue;
        out.push({ d: pathOf([pa.p, pb.p]), alpha });
      }
    }
  }
  return out;
}

export function tubePieces(points3, radius, P, { maxLength = 30, spacing = 2.2, closed = false, breaks = [], gaps = [], caps = true, stripes = "auto", rings = null } = {}) {
  const dense = G.resample(points3, spacing);
  const total = G.pathLength(dense);
  const radiusAt = typeof radius === "function" ? radius : () => radius;
  const k = G.scaleOf(P);
  const wide = stripes === "auto" ? Math.max(...Array.from({ length: 33 }, (_, index) => radiusAt(index / 32))) * k > 1.3 : Boolean(stripes);
  const lead = 0.5;
  const out = [];
  const looped = closed && !gaps.length;
  const wrap = (at) => (looped ? ((at % total) + total) % total : Math.max(0, Math.min(total, at)));
  const step = Math.max(0.05, Math.min(spacing, total / 4) * 0.5);
  const frameAt = (at, p, width) => {
    const a = iso(G.pointAlong(dense, wrap(at - step)), P);
    const b = iso(G.pointAlong(dense, wrap(at + step)), P);
    let tx = b[0] - a[0];
    let ty = b[1] - a[1];
    const length = Math.hypot(tx, ty) || 1;
    tx /= length;
    ty /= length;
    return { p, nx: -ty, ny: tx, half: width };
  };
  const partOf = (start, end) => {
    let points = G.cut(dense, Math.max(0, start), Math.min(total, end));
    if (looped && start < 0) points = [...G.cut(dense, total + start, total), ...points.slice(1)];
    if (looped && end > total) points = [...points, ...G.cut(dense, 0, end - total).slice(1)];
    let walked = looped ? start : Math.max(0, start);
    const along = points.map((point, index) => {
      if (index > 0) walked += G.len3(G.sub3(point, points[index - 1]));
      return walked;
    });
    const flat = points.map((point) => iso(point, P));
    const widths = along.map((at) => radiusAt(looped ? wrap(at) / total : Math.min(1, Math.max(0, at / total))) * k);
    const frames = flat.map((p, index) => frameAt(along[index], p, widths[index]));
    return { points, flat, widths, frames };
  };
  for (const piece of spansOf(total, gaps, breaks, maxLength, closed)) {
    const openStart = piece.open0 && (piece.from > 0 || looped);
    const openEnd = piece.open1 && (piece.to < total || looped);
    const body = partOf(openStart ? piece.from - lead : piece.from, openEnd ? piece.to + lead : piece.to);
    if (body.points.length < 2) continue;
    const stripe = openStart || openEnd ? partOf(openStart ? piece.from - 2 * lead : piece.from, openEnd ? piece.to + 2 * lead : piece.to) : body;
    const { points: part, flat, widths, frames } = body;
    const sides = sideOf(frames);
    const bands = stripe === body ? sides : sideOf(stripe.frames);
    let bodyPath = poly([...sides.left, ...sides.right.slice().reverse()]);
    let edges = pathOf(sides.left) + pathOf(sides.right);
    const ends = [!piece.open0 && !looped && (piece.from <= 1e-6 || piece.gap0), !piece.open1 && !looped && (piece.to >= total - 1e-6 || piece.gap1)];
    const capped = [ends[0] && (piece.gap0 || [caps].flat()[0]), ends[1] && (piece.gap1 || [caps].flat().at(-1))];
    const sign = Math.sign(area([...sides.left, ...sides.right.slice().reverse()]));
    const tips = [null, null];
    [0, 1].forEach((side) => {
      if (!capped[side]) return;
      const at = side ? part.length - 1 : 0;
      const other = side ? part.length - 2 : 1;
      const tangent = G.unit3(G.sub3(part[side ? at : other], part[side ? other : at]));
      const outward = side ? tangent : G.mul3(tangent, -1);
      const inward = [flat[other][0] - flat[at][0], flat[other][1] - flat[at][1]];
      const { ring, arc, inner } = capOf(part[at], tangent, widths[at] / k, inward, P);
      const oriented = Math.sign(area(ring)) === sign ? ring : ring.slice().reverse();
      bodyPath += poly(oriented);
      edges += pathOf(arc);
      const open = G.dot3(outward, G.viewOf(P)) > 0;
      if (open) edges += pathOf(inner);
      tips[side] = { half: open ? inner : arc, frame: frames[at] };
    });
    const band = (a, b, offsets) => {
      const head = tips[1] ? stripeEnd(tips[1].half, tips[1].frame.p, tips[1].frame.nx, tips[1].frame.ny, offsets(tips[1].frame)[0], offsets(tips[1].frame)[1]) : [];
      const tail = tips[0] ? stripeEnd(tips[0].half, tips[0].frame.p, tips[0].frame.nx, tips[0].frame.ny, offsets(tips[0].frame)[1], offsets(tips[0].frame)[0]) : [];
      return poly([...a, ...head, ...b.slice().reverse(), ...tail]);
    };
    const toward = ({ nx, ny }) => nx * LIGHT[0] + ny * LIGHT[1];
    const shineOffsets = (frame) => [frame.half * 0.44 * toward(frame) + frame.half * 0.24, frame.half * 0.44 * toward(frame) - frame.half * 0.24];
    const shadeOffsets = (frame) => [-frame.half * 0.62 * toward(frame) + frame.half * 0.2, -frame.half * 0.62 * toward(frame) - frame.half * 0.2];
    const ringSpan = Object.assign([openStart ? piece.from - 2 * lead : piece.from, openEnd ? piece.to + 2 * lead : piece.to], { real0: !openStart, real1: !openEnd });
    out.push({
      rings: rings ? ringsOf(dense, total, radiusAt, P, ringSpan, rings) : [],
      ringTone: rings?.tone ?? "lo",
      mid: G.pointAlong(dense, (piece.from + piece.to) / 2),
      body: bodyPath,
      shine: band(bands.shineA, bands.shineB, shineOffsets),
      shade: band(bands.shadeA, bands.shadeB, shadeOffsets),
      edges,
      wide,
      points: part,
      radii: widths.map((width) => width / k),
      ends,
      from: piece.from,
      to: piece.to,
    });
  }
  return out;
}

export function ringsSvg(rings, tone = "lo") {
  const groups = new Map();
  for (const { d, alpha } of rings ?? []) {
    const level = Math.min(1, Math.round(alpha * 20) / 20);
    if (level <= 0) continue;
    groups.set(level, (groups.get(level) ?? "") + d);
  }
  return [...groups].map(([alpha, d]) => `<path class="tb-ring" data-tone="${tone}" d="${d}"${alpha < 1 ? ` opacity="${alpha.toFixed(2)}"` : ""}/>`).join("");
}

export function tubeSvg(piece, { tone = "hi" } = {}) {
  const shine = piece.wide ? `<path class="tb-shine" d="${piece.shine}"/><path class="tb-shade" d="${piece.shade}"/>` : "";
  return `<g class="tb"><path class="tb-body" d="${piece.body}"/>${shine}${ringsSvg(piece.rings, piece.ringTone)}<path class="iso-line tb-edge" data-tone="${tone}" d="${piece.edges}"/></g>`;
}
