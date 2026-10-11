import { cameraOf, roundedPlan } from "./iso-kit.mjs";
import { add3, sub3, mul3, dot3, len3, unit3, pathLength, resample, curveRuns, cut as cutAlong, tangentAlong } from "./lathe.mjs";
import { runsOf } from "./tube.mjs";

const BIG = 1e9;

export function body(F, poly, meta = {}) {
  return { kind: "body", F, poly: poly.map(([s, r]) => [s, Math.max(0, r)]), ...meta };
}

export function traced(meridian, { radius = null, error = 0.02, depth = 6 } = {}) {
  const segments = radius ? null : curveRuns(meridian);
  const out = [[meridian[0][0], meridian[0][1]]];
  for (let index = 0; index < meridian.length - 1; index++) {
    const [s0] = meridian[index];
    const [s1, r1] = meridian[index + 1];
    const along = Math.abs(s1 - s0) > 1e-9;
    const at = along && radius ? radius : along && segments[index] ? segments[index].radius : null;
    if (at) {
      const split = (a, b, level) => {
        const mid = (a + b) / 2;
        if (level >= depth || Math.abs(at(mid) - (at(a) + at(b)) / 2) <= error) return;
        split(a, mid, level + 1);
        out.push([mid, at(mid)]);
        split(mid, b, level + 1);
      };
      split(s0, s1, 0);
    }
    out.push([s1, r1]);
  }
  return out;
}

export function lathe(F, meridian, { radius = null, ...meta } = {}) {
  return body(F, traced(meridian, { radius }), meta);
}

export function solid(F, profile, { radius = null, ...meta } = {}) {
  const first = profile[0];
  const last = profile[profile.length - 1];
  return body(F, traced([[first[0], 0], ...profile, [last[0], 0]], { radius: radius && ((s) => radius(Math.max(first[0], Math.min(last[0], s)))) }), meta);
}

export function prism(F, polygon, s0, s1, meta = {}) {
  return { kind: "prism", F, poly: polygon.map(([x, y]) => [x, y]), s0: Math.min(s0, s1), s1: Math.max(s0, s1), ...meta };
}

const FLOOR = { o: [0, 0, 0], a: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] };

export const extruded = (ring, z, height, meta = {}) => prism(FLOOR, ring, z, z + height, meta);

export const slab = (plan, z, height, meta = {}, steps = 8) => extruded(roundedPlan(plan, steps), z, height, meta);

export const cylinder = (cx, cy, r, z, height, meta = {}) => disc({ o: [cx, cy, 0], a: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] }, z, z + height, r, meta);

export const disc = (F, s0, s1, r, meta = {}) => body(F, [[s0, 0], [s0, r], [s1, r], [s1, 0]], meta);

export const ring = (F, s0, s1, rIn, rOut, meta = {}) => body(F, [[s0, rIn], [s0, rOut], [s1, rOut], [s1, rIn]], meta);

export function box(min, max, meta = {}) {
  const centre = mul3(add3(min, max), 0.5);
  return { kind: "box", o: centre, axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], half: mul3(sub3(max, min), 0.5), ...meta };
}

export const orientedBox = (centre, axes, half, meta = {}) => ({ kind: "box", o: centre, axes: axes.map(unit3), half, ...meta });

export const ball = (at, r, meta = {}) => ({ kind: "ball", o: at, r, ...meta });

export function tube(points, radius, meta = {}) {
  const total = pathLength(points) || 1;
  let walked = 0;
  const radii = Array.isArray(radius)
    ? radius
    : points.map((point, index) => {
        if (index > 0) walked += len3(sub3(point, points[index - 1]));
        return typeof radius === "function" ? radius(walked / total) : radius;
      });
  return { kind: "tube", points, radii, round: meta.round ?? [false, false], ...meta };
}

export const slide = (F, from, to, r, meta = {}) => ({ ...disc(F, from, to, r, meta), slide: { from, to, r } });

export function moved(shape, offset) {
  if (!offset || (!offset[0] && !offset[1] && !offset[2])) return shape;
  if (shape.slide) {
    const { from, to, r } = shape.slide;
    const reach = Math.max(from + 1e-3, to + dot3(offset, shape.F.a));
    return { ...shape, poly: [[from, 0], [from, r], [reach, r], [reach, 0]], slide: { from, to: reach, r }, _b: undefined, _f: undefined };
  }
  const cut = shape.cut ? (Array.isArray(shape.cut) ? shape.cut.map((one) => moved(one, offset)) : moved(shape.cut, offset)) : undefined;
  if (shape.kind === "body" || shape.kind === "prism") return { ...shape, cut, F: { ...shape.F, o: add3(shape.F.o, offset) }, _b: undefined, _f: undefined };
  if (shape.kind === "tube") return { ...shape, cut, points: shape.points.map((p) => add3(p, offset)), tips: shape.tips && shape.tips.map((tip) => tip && add3(tip, offset)), _b: undefined };
  return { ...shape, cut, o: add3(shape.o, offset), _b: undefined };
}

function polyDistance(poly, s, r) {
  let best = BIG;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [s0, r0] = poly[j];
    const [s1, r1] = poly[i];
    const ds = s1 - s0;
    const dr = r1 - r0;
    const span = ds * ds + dr * dr;
    const t = span > 0 ? Math.max(0, Math.min(1, ((s - s0) * ds + (r - r0) * dr) / span)) : 0;
    const es = s - s0 - ds * t;
    const er = r - r0 - dr * t;
    const d = es * es + er * er;
    if (d < best && (r0 > 1e-9 || r1 > 1e-9)) best = d;
    if (r0 > r !== r1 > r && s < s0 + ((r - r0) * ds) / (dr || 1e-12)) inside = !inside;
  }
  const d = Math.sqrt(best);
  return inside ? -d : d;
}

function polygonDistance(poly, x, y) {
  let best = BIG;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [x0, y0] = poly[j];
    const [x1, y1] = poly[i];
    const dx = x1 - x0;
    const dy = y1 - y0;
    const span = dx * dx + dy * dy;
    const t = span > 0 ? Math.max(0, Math.min(1, ((x - x0) * dx + (y - y0) * dy) / span)) : 0;
    const ex = x - x0 - dx * t;
    const ey = y - y0 - dy * t;
    const d = ex * ex + ey * ey;
    if (d < best) best = d;
    if (y0 > y !== y1 > y && x < x0 + ((y - y0) * dx) / (dy || 1e-12)) inside = !inside;
  }
  const d = Math.sqrt(best);
  return inside ? -d : d;
}

function cappedSegment(p, a, b, r) {
  const bx = b[0] - a[0];
  const by = b[1] - a[1];
  const bz = b[2] - a[2];
  const px = p[0] - a[0];
  const py = p[1] - a[1];
  const pz = p[2] - a[2];
  const baba = bx * bx + by * by + bz * bz || 1e-12;
  const paba = px * bx + py * by + pz * bz;
  const x = Math.hypot(px * baba - bx * paba, py * baba - by * paba, pz * baba - bz * paba) - r * baba;
  const y = Math.abs(paba - baba * 0.5) - baba * 0.5;
  const x2 = x * x;
  const y2 = y * y * baba;
  const d = Math.max(x, y) < 0 ? -Math.min(x2, y2) : (x > 0 ? x2 : 0) + (y > 0 ? y2 : 0);
  return (Math.sign(d) * Math.sqrt(Math.abs(d))) / baba;
}

function segmentPoint(p, a, b) {
  const ba = sub3(b, a);
  const pa = sub3(p, a);
  const t = Math.max(0, Math.min(1, dot3(pa, ba) / (dot3(ba, ba) || 1e-12)));
  return { d: len3(sub3(pa, mul3(ba, t))), t };
}

export function sdf(shape, p) {
  const d = baseSdf(shape, p);
  return shape.cut ? Math.max(d, -cutDistance(shape, p)) : d;
}

function cutDistance(shape, p) {
  if (!Array.isArray(shape.cut)) return sdf(shape.cut, p);
  let best = BIG;
  for (const cut of shape.cut) best = Math.min(best, sdf(cut, p));
  return best;
}

function baseSdf(shape, p) {
  if (shape.kind === "body") {
    const { F } = shape;
    const q = sub3(p, F.o);
    const s = dot3(q, F.a);
    const x = dot3(q, F.u);
    const y = dot3(q, F.v);
    return polyDistance(shape.poly, s, Math.hypot(x, y));
  }
  if (shape.kind === "ball") {
    let d = len3(sub3(p, shape.o)) - shape.r;
    if (shape.flats) for (const [nx, ny, nz, cut] of shape.flats) d = Math.max(d, nx * (p[0] - shape.o[0]) + ny * (p[1] - shape.o[1]) + nz * (p[2] - shape.o[2]) - cut);
    return d;
  }
  if (shape.kind === "prism") {
    const { F } = shape;
    const q = sub3(p, F.o);
    const s = dot3(q, F.a);
    const flat = polygonDistance(shape.poly, dot3(q, F.u), dot3(q, F.v));
    const along = Math.max(shape.s0 - s, s - shape.s1);
    return Math.hypot(Math.max(flat, 0), Math.max(along, 0)) + Math.min(Math.max(flat, along), 0);
  }
  if (shape.kind === "box") {
    const q = sub3(p, shape.o);
    const d = shape.axes.map((axis, index) => Math.abs(dot3(q, axis)) - shape.half[index]);
    const outside = Math.hypot(Math.max(d[0], 0), Math.max(d[1], 0), Math.max(d[2], 0));
    return outside + Math.min(Math.max(d[0], d[1], d[2]), 0);
  }
  const { points, radii, round } = shape;
  let best = BIG;
  const last = points.length - 1;
  for (let index = 0; index < last; index++) {
    const r = (radii[index] + radii[index + 1]) / 2;
    const d = cappedSegment(p, points[index], points[index + 1], r);
    if (d < best) best = d;
  }
  if (round[0]) best = Math.min(best, len3(sub3(p, points[0])) - radii[0]);
  if (round[1]) best = Math.min(best, len3(sub3(p, points[last])) - radii[last]);
  return best;
}

export function boundsOf(shape) {
  if (shape._b) return shape._b;
  let min;
  let max;
  if (shape.kind === "body") {
    const { F, poly } = shape;
    const s0 = Math.min(...poly.map(([s]) => s));
    const s1 = Math.max(...poly.map(([s]) => s));
    const rm = Math.max(...poly.map(([, r]) => r));
    const a = add3(F.o, mul3(F.a, s0));
    const b = add3(F.o, mul3(F.a, s1));
    const spread = [0, 1, 2].map((i) => rm * Math.sqrt(Math.max(0, 1 - F.a[i] * F.a[i])));
    min = [0, 1, 2].map((i) => Math.min(a[i], b[i]) - spread[i]);
    max = [0, 1, 2].map((i) => Math.max(a[i], b[i]) + spread[i]);
  } else if (shape.kind === "ball") {
    min = shape.o.map((v) => v - shape.r);
    max = shape.o.map((v) => v + shape.r);
  } else if (shape.kind === "prism") {
    const { F } = shape;
    min = [BIG, BIG, BIG];
    max = [-BIG, -BIG, -BIG];
    for (const s of [shape.s0, shape.s1])
      for (const [x, y] of shape.poly) {
        const p = add3(add3(F.o, mul3(F.a, s)), add3(mul3(F.u, x), mul3(F.v, y)));
        for (let i = 0; i < 3; i++) (min[i] = Math.min(min[i], p[i])), (max[i] = Math.max(max[i], p[i]));
      }
  } else if (shape.kind === "box") {
    const extent = [0, 1, 2].map((i) => shape.axes.reduce((sum, axis, j) => sum + Math.abs(axis[i]) * shape.half[j], 0));
    min = shape.o.map((v, i) => v - extent[i]);
    max = shape.o.map((v, i) => v + extent[i]);
  } else {
    min = [BIG, BIG, BIG];
    max = [-BIG, -BIG, -BIG];
    shape.points.forEach((p, index) => {
      for (let i = 0; i < 3; i++) {
        min[i] = Math.min(min[i], p[i] - shape.radii[index]);
        max[i] = Math.max(max[i], p[i] + shape.radii[index]);
      }
    });
  }
  shape._b = { min, max };
  return shape._b;
}

export function cameraBasis(P) {
  const { sinA, cosA, sinE, cosE, k } = cameraOf(P);
  return { R: [sinA, -cosA, 0], D: [cosA * sinE, sinA * sinE, -cosE], V: [cosA * cosE, sinA * cosE, sinE], k, origin: P.origin };
}

function screenBox(shape, cam, offset) {
  const { min, max } = boundsOf(shape);
  let x0 = BIG;
  let y0 = BIG;
  let x1 = -BIG;
  let y1 = -BIG;
  const visit = (p) => {
    const x = cam.origin[0] + dot3(p, cam.R) * cam.k;
    const y = cam.origin[1] + dot3(p, cam.D) * cam.k;
    x0 = Math.min(x0, x);
    y0 = Math.min(y0, y);
    x1 = Math.max(x1, x);
    y1 = Math.max(y1, y);
  };
  if (shape.kind === "tube") {
    shape.points.forEach((p, index) => {
      const r = shape.radii[index];
      const q = add3(p, offset);
      visit(add3(q, mul3(cam.R, r)));
      visit(add3(q, mul3(cam.R, -r)));
      visit(add3(q, mul3(cam.D, r)));
      visit(add3(q, mul3(cam.D, -r)));
    });
  } else {
    for (const x of [min[0], max[0]]) for (const y of [min[1], max[1]]) for (const z of [min[2], max[2]]) visit(add3([x, y, z], offset));
  }
  return [x0, y0, x1, y1];
}

function spanOf(c, V, min, max) {
  let t0 = -BIG;
  let t1 = BIG;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(V[i]) < 1e-12) {
      if (c[i] < min[i] || c[i] > max[i]) return null;
      continue;
    }
    let a = (min[i] - c[i]) / V[i];
    let b = (max[i] - c[i]) / V[i];
    if (a > b) [a, b] = [b, a];
    t0 = Math.max(t0, a);
    t1 = Math.min(t1, b);
  }
  return t0 <= t1 ? [t0, t1] : null;
}

function tubeDistance(shape, p, segments) {
  const { points, radii, round } = shape;
  const last = points.length - 1;
  let best = BIG;
  for (const index of segments) {
    const d = cappedSegment(p, points[index], points[index + 1], (radii[index] + radii[index + 1]) / 2);
    if (d < best) best = d;
    for (const [joint, flag] of [[0, round[0]], [last, round[1]]]) {
      if (!flag || (joint !== index && joint !== index + 1)) continue;
      const q = points[joint];
      const e = Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) - radii[joint];
      if (e < best) best = e;
    }
  }
  return best;
}

function bodyDepth(shape, c, V, eps) {
  const { F, poly } = shape;
  if (!shape._f) {
    const s = poly.map(([value]) => value);
    shape._f = { aV: dot3(F.a, V), uV: dot3(F.u, V), vV: dot3(F.v, V), s0: Math.min(...s), s1: Math.max(...s), rm: Math.max(...poly.map(([, r]) => r)) };
  }
  const { aV, uV, vV, s0, s1, rm } = shape._f;
  const q = [c[0] - F.o[0], c[1] - F.o[1], c[2] - F.o[2]];
  const sc = dot3(q, F.a);
  const xc = dot3(q, F.u);
  const yc = dot3(q, F.v);
  let lo = -BIG;
  let hi = BIG;
  if (Math.abs(aV) > 1e-9) {
    const a = (s0 - sc) / aV;
    const b = (s1 - sc) / aV;
    lo = Math.min(a, b);
    hi = Math.max(a, b);
  } else if (sc < s0 || sc > s1) return null;
  const qa = uV * uV + vV * vV;
  if (qa > 1e-12) {
    const qb = 2 * (xc * uV + yc * vV);
    const qc = xc * xc + yc * yc - rm * rm;
    const disc = qb * qb - 4 * qa * qc;
    if (disc < 0) return null;
    const root = Math.sqrt(disc);
    lo = Math.max(lo, (-qb - root) / (2 * qa));
    hi = Math.min(hi, (-qb + root) / (2 * qa));
  } else if (xc * xc + yc * yc > rm * rm) return null;
  if (lo > hi) return null;
  let t = hi + eps;
  for (let step = 0; step < 160 && t >= lo - eps; step++) {
    let d = polyDistance(poly, sc + t * aV, Math.hypot(xc + t * uV, yc + t * vV));
    if (shape.cut) d = Math.max(d, -cutDistance(shape, [c[0] + V[0] * t, c[1] + V[1] * t, c[2] + V[2] * t]));
    if (d < eps) return t;
    t -= Math.max(d, eps * 0.5);
  }
  return null;
}

function rayDepth(shape, c, V, eps, segments) {
  if (shape.kind === "body") return bodyDepth(shape, c, V, eps);
  const span = spanOf(c, V, boundsOf(shape).min, boundsOf(shape).max);
  if (!span) return null;
  let t = span[1] + eps;
  for (let step = 0; step < 160 && t >= span[0] - eps; step++) {
    const p = [c[0] + V[0] * t, c[1] + V[1] * t, c[2] + V[2] * t];
    const d = segments ? tubeDistance(shape, p, segments) : sdf(shape, p);
    if (d < eps) return t;
    t -= Math.max(d, eps * 0.5);
  }
  return null;
}

function flatDistance(x, y, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const span = dx * dx + dy * dy;
  const t = span > 0 ? Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / span)) : 0;
  return Math.hypot(x - a[0] - dx * t, y - a[1] - dy * t);
}

export const shareOf = (item, state) => (typeof state === "number" ? state : state.shares[item.ride ?? item.piece] ?? 0);

export const offsetOf = (item, state) => mul3(item.move ?? [0, 0, 0], shareOf(item, state));

export const labelOf = (state) => (typeof state === "number" ? `share ${Math.round(state * 1000) / 1000}` : state.label);

export function springStates(order, { stagger = 0.035, stiffness = 64, damping = 14.2, substeps = 4, fps = 60, settle = 1e-3, stride = 1, reversals = [] } = {}) {
  const out = [];
  const scripts = [[true], [false], ...reversals.flatMap((at) => [[true, at], [false, at]])];
  for (const [on, back] of scripts) {
    const springs = Object.fromEntries(order.map((id) => [id, { x: on ? 0 : 1, v: 0, target: on ? 0 : 1, pending: null }]));
    const toggle = (value, clock) => {
      const goal = value ? 1 : 0;
      let index = 0;
      for (const id of value ? order : [...order].reverse()) {
        if (springs[id].target === goal) springs[id].pending = null;
        else springs[id].pending = { value: goal, at: clock + index++ * stagger };
      }
    };
    const name = back === undefined ? (on ? "apart" : "together") : `${on ? "apart" : "together"}, back at ${back.toFixed(2)} s,`;
    const dt = 1 / fps;
    let clock = 0;
    let goal = on ? 1 : 0;
    toggle(on, 0);
    for (let frame = 0; frame < 100000; frame++) {
      if (back !== undefined && clock < back && clock + dt >= back) toggle(!on, back), (goal = on ? 0 : 1);
      clock += dt;
      for (const spring of Object.values(springs)) {
        if (spring.pending && clock >= spring.pending.at) (spring.target = spring.pending.value), (spring.pending = null);
        for (let index = 0; index < substeps; index++) {
          const h = dt / substeps;
          spring.v += ((spring.target - spring.x) * stiffness - spring.v * damping) * h;
          spring.x += spring.v * h;
        }
      }
      const settled = (back === undefined || clock >= back) && Object.values(springs).every((spring) => !spring.pending && Math.abs(spring.x - goal) < settle && Math.abs(spring.v) < settle);
      if (settled) break;
      if ((frame + 1) % stride === 0) out.push({ label: `${name} ${clock.toFixed(3)} s`, shares: Object.fromEntries(Object.entries(springs).map(([id, spring]) => [id, spring.x])) });
    }
  }
  return out;
}

export const uniformStates = (count = 10) => Array.from({ length: count + 1 }, (_, index) => index / count);

export const reversalsOf = (order, stagger = 0.035, margin = 0.008) => order.flatMap((_, index) => [index * stagger - margin, index * stagger + margin]).filter((at) => at > 0);

export function prefixStates(order, { shares = [0.25, 0.5, 0.75, 1] } = {}) {
  const out = [];
  for (let count = 1; count < order.length; count++)
    for (const s of shares)
      out.push({ label: `first ${count - 1} out, ${order[count - 1]} at ${s}`, shares: Object.fromEntries(order.map((id, index) => [id, index < count - 1 ? 1 : index === count - 1 ? s : 0])) });
  return out;
}

function rasterOf(item, cam, step, offset) {
  const shapes = item.shapes.map((shape) => moved(shape, offset));
  const flat = (p) => [cam.origin[0] + dot3(p, cam.R) * cam.k, cam.origin[1] + dot3(p, cam.D) * cam.k];
  const boxes = shapes.map((shape) => screenBox(shape, cam, [0, 0, 0]));
  const guides = shapes.map((shape) => {
    if (shape.kind === "tube") {
      const points = shape.points.map(flat);
      const widths = shape.radii.map((r) => r * cam.k);
      const boxes = new Float64Array((points.length - 1) * 4);
      for (let i = 0; i < points.length - 1; i++) {
        const pad = Math.max(widths[i], widths[i + 1]) + step * 0.5;
        boxes[i * 4] = Math.min(points[i][0], points[i + 1][0]) - pad;
        boxes[i * 4 + 1] = Math.max(points[i][0], points[i + 1][0]) + pad;
        boxes[i * 4 + 2] = Math.min(points[i][1], points[i + 1][1]) - pad;
        boxes[i * 4 + 3] = Math.max(points[i][1], points[i + 1][1]) + pad;
      }
      return { points, widths, boxes };
    }
    if (shape.kind === "body") {
      const s = shape.poly.map(([value]) => value);
      return { axis: [flat(add3(shape.F.o, mul3(shape.F.a, Math.min(...s)))), flat(add3(shape.F.o, mul3(shape.F.a, Math.max(...s))))], reach: Math.max(...shape.poly.map(([, r]) => r)) * cam.k };
    }
    return null;
  });
  const x0 = Math.floor(Math.min(...boxes.map((b) => b[0])) / step);
  const y0 = Math.floor(Math.min(...boxes.map((b) => b[1])) / step);
  const x1 = Math.ceil(Math.max(...boxes.map((b) => b[2])) / step);
  const y1 = Math.ceil(Math.max(...boxes.map((b) => b[3])) / step);
  const w = x1 - x0 + 1;
  const h = y1 - y0 + 1;
  return { x0, y0, x1, y1, w, h, boxes, shapes, guides, depth: null };
}

function depthAt(raster, cam, step, cx, cy) {
  if (!raster.depth) {
    raster.depth = new Float32Array(raster.w * raster.h);
    raster.done = new Uint8Array(raster.w * raster.h);
  }
  const cell = (cy - raster.y0) * raster.w + (cx - raster.x0);
  if (raster.done[cell]) return raster.depth[cell];
  const x = cx * step;
  const y = cy * step;
  const c = add3(mul3(cam.R, (x - cam.origin[0]) / cam.k), mul3(cam.D, (y - cam.origin[1]) / cam.k));
  let best = NaN;
  for (let index = 0; index < raster.shapes.length; index++) {
    const shape = raster.shapes[index];
    const b = raster.boxes[index];
    if (x < b[0] || x > b[2] || y < b[1] || y > b[3]) continue;
    const guide = raster.guides[index];
    let segments = null;
    if (shape.kind === "tube") {
      segments = [];
      const boxes = guide.boxes;
      for (let i = 0; i < guide.points.length - 1; i++) {
        if (x < boxes[i * 4] || x > boxes[i * 4 + 1] || y < boxes[i * 4 + 2] || y > boxes[i * 4 + 3]) continue;
        if (flatDistance(x, y, guide.points[i], guide.points[i + 1]) <= Math.max(guide.widths[i], guide.widths[i + 1]) + step * 0.5) segments.push(i);
      }
      if (!segments.length) continue;
    } else if (shape.kind === "body" && flatDistance(x, y, guide.axis[0], guide.axis[1]) > guide.reach + step) continue;
    const t = rayDepth(shape, c, cam.V, 0.02, segments);
    if (t !== null && !(t <= best)) best = t;
  }
  raster.depth[cell] = best;
  raster.done[cell] = 1;
  return best;
}

function regionOf(flags, gw, gh, value, step) {
  const n = gw * gh;
  const on = new Uint8Array(n);
  let all = 0;
  let inner = 0;
  for (let cell = 0; cell < n; cell++) if (flags[cell] === value) (on[cell] = 1), all++;
  if (!all) return { all, inner, area: 0 };
  const grow = new Uint8Array(n);
  for (let y = 0; y < gh; y++)
    for (let x = 0; x < gw; x++) {
      const cell = y * gw + x;
      if (on[cell] && x > 0 && x < gw - 1 && y > 0 && y < gh - 1 && on[cell - 1] && on[cell + 1] && on[cell - gw] && on[cell + gw]) inner++;
      if (on[cell] || (x > 0 && on[cell - 1]) || (x < gw - 1 && on[cell + 1]) || (y > 0 && on[cell - gw]) || (y < gh - 1 && on[cell + gw])) grow[cell] = 1;
    }
  const shut = new Uint8Array(n);
  for (let y = 0; y < gh; y++)
    for (let x = 0; x < gw; x++) {
      const cell = y * gw + x;
      shut[cell] = on[cell] || (grow[cell] && (x === 0 || grow[cell - 1]) && (x === gw - 1 || grow[cell + 1]) && (y === 0 || grow[cell - gw]) && (y === gh - 1 || grow[cell + gw])) ? 1 : 0;
    }
  const core = new Uint8Array(n);
  for (let y = 1; y < gh - 1; y++)
    for (let x = 1; x < gw - 1; x++) {
      const cell = y * gw + x;
      core[cell] = shut[cell] && shut[cell - 1] && shut[cell + 1] && shut[cell - gw] && shut[cell + gw] ? 1 : 0;
    }
  const seen = new Uint8Array(n);
  const stack = new Int32Array(n);
  let largest = 0;
  for (let start = 0; start < n; start++) {
    if (!core[start] || seen[start]) continue;
    let top = 0;
    let size = 0;
    stack[top++] = start;
    seen[start] = 1;
    while (top) {
      const cell = stack[--top];
      size++;
      const x = cell % gw;
      const y = (cell - x) / gw;
      const visit = (next) => {
        if (core[next] && !seen[next]) (seen[next] = 1), (stack[top++] = next);
      };
      if (x > 0) visit(cell - 1);
      if (x < gw - 1) visit(cell + 1);
      if (y > 0) visit(cell - gw);
      if (y < gh - 1) visit(cell + gw);
    }
    if (size > largest) largest = size;
  }
  return { all, inner, area: largest * step * step };
}

function measure(ra, rb, dx, dy, dz, cam, step, tolerance, least) {
  if (ra.x1 < rb.x0 + dx || rb.x1 + dx < ra.x0 || ra.y1 < rb.y0 + dy || rb.y1 + dy < ra.y0) return null;
  const ix0 = Math.max(ra.x0, rb.x0 + dx);
  const ix1 = Math.min(ra.x1, rb.x1 + dx);
  const iy0 = Math.max(ra.y0, rb.y0 + dy);
  const iy1 = Math.min(ra.y1, rb.y1 + dy);
  const gw = ix1 - ix0 + 1;
  const gh = iy1 - iy0 + 1;
  const flags = new Int8Array(gw * gh);
  const worst = [0, 0];
  const at = [null, null];
  let any = false;
  for (let y = iy0; y <= iy1; y++) {
    for (let x = ix0; x <= ix1; x++) {
      const ta = depthAt(ra, cam, step, x, y);
      if (Number.isNaN(ta)) continue;
      const tb = depthAt(rb, cam, step, x - dx, y - dy) + dz;
      if (Number.isNaN(tb)) continue;
      const cell = (y - iy0) * gw + (x - ix0);
      if (ta > tb + tolerance) {
        flags[cell] = 1;
        any = true;
        if (ta - tb > worst[0]) (worst[0] = ta - tb), (at[0] = [x * step, y * step]);
      } else if (tb > ta + tolerance) {
        flags[cell] = -1;
        any = true;
        if (tb - ta > worst[1]) (worst[1] = tb - ta), (at[1] = [x * step, y * step]);
      }
    }
  }
  if (!any) return null;
  const front = [regionOf(flags, gw, gh, 1, step), regionOf(flags, gw, gh, -1, step)];
  if (front[0].area < least && front[1].area < least) return null;
  return front.map((region, side) => ({ ...region, depth: worst[side], at: at[side] }));
}

export function overlapsOf(items, P, { step = 0.25, tolerance = 0.3, least = 0.25, states = [0], skip = () => false } = {}) {
  const cam = cameraBasis(P);
  const live = items.map((item, index) => ({ item, index })).filter(({ item }) => item.shapes && item.shapes.length);
  const homes = live.map(() => null);
  let uid = 0;
  const stamp = (raster) => Object.assign(raster, { uid: uid++ });
  const home = (index) => homes[index] ?? (homes[index] = stamp(rasterOf(live[index].item, cam, step, [0, 0, 0])));
  const sliding = live.map(({ item }) => item.shapes.some((shape) => shape.slide));
  const slid = new Map();
  const rasterAt = (index, offset) => {
    if (!sliding[index]) return home(index);
    const id = `${index}|${offset.map((v) => v.toFixed(2)).join(",")}`;
    if (!slid.has(id)) slid.set(id, stamp(rasterOf(live[index].item, cam, step, offset)));
    return slid.get(id);
  };
  const shiftOf = (offset) => ({ gx: Math.round((dot3(offset, cam.R) * cam.k) / step), gy: Math.round((dot3(offset, cam.D) * cam.k) / step), dz: dot3(offset, cam.V) });
  const memo = new Map();
  const pairs = [];
  for (const state of states) {
    const offsets = live.map(({ item }) => offsetOf(item, state));
    const shifts = live.map((_, index) => (sliding[index] ? { gx: 0, gy: 0, dz: 0 } : shiftOf(offsets[index])));
    for (let a = 0; a < live.length; a++) {
      const ra = rasterAt(a, offsets[a]);
      const sa = shifts[a];
      for (let b = a + 1; b < live.length; b++) {
        const rb = rasterAt(b, offsets[b]);
        const sb = shifts[b];
        if (ra.x1 + sa.gx < rb.x0 + sb.gx || rb.x1 + sb.gx < ra.x0 + sa.gx || ra.y1 + sa.gy < rb.y0 + sb.gy || rb.y1 + sb.gy < ra.y0 + sa.gy) continue;
        if (skip(live[a].item, live[b].item)) continue;
        const dx = sb.gx - sa.gx;
        const dy = sb.gy - sa.gy;
        const dz = sb.dz - sa.dz;
        const id = `${ra.uid}|${rb.uid}|${dx},${dy},${dz.toFixed(3)}`;
        let hit = memo.get(id);
        if (hit === undefined) {
          hit = measure(ra, rb, dx, dy, dz, cam, step, tolerance, least);
          memo.set(id, hit);
        }
        if (!hit) continue;
        pairs.push({ state, a: live[a].index, b: live[b].index, front: hit.map((side) => ({ ...side, at: side.at && [side.at[0] + sa.gx * step, side.at[1] + sa.gy * step] })) });
      }
    }
  }
  return { pairs, least, states, step, V: cam.V };
}

const liftOf = (item, V) => dot3(item.move ?? [0, 0, 0], V);

export const keyAt = (item, state, V) => item.key + liftOf(item, V) * shareOf(item, state);

export function orderKeys(items, overlaps, { gap = 0.02, rounds = 0 } = {}) {
  const { pairs, least, V } = overlaps;
  const start = items.map((item) => item.key);
  const strongest = new Map();
  const cycles = [];
  for (const pair of pairs) {
    const aFront = pair.front[0].area >= least;
    const bFront = pair.front[1].area >= least;
    if (aFront && bFront) {
      cycles.push(pair);
      continue;
    }
    const [front, back] = aFront ? [pair.a, pair.b] : [pair.b, pair.a];
    const weight = pair.front[aFront ? 0 : 1].area;
    const need = gap + shareOf(items[back], pair.state) * liftOf(items[back], V) - shareOf(items[front], pair.state) * liftOf(items[front], V);
    const id = `${front}|${back}`;
    const prior = strongest.get(id);
    if (!prior) strongest.set(id, { front, back, weight, pair, need });
    else {
      prior.weight = Math.max(prior.weight, weight);
      if (need > prior.need) (prior.need = need), (prior.pair = pair);
    }
  }
  const rules = [...strongest.values()];
  const limit = rounds || Math.min(items.length + 1, 400);
  const dropped = [];
  let keys = start.slice();
  let active = rules.slice();
  for (let attempt = 0; attempt < 400; attempt++) {
    keys = start.slice();
    const by = new Array(items.length).fill(null);
    let changed = true;
    let round = 0;
    let last = -1;
    for (; changed && round < limit; round++) {
      changed = false;
      for (const rule of active) {
        if (keys[rule.front] - keys[rule.back] >= rule.need - 1e-9) continue;
        keys[rule.front] = keys[rule.back] + rule.need;
        by[rule.front] = rule;
        last = rule.front;
        changed = true;
      }
    }
    if (!changed) break;
    const seen = new Map();
    let node = last;
    let step = 0;
    while (node !== -1 && by[node] && !seen.has(node) && step < items.length * 2) {
      seen.set(node, step++);
      node = by[node].back;
    }
    const loop = [];
    if (node !== -1 && seen.has(node)) {
      let walk = node;
      do {
        loop.push(by[walk]);
        walk = by[walk].back;
      } while (walk !== node && loop.length <= items.length);
    }
    const victim = (loop.length ? loop : active.filter((rule) => keys[rule.front] - keys[rule.back] < rule.need - 1e-9)).reduce((low, rule) => (rule.weight < low.weight ? rule : low));
    dropped.push(victim);
    active = active.filter((rule) => rule !== victim);
  }
  return { keys, rules: rules.length, cycles, dropped };
}

export function depthOrder(items, P, options = {}) {
  const overlaps = options.overlaps ?? overlapsOf(items, P, options);
  const { pairs, least, states, V } = overlaps;
  const allow = options.allow ?? [];
  const dropped = new Set((options.solved?.dropped ?? []).map((rule) => `${rule.front}|${rule.back}`));
  const missing = items.filter((item) => !item.shapes || !item.shapes.length).map((item) => item.name);
  const ranks = new Map();
  for (const state of states) {
    const order = items.map((item, index) => ({ index, key: keyAt(item, state, V) })).sort((x, y) => x.key - y.key || x.index - y.index);
    const rank = new Int32Array(items.length);
    order.forEach(({ index }, position) => (rank[index] = position));
    ranks.set(state, rank);
  }
  const rest = options.rest ?? states.filter((state) => state === 0 || state === 1);
  const problems = [];
  const passing = [];
  for (const pair of pairs) {
    const rank = ranks.get(pair.state);
    const aFirst = rank[pair.a] < rank[pair.b];
    const wrong = pair.front[aFirst ? 0 : 1];
    const right = pair.front[aFirst ? 1 : 0];
    if (wrong.area < least) continue;
    const front = items[aFirst ? pair.a : pair.b];
    const back = items[aFirst ? pair.b : pair.a];
    const excused = allow.find(([x, y]) => (matches([x], front.name) && matches([y], back.name)) || (matches([y], front.name) && matches([x], back.name)));
    const entry = {
      state: pair.state,
      label: labelOf(pair.state),
      front: front.name,
      back: back.name,
      frontKey: keyAt(front, pair.state, V),
      backKey: keyAt(back, pair.state, V),
      cells: wrong.all,
      area: wrong.area,
      against: right.area,
      cycle: right.area >= least,
      depth: wrong.depth,
      at: wrong.at,
    };
    const front_ = aFirst ? pair.a : pair.b;
    const back_ = aFirst ? pair.b : pair.a;
    entry.unorderable = !rest.includes(pair.state) && dropped.has(`${front_}|${back_}`);
    const replayed = typeof pair.state !== "number";
    (!excused && (rest.includes(pair.state) || replayed || (!entry.cycle && !entry.unorderable)) ? problems : passing).push(entry);
  }
  return { problems, passing, missing };
}

function samplesOf(shape, spacing) {
  const dense = resample(shape.points, spacing);
  const total = pathLength(shape.points) || 1;
  const lengths = [0];
  for (let index = 1; index < shape.points.length; index++) lengths.push(lengths[index - 1] + len3(sub3(shape.points[index], shape.points[index - 1])));
  const radiusAt = (at) => {
    let index = 1;
    while (index < lengths.length - 1 && lengths[index] < at) index++;
    const span = lengths[index] - lengths[index - 1] || 1;
    const t = Math.max(0, Math.min(1, (at - lengths[index - 1]) / span));
    return shape.radii[index - 1] + (shape.radii[index] - shape.radii[index - 1]) * t;
  };
  const step = total / Math.max(1, dense.length - 1);
  return dense.map((p, index) => {
    const t = unit3(sub3(dense[Math.min(dense.length - 1, index + 1)], dense[Math.max(0, index - 1)]));
    return { p, t, along: index * step, edge: Math.min(index * step, total - index * step), r: radiusAt(index * step) };
  });
}

const overlaps = (a, b, pad) => a.min.every((v, i) => v - pad <= b.max[i] && b.min[i] - pad <= a.max[i]);

function centreline(shape, p) {
  let best = { d: BIG, r: 0 };
  for (let index = 0; index < shape.points.length - 1; index++) {
    const { d, t } = segmentPoint(p, shape.points[index], shape.points[index + 1]);
    if (d < best.d) best = { d, r: shape.radii[index] + (shape.radii[index + 1] - shape.radii[index]) * t };
  }
  return best;
}

const matches = (list, name) =>
  (list ?? []).some((entry) => {
    if (typeof entry === "function") return entry(name);
    if (entry.endsWith("*")) return name.startsWith(entry.slice(0, -1));
    return name === entry || name.startsWith(entry + ".") || name.startsWith(entry + ":") || name.startsWith(entry + "#");
  });

const sameOffset = (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2]) < 1e-6;

const placed = (shape, state) => (state === undefined || !shape.move ? shape : moved(shape, offsetOf(shape, state)));

const offsetAt = (shape, state) => (state === undefined || !shape.move ? [0, 0, 0] : offsetOf(shape, state));

function tipDistance(other, p) {
  if (other.kind === "tube") {
    const { d, r } = centreline(other, p);
    return d - r;
  }
  return sdf(other, p);
}

function rimOf(tube, index) {
  const tip = tube.tips[index];
  const axis = tube.tipAxes ? tube.tipAxes[index] : unit3(index ? sub3(tube.points[tube.points.length - 1], tube.points[tube.points.length - 2]) : sub3(tube.points[1], tube.points[0]));
  const radius = tube.tipRims ? tube.tipRims[index] : tube.radii[index ? tube.radii.length - 1 : 0];
  const helper = Math.abs(axis[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
  const u = unit3([helper[1] * axis[2] - helper[2] * axis[1], helper[2] * axis[0] - helper[0] * axis[2], helper[0] * axis[1] - helper[1] * axis[0]]);
  const v = [axis[1] * u[2] - axis[2] * u[1], axis[2] * u[0] - axis[0] * u[2], axis[0] * u[1] - axis[1] * u[0]];
  return Array.from({ length: 12 }, (_, k) => add3(tip, add3(mul3(u, radius * Math.cos((k * Math.PI) / 6)), mul3(v, radius * Math.sin((k * Math.PI) / 6)))));
}

export function mountsOf(tubes, solids, { tolerance = 0.15 } = {}) {
  const byName = new Map(tubes.map((tube) => [tube.name, tube]));
  const out = new Map();
  for (const tube of tubes) {
    const found = [null, null];
    (tube.tips ?? []).forEach((tip, index) => {
      if (!tip) return;
      const rim = rimOf(tube, index);
      let best = null;
      const consider = (other) => {
        if (other === tube || (other.owner && other.owner === tube.owner) || other.name === tube.owner || other.anchor === false) return;
        const { min, max } = boundsOf(other);
        if (!overlaps({ min: tip, max: tip }, { min, max }, 3 + (tube.tipRims ? tube.tipRims[index] : 0))) return;
        const ds = rim.map((p) => tipDistance(other, p));
        const score = (Math.abs(tipDistance(other, tip)) + ds.reduce((sum, d) => sum + Math.abs(d), 0) / ds.length) / 2;
        if (!best || score < best.score) best = { shape: other, score, d: tipDistance(other, tip), high: Math.max(...ds), low: Math.min(...ds) };
      };
      solids.forEach(consider);
      tubes.forEach(consider);
      if (best && byName.get(best.shape.owner) && best.shape.kind !== "tube") best.host = byName.get(best.shape.owner);
      found[index] = best && { ...best, tip, landed: best.high <= tolerance && best.low >= -tolerance };
    });
    out.set(tube, found);
  }
  return out;
}

function ringAround(sample, r) {
  const axis = sample.t;
  const helper = Math.abs(axis[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
  const u = unit3([helper[1] * axis[2] - helper[2] * axis[1], helper[2] * axis[0] - helper[0] * axis[2], helper[0] * axis[1] - helper[1] * axis[0]]);
  const v = [axis[1] * u[2] - axis[2] * u[1], axis[2] * u[0] - axis[0] * u[2], axis[0] * u[1] - axis[1] * u[0]];
  return Array.from({ length: 12 }, (_, k) => add3(sample.p, add3(mul3(u, r * Math.cos((k * Math.PI) / 6)), mul3(v, r * Math.sin((k * Math.PI) / 6)))));
}

export function clearances(tubes, solids, { ratio = 0.5, least = 0.2, spacing = 0.5, touching = -0.05, through = -0.1, state, only, mounts = mountsOf(tubes, solids) } = {}) {
  const gapOf = (r) => Math.max(least, ratio * r);
  const problems = new Map();
  const note = (tube, other, kind, clearance, need, sample) => {
    if (only && kind !== only) return;
    const id = `${tube.name}|${other.name}`;
    const prior = problems.get(id);
    if (prior && prior.clearance <= clearance) {
      prior.from = Math.min(prior.from, sample.along);
      prior.to = Math.max(prior.to, sample.along);
      return;
    }
    problems.set(id, { tube: tube.name, other: other.name, kind, clearance, need, at: sample.p, from: prior ? Math.min(prior.from, sample.along) : sample.along, to: prior ? Math.max(prior.to, sample.along) : sample.along, length: pathLength(tube.points), state: state === undefined ? "" : labelOf(state) });
  };
  const poseOf = new Map([...tubes, ...solids].map((shape) => [shape, placed(shape, state)]));
  const offsets = new Map([...tubes, ...solids].map((shape) => [shape, offsetAt(shape, state)]));
  const together = (a, b) => len3(sub3(offsets.get(a), offsets.get(b))) < 0.05;
  const runsBy = new Map();
  for (const tube of tubes) runsBy.set(tube.owner, [...(runsBy.get(tube.owner) ?? []), tube]);
  const tipsOf = (owner) => (runsBy.get(owner) ?? []).flatMap((run) => (mounts.get(run) ?? []).filter(Boolean).map((mount) => ({ run, mount })));
  const seatedOn = (shape, owner) => shape && shape.cut && [shape.cut].flat().some((cut) => cut.owner === owner);
  const zone = (r) => r + gapOf(r) + 1;
  const ownerOf = (other) => other.owner ?? other.name;
  const related = (tube, other, sample) => {
    if (seatedOn(other, tube.owner)) return true;
    const tips = tipsOf(tube.owner);
    const near = (entry) => len3(sub3(sample.p, add3(entry.mount.tip, offsets.get(entry.run)))) <= zone(sample.r) + (entry.run.tipRims ? Math.max(...entry.run.tipRims) : 0);
    for (const entry of tips) {
      if (!near(entry) || !together(tube, entry.mount.shape)) continue;
      const mount = entry.mount.shape;
      if (mount === other || (entry.mount.host && (entry.mount.host === other || other.owner === entry.mount.host.owner))) return true;
      if (tipsOf(ownerOf(other)).some((theirs) => theirs.mount.shape === mount)) return true;
    }
    for (const entry of tipsOf(ownerOf(other))) {
      const mount = entry.mount.shape;
      const lands = mount.owner === tube.owner || seatedOn(mount, tube.owner);
      if (!lands || !together(entry.run, tube)) continue;
      if (len3(sub3(sample.p, add3(entry.mount.tip, offsets.get(entry.run)))) <= zone(sample.r) + (entry.run.tipRims ? Math.max(...entry.run.tipRims) : 0) + sample.r) return true;
    }
    return false;
  };
  const nearTip = (tube, sample) => {
    const at = (tube.offset ?? 0) + sample.along;
    const total = tube.total ?? pathLength(tube.points);
    const tips = mounts.get(tube) ?? [null, null];
    return (tips[0] && at <= zone(sample.r)) || (tips[1] && total - at <= zone(sample.r));
  };
  const clearanceTo = (other, sample) => {
    if (sample.edge >= sample.r) return sdf(other, sample.p) - sample.r;
    return Math.min(...ringAround(sample, sample.r).map((q) => sdf(other, q)));
  };
  for (const tube of tubes) {
    const posed = poseOf.get(tube);
    const samples = samplesOf(posed, spacing);
    const tb = boundsOf(posed);
    for (const shape of solids) {
      if (shape.owner && shape.owner === tube.owner) continue;
      if (matches(tube.touch, shape.name) || matches(shape.touch, tube.name)) continue;
      const other = poseOf.get(shape);
      const ob = boundsOf(other);
      if (!overlaps(tb, ob, 4)) continue;
      for (const sample of samples) {
        if (!overlaps({ min: sample.p, max: sample.p }, ob, sample.r + 3)) continue;
        const gap = gapOf(sample.r);
        const clearance = clearanceTo(other, sample);
        if (clearance >= gap) continue;
        if (related(tube, shape, sample)) continue;
        if (nearTip(tube, sample) && clearance >= through) continue;
        note(tube, shape, clearance < 0 ? "through" : "close", clearance, gap, sample);
      }
    }
  }
  for (let a = 0; a < tubes.length; a++) {
    const ta = poseOf.get(tubes[a]);
    const samples = samplesOf(ta, spacing);
    for (let b = 0; b < tubes.length; b++) {
      if (a === b) continue;
      const tb = poseOf.get(tubes[b]);
      if (tubes[a].owner && tubes[a].owner === tubes[b].owner && tubes[a].joined) continue;
      if (matches(tubes[a].touch, tubes[b].name) || matches(tubes[b].touch, tubes[a].name)) continue;
      if (!overlaps(boundsOf(ta), boundsOf(tb), 3)) continue;
      const bundled = tubes[a].bundle && tubes[a].bundle === tubes[b].bundle;
      for (const sample of samples) {
        if (!overlaps({ min: sample.p, max: sample.p }, boundsOf(tb), sample.r + 3)) continue;
        const clearance = sample.edge >= sample.r ? sdf(tb, sample.p) - sample.r : Math.min(...ringAround(sample, sample.r).map((q) => sdf(tb, q)));
        const r = centreline(tb, sample.p).r;
        const gap = bundled ? touching : gapOf(Math.min(sample.r, r));
        if (clearance >= gap) continue;
        if (related(tubes[a], tubes[b], sample)) continue;
        if (nearTip(tubes[a], sample) && clearance >= through) continue;
        if (a > b && problems.has(`${tubes[b].name}|${tubes[a].name}`)) continue;
        note(tubes[a], tubes[b], clearance < 0 ? "through" : "close", clearance, gap, sample);
      }
    }
  }
  const seen = new Set();
  return [...problems.values()]
    .sort((x, y) => x.clearance - y.clearance)
    .filter((c) => {
      const id = [c.tube, c.other].sort().join("|");
      if (seen.has(id)) return false;
      seen.add(id);
      return true;
    });
}

export function terminals(tubes, solids, { tolerance = 0.15, mounts = mountsOf(tubes, solids, { tolerance }) } = {}) {
  const out = [];
  for (const tube of tubes) {
    (mounts.get(tube) ?? []).forEach((mount, index) => {
      if (!tube.tips || !tube.tips[index] || (mount && mount.landed)) return;
      out.push({ tube: tube.name, end: index ? "end" : "start", at: tube.tips[index], d: mount ? mount.d : Infinity, high: mount ? mount.high : Infinity, low: mount ? mount.low : -Infinity, nearest: mount ? mount.shape.name : "nothing" });
    });
  }
  return out;
}

export function looseEnds(tubes, solids, { state = 1, tolerance = 0.15 } = {}) {
  const out = [];
  for (const tube of tubes) {
    const offset = offsetAt(tube, state);
    (tube.tips ?? []).forEach((tip, index) => {
      if (!tip) return;
      const hit = solids.some((shape) => {
        if ((shape.owner && shape.owner === tube.owner) || shape.name === tube.owner) return false;
        if (!sameOffset(offsetAt(shape, state), offset)) return false;
        const { min, max } = boundsOf(shape);
        return overlaps({ min: tip, max: tip }, { min, max }, 1) && sdf(shape, tip) <= tolerance;
      });
      if (!hit) out.push({ tube: tube.name, end: index ? "end" : "start", at: add3(tip, offset), state: labelOf(state), free: Boolean(tube.free && tube.free[index]) });
    });
  }
  return out;
}

function surfaceOf(shape, spacing = 0.6) {
  const points = [];
  if (shape.kind === "body") {
    const { F, poly } = shape;
    const u = F.u ?? unit3(Math.abs(F.a[2]) > 0.9 ? [1, 0, 0] : [-F.a[1], F.a[0], 0]);
    const v = F.v ?? [F.a[1] * u[2] - F.a[2] * u[1], F.a[2] * u[0] - F.a[0] * u[2], F.a[0] * u[1] - F.a[1] * u[0]];
    for (let i = 0; i < poly.length - 1; i++) {
      const [s0, r0] = poly[i];
      const [s1, r1] = poly[i + 1];
      const steps = Math.max(1, Math.ceil(Math.hypot(s1 - s0, r1 - r0) / spacing));
      for (let j = 0; j <= steps; j++) {
        const s = s0 + ((s1 - s0) * j) / steps;
        const r = r0 + ((r1 - r0) * j) / steps;
        const around = Math.max(1, Math.min(160, Math.ceil((2 * Math.PI * r) / spacing)));
        for (let k = 0; k < around; k++) {
          const t = (k / around) * 2 * Math.PI;
          points.push(add3(add3(F.o, mul3(F.a, s)), add3(mul3(u, r * Math.cos(t)), mul3(v, r * Math.sin(t)))));
        }
      }
    }
  } else if (shape.kind === "box") {
    for (let axis = 0; axis < 3; axis++)
      for (const sign of [-1, 1]) {
        const o1 = (axis + 1) % 3;
        const o2 = (axis + 2) % 3;
        const n1 = Math.max(1, Math.ceil((2 * shape.half[o1]) / spacing));
        const n2 = Math.max(1, Math.ceil((2 * shape.half[o2]) / spacing));
        for (let i = 0; i <= n1; i++)
          for (let j = 0; j <= n2; j++) {
            const c = [0, 0, 0];
            c[axis] = sign * shape.half[axis];
            c[o1] = -shape.half[o1] + (2 * shape.half[o1] * i) / n1;
            c[o2] = -shape.half[o2] + (2 * shape.half[o2] * j) / n2;
            points.push(add3(shape.o, add3(add3(mul3(shape.axes[0], c[0]), mul3(shape.axes[1], c[1])), mul3(shape.axes[2], c[2]))));
          }
      }
  } else if (shape.kind === "prism") {
    const { F, poly, s0, s1 } = shape;
    const at = (s, x, y) => add3(add3(F.o, mul3(F.a, s)), add3(mul3(F.u, x), mul3(F.v, y)));
    const rows = Math.max(1, Math.ceil((s1 - s0) / spacing));
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [x0, y0] = poly[j];
      const [x1, y1] = poly[i];
      const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / spacing));
      for (let k = 0; k < steps; k++)
        for (let row = 0; row <= rows; row++) points.push(at(s0 + ((s1 - s0) * row) / rows, x0 + ((x1 - x0) * k) / steps, y0 + ((y1 - y0) * k) / steps));
    }
    const xs = poly.map(([x]) => x);
    const ys = poly.map(([, y]) => y);
    for (let x = Math.min(...xs); x <= Math.max(...xs); x += spacing)
      for (let y = Math.min(...ys); y <= Math.max(...ys); y += spacing) if (polygonDistance(poly, x, y) < 0) points.push(at(s0, x, y), at(s1, x, y));
  } else if (shape.kind === "ball") {
    const count = Math.max(24, Math.min(600, Math.ceil((4 * Math.PI * shape.r * shape.r) / (spacing * spacing))));
    for (let i = 0; i < count; i++) {
      const z = 1 - (2 * (i + 0.5)) / count;
      const ring = Math.sqrt(1 - z * z);
      const t = i * 2.399963;
      points.push(add3(shape.o, mul3([ring * Math.cos(t), ring * Math.sin(t), z], shape.r)));
    }
    if (shape.flats) {
      const kept = points.filter((p) => shape.flats.every(([nx, ny, nz, cut]) => nx * (p[0] - shape.o[0]) + ny * (p[1] - shape.o[1]) + nz * (p[2] - shape.o[2]) <= cut + 1e-9));
      points.length = 0;
      points.push(...kept);
      for (const [nx, ny, nz, cut] of shape.flats) {
        const n = [nx, ny, nz];
        const helper = Math.abs(nz) > 0.9 ? [1, 0, 0] : [0, 0, 1];
        const u = unit3([helper[1] * nz - helper[2] * ny, helper[2] * nx - helper[0] * nz, helper[0] * ny - helper[1] * nx]);
        const v = [ny * u[2] - nz * u[1], nz * u[0] - nx * u[2], nx * u[1] - ny * u[0]];
        const rho = Math.sqrt(Math.max(0, shape.r * shape.r - cut * cut));
        const centre = add3(shape.o, mul3(n, cut));
        for (let x = -rho; x <= rho; x += spacing)
          for (let y = -rho; y <= rho; y += spacing) {
            if (x * x + y * y > rho * rho) continue;
            const p = add3(centre, add3(mul3(u, x), mul3(v, y)));
            if (shape.flats.every(([mx, my, mz, other]) => mx * (p[0] - shape.o[0]) + my * (p[1] - shape.o[1]) + mz * (p[2] - shape.o[2]) <= other + 1e-6)) points.push(p);
          }
      }
    }
  } else {
    for (const { p, r } of samplesOf(shape, spacing)) for (let k = 0; k < 8; k++) points.push(add3(p, mul3([Math.cos((k * Math.PI) / 4), Math.sin((k * Math.PI) / 4), 0], r)));
  }
  return shape.cut ? points.filter((p) => cutDistance(shape, p) >= -1e-3) : points;
}

const seatedOn = (shape, host) => Boolean(shape.seated && shape.cut && [shape.cut].flat().some((cut) => cut === host || (cut.name && cut.name === host.name)));

export const grounded = (shape) => Boolean(shape.ground) || (shape.name ?? "").startsWith("stand.");

export function supports(solids, tubes, { ground = grounded, touch = 0.15, sunk = 0.5 } = {}) {
  const nodes = [...solids, ...tubes.filter((tube) => !tube.limp)];
  const surfaces = nodes.map((shape) => surfaceOf(shape));
  const links = nodes.map(() => []);
  const deep = [];
  for (let a = 0; a < nodes.length; a++) {
    const ba = boundsOf(nodes[a]);
    for (let b = a + 1; b < nodes.length; b++) {
      const bb = boundsOf(nodes[b]);
      if (!overlaps(ba, bb, touch + 0.05)) continue;
      const same = (nodes[a].owner && (nodes[a].owner === nodes[b].owner || nodes[a].owner === nodes[b].name)) || (nodes[b].owner && nodes[b].owner === nodes[a].name);
      let gap = BIG;
      for (const p of surfaces[a]) gap = Math.min(gap, sdf(nodes[b], p));
      for (const p of surfaces[b]) gap = Math.min(gap, sdf(nodes[a], p));
      if (gap <= touch) links[a].push(b), links[b].push(a);
      const solidPair = nodes[a].kind !== "tube" && nodes[b].kind !== "tube";
      if (solidPair && !same && gap < -sunk && !seatedOn(nodes[a], nodes[b]) && !seatedOn(nodes[b], nodes[a])) deep.push({ a: nodes[a].name, b: nodes[b].name, depth: -gap });
    }
  }
  const held = new Uint8Array(nodes.length);
  const queue = [];
  nodes.forEach((shape, index) => ground(shape) && ((held[index] = 1), queue.push(index)));
  while (queue.length) {
    const index = queue.pop();
    for (const next of links[index]) if (!held[next]) (held[next] = 1), queue.push(next);
  }
  const loose = nodes.filter((_, index) => !held[index] && nodes[index].kind !== "tube").map((shape) => shape.name);
  const hanging = nodes.filter((_, index) => !held[index] && nodes[index].kind === "tube").map((shape) => shape.name);
  return { loose, hanging, sunk: deep.sort((x, y) => y.depth - x.depth) };
}

export function folds(routes, P, { fold = 140, turn = 110, pinch = 2, items = null } = {}) {
  const cam = cameraBasis(P);
  const entries = items ? entriesOf(items, cam) : null;
  const shown = (name, p) => !entries || frontAt(entries, cam, cam.origin[0] + dot3(p, cam.R) * cam.k, cam.origin[1] + dot3(p, cam.D) * cam.k) === name;
  const flat = (p) => [dot3(p, cam.R) * cam.k, dot3(p, cam.D) * cam.k];
  const angle = (a, b) => (Math.acos(Math.max(-1, Math.min(1, (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(a[0], a[1]) * Math.hypot(b[0], b[1]) || 1)))) * 180) / Math.PI;
  const out = [];
  for (const { name, points, r } of routes) {
    const step = Math.max(0.25, r * 0.5);
    const dense = resample(points, step);
    const screen = dense.map(flat);
    const reach = Math.max(2, Math.round((4 * r) / step));
    const width = r * cam.k;
    let worst = null;
    for (const span of [1.5, 3, 6]) {
      const near = Math.max(1, Math.round((span * r) / step));
      for (let i = near; i < dense.length - near; i++) {
        const p0 = screen[i - near];
        const p1 = screen[i];
        const p2 = screen[i + near];
        const a = [p1[0] - p0[0], p1[1] - p0[1]];
        const b = [p2[0] - p1[0], p2[1] - p1[1]];
        const c = [p2[0] - p0[0], p2[1] - p0[1]];
        const la = Math.hypot(a[0], a[1]);
        const lb = Math.hypot(b[0], b[1]);
        if (la < width * 0.3 || lb < width * 0.3) continue;
        const bend = angle(a, b);
        const area = Math.abs(a[0] * b[1] - a[1] * b[0]) / 2;
        const radius = area > 1e-9 ? (la * lb * Math.hypot(c[0], c[1])) / (4 * area) : Infinity;
        if (bend > 90 && radius < pinch * width && (!worst || radius < worst.radius) && shown(name, dense[i])) worst = { name, kind: "pinch", flatTurn: bend, radius, width, at: dense[i], index: i };
      }
    }
    if (worst) {
      let toStart = 0;
      for (let i = 1; i <= worst.index; i++) toStart += Math.hypot(screen[i][0] - screen[i - 1][0], screen[i][1] - screen[i - 1][1]);
      let toEnd = 0;
      for (let i = worst.index + 1; i < screen.length; i++) toEnd += Math.hypot(screen[i][0] - screen[i - 1][0], screen[i][1] - screen[i - 1][1]);
      if (Math.min(toStart, toEnd) > 3 * width) out.push(worst);
    }
    for (let i = reach; i < dense.length - reach; i++) {
      const a = [screen[i][0] - screen[i - reach][0], screen[i][1] - screen[i - reach][1]];
      const b = [screen[i + reach][0] - screen[i][0], screen[i + reach][1] - screen[i][1]];
      if (Math.hypot(a[0], a[1]) < 1e-6 || Math.hypot(b[0], b[1]) < 1e-6) continue;
      const flatTurn = angle(a, b);
      const A3 = sub3(dense[i], dense[i - reach]);
      const B3 = sub3(dense[i + reach], dense[i]);
      const realTurn = (Math.acos(Math.max(-1, Math.min(1, dot3(A3, B3) / (len3(A3) * len3(B3) || 1)))) * 180) / Math.PI;
      if (flatTurn > fold && realTurn < turn && shown(name, dense[i])) {
        out.push({ name, kind: "fold", flatTurn, realTurn, at: dense[i] });
        i += 2 * reach;
      }
    }
  }
  return out;
}

export function linksOf(tubes, mounts) {
  const links = new Map();
  const link = (a, b) => {
    if (!a || !b || a === b) return;
    if (!links.has(a)) links.set(a, new Set());
    if (!links.has(b)) links.set(b, new Set());
    links.get(a).add(b);
    links.get(b).add(a);
  };
  const ownerOf = (shape) => shape.owner ?? shape.name;
  const seats = new Map();
  for (const tube of tubes)
    for (const mount of mounts.get(tube) ?? []) {
      if (!mount) continue;
      const host = ownerOf(mount.shape);
      link(tube.owner, host);
      if (!seats.has(host)) seats.set(host, []);
      seats.get(host).push(tube.owner);
      if (mount.shape.cut) for (const cut of [mount.shape.cut].flat()) link(tube.owner, ownerOf(cut));
    }
  for (const owners of seats.values()) for (const a of owners) for (const b of owners) link(a, b);
  return links;
}

const entriesOf = (items, cam) => items.flatMap((item) => (item.shapes ?? []).map((shape) => ({ shape, box: screenBox(shape, cam, [0, 0, 0]).map((v, i) => v + (i < 2 ? -1 : 1)), owner: item.route ?? shape.owner ?? item.name, bundle: shape.bundle ?? null, item: item.name })));

function frontEntry(entries, cam, x, y) {
  const c = add3(mul3(cam.R, (x - cam.origin[0]) / cam.k), mul3(cam.D, (y - cam.origin[1]) / cam.k));
  let best = -BIG;
  let who = null;
  for (const entry of entries) {
    const { shape, box } = entry;
    if (x < box[0] || x > box[2] || y < box[1] || y > box[3]) continue;
    const t = rayDepth(shape, c, cam.V, 0.02, null);
    if (t !== null && t > best) (best = t), (who = entry);
  }
  return who;
}

function frontAt(entries, cam, x, y) {
  const entry = frontEntry(entries, cam, x, y);
  return entry ? entry.owner : null;
}

const belongs = (entry, name, bundle) => Boolean(entry) && (entry.owner === name || entry.item === name || (bundle && entry.bundle === bundle));

export function slenderOf(solids, { ratio = 4, least = 0.3 } = {}) {
  const out = [];
  const axisLine = (centre, axis, half, r, shape) => {
    out.push({ name: shape.owner ?? shape.name, solid: shape.name, owners: new Set([shape.name, shape.owner].filter(Boolean)), points: [add3(centre, mul3(axis, -half)), add3(centre, mul3(axis, half))], r, move: shape.move, ride: shape.ride, piece: shape.piece, rod: true });
  };
  for (const shape of solids) {
    if (!shape || shape.kind === "tube" || shape.kind === "ball" || shape.fitting) continue;
    if (shape.kind === "body") {
      const s = shape.poly.map(([value]) => value);
      const s0 = Math.min(...s);
      const s1 = Math.max(...s);
      const r = Math.max(...shape.poly.map(([, value]) => value));
      if (r >= least && s1 - s0 >= ratio * r) axisLine(add3(shape.F.o, mul3(shape.F.a, (s0 + s1) / 2)), shape.F.a, (s1 - s0) / 2, r, shape);
      continue;
    }
    let axes;
    let half;
    let centre;
    if (shape.kind === "prism") {
      const xs = shape.poly.map(([x]) => x);
      const ys = shape.poly.map(([, y]) => y);
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
      axes = [shape.F.u, shape.F.v, shape.F.a];
      half = [(x1 - x0) / 2, (y1 - y0) / 2, (shape.s1 - shape.s0) / 2];
      centre = add3(shape.F.o, add3(add3(mul3(shape.F.u, (x0 + x1) / 2), mul3(shape.F.v, (y0 + y1) / 2)), mul3(shape.F.a, (shape.s0 + shape.s1) / 2)));
    } else if (shape.kind === "box") {
      axes = shape.axes;
      half = shape.half;
      centre = shape.o;
    } else continue;
    const order = [0, 1, 2].sort((a, b) => half[b] - half[a]);
    const r = half[order[1]];
    if (r >= least && half[order[0]] >= ratio * r) axisLine(centre, axes[order[0]], half[order[0]], r, shape);
  }
  return out;
}

export function crowding(routes, P, { gap = 1, bend = 30, links = new Map(), items = null, allow = [], state, clear = 2, rods = [] } = {}) {
  const cam = cameraBasis(P);
  const shift = (thing) => (state === undefined ? [0, 0, 0] : offsetOf(thing, state));
  const entries = items ? entriesOf(items.map((item) => ({ ...item, shapes: (item.shapes ?? []).map((shape) => moved(shape, shift(item))) })), cam) : null;
  const bundles = new Map(routes.map((route) => [route.name, route.bundle]));
  const shown = (owner, x, y, rod) => {
    if (!entries) return true;
    const entry = frontEntry(entries, cam, x, y);
    if (rod) return Boolean(entry) && (entry.shape.name === rod.solid || (!entry.shape.name && entry.item === rod.solid));
    return belongs(entry, owner, bundles.get(owner));
  };
  const flat = (p) => [cam.origin[0] + dot3(p, cam.R) * cam.k, cam.origin[1] + dot3(p, cam.D) * cam.k];
  const lines = [...routes, ...rods].map((route) => {
    const step = Math.max(0.25, route.r * 0.5);
    const offset = shift(route);
    const dense = resample(route.points.map((p) => add3(p, offset)), step);
    return { ...route, step, dense, screen: dense.map(flat), width: route.r * cam.k };
  });
  const marksOf = (line) => {
    const out = [];
    const reach = Math.max(2, Math.round((2 * line.r) / line.step));
    const last = line.dense.length - 1;
    for (let i = 0; i <= last; i++) {
      if (i <= reach || i >= last - reach) {
        out.push({ i, where: i <= reach ? "start" : "end" });
        continue;
      }
      const A = sub3(line.dense[i], line.dense[i - reach]);
      const B = sub3(line.dense[i + reach], line.dense[i]);
      const turn = (Math.acos(Math.max(-1, Math.min(1, dot3(A, B) / (len3(A) * len3(B) || 1)))) * 180) / Math.PI;
      if (turn > bend) out.push({ i, where: "bend" });
    }
    return out;
  };
  const found = new Map();
  for (const a of lines) {
    if (a.rod) continue;
    const marks = marksOf(a);
    for (const b of lines) {
      if (a === b || (a.bundle && a.bundle === b.bundle) || links.get(a.name)?.has(b.name)) continue;
      if (b.rod && [...b.owners].some((name) => name === a.name || links.get(a.name)?.has(name))) continue;
      if (allow.some(([x, y]) => [b.name, b.solid].filter(Boolean).some((name) => (matches([x], a.name) && matches([y], name)) || (matches([y], a.name) && matches([x], name))))) continue;
      for (const { i, where } of marks) {
        const [x, y] = a.screen[i];
        let best = { d: BIG, j: -1 };
        for (let j = 0; j < b.screen.length; j++) {
          const d = Math.hypot(b.screen[j][0] - x, b.screen[j][1] - y);
          if (d < best.d) best = { d, j };
        }
        const touch = best.d - a.width - b.width;
        if (touch >= gap) continue;
        if (where !== "bend" && touch <= -2 * Math.min(a.width, b.width)) {
          const tip = a.screen[where === "start" ? 0 : a.screen.length - 1];
          let clearTip = BIG;
          for (const q of b.screen) clearTip = Math.min(clearTip, Math.hypot(q[0] - tip[0], q[1] - tip[1]));
          const reachB = Math.max(2, Math.round((2 * b.r) / b.step));
          const j0 = Math.max(0, best.j - reachB);
          const j1 = Math.min(b.dense.length - 1, best.j + reachB);
          const straight = j1 - j0 >= 2 && dot3(unit3(sub3(b.dense[best.j], b.dense[j0])), unit3(sub3(b.dense[j1], b.dense[best.j]))) > Math.cos((bend * Math.PI) / 180);
          if (straight && clearTip > b.width + a.width + 1) continue;
        }
        const [bx, by] = b.screen[best.j];
        const u = [(bx - x) / (best.d || 1), (by - y) / (best.d || 1)];
        const inA = Math.max(0.2, a.width - 0.35);
        const inB = Math.max(0.2, b.width - 0.35);
        if (!shown(a.name, x + u[0] * inA, y + u[1] * inA)) continue;
        if (b.rod) {
          if (!shown(b.name, bx - u[0] * inB, by - u[1] * inB, b) && !shown(b.name, bx + u[0] * inB, by + u[1] * inB, b)) continue;
        } else if (!shown(b.name, bx - u[0] * inB, by - u[1] * inB)) continue;
        const depth = Math.abs(dot3(sub3(a.dense[i], b.dense[best.j]), cam.V));
        const need = 2 * (a.r + b.r);
        const id = [a.name, b.name].sort().join("|");
        const prior = found.get(id);
        if (!prior || Math.abs(touch) < Math.abs(prior.gap)) found.set(id, { a: a.name, b: b.rod ? `${b.solid} (rod)` : b.name, where, gap: touch, depth, need, screen: [x, y], at: a.dense[i], state: state === undefined ? "" : labelOf(state) });
      }
    }
  }
  if (entries)
    for (const fitting of entries.filter((entry) => entry.shape.fitting && entry.shape.kind === "body")) {
      const { shape } = fitting;
      const reach = Math.max(...shape.poly.map(([, r]) => r));
      const [x, y] = flat(shape.F.o);
      for (const b of lines) {
        if (b.rod) continue;
        if ((fitting.bundle && b.bundle === fitting.bundle) || b.name === fitting.owner || links.get(fitting.owner)?.has(b.name)) continue;
        if (allow.some(([p, q]) => (matches([p], fitting.item) && matches([q], b.name)) || (matches([q], fitting.item) && matches([p], b.name)))) continue;
        let best = { d: BIG, j: -1 };
        for (let j = 0; j < b.screen.length; j++) {
          const d = Math.hypot(b.screen[j][0] - x, b.screen[j][1] - y);
          if (d < best.d) best = { d, j };
        }
        const touch = best.d - reach * cam.k - b.width;
        if (touch >= clear) continue;
        const front = frontEntry(entries, cam, x, y);
        if (!(front === fitting || belongs(front, fitting.owner, fitting.bundle))) continue;
        const id = [fitting.item, b.name].sort().join("|");
        const prior = found.get(id);
        if (!prior || Math.abs(touch) < Math.abs(prior.gap)) found.set(id, { a: fitting.item, b: b.name, where: "fitting", gap: touch, depth: Math.abs(dot3(sub3(shape.F.o, b.dense[best.j]), cam.V)), need: 0, screen: [x, y], at: shape.F.o, state: state === undefined ? "" : labelOf(state) });
      }
    }
  return [...found.values()].sort((x, y) => Math.abs(x.gap) - Math.abs(y.gap));
}

const multiply = (m, n) => [m[0] * n[0] + m[2] * n[1], m[1] * n[0] + m[3] * n[1], m[0] * n[2] + m[2] * n[3], m[1] * n[2] + m[3] * n[3], m[0] * n[4] + m[2] * n[5] + m[4], m[1] * n[4] + m[3] * n[5] + m[5]];

function matrixOf(text) {
  let m = [1, 0, 0, 1, 0, 0];
  if (!text) return m;
  for (const [, kind, args] of text.matchAll(/(matrix|translate|scale|rotate)\s*\(([^)]*)\)/g)) {
    const v = (args.match(/-?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?/gi) ?? []).map(Number);
    let n = [1, 0, 0, 1, 0, 0];
    if (kind === "matrix" && v.length === 6) n = v;
    else if (kind === "translate") n = [1, 0, 0, 1, v[0] ?? 0, v[1] ?? 0];
    else if (kind === "scale") n = [v[0] ?? 1, 0, 0, v[1] ?? v[0] ?? 1, 0, 0];
    else if (kind === "rotate") {
      const a = ((v[0] ?? 0) * Math.PI) / 180;
      const [cx, cy] = [v[1] ?? 0, v[2] ?? 0];
      n = multiply(multiply([1, 0, 0, 1, cx, cy], [Math.cos(a), Math.sin(a), -Math.sin(a), Math.cos(a), 0, 0]), [1, 0, 0, 1, -cx, -cy]);
    }
    m = multiply(m, n);
  }
  return m;
}

const NUMBER = /-?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?/gi;

function pointsOfShape(tag, attrs) {
  const get = (name) => {
    const hit = attrs.match(new RegExp(`(?:^|\\s)${name}="([^"]*)"`));
    return hit ? hit[1] : null;
  };
  const num = (name) => Number(get(name) ?? 0);
  if (tag === "path") {
    const d = get("d") ?? "";
    const out = [];
    let x = 0;
    let y = 0;
    for (const [, command, args] of d.matchAll(/([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)/g)) {
      const v = (args.match(NUMBER) ?? []).map(Number);
      const rel = command === command.toLowerCase();
      const C = command.toUpperCase();
      const pair = (i) => [v[i] + (rel ? x : 0), v[i + 1] + (rel ? y : 0)];
      const take = (p) => {
        out.push(p);
        return p;
      };
      if (C === "H") for (const value of v) [x] = take([value + (rel ? x : 0), y]);
      else if (C === "V") for (const value of v) [, y] = take([x, value + (rel ? y : 0)]);
      else if (C === "A") for (let i = 0; i + 6 < v.length; i += 7) [x, y] = take([v[i + 5] + (rel ? x : 0), v[i + 6] + (rel ? y : 0)]);
      else if (C !== "Z") {
        const width = C === "C" ? 6 : C === "S" || C === "Q" ? 4 : 2;
        for (let i = 0; i + 1 < v.length; i += width) {
          for (let j = 0; j < width; j += 2) out.push(pair(i + j));
          [x, y] = pair(i + width - 2);
        }
      }
    }
    return out;
  }
  if (tag === "circle" || tag === "ellipse") {
    const r = Number(get("r") ?? get("rx") ?? 0);
    const ry = Number(get("r") ?? get("ry") ?? 0);
    return [[num("cx") - r, num("cy") - ry], [num("cx") + r, num("cy") + ry]];
  }
  if (tag === "line") return [[num("x1"), num("y1")], [num("x2"), num("y2")]];
  if (tag === "rect") return [[num("x"), num("y")], [num("x") + num("width"), num("y") + num("height")]];
  if (tag === "polyline" || tag === "polygon") {
    const v = (get("points") ?? "").match(NUMBER)?.map(Number) ?? [];
    const out = [];
    for (let i = 0; i + 1 < v.length; i += 2) out.push([v[i], v[i + 1]]);
    return out;
  }
  return [];
}

const boxOfPoints = (points, m) => {
  let box = null;
  for (const [x, y] of points) {
    const px = m[0] * x + m[2] * y + m[4];
    const py = m[1] * x + m[3] * y + m[5];
    box = box ? [Math.min(box[0], px), Math.min(box[1], py), Math.max(box[2], px), Math.max(box[3], py)] : [px, py, px, py];
  }
  return box;
};
const unionBox = (a, b) => (!a ? b : !b ? a : [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])]);
const meetBox = (a, b) => (!a || !b ? a ?? null : [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.min(a[3], b[3])]);
const areaOf = (box) => (box ? Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]) : 0);

export function markupBox(svg) {
  const tags = [...svg.matchAll(/<(\/?)([a-zA-Z]+)([^>]*?)(\/?)>/g)];
  const clips = new Map();
  let clipId = null;
  for (const [, close, tag, attrs, self] of tags) {
    if (tag === "clipPath" && !close) {
      clipId = self ? null : (attrs.match(/id="([^"]*)"/) ?? [])[1] ?? null;
      continue;
    }
    if (clipId) {
      if (tag === "clipPath" && close) {
        clipId = null;
        continue;
      }
      clips.set(clipId, unionBox(clips.get(clipId), boxOfPoints(pointsOfShape(tag, attrs), matrixOf((attrs.match(/transform="([^"]*)"/) ?? [])[1]))));
    }
  }
  const stack = [{ m: [1, 0, 0, 1, 0, 0], clip: null, skip: false }];
  let box = null;
  for (const [, close, tag, attrs, self] of tags) {
    const top = stack[stack.length - 1];
    if (close) {
      if (stack.length > 1) stack.pop();
      continue;
    }
    const m = multiply(top.m, matrixOf((attrs.match(/transform="([^"]*)"/) ?? [])[1]));
    const ref = (attrs.match(/clip-path="url\(#([^)]*)\)"/) ?? [])[1];
    const own = ref && clips.get(ref) ? boxOfPoints([[clips.get(ref)[0], clips.get(ref)[1]], [clips.get(ref)[2], clips.get(ref)[3]]], m) : null;
    const frame = { m, clip: own ? meetBox(top.clip ?? own, own) : top.clip, skip: top.skip || ["defs", "clipPath", "mask", "linearGradient", "radialGradient", "pattern", "text", "title"].includes(tag) || /class="[^"]*iso-halo/.test(attrs) };
    if (!frame.skip) {
      const shape = boxOfPoints(pointsOfShape(tag, attrs), m);
      if (shape) box = unionBox(box, frame.clip ? meetBox(shape, frame.clip) : shape);
    }
    if (!self) stack.push(frame);
  }
  return box;
}

export function footprintOf(shape, cam) {
  const flat = (p) => [cam.origin[0] + dot3(p, cam.R) * cam.k, cam.origin[1] + dot3(p, cam.D) * cam.k];
  if (shape.kind === "body") {
    const { F } = shape;
    const points = [];
    for (const [s, r] of shape.poly) {
      if (r <= 0) {
        points.push(flat(add3(F.o, mul3(F.a, s))));
        continue;
      }
      for (let i = 0; i < 16; i++) {
        const a = (i / 16) * Math.PI * 2;
        points.push(flat(add3(add3(F.o, mul3(F.a, s)), add3(mul3(F.u, r * Math.cos(a)), mul3(F.v, r * Math.sin(a))))));
      }
    }
    return boxOfPoints(points, [1, 0, 0, 1, 0, 0]);
  }
  if (shape.kind === "prism") {
    const { F } = shape;
    const points = [];
    for (const s of [shape.s0, shape.s1]) for (const [x, y] of shape.poly) points.push(flat(add3(add3(F.o, mul3(F.a, s)), add3(mul3(F.u, x), mul3(F.v, y)))));
    return boxOfPoints(points, [1, 0, 0, 1, 0, 0]);
  }
  const [x0, y0, x1, y1] = screenBox(shape, cam, [0, 0, 0]);
  return [x0, y0, x1, y1];
}

export function coverage(items, P, { least = 0.6, pad = 1, allow = [] } = {}) {
  const cam = cameraBasis(P);
  const out = [];
  for (const item of items) {
    if (typeof item.svg !== "string" || !item.shapes || !item.shapes.length || matches(allow, item.name)) continue;
    const drawn = markupBox(item.svg);
    if (!drawn) continue;
    const grow = (box) => [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad];
    const shapes = item.shapes.map((shape) => footprintOf(shape, cam)).reduce(unionBox, null);
    const share = areaOf(meetBox(grow(drawn), grow(shapes))) / (areaOf(grow(drawn)) || 1);
    if (share < least) out.push({ name: item.name, share, drawn, shapes });
  }
  return out.sort((a, b) => a.share - b.share);
}

export function selfOverlaps(items, P, { entries: given = null, fold = 0.25, least = 0.5 } = {}) {
  const cam = cameraBasis(P);
  const entries = given ?? entriesOf(items, cam);
  const flat = (p) => [cam.origin[0] + dot3(p, cam.R) * cam.k, cam.origin[1] + dot3(p, cam.D) * cam.k];
  const out = [];
  for (const item of items) {
    if (!item.route || !item.shapes || !item.shapes.length || item.shapes[0].kind !== "tube") continue;
    const shape = item.shapes[0];
    const r = shape.radii.reduce((sum, value) => sum + value, 0) / shape.radii.length;
    const dense = resample(shape.points, Math.max(0.1, r * 0.2));
    const screen = dense.map(flat);
    const n = screen.length;
    if (n < 4) continue;
    const width = r * cam.k;
    const span = 3;
    const normals = screen.map((_, i) => {
      const a = screen[Math.max(0, i - span)];
      const b = screen[Math.min(n - 1, i + span)];
      const length = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      return [-(b[1] - a[1]) / length, (b[0] - a[0]) / length];
    });
    let worst = null;
    for (const side of [1, -1]) {
      const edge = screen.map((p, i) => [p[0] + side * normals[i][0] * width, p[1] + side * normals[i][1] * width]);
      for (let i = span; i < n - span - 1; i++) {
        const along = [screen[i + 1][0] - screen[i][0], screen[i + 1][1] - screen[i][1]];
        const step = [edge[i + 1][0] - edge[i][0], edge[i + 1][1] - edge[i][1]];
        const length = Math.hypot(along[0], along[1]);
        if (length < 1e-6) continue;
        const forward = (step[0] * along[0] + step[1] * along[1]) / length;
        if (forward > -fold * length) continue;
        const x = (screen[i][0] + screen[i + 1][0]) / 2;
        const y = (screen[i][1] + screen[i + 1][1]) / 2;
        const front = frontEntry(entries, cam, x, y);
        if (front && front.owner !== item.route && front.item !== item.name) continue;
        const radius = Math.abs(forward / length - 1) > 1e-6 ? width / Math.abs(1 - forward / length) : Infinity;
        if (width - radius < least) continue;
        if (!worst || radius < worst.radius) worst = { name: item.name, route: item.route, radius, width, at: dense[i], screen: [x, y] };
      }
    }
    if (worst) out.push(worst);
  }
  return out;
}

export function solidClearances(solids, { state, sunk = 0.5, spacing = 0.6, cell = 4 } = {}) {
  const list = solids.filter((shape) => shape.kind !== "tube").map((shape) => (shape.slide && state !== undefined && shape.move ? { ...moved(shape, offsetOf(shape, state)), move: undefined } : shape));
  if (!solidClearances.cache) solidClearances.cache = new WeakMap();
  const cache = solidClearances.cache;
  const gridOf = (shape) => {
    let grid = cache.get(shape);
    if (grid) return grid;
    grid = new Map();
    for (const p of surfaceOf(shape, spacing)) {
      const key = p.map((v) => Math.floor(v / cell)).join(",");
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(p);
    }
    cache.set(shape, grid);
    return grid;
  };
  const pointsIn = (shape, box) => {
    const grid = gridOf(shape);
    const lo = box.min.map((v) => Math.floor(v / cell));
    const hi = box.max.map((v) => Math.floor(v / cell));
    const out = [];
    if ((hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1) > grid.size) {
      for (const points of grid.values()) for (const p of points) if (p.every((v, i) => v >= box.min[i] && v <= box.max[i])) out.push(p);
      return out;
    }
    for (let x = lo[0]; x <= hi[0]; x++)
      for (let y = lo[1]; y <= hi[1]; y++)
        for (let z = lo[2]; z <= hi[2]; z++) {
          const points = grid.get(`${x},${y},${z}`);
          if (points) for (const p of points) if (p.every((v, i) => v >= box.min[i] && v <= box.max[i])) out.push(p);
        }
    return out;
  };
  const offsets = list.map((shape) => offsetAt(shape, state));
  const bounds = list.map((shape) => boundsOf(shape));
  const found = [];
  for (let a = 0; a < list.length; a++)
    for (let b = a + 1; b < list.length; b++) {
      const rel = sub3(offsets[a], offsets[b]);
      if (Math.abs(rel[0]) + Math.abs(rel[1]) + Math.abs(rel[2]) < 1e-6) continue;
      const box = { min: [0, 1, 2].map((i) => Math.max(bounds[a].min[i] + rel[i], bounds[b].min[i])), max: [0, 1, 2].map((i) => Math.min(bounds[a].max[i] + rel[i], bounds[b].max[i])) };
      if (box.min.some((v, i) => v > box.max[i] - sunk)) continue;
      if (seatedOn(list[a], list[b]) || seatedOn(list[b], list[a])) continue;
      let worst = -sunk;
      let at = null;
      const local = { min: sub3(box.min, rel), max: sub3(box.max, rel) };
      for (const p of pointsIn(list[a], local)) {
        const d = sdf(list[b], add3(p, rel));
        if (d < worst) (worst = d), (at = add3(p, offsets[a]));
      }
      for (const p of pointsIn(list[b], box)) {
        const d = sdf(list[a], sub3(p, rel));
        if (d < worst) (worst = d), (at = add3(p, offsets[b]));
      }
      if (at) found.push({ tube: list[a].name, other: list[b].name, kind: "through", solid: true, clearance: worst, need: -sunk, at, from: 0, to: 0, length: 0, state: state === undefined ? "" : labelOf(state) });
    }
  return found.sort((x, y) => x.clearance - y.clearance);
}

export function wobble(solids, { samples = 20 } = {}) {
  const out = [];
  for (const shape of solids) {
    const profile = shape.curve;
    if (!profile) continue;
    const segments = curveRuns(profile);
    let run = [];
    const flush = () => {
      if (run.length) {
        const values = [];
        for (const index of run) {
          const [s0] = profile[index];
          const [s1] = profile[index + 1];
          for (let j = values.length ? 1 : 0; j <= samples; j++) {
            const s = s0 + ((s1 - s0) * j) / samples;
            values.push([s, segments[index].slope(s)]);
          }
        }
        const scale = Math.max(1, ...values.map(([, g]) => Math.abs(g))) * 1e-6;
        const turns = [];
        let last = 0;
        for (let i = 1; i < values.length; i++) {
          const d = values[i][1] - values[i - 1][1];
          if (Math.abs(d) <= scale) continue;
          const sign = Math.sign(d);
          if (last && sign !== last) turns.push(Math.round(values[i - 1][0] * 100) / 100);
          last = sign;
        }
        if (turns.length) out.push({ name: shape.name, at: turns, from: profile[run[0]][0], to: profile[run[run.length - 1] + 1][0] });
      }
      run = [];
    };
    segments.forEach((segment, index) => (segment ? run.push(index) : flush()));
    flush();
  }
  return out;
}

export function seats(solids) {
  const out = [];
  for (const shape of solids) {
    if (!shape.seated || !shape.cut) continue;
    for (const host of [shape.cut].flat()) {
      const mine = mul3(shape.move ?? [0, 0, 0], 1);
      const theirs = mul3(host.move ?? [0, 0, 0], 1);
      if ((shape.ride ?? shape.piece) === (host.ride ?? host.piece) && sameOffset(mine, theirs)) continue;
      out.push({ name: shape.name, host: host.name, piece: shape.ride ?? shape.piece, hostPiece: host.ride ?? host.piece });
    }
  }
  return out;
}

export function report({ clearance = [], depth = { problems: [], missing: [] }, terminals: ends, folds: bends, supports: held, loose, apart, flight, crowding: crowded, self, wobble: wobbly, seats: seated, cover: covered } = {}, { limit = 4000 } = {}) {
  const lines = [];
  const f = (v) => (Math.round(v * 100) / 100).toFixed(2);
  const p3 = (p) => `[${p.map((v) => v.toFixed(1)).join(", ")}]`;
  const clash = (c) => (c.solid ? `  solid   ${c.state ? c.state + "  " : ""}${c.tube}  ×  ${c.other}   sunk ${f(-c.clearance)} deep   at ${p3(c.at)}` : `  ${c.kind.padEnd(7)} ${c.state ? c.state + "  " : ""}${c.tube}  ×  ${c.other}   clearance ${f(c.clearance)} < ${f(c.need)}   along ${f(c.from)}–${f(c.to)} of ${f(c.length)}   at ${p3(c.at)}`);
  lines.push(`clearance: ${clearance.length} problem${clearance.length === 1 ? "" : "s"}`);
  for (const c of clearance.slice(0, limit)) lines.push(clash(c));
  if (apart) {
    lines.push(`clearance apart: ${apart.length} problem${apart.length === 1 ? "" : "s"}`);
    for (const c of apart.slice(0, limit)) lines.push(clash(c));
  }
  if (ends) {
    lines.push(`terminals: ${ends.length} end${ends.length === 1 ? "" : "s"} not on a mount`);
    for (const e of ends.slice(0, limit)) lines.push(`  ${e.tube} ${e.end}  ${Number.isFinite(e.d) ? f(e.d) : "∞"} from ${e.nearest} (rim ${Number.isFinite(e.high) ? `${f(e.low)}…${f(e.high)}` : "-"})   at ${p3(e.at)}`);
  }
  if (loose) {
    const hanging = loose.filter((e) => !e.free);
    lines.push(`loose ends: ${hanging.length}`);
    for (const e of hanging.slice(0, limit)) lines.push(`  ${e.state}  ${e.tube} ${e.end} touches nothing that moves with it   at ${p3(e.at)}`);
  }
  if (bends) {
    lines.push(`screen folds: ${bends.length}`);
    for (const b of bends.slice(0, limit))
      lines.push(b.kind === "pinch" ? `  ${b.name}  turns ${b.flatTurn.toFixed(0)}° on screen round a ${f(b.radius)} px bend, under ${f(b.width * 2)} px for a ${f(b.width)} px tube   at ${p3(b.at)}` : `  ${b.name}  turns ${b.flatTurn.toFixed(0)}° on screen for ${b.realTurn.toFixed(0)}° in 3D   at ${p3(b.at)}`);
  }
  if (held) {
    const hanging = held.hanging ?? [];
    lines.push(`support: ${held.loose.length} unsupported solid${held.loose.length === 1 ? "" : "s"} · ${hanging.length} hanging tube${hanging.length === 1 ? "" : "s"} · ${held.sunk.length} sunk pair${held.sunk.length === 1 ? "" : "s"}`);
    for (const name of held.loose.slice(0, limit)) lines.push(`  loose   ${name}`);
    for (const name of hanging.slice(0, limit)) lines.push(`  hanging ${name}`);
    for (const d of held.sunk.slice(0, limit)) lines.push(`  sunk    ${d.a} × ${d.b}   ${f(d.depth)} deep`);
  }
  lines.push(`depth order: ${depth.problems.length} problem${depth.problems.length === 1 ? "" : "s"}`);
  for (const d of depth.problems.slice(0, limit))
    lines.push(`  ${d.cycle ? "cycle " : "order "} ${d.label}  ${d.front} (k ${f(d.frontKey)}) is in front of ${d.back} (k ${f(d.backKey)}) but drawn first   ${f(d.area)} px²${d.against ? ` (${f(d.against)} px² the other way)` : ""}, up to ${f(d.depth)} nearer, at ${d.at.map((v) => v.toFixed(1)).join(",")}`);
  if (flight) {
    lines.push(`through in mid-flight: ${flight.length} pair${flight.length === 1 ? "" : "s"}`);
    for (const c of flight.slice(0, limit)) lines.push(clash(c));
  }
  if (self) {
    lines.push(`self-overlap: ${self.length} chunk${self.length === 1 ? "" : "s"} whose outline crosses itself on screen`);
    for (const c of self.slice(0, limit)) lines.push(`  ${c.name}  its outline folds back on screen: the line turns on a ${f(c.radius)} px radius, under its ${f(c.width)} px half-width   at ${p3(c.at)}, screen ${c.screen.map((v) => v.toFixed(1)).join(",")}`);
  }
  if (wobbly) {
    lines.push(`wobble: ${wobbly.length} smoothed profile${wobbly.length === 1 ? "" : "s"} whose slope turns between its ends`);
    for (const w of wobbly.slice(0, limit)) lines.push(`  ${w.name}  dr/ds turns at s = ${w.at.join(", ")} on the run ${f(w.from)}–${f(w.to)}; give it an analytic profile and smooth.slope`);
  }
  if (seated) {
    lines.push(`saddles off their host apart: ${seated.length}`);
    for (const d of seated.slice(0, limit)) lines.push(`  ${d.name} (${d.piece}) is cut to ${d.host} (${d.hostPiece}); apart, its cut base is on show`);
  }
  if (covered) {
    lines.push(`stand-in shapes: ${covered.length} item${covered.length === 1 ? "" : "s"} whose recorded shape covers too little of what it draws`);
    for (const c of covered.slice(0, limit)) lines.push(`  ${c.name}  its shapes cover ${Math.round(c.share * 100)}% of its drawing (drawn ${c.drawn.map((v) => v.toFixed(0)).join(",")}, recorded ${c.shapes.map((v) => v.toFixed(0)).join(",")}); record the true shape of what you draw`);
  }
  if (crowded) {
    lines.push(`crowding: ${crowded.length} pair${crowded.length === 1 ? "" : "s"} where a bend or end touches another line on screen`);
    for (const c of crowded.slice(0, limit)) lines.push(`  ${c.state ? c.state + "  " : ""}${c.a} ${c.where}  ×  ${c.b}   outlines ${c.gap < 0 ? `overlap ${f(-c.gap)}` : `${f(c.gap)} apart`} px, ${f(c.depth)} apart in depth   at screen ${c.screen.map((v) => v.toFixed(1)).join(",")}`);
  }
  if (depth.passing && depth.passing.length) {
    const pairs = (list) => new Set(list.map((d) => [d.front, d.back].sort().join("|"))).size;
    lines.push(`in passing (uniform explode states only, not failing): ${pairs(depth.passing.filter((d) => d.cycle))} interlocked pairs, ${pairs(depth.passing.filter((d) => !d.cycle))} pairs no linear key can order`);
    for (const d of depth.passing.slice(0, limit)) lines.push(`  ${d.cycle ? "cycle " : "unorderable"} ${d.label}  ${d.front} / ${d.back}   ${f(d.area)} px²`);
  }
  if (depth.missing.length) lines.push(`items with no shape: ${depth.missing.length}  ${depth.missing.slice(0, 20).join(", ")}`);
  if (loose && loose.some((e) => e.free)) {
    const free = loose.filter((e) => e.free);
    lines.push(`free ends apart (marked free, not failing): ${free.length}`);
    for (const e of free.slice(0, limit)) lines.push(`  ${e.tube} ${e.end}   at ${p3(e.at)}`);
  }
  return lines.join("\n");
}

export const failures = ({ clearance = [], depth = { problems: [], missing: [] }, terminals: ends = [], folds: bends = [], supports: held = { loose: [], sunk: [] }, loose = [], apart = [], flight = [], crowding: crowded = [], self = [], wobble: wobbly = [], seats: seated = [], cover: covered = [] }) =>
  covered.length + clearance.length + apart.length + flight.length + crowded.length + ends.length + loose.filter((e) => !e.free).length + bends.length + held.loose.length + (held.hanging ?? []).length + held.sunk.length + depth.problems.length + depth.missing.length + self.length + wobbly.length + seated.length;

export const sameRun = (a, b) => Boolean(a.route) && a.route === b.route && Math.abs(a.chunk - b.chunk) <= 1;

export function recorder(P, { order = [], spring = {}, ground = grounded } = {}) {
  const V = cameraBasis(P).V;
  const counts = new Map();
  const R = { P, V, order, spring, ground, items: [], solids: [], tubes: [], routes: [], settled: null };
  R.nameFor = (base) => {
    const count = (counts.get(base) ?? 0) + 1;
    counts.set(base, count);
    return count === 1 ? base : `${base}#${count}`;
  };
  R.solid = (shape) => {
    R.solids.push(shape);
    return shape;
  };
  R.put = ({ at = null, bias = 0, key, name, piece = "", move = [0, 0, 0], ride, shapes = [], ...rest } = {}) => {
    const item = { ...rest, piece, ride: ride ?? piece, key: key ?? (at ? dot3(at, V) : 0) + bias, move, shapes, name: name ?? R.nameFor(`${piece || "item"}.item`) };
    for (const shape of shapes) (shape.move = item.move), (shape.ride = item.ride), (shape.piece = piece);
    R.items.push(item);
    return item;
  };
  R.route = (name, points, radius, { owner = name, touch, bundle, limp = false, gaps = [], closed = false, ends = [true, true], rims, free = [false, false], piece = "", move = [0, 0, 0], ride = piece } = {}) => {
    const total = pathLength(points);
    const radiusOf = typeof radius === "function" ? radius : () => radius;
    const runs = runsOf(total, gaps);
    const recorded = runs.map(([from, to], index) => ({
      ...tube(cutAlong(points, from, to), (t) => radiusOf((from + (to - from) * t) / total), { name, owner, touch, bundle, limp, joined: true }),
      tips: closed ? [null, null] : [index === 0 && ends[0] ? points[0] : null, index === runs.length - 1 && ends[1] ? points[points.length - 1] : null],
      tipAxes: [tangentAlong(points, 0), tangentAlong(points, total)],
      tipRims: rims ?? [radiusOf(0), radiusOf(1)],
      free: [index === 0 && free[0], index === runs.length - 1 && free[1]],
      offset: from,
      total,
      move,
      ride,
      piece,
    }));
    R.tubes.push(...recorded);
    R.routes.push({ name, points, r: radiusOf(0.5), bundle, move, ride, piece });
    return recorded;
  };
  return R;
}

export function explodeStates(order = [], spring = {}, { uniform = 10, stride = 4 } = {}) {
  if (!order.length) return { depth: [0], frames: [], apart: false };
  const reversals = reversalsOf(order, spring.stagger ?? 0.035);
  const prefix = prefixStates(order);
  return {
    depth: [...uniformStates(uniform), ...springStates(order, { ...spring, stride, reversals }), ...prefix],
    frames: [...springStates(order, { ...spring, reversals }), ...prefix],
    apart: true,
  };
}

export function settle(R, P = R.P, { states = explodeStates(R.order, R.spring).depth, skip = sameRun, ...options } = {}) {
  const items = R.items.slice();
  const overlaps = overlapsOf(items, P, { states, skip, ...options });
  const solved = orderKeys(items, overlaps);
  items.forEach((item, index) => (item.key = solved.keys[index]));
  R.settled = { items, overlaps, solved, states };
  return R.settled;
}

const worstOf = (lists) => [...lists.flat().reduce((all, c) => (all.has(`${c.tube}|${c.other}`) && all.get(`${c.tube}|${c.other}`).clearance <= c.clearance ? all : all.set(`${c.tube}|${c.other}`, c)), new Map()).values()].sort((x, y) => x.clearance - y.clearance);

export function audit(R, P = R.P, { allow = [], crowd = [], cover = [], ground = R.ground, frames = explodeStates(R.order, R.spring).frames, apart = R.order.length > 0 } = {}) {
  const { items, overlaps, solved } = R.settled ?? settle(R, P);
  const mounts = mountsOf(R.tubes, R.solids);
  const links = linksOf(R.tubes, mounts);
  return {
    clearance: clearances(R.tubes, R.solids, { mounts }),
    apart: apart ? [...clearances(R.tubes, R.solids, { state: 1, mounts, only: "through" }), ...solidClearances(R.solids, { state: 1 })] : [],
    flight: worstOf(frames.map((state) => [...clearances(R.tubes, R.solids, { state, mounts, only: "through" }), ...solidClearances(R.solids, { state })])),
    crowding: (apart ? [0, 1] : [0]).flatMap((state) => crowding(R.routes, P, { links, items, allow: crowd, state, rods: slenderOf(R.solids) })),
    cover: coverage(items, P, { allow: cover }),
    terminals: terminals(R.tubes, R.solids, { mounts }),
    loose: looseEnds(R.tubes, R.solids, { state: 1 }),
    folds: folds(R.routes, P, { items }),
    self: selfOverlaps(items, P),
    wobble: wobble(R.solids),
    seats: seats(R.solids),
    supports: supports(R.solids, R.tubes, { ground }),
    depth: depthOrder(items, P, { overlaps, solved, allow }),
  };
}

export function auditOrExit(R, P = R.P, { flag = "--audit", argv = process.argv, ...options } = {}) {
  if (!argv.includes(flag)) return null;
  const started = Date.now();
  const result = audit(R, P, options);
  const { overlaps, solved, states } = R.settled;
  console.log(report(result));
  console.log(`audit · ${R.tubes.length} tubes · ${R.solids.length} solids · ${R.items.length} items · ${states.length} depth states · ${overlaps.pairs.length} overlaps · ${solved.rules} order rules · ${((Date.now() - started) / 1000).toFixed(1)} s`);
  const count = failures(result);
  console.log(count ? `audit failed: ${count} problem${count === 1 ? "" : "s"}` : "audit passed");
  if (count) process.exitCode = 1;
  return result;
}
