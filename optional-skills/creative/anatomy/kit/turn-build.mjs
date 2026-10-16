import { TURN, TURN_CSS } from "./turn.mjs";
import * as k from "./iso-kit.mjs";
import * as G from "./lathe.mjs";
import * as A from "./audit.mjs";

const TAU = Math.PI * 2;
const r4 = (v) => Math.round(v * 1e4) / 1e4;
const r3 = (v) => Math.round(v * 1e3) / 1e3;
const pack3 = (value) => JSON.parse(JSON.stringify(value, (key, v) => (typeof v === "number" ? r3(v) : v instanceof Float64Array || v instanceof Float32Array ? Array.from(v, r3) : v)));
const pack = (value) => JSON.parse(JSON.stringify(value, (key, v) => (typeof v === "number" ? r4(v) : v instanceof Float64Array || v instanceof Float32Array ? Array.from(v, r4) : v)));
const esc = (text) => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function octOf(n) {
  const l = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]) || 1;
  let x = n[0] / l;
  let y = n[1] / l;
  if (n[2] < 0) {
    const ox = (1 - Math.abs(y)) * (x >= 0 ? 1 : -1);
    const oy = (1 - Math.abs(x)) * (y >= 0 ? 1 : -1);
    x = ox;
    y = oy;
  }
  return [Math.round((x * 0.5 + 0.5) * 65535), Math.round((y * 0.5 + 0.5) * 65535)];
}

function packPairs(byLayer, stride, grouped) {
  const out = {};
  for (const [name, list] of Object.entries(byLayer)) {
    const count = list.length / stride;
    const bytes = new Uint8Array(count * (grouped ? 10 : 8));
    const view = new DataView(bytes.buffer);
    for (let e = 0; e < count; e++) {
      const at = e * stride;
      let o = e * (grouped ? 10 : 8);
      view.setUint16(o, list[at]);
      view.setUint16(o + 2, list[at + 1]);
      o += 4;
      if (grouped) {
        view.setUint16(o, list[at + 2] + 1);
        o += 2;
      }
      const [u, v] = octOf(list.slice(at + stride - 3, at + stride));
      view.setUint16(o, u);
      view.setUint16(o + 2, v);
    }
    out[name] = Buffer.from(bytes).toString("base64");
  }
  return out;
}

function signedArea(points) {
  let s = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    s += a[0] * b[1] - b[0] * a[1];
  }
  return s / 2;
}

function convexOrThrow(points, name) {
  const n = points.length;
  let sign = 0;
  for (let i = 0; i < n; i++) {
    const a = points[i];
    const b = points[(i + 1) % n];
    const c = points[(i + 2) % n];
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (Math.abs(cross) < 1e-9) continue;
    if (!sign) sign = Math.sign(cross);
    else if (Math.sign(cross) !== sign) throw new Error(`turning: part ${name} is not convex; split it into convex pieces`);
  }
}

export function roundedPolygon(polygon, radius = 0, steps = 4) {
  const poly = signedArea(polygon) < 0 ? polygon.slice().reverse() : polygon.slice();
  if (!(radius > 0)) return poly.map(([x, y]) => [x, y]);
  const n = poly.length;
  const normals = poly.map((a, i) => {
    const b = poly[(i + 1) % n];
    const l = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
    return [(b[1] - a[1]) / l, -(b[0] - a[0]) / l];
  });
  const out = [];
  for (let i = 0; i < n; i++) {
    const na = normals[(i - 1 + n) % n];
    const nb = normals[i];
    const denom = 1 + na[0] * nb[0] + na[1] * nb[1];
    const c = [poly[i][0] - (radius * (na[0] + nb[0])) / denom, poly[i][1] - (radius * (na[1] + nb[1])) / denom];
    const a0 = Math.atan2(na[1], na[0]);
    let a1 = Math.atan2(nb[1], nb[0]);
    while (a1 < a0 - 1e-9) a1 += TAU;
    for (let s = 0; s <= steps; s++) {
      const t = a0 + ((a1 - a0) * s) / steps;
      out.push([c[0] + radius * Math.cos(t), c[1] + radius * Math.sin(t)]);
    }
  }
  return out;
}

const fine = (v) => {
  const t = Math.round(v * 1000) / 1000;
  return t === 0 ? "0" : String(t);
};
const finePath = (points) => points.map(([x, y], i) => `${i ? "L" : "M"}${fine(x)} ${fine(y)}`).join("") + "Z";

const dot3 = G.dot3;
const sub3 = G.sub3;
const add3 = G.add3;
const mul3 = G.mul3;

function marginOf(n, A, B) {
  let hiA = -Infinity;
  for (const a of A) {
    const v = n[0] * a[0] + n[1] * a[1] + n[2] * a[2];
    if (v > hiA) hiA = v;
  }
  let loB = Infinity;
  for (const b of B) {
    const v = n[0] * b[0] + n[1] * b[1] + n[2] * b[2];
    if (v < loB) loB = v;
  }
  return { margin: loB - hiA, d: (loB + hiA) / 2 };
}

function closest(A, B, rounds = 200) {
  let w = sub3(B[0], A[0]);
  for (let it = 0; it < rounds; it++) {
    const l = G.len3(w);
    if (l < 1e-12) break;
    let hi = -Infinity;
    let a = A[0];
    for (const p of A) {
      const v = dot3(p, w);
      if (v > hi) {
        hi = v;
        a = p;
      }
    }
    let lo = Infinity;
    let b = B[0];
    for (const p of B) {
      const v = dot3(p, w);
      if (v < lo) {
        lo = v;
        b = p;
      }
    }
    const s = sub3(b, a);
    const ds = sub3(w, s);
    const dd = dot3(ds, ds);
    if (dd < 1e-14) break;
    const t = Math.max(0, Math.min(1, dot3(w, ds) / dd));
    const next = sub3(w, mul3(ds, t));
    if (G.len3(sub3(next, w)) < 1e-10) break;
    w = next;
  }
  return w;
}

export function gjk(A, B, warm = null) {
  const w = warm && G.len3(warm) > 1e-9 ? warm : closest(A, B);
  const n = G.unit3(G.len3(w) > 1e-12 ? w : [0, 0, 1]);
  const { margin, d } = marginOf(n, A, B);
  return { n, d, margin };
}

function quickSeparate(A, B, hints = []) {
  const centre = (list) => mul3(list.reduce((sum, p) => add3(sum, p), [0, 0, 0]), 1 / list.length);
  const candidates = [sub3(centre(B), centre(A)), ...hints.flatMap((h) => [h, mul3(h, -1)])].filter((n) => G.len3(n) > 1e-9).map(G.unit3);
  let best = null;
  for (const n of candidates) {
    const { margin } = marginOf(n, A, B);
    if (!best || margin > best.margin) best = { n, margin };
  }
  if (best.margin >= 0.1) return best;
  let { n, margin } = best;
  for (const step of [0.3, 0.1, 0.03]) {
    for (let round = 0; round < 12; round++) {
      const F = G.frameAlong([0, 0, 0], n);
      let improved = false;
      for (let j = 0; j < 8; j++) {
        const t = (j / 8) * TAU;
        const trial = G.unit3(add3(n, add3(mul3(F.u, step * Math.cos(t)), mul3(F.v, step * Math.sin(t)))));
        const value = marginOf(trial, A, B).margin;
        if (value > margin + 1e-9) {
          margin = value;
          n = trial;
          improved = true;
        }
      }
      if (!improved) break;
    }
  }
  return { n, margin };
}

export function separate(A, B, hints = []) {
  const centre = (list) => mul3(list.reduce((s, p) => add3(s, p), [0, 0, 0]), 1 / list.length);
  const candidates = [sub3(centre(B), centre(A)), closest(A, B), ...hints.flatMap((h) => [h, mul3(h, -1)])].filter((n) => G.len3(n) > 1e-9).map(G.unit3);
  let best = null;
  for (const n of candidates) {
    const { margin } = marginOf(n, A, B);
    if (!best || margin > best.margin) best = { n, margin };
  }
  let { n, margin } = best;
  for (const step of [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]) {
    for (let round = 0; round < 40; round++) {
      const F = G.frameAlong([0, 0, 0], n);
      let improved = false;
      for (let j = 0; j < 8; j++) {
        const t = (j / 8) * TAU;
        const trial = G.unit3(add3(n, add3(mul3(F.u, step * Math.cos(t)), mul3(F.v, step * Math.sin(t)))));
        const value = marginOf(trial, A, B).margin;
        if (value > margin + 1e-9) {
          margin = value;
          n = trial;
          improved = true;
        }
      }
      if (!improved) break;
    }
  }
  return { n, d: marginOf(n, A, B).d, margin };
}

function frameFrom(F) {
  const a = G.unit3(F.a);
  let u = F.u ? G.unit3(sub3(F.u, mul3(a, dot3(F.u, a)))) : null;
  if (!u || G.len3(u) < 1e-9) u = G.frameAlong([0, 0, 0], a).u;
  const v = F.v && Math.abs(dot3(G.cross3(a, u), F.v)) > 0.999 ? G.unit3(F.v) : G.cross3(a, u);
  return { o: F.o.slice(), a, u, v };
}

function sampleShape(shape, count = 48) {
  const out = [];
  if (shape.kind === "body") {
    const F = frameFrom(shape.F);
    for (const [s, r] of shape.poly) {
      if (r < 1e-9) {
        out.push(G.pointOf(F, s, 0, 0));
        continue;
      }
      for (let j = 0; j < count; j++) out.push(G.pointOf(F, s, r, (j / count) * TAU));
    }
  } else if (shape.kind === "prism") {
    const F = shape.F;
    for (const s of [shape.s0, shape.s1]) for (const [x, y] of shape.poly) out.push(add3(add3(F.o, mul3(F.a, s)), add3(mul3(F.u, x), mul3(F.v, y))));
  } else if (shape.kind === "ball") {
    const golden = Math.PI * (3 - Math.sqrt(5));
    const total = Math.max(count * 3, 96);
    for (let i = 0; i < total; i++) {
      const y = 1 - (2 * (i + 0.5)) / total;
      const rad = Math.sqrt(1 - y * y);
      const t = golden * i;
      const n = [Math.cos(t) * rad, Math.sin(t) * rad, y];
      if ((shape.flats ?? []).some(([nx, ny, nz, d]) => nx * n[0] * shape.r + ny * n[1] * shape.r + nz * n[2] * shape.r > d + 1e-9)) continue;
      out.push(add3(shape.o, mul3(n, shape.r)));
    }
    for (const [nx, ny, nz, d] of shape.flats ?? []) {
      const F = G.frameAlong(add3(shape.o, mul3([nx, ny, nz], d)), [nx, ny, nz]);
      const rho = Math.sqrt(Math.max(0, shape.r * shape.r - d * d));
      for (let j = 0; j < count; j++) {
        const p = G.pointOf(F, 0, rho, (j / count) * TAU);
        if ((shape.flats ?? []).every(([mx, my, mz, e]) => mx * (p[0] - shape.o[0]) + my * (p[1] - shape.o[1]) + mz * (p[2] - shape.o[2]) <= e + 1e-6)) out.push(p);
      }
    }
  } else if (shape.kind === "box") {
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) out.push(add3(shape.o, add3(add3(mul3(shape.axes[0], sx * shape.half[0]), mul3(shape.axes[1], sy * shape.half[1])), mul3(shape.axes[2], sz * shape.half[2]))));
  } else if (shape.kind === "tube") {
    const pts = shape.points;
    for (let i = 0; i < pts.length; i++) {
      const t = G.unit3(sub3(pts[Math.min(pts.length - 1, i + 1)], pts[Math.max(0, i - 1)]));
      const F = G.frameAlong(pts[i], t);
      const ring = Math.max(8, Math.round(count / 4));
      for (let j = 0; j < ring; j++) out.push(G.pointOf(F, 0, shape.radii[i], (j / ring) * TAU));
    }
  }
  return out;
}

function hull2(points) {
  return G.hullOf(points);
}

function thinCloud(points, most = 64) {
  if (points.length <= most) return points;
  const out = [];
  const step = points.length / most;
  for (let i = 0; i < most; i++) out.push(points[Math.floor(i * step)]);
  return out;
}

export function turning(P, { recorder = null, scale = 1 } = {}) {
  const T = { P, R: recorder, scale, groups: [], layers: [], order: [], parts: [], built: null };
  const groupOf = (g) => (g === undefined || g === null ? null : typeof g === "string" ? T.groups.find((x) => x.name === g) : g);
  const layerOf = (l) => (typeof l === "string" ? T.layers.find((x) => x.name === l) : l);
  const origin = (group) => (group ? group.origin : [0, 0, 0]);
  const local = (group, p) => sub3(p, origin(group));
  T.group = (name, options = {}) => {
    const parent = groupOf(options.parent);
    const po = origin(parent);
    let kind;
    let spec;
    if (options.turn) {
      kind = "turn";
      const pivot = options.turn.pivot ?? [0, 0];
      spec = { origin: [pivot[0], pivot[1], 0], pivot: [pivot[0], pivot[1]], range: options.turn.range ?? null };
    } else if (options.slide) {
      kind = "slide";
      spec = { origin: po.slice(), direction: G.unit3(options.slide.direction), travel: options.slide.travel ?? 0, range: [0, options.slide.travel ?? 0] };
    } else if (options.hinge) {
      kind = "hinge";
      spec = { origin: options.hinge.point.slice(), point: options.hinge.point.slice(), axis: G.unit3(options.hinge.axis), range: options.hinge.range ?? [0, 0] };
    } else if (options.free) {
      kind = "free";
      const at = options.free.origin ?? po;
      spec = { origin: at.slice(), range: null };
    } else throw new Error(`turning: group ${name} needs turn, slide, hinge or free`);
    const upright = kind !== "free" && (parent ? parent.upright : true) && (kind !== "hinge" || Math.abs(spec.axis[2]) > 0.9999);
    const loose = kind === "free" || Boolean(parent && parent.loose);
    const group = { name, index: T.groups.length, parent, kind, upright, loose, ...spec };
    T.groups.push(group);
    return group;
  };
  T.layer = (name) => {
    const layer = { name, index: T.layers.length, live: true };
    T.layers.push(layer);
    return layer;
  };
  T.stack = (names) => {
    T.order = names.slice();
    return T;
  };
  const common = (options, kind) => {
    const group = groupOf(options.group);
    const layer = layerOf(options.layer ?? T.layers[0]);
    if (!layer) throw new Error(`turning: part ${options.name} has no live layer`);
    const name = options.name ?? `${kind}#${T.parts.length}`;
    const material = options.material ?? null;
    if (material && !TURN.materials.includes(material)) throw new Error(`turning: part ${name} has material ${material}; use one of ${TURN.materials.join(", ")}`);
    return { id: T.parts.length, name, kind, group, layer, owner: options.owner, tone: options.tone ?? "hi", crease: options.crease ?? "faint", lit: Boolean(options.lit), material, details: options.details ?? {}, shapes: [], planes: [], dots: [], samples: [], hints: [] };
  };
  const record = (part, shape) => {
    shape.name = shape.name ?? part.name;
    if (part.owner && !shape.owner) shape.owner = part.owner;
    part.shapes.push(shape);
    if (T.R) T.R.solid(shape);
    return shape;
  };
  const toLocalDots = (part, F, list) => {
    for (const dot of list ?? []) {
      const [s, x, y] = dot.at;
      const p = add3(add3(F.o, mul3(F.a, s)), add3(mul3(F.u, x), mul3(F.v, y)));
      part.dots.push({ p: local(part.group, p), n: dot.normal ? G.unit3(dot.normal) : null, size: dot.size ?? 0.5, tone: dot.tone ?? "mid", fade: dot.fade ?? [0.04, 0.42] });
    }
  };
  T.prism = (options) => {
    const part = common(options, "prism");
    let F;
    let ring;
    let s0;
    let s1;
    if (options.plan) {
      F = { o: [0, 0, 0], a: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] };
      ring = k.roundedPlan(options.plan, options.steps ?? 4);
      s0 = options.z ?? options.s0 ?? 0;
      s1 = options.h !== undefined ? s0 + options.h : options.s1;
    } else {
      F = frameFrom(options.F);
      ring = roundedPolygon(options.polygon, options.round ?? 0, options.steps ?? 4);
      s0 = options.s0;
      s1 = options.s1;
    }
    if (s1 < s0) [s0, s1] = [s1, s0];
    if (signedArea(ring) < 0) ring = ring.slice().reverse();
    convexOrThrow(ring, part.name);
    const bevel = options.bevel ?? part.details.bevel ?? 0;
    const upright = Math.abs(F.a[2]) > 0.9999 && (!part.group || part.group.upright);
    record(part, A.prism(F, ring, s0, s1, { name: part.name }));
    const corner = (s, [x, y]) => add3(add3(F.o, mul3(F.a, s)), add3(mul3(F.u, x), mul3(F.v, y)));
    part.world = { F, ring, s0, s1 };
    if (upright) {
      const flip = F.a[2] < 0;
      const pts = ring.map((p) => local(part.group, corner(flip ? s1 : s0, p)));
      let xy = pts.map(([x, y]) => [x, y]);
      if (signedArea(xy) < 0) xy = xy.reverse();
      const z0 = pts[0][2];
      const z1 = z0 + (s1 - s0);
      part.spec = { up: 1, ring: xy.flat(), z0, z1, bevel };
      part.planPath = finePath(xy);
      part.localSeams = (part.details.seams ?? []).map((s) => (flip ? z1 - (s - s0) : z0 + (s - s0)));
    } else {
      const FL = { o: local(part.group, F.o), a: F.a, u: F.u, v: F.v };
      part.spec = { up: 0, F: FL, ring: ring.flat(), s0, s1, bevel };
      part.localSeams = (part.details.seams ?? []).slice();
    }
    toLocalDots(part, F, part.details.dots);
    for (const s of [s0, s1]) for (const p of ring) part.samples.push(local(part.group, corner(s, p)));
    part.hints.push(F.a);
    const n = ring.length;
    for (let i = 0; i < n; i++) {
      const a = ring[i];
      const b = ring[(i + 1) % n];
      const l = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (l > 1e-6) part.hints.push(G.unit3(add3(mul3(F.u, (b[1] - a[1]) / l), mul3(F.v, -(b[0] - a[0]) / l))));
    }
    T.parts.push(part);
    return part;
  };
  const detailParts = (part, F, radiusAt, ends) => {
    const details = part.details ?? {};
    for (const bolt of details.bolts ?? []) {
      const { s, r, count, phase = 0, size = 0.5, tone = "mid", fade = [0.04, 0.42] } = bolt;
      const atEnd = ends ? (Math.abs(s - ends[0]) < 1e-6 ? -1 : Math.abs(s - ends[1]) < 1e-6 ? 1 : 0) : 0;
      const onCap = atEnd !== 0 && (!radiusAt || r < radiusAt(s) - 1e-6);
      const slope = radiusAt ? (radiusAt(s + 1e-3) - radiusAt(s - 1e-3)) / 2e-3 : 0;
      for (let j = 0; j < count; j++) {
        const t = phase + (j / count) * TAU;
        const p = G.pointOf(F, s, r, t);
        const radial = G.radialOf(F, t);
        const n = onCap ? mul3(F.a, atEnd) : G.unit3(sub3(radial, mul3(F.a, slope)));
        part.dots.push({ p: local(part.group, p), n, size, tone, fade });
      }
    }
    toLocalDots(part, F, details.dots);
    for (const ring of details.rings ?? []) {
      const [s, x, y] = ring.at;
      const o = add3(add3(F.o, mul3(F.a, s)), add3(mul3(F.u, x), mul3(F.v, y)));
      const n = G.unit3(ring.normal ?? F.a);
      const B = G.frameAlong([0, 0, 0], n);
      const steps = Math.max(12, Math.ceil((TAU * ring.r * k.cameraOf(P).k) / 1.2));
      const pts = Array.from({ length: steps }, (_, j) => [ring.r * Math.cos((j / steps) * TAU), ring.r * Math.sin((j / steps) * TAU)]);
      T.plane({ part, o, u: B.u, v: B.v, normal: n, fade: ring.fade ?? [0.04, 0.3], svg: `<path class="iso-line" data-tone="${ring.tone ?? "lo"}" d="${finePath(pts)}"/>` });
    }
    for (const rule of details.rules ?? []) {
      const pts = rule.points.map(([s, x, y]) => add3(add3(F.o, mul3(F.a, s)), add3(mul3(F.u, x), mul3(F.v, y))));
      let n = rule.normal ? G.unit3(rule.normal) : null;
      if (!n) {
        for (let i = 2; i < pts.length && !n; i++) {
          const c = G.cross3(sub3(pts[1], pts[0]), sub3(pts[i], pts[0]));
          if (G.len3(c) > 1e-9) n = G.unit3(c);
        }
        n = n ?? F.a;
      }
      const B = G.frameAlong([0, 0, 0], n);
      const o = pts[0];
      const flat = pts.map((p) => [dot3(sub3(p, o), B.u), dot3(sub3(p, o), B.v)]);
      const d = flat.map(([x, y], i) => `${i ? "L" : "M"}${fine(x)} ${fine(y)}`).join("");
      const free = rule.free === true ? ' data-free=""' : rule.free ? ` data-free="${rule.free}"` : "";
      T.plane({ part, o, u: B.u, v: B.v, normal: n, fade: rule.fade ?? [0.04, 0.3], svg: `<path class="iso-line" data-tone="${rule.tone ?? "lo"}"${free} d="${d}"/>` });
    }
  };
  T.round = (options) => {
    const F = frameFrom(options.F);
    const r0 = Array.isArray(options.r) ? options.r[0] : options.r0 ?? options.r;
    const r1 = Array.isArray(options.r) ? options.r[1] : options.r1 ?? options.r;
    let s0 = options.s0;
    let s1 = options.s1;
    const ends = (options.ends ?? ["flat", "flat"]).slice();
    if (s1 < s0) throw new Error(`turning: round ${options.name} needs s0 < s1`);
    ends.forEach((end, j) => {
      if (end === "dome" && Math.abs(r0 - r1) > 1e-9) throw new Error(`turning: round ${options.name} can only have a dome where the end radius equals the side's (a capsule)`);
    });
    const profile = [];
    const arc = (sAt, sign, r) => Array.from({ length: 9 }, (_, j) => (j / 8) * (Math.PI / 2)).map((b) => [sAt + sign * r * Math.cos(b), r * Math.sin(b), 1]);
    if (ends[0] === "dome") profile.push(...arc(s0, -1, r0).slice(0, -1).map((p, j) => (j === 0 ? [p[0], 0] : p)), [s0, r0, 1]);
    else profile.push([s0, r0]);
    if (ends[1] === "dome") profile.push([s1, r1, 1], ...arc(s1, 1, r1).reverse().slice(1).map((p, j, list) => (j === list.length - 1 ? [p[0], 0] : p)));
    else profile.push([s1, r1]);
    const vertical = Math.abs(F.a[2]) > 0.9999;
    const groupOk = !options.group || groupOf(options.group).upright;
    const radiusAt = (s) => r0 + ((r1 - r0) * (Math.max(s0, Math.min(s1, s)) - s0)) / (s1 - s0 || 1);
    if (vertical && groupOk && options.fixed !== false) {
      const meridian = [[profile[0][0], 0], ...profile.filter((p) => p[1] > 1e-9), [profile[profile.length - 1][0], 0]];
      const seams = (options.details?.seams ?? []).map((sv) => G.arcOf(F, sv, radiusAt(sv), P, { slope: (r1 - r0) / (s1 - s0 || 1), least: 0 })).join("");
      const paths = G.lathe(meridian, F, P, ends.includes("dome") ? { smooth: true } : {});
      const svg = k.solidSvg(paths, { tone: options.tone ?? "hi", crease: options.crease ?? "faint", lit: options.lit, inner: seams ? k.lineSvg(seams, { tone: "faint" }) : "" });
      const ribs = options.details?.ribs ? { ...options.details.ribs, r0: options.details.ribs.r0 ?? radiusAt(options.details.ribs.s0), r1: options.details.ribs.r1 ?? radiusAt(options.details.ribs.s1) } : undefined;
      const part = T.fixed({ ...options, svg, anchor: G.pointOf(F, s0, 0, 0), F, details: { ...(options.details ?? {}), ribs, seams: undefined }, shapes: [A.solid(F, profile, { name: options.name })], skipDetails: true });
      part.round = { F, s0, s1, r0, r1, ends };
      detailParts(part, F, radiusAt, [s0, s1]);
      return part;
    }
    const part = common(options, "round");
    record(part, A.solid(F, profile, { name: part.name }));
    part.world = { F, s0, s1, r0, r1, ends, profile };
    part.spec = { F: { o: local(part.group, F.o), a: F.a, u: F.u, v: F.v }, s0, s1, r0, r1, ends };
    detailParts(part, F, radiusAt, [s0, s1]);
    const ring = (s, r, count = 48) => Array.from({ length: count }, (_, j) => G.pointOf(F, s, r, (j / count) * TAU));
    const pts = [...ring(s0, r0), ...ring(s1, r1)];
    for (const [end, sAt, r] of [[0, s0, r0], [1, s1, r1]]) {
      if (ends[end] !== "dome") continue;
      const sign = end ? 1 : -1;
      for (let j = 1; j <= 6; j++) {
        const b = (j / 6) * (Math.PI / 2);
        if (j === 6) pts.push(add3(F.o, mul3(F.a, sAt + sign * r)));
        else pts.push(...ring(sAt + sign * r * Math.sin(b), r * Math.cos(b), 32));
      }
    }
    part.samples = pts.map((p) => local(part.group, p));
    part.hints.push(F.a);
    T.parts.push(part);
    return part;
  };
  T.ball = (options) => {
    const c = options.c;
    const r = options.r;
    const flats = (options.flats ?? []).map(({ n, d }) => ({ n: G.unit3(n), d }));
    const shape = A.ball(c, r, { name: options.name, flats: flats.map(({ n, d }) => [n[0], n[1], n[2], d]) });
    if (!flats.length && options.fixed !== false) {
      const svg = k.solidSvg(G.sphereOf(c, r, P, { steps: 48 }), { tone: options.tone ?? "hi", crease: options.crease ?? "faint", lit: options.lit });
      return T.fixed({ ...options, svg, anchor: c, shapes: [shape] });
    }
    const part = common(options, "ball");
    record(part, shape);
    part.world = { c, r, flats };
    part.spec = { c: local(part.group, c), r, flats };
    part.samples = sampleShape(shape, 48).map((p) => local(part.group, p));
    for (const { n } of flats) part.hints.push(n);
    detailParts(part, G.frameAlong(c, flats.length ? flats[0].n : [0, 0, 1]), null, null);
    T.parts.push(part);
    return part;
  };
  T.lathe = (options) => {
    const F = frameFrom(options.F);
    const profile = options.profile;
    const curve = G.profileCurve(profile.map(([sv, rv]) => [sv, rv]), { slope: options.slope ?? null, radius: options.radius ?? null });
    const kk = k.cameraOf(P).k;
    const knots = [curve.s0];
    const split = (a, b, level) => {
      const mid = (a + b) / 2;
      if (level < 7 && (Math.abs(curve.radius(mid) - (curve.radius(a) + curve.radius(b)) / 2) * kk > 0.1 || b - a > 4)) {
        split(a, mid, level + 1);
        knots.push(mid);
        split(mid, b, level + 1);
      }
    };
    const stops = [curve.s0, ...profile.map(([sv]) => sv).filter((sv) => sv > curve.s0 && sv < curve.s1), curve.s1];
    for (let i = 0; i < stops.length - 1; i++) {
      split(stops[i], stops[i + 1], 0);
      knots.push(stops[i + 1]);
    }
    for (let i = 1; i < knots.length - 1; i++) {
      const a = curve.radius(knots[i - 1]);
      const b = curve.radius(knots[i]);
      const c = curve.radius(knots[i + 1]);
      const h0 = knots[i] - knots[i - 1];
      const h1 = knots[i + 1] - knots[i];
      if ((c - b) / h1 - (b - a) / h0 > 1e-6) throw new Error(`turning: lathe ${options.name} is not convex near s = ${knots[i].toFixed(2)} (r(s) must be concave); split it`);
    }
    const levels = [];
    for (const sv of knots) {
      const g = curve.slope(sv);
      const norm = Math.hypot(1, g);
      levels.push(sv, curve.radius(sv), 1 / norm, -g / norm);
    }
    const caps = [curve.radius(curve.s0) > 1e-6, curve.radius(curve.s1) > 1e-6];
    const shape = A.solid(F, profile.map(([sv, rv]) => [sv, rv, 1]), { name: options.name, radius: curve.radius });
    const vertical = Math.abs(F.a[2]) > 0.9999;
    const groupOk = !options.group || groupOf(options.group).upright;
    if (vertical && groupOk && options.fixed !== false) {
      const meridian = [[curve.s0, 0], ...knots.map((sv) => [sv, curve.radius(sv), 1]), [curve.s1, 0]];
      meridian[1] = [meridian[1][0], meridian[1][1]];
      meridian[meridian.length - 2] = [meridian[meridian.length - 2][0], meridian[meridian.length - 2][1]];
      const seams = (options.details?.seams ?? []).map((sv) => G.arcOf(F, sv, curve.radius(sv), P, { slope: curve.slope(sv), least: 0 })).join("");
      const svg = k.solidSvg(G.lathe(meridian, F, P, { smooth: { slope: curve.slope, radius: curve.radius } }), { tone: options.tone ?? "hi", crease: options.crease ?? "faint", lit: options.lit, inner: seams ? k.lineSvg(seams, { tone: "faint" }) : "" });
      const part = T.fixed({ ...options, svg, anchor: G.pointOf(F, curve.s0, 0, 0), F, details: { ...(options.details ?? {}), seams: undefined }, shapes: [shape], skipDetails: true });
      detailParts(part, F, curve.radius, [curve.s0, curve.s1]);
      return part;
    }
    const part = common(options, "lathe");
    record(part, shape);
    part.spec = { F: { o: local(part.group, F.o), a: F.a, u: F.u, v: F.v }, levels, caps };
    part.world = { F, profile, curve, knots };
    const pts = [];
    for (const sv of knots) for (let j = 0; j < 32; j++) pts.push(G.pointOf(F, sv, curve.radius(sv), (j / 32) * TAU));
    part.samples = pts.map((p) => local(part.group, p));
    part.hints.push(F.a);
    detailParts(part, F, curve.radius, [curve.s0, curve.s1]);
    T.parts.push(part);
    return part;
  };
  T.routes = [];
  T.tube = (options) => {
    const { route, r, chunk = 6, turn = 20, gaps = [], breaks = [], caps = true, rings = null, hue = null, spacing = 0.8, owner, touch, bundle, rims, free } = options;
    const group = groupOf(options.group);
    const name = options.name ?? `tube#${T.routes.length}`;
    const dense = G.resample(route, spacing);
    const total = G.pathLength(dense);
    const turnBreaks = [];
    let walked = 0;
    let bent = 0;
    for (let i = 1; i < dense.length - 1; i++) {
      walked += G.len3(sub3(dense[i], dense[i - 1]));
      const a = G.unit3(sub3(dense[i], dense[i - 1]));
      const b = G.unit3(sub3(dense[i + 1], dense[i]));
      bent += (Math.acos(Math.max(-1, Math.min(1, dot3(a, b)))) * 180) / Math.PI;
      if (bent > turn) {
        turnBreaks.push(walked);
        bent = 0;
      }
    }
    const allBreaks = [...breaks, ...turnBreaks].sort((x, y) => x - y);
    const pieces = spansOf(total, gaps, allBreaks, chunk);
    const wide = r * k.cameraOf(P).k > 1.3;
    const routeIndex = T.routes.length;
    const capList = [caps].flat();
    T.routes.push({ name, group, points: dense.map((p) => local(group, p)), r, spacing, wide, hue, rings, world: route, gaps, breaks: allBreaks, chunk, caps });
    if (T.R) T.R.route(name, route, r, { gaps, limp: options.limp ?? true, owner: owner ?? name, touch, bundle, rims, free });
    const chunks = pieces.map((piece, index) => {
      const part = common({ ...options, name: `${name}:${index}` }, "tube");
      part.route = name;
      part.chunk = index;
      part.hue = hue;
      part.spec = { route: routeIndex, from: piece.from, to: piece.to, open0: piece.open0, open1: piece.open1, gap0: piece.gap0, gap1: piece.gap1, caps: [capList[0], capList[capList.length - 1]], wide, rings: rings ? { tone: rings.tone ?? "lo" } : null };
      const core = G.cut(dense, piece.from, piece.to);
      part.shapes.push(A.tube(core, r, { name: part.name, owner: owner ?? name }));
      part.world = { route, r, index };
      for (let i = 0; i < core.length; i++) {
        const t = G.unit3(sub3(core[Math.min(core.length - 1, i + 1)], core[Math.max(0, i - 1)]));
        const F = G.frameAlong(core[i], t);
        for (let j = 0; j < 10; j++) part.samples.push(local(group, G.pointOf(F, 0, r, (j / 10) * TAU)));
      }
      part.hints.push(G.unit3(sub3(core[1], core[0])), G.unit3(sub3(core[core.length - 1], core[core.length - 2])));
      T.parts.push(part);
      return part;
    });
    return { name, chunks, total, dense };
  };
  T.fixed = (options) => {
    const part = common(options, "fixed");
    const anchor = options.anchor;
    for (const shape of options.shapes ?? []) record(part, shape);
    const world = part.shapes.flatMap((shape) => sampleShape(shape, 48));
    const flat = world.map((p) => k.iso(p, P));
    const hull = options.hull ?? hull2(flat);
    const box = A.markupBox(options.svg) ?? [Math.min(...hull.map((p) => p[0])), Math.min(...hull.map((p) => p[1])), Math.max(...hull.map((p) => p[0])), Math.max(...hull.map((p) => p[1]))];
    part.svg = options.svg;
    const F = options.F ? frameFrom(options.F) : null;
    part.spec = { anchor: local(part.group, anchor), home: k.iso(anchor, P), box: [box[0] - 0.5, box[1] - 0.5, box[2] + 0.5, box[3] + 0.5], hull: hull.flat(), F: F ? { o: local(part.group, F.o), a: F.a, u: F.u, v: F.v } : null };
    if (F && !options.skipDetails) detailParts(part, F, null, null);
    part.samples = world.map((p) => local(part.group, p));
    if (F) part.hints.push(F.a);
    T.parts.push(part);
    return part;
  };
  T.item = (options) => {
    const part = common(options, "item");
    const cloud = options.cloud ?? [[0, 0, 0]];
    part.spec = { item: 1 };
    part.cloud = cloud.map((p) => local(part.group, p)).flat();
    part.samples = cloud.map((p) => local(part.group, p));
    T.parts.push(part);
    return part;
  };
  T.plane = ({ part, o, u, v, svg, fade = [0.04, 0.3], normal = null }) => {
    const U = G.unit3(u);
    const W = G.unit3(v);
    const n = normal ? G.unit3(normal) : G.unit3(G.cross3(U, W));
    part.planes.push({ o: local(part.group, o), u: U, v: W, n, fade, svg });
    return part;
  };
  T.billboard = ({ part, at, normal, svg, fade = [0.04, 0.3] }) => {
    part.planes.push({ o: local(part.group, at), u: [1, 0, 0], v: [0, 1, 0], n: G.unit3(normal), fade, svg, billboard: true });
    return part;
  };
  const drop = (part) => {
    T.parts.splice(T.parts.indexOf(part), 1);
    T.parts.forEach((p, index) => (p.id = index));
  };
  T.split = (part, { n, d }) => {
    const N = G.unit3(n);
    if (part.kind === "prism") {
      const { F, ring, s0, s1 } = part.world;
      const along = dot3(N, F.a);
      const base = { name: part.name, group: part.group, layer: part.layer, tone: part.tone, crease: part.crease, lit: part.lit, material: part.material, details: { ...part.details }, bevel: part.spec.bevel };
      let halves;
      if (Math.abs(Math.abs(along) - 1) < 1e-6) {
        const sCut = (d - dot3(N, F.o)) / along;
        if (!(sCut > s0 + 1e-6 && sCut < s1 - 1e-6)) throw new Error(`turning: split plane misses ${part.name}`);
        halves = [{ s0, s1: sCut }, { s0: sCut, s1 }].map((span) => ({ ...base, F, polygon: ring, s0: span.s0, s1: span.s1 }));
      } else if (Math.abs(along) < 1e-6) {
        const nu = dot3(N, F.u);
        const nv = dot3(N, F.v);
        const c = d - dot3(N, F.o);
        const side = (p) => nu * p[0] + nv * p[1] - c;
        const clip = (sign) => {
          const out = [];
          for (let i = 0; i < ring.length; i++) {
            const a = ring[i];
            const b = ring[(i + 1) % ring.length];
            const sa = sign * side(a);
            const sb = sign * side(b);
            if (sa <= 0) out.push(a);
            if ((sa < 0 && sb > 0) || (sa > 0 && sb < 0)) {
              const t = sa / (sa - sb);
              out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
            }
          }
          return out;
        };
        halves = [clip(1), clip(-1)].map((polygon) => ({ ...base, F, polygon, s0, s1 }));
        if (halves.some((h) => h.polygon.length < 3)) throw new Error(`turning: split plane misses ${part.name}`);
      } else throw new Error(`turning: a prism can only be split by a plane along or across its axis (${part.name})`);
      part.dead = true;
      if (T.R) T.R.solids.splice(T.R.solids.indexOf(part.shapes[0]), 1);
      const made = halves.map((h, index) => T.prism({ ...h, name: `${part.name}.${index ? "b" : "a"}`, round: 0, details: index ? { ...h.details, dots: [] } : h.details }));
      for (const half of made) {
        half.spec.cuts = [...(part.spec.cuts ?? []), { n: N, d: d - dot3(N, half.group ? half.group.origin : [0, 0, 0]) }];
        half.hints.unshift(N);
      }
      drop(part);
      return made;
    }
    if (part.kind === "round") {
      const { F, s0, s1, r0, r1, ends } = part.world;
      const along = dot3(N, F.a);
      if (Math.abs(Math.abs(along) - 1) > 1e-6) throw new Error(`turning: a round can only be split across its axis (${part.name})`);
      const sCut = (d - dot3(N, F.o)) / along;
      if (!(sCut > s0 + 1e-6 && sCut < s1 - 1e-6)) throw new Error(`turning: split plane misses ${part.name}`);
      const rAt = r0 + ((r1 - r0) * (sCut - s0)) / (s1 - s0);
      part.dead = true;
      if (T.R) T.R.solids.splice(T.R.solids.indexOf(part.shapes[0]), 1);
      const base = { group: part.group, layer: part.layer, tone: part.tone, crease: part.crease, lit: part.lit, material: part.material, F, fixed: false };
      const made = [
        T.round({ ...base, name: `${part.name}.a`, s0, s1: sCut, r: [r0, rAt], ends: [ends[0], "open"], details: part.details }),
        T.round({ ...base, name: `${part.name}.b`, s0: sCut, s1, r: [rAt, r1], ends: ["open", ends[1]] }),
      ];
      for (const half of made) {
        half.spec.cuts = [...(part.spec.cuts ?? []), { n: N, d: d - dot3(N, half.group ? half.group.origin : [0, 0, 0]) }];
        half.hints.unshift(N);
      }
      drop(part);
      return made;
    }
    throw new Error(`turning: split supports prisms and rounds (${part.name})`);
  };
  T.build = (options = {}) => build(T, options);
  return T;
}

function spansOf(total, gaps, breaks, maxLength) {
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
  return pieces;
}

function groupData(T) {
  return T.groups.map((g) => ({
    name: g.name,
    parent: g.parent ? g.parent.index : -1,
    kind: g.kind,
    origin: g.origin,
    pivot: g.pivot,
    direction: g.direction,
    travel: g.travel,
    point: g.point,
    axis: g.axis,
    range: g.range,
    upright: g.upright,
    loose: g.loose || undefined,
  }));
}

function partData(part) {
  const details = {};
  const seams = part.kind === "prism" ? part.localSeams : part.kind === "fixed" ? null : part.details.seams;
  if (seams && seams.length) details.seams = seams;
  if (part.details.ribs) details.ribs = part.details.ribs;
  const out = { name: part.name, kind: part.kind, group: part.group ? part.group.index : -1, layer: part.layer.index };
  if (part.tone !== "hi") out.tone = part.tone;
  if (part.crease !== "faint") out.crease = part.crease;
  if (part.lit) out.lit = true;
  if (part.material) out.mat = part.material;
  out.spec = part.spec;
  if (Object.keys(details).length) out.details = details;
  if (part.planes.length) out.planes = part.planes.map(({ o, u, v, n, fade, billboard }) => ({ o, u, v, n, fade, billboard }));
  if (part.dots.length) out.dots = part.dots.map(({ p, n, size, tone, fade }) => ({ p, n, size, tone, fade }));
  if (part.kind === "item") out.cloud = part.cloud;
  return out;
}

function statesOf(T, poses) {
  if (typeof poses === "function") poses = poses();
  if (Array.isArray(poses) && poses.length) return poses;
  const out = [];
  for (let deg = 0; deg < 360; deg += 2) out.push(Object.fromEntries(T.groups.map((g) => [g.name, g.kind === "turn" ? deg : g.range ? g.range[0] + ((g.range[1] - g.range[0]) * (deg % 90)) / 90 : 0])));
  return out;
}

function tumbleStates(T, group, count = 160) {
  let root = group;
  while (root && root.kind !== "free") root = root.parent;
  if (!root) return [];
  let seed = 977;
  const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const out = [];
  for (let i = 0; i < count; i++) {
    const u = rand();
    const a = rand() * TAU;
    const b = rand() * TAU;
    const q = [Math.sqrt(1 - u) * Math.sin(a), Math.sqrt(1 - u) * Math.cos(a), Math.sqrt(u) * Math.sin(b), Math.sqrt(u) * Math.cos(b)];
    const values = { [root.name]: { q } };
    for (const g of T.groups) if (g.kind === "hinge" && g.loose) values[g.name] = g.range[0] + (g.range[1] - g.range[0]) * rand();
    out.push(values);
  }
  return out;
}

function groupStates(T, group, states, { tumble = false } = {}) {
  if (!group) return [{}];
  let chain = [];
  for (let g = group; g; g = g.parent) chain.push(g);
  if (group.loose) return tumble ? [...states, ...tumbleStates(T, group)] : states.length ? states : [{}];
  if (chain.length === 1 && group.kind === "turn") return Array.from({ length: 360 }, (_, deg) => ({ [group.name]: deg }));
  if (chain.length === 1) {
    const [lo, hi] = group.range ?? [0, 0];
    return Array.from({ length: 33 }, (_, i) => ({ [group.name]: lo + ((hi - lo) * i) / 32 }));
  }
  return states;
}

function sweptGroups(T, layer, states, scene) {
  const V = G.viewOf(T.P);
  const cam = k.cameraOf(T.P);
  const byGroup = new Map();
  for (const part of T.parts) {
    if (part.layer !== layer || part.dead) continue;
    const key = part.group ? part.group.index : -1;
    if (!byGroup.has(key)) byGroup.set(key, []);
    byGroup.get(key).push(part);
  }
  const out = [];
  for (const [key, parts] of byGroup) {
    const group = key >= 0 ? T.groups[key] : null;
    const local = parts.flatMap((part) => part.samples);
    const root = group && group.kind === "turn" && !group.parent;
    const list = root ? Array.from({ length: 72 }, (_, i) => ({ [group.name]: i * 5 })) : group ? groupStates(T, group, states) : [{}];
    let zmin = Infinity;
    let zmax = -Infinity;
    const box = [Infinity, Infinity, -Infinity, -Infinity];
    const cloud = [];
    const stride = Math.max(1, Math.floor((local.length * list.length) / 4000));
    let counter = 0;
    for (const values of list) {
      const cams = TURN.cams(scene, values);
      const c = key >= 0 ? cams[key] : cams.world;
      for (const q of local) {
        const p = [c.R[0] * q[0] + c.R[1] * q[1] + c.R[2] * q[2] + c.t[0], c.R[3] * q[0] + c.R[4] * q[1] + c.R[5] * q[2] + c.t[1], c.R[6] * q[0] + c.R[7] * q[1] + c.R[8] * q[2] + c.t[2]];
        if (p[2] < zmin) zmin = p[2];
        if (p[2] > zmax) zmax = p[2];
        const [x, y] = k.iso(p, T.P);
        if (x < box[0]) box[0] = x;
        if (y < box[1]) box[1] = y;
        if (x > box[2]) box[2] = x;
        if (y > box[3]) box[3] = y;
        if (counter++ % stride === 0) cloud.push(p);
      }
    }
    let radius = 0;
    let envelope = cloud;
    if (root) {
      const slices = new Map();
      for (const q of local) {
        const band = Math.floor(q[2]);
        slices.set(band, Math.max(slices.get(band) ?? 0, Math.hypot(q[0], q[1])));
        radius = Math.max(radius, Math.hypot(q[0], q[1]));
      }
      envelope = [];
      for (const [band, r] of slices)
        for (const z of [band, band + 1])
          for (let j = 0; j < 48; j++) envelope.push([group.origin[0] + r * Math.cos((j / 48) * TAU), group.origin[1] + r * Math.sin((j / 48) * TAU), Math.max(zmin, Math.min(zmax, z))]);
      const pad = radius * cam.k;
      const c = k.iso([group.origin[0], group.origin[1], 0], T.P);
      box[0] = Math.min(box[0], c[0] - pad);
      box[2] = Math.max(box[2], c[0] + pad);
    }
    out.push({ key, group, parts, root, radius, zmin, zmax, box, envelope, V });
  }
  return out;
}

function relation(points, box, swept, P) {
  const V = G.viewOf(P);
  let zmin = Infinity;
  let zmax = -Infinity;
  for (const p of points) {
    zmin = Math.min(zmin, p[2]);
    zmax = Math.max(zmax, p[2]);
  }
  if (zmax <= swept.zmin + 1e-6) return { side: "behind", rule: "horizontal plane" };
  if (zmin >= swept.zmax - 1e-6) return { side: "front", rule: "horizontal plane" };
  if (swept.root) {
    const a = ((P.azimuth ?? 45) * Math.PI) / 180;
    const along = points.map((p) => (p[0] - swept.group.origin[0]) * Math.cos(a) + (p[1] - swept.group.origin[1]) * Math.sin(a));
    if (Math.max(...along) <= -swept.radius) return { side: "behind", rule: "cylinder" };
    if (Math.min(...along) >= swept.radius) return { side: "front", rule: "cylinder" };
  }
  if (box && (box[2] < swept.box[0] - 1 || swept.box[2] < box[0] - 1 || box[3] < swept.box[1] - 1 || swept.box[3] < box[1] - 1)) return { side: "apart", rule: "never overlap" };
  const hit = separate(points, swept.envelope, [[0, 0, 1], [Math.cos(((P.azimuth ?? 45) * Math.PI) / 180), Math.sin(((P.azimuth ?? 45) * Math.PI) / 180), 0]]);
  if (hit.margin >= -0.08) return { side: dot3(hit.n, V) > 0 ? "behind" : "front", rule: "swept envelope", margin: hit.margin };
  return null;
}

function proveLayers(T, statics, states, scene) {
  const order = T.order.length ? T.order : T.layers.map((layer) => layer.name);
  const rank = new Map(order.map((name, index) => [name, index]));
  const problems = [];
  const proofs = [];
  const swept = new Map(T.layers.map((layer) => [layer.name, sweptGroups(T, layer, states, scene)]));
  for (const layer of T.layers) if (!rank.has(layer.name)) problems.push(`live layer ${layer.name} is missing from T.stack`);
  for (const item of statics) {
    if (!item.shapes || !item.shapes.length) continue;
    const layerName = item.layer;
    if (!rank.has(layerName)) {
      problems.push(`static ${item.name} has layer ${layerName ?? "(none)"}, which is not in T.stack`);
      continue;
    }
    const points = item.shapes.flatMap((shape) => sampleShape(shape, 24));
    const flat = points.map((p) => k.iso(p, T.P));
    const box = [Math.min(...flat.map((p) => p[0])), Math.min(...flat.map((p) => p[1])), Math.max(...flat.map((p) => p[0])), Math.max(...flat.map((p) => p[1]))];
    for (const layer of T.layers) {
      const want = rank.get(layerName) < rank.get(layer.name) ? "behind" : "front";
      for (const group of swept.get(layer.name)) {
        const proof = relation(points, box, group, T.P);
        const label = `${item.name} × ${layer.name}${group.group ? `/${group.group.name}` : ""}`;
        if (!proof) {
          const zs = points.map((p) => p[2]);
          const top = Math.max(...zs);
          const bottom = Math.min(...zs);
          const where = top > group.zmax + 1e-6 ? `at z = ${group.zmax.toFixed(2)}` : bottom < group.zmin - 1e-6 ? `at z = ${group.zmin.toFixed(2)}` : `into a piece behind and a piece in front of ${group.group ? `the ${group.group.name} axis` : "the layer"}`;
          problems.push(`static ${item.name} interleaves live layer ${layer.name}: split ${item.name} ${where}, or move it`);
          continue;
        }
        if (proof.side !== "apart" && proof.side !== want) {
          problems.push(`static ${item.name} is ${proof.side === "behind" ? "behind" : "in front of"} live layer ${layer.name} (${proof.rule}) but its layer ${layerName} is ${want === "behind" ? "under" : "over"} it: move it to a layer ${proof.side === "behind" ? "under" : "over"} ${layer.name}`);
          continue;
        }
        proofs.push({ static: item.name, layer: layer.name, group: group.group ? group.group.name : "world", rule: proof.rule, side: proof.side });
        void label;
      }
    }
  }
  const live = T.layers.slice().sort((a, b) => rank.get(a.name) - rank.get(b.name));
  for (let i = 0; i < live.length; i++)
    for (let j = i + 1; j < live.length; j++)
      for (const lower of swept.get(live[i].name))
        for (const upper of swept.get(live[j].name)) {
          let ok = lower.zmax <= upper.zmin + 1e-6 ? "horizontal plane" : null;
          if (!ok && (lower.box[2] < upper.box[0] - 1 || upper.box[2] < lower.box[0] - 1 || lower.box[3] < upper.box[1] - 1 || upper.box[3] < lower.box[1] - 1)) ok = "never overlap";
          if (!ok) {
            const hit = separate(lower.envelope, upper.envelope, [[0, 0, 1]]);
            if (hit.margin >= -0.08 && dot3(hit.n, G.viewOf(T.P)) > 0) ok = "swept envelope";
          }
          if (!ok) problems.push(`live layer ${live[i].name} interleaves live layer ${live[j].name} (${lower.group ? lower.group.name : "world"} × ${upper.group ? upper.group.name : "world"}): move parts between them or reorder T.stack`);
          else proofs.push({ static: `layer ${live[i].name}`, layer: live[j].name, group: upper.group ? upper.group.name : "world", rule: ok, side: "behind" });
        }
  return { problems, proofs };
}

function build(T, { statics = [], verify = false, poses = null, strict = true, fullClouds = false } = {}) {
  const P = T.P;
  const started = Date.now();
  const states = statesOf(T, poses);
  const groups = groupData(T);
  const parts = T.parts.map(partData);
  const sweep = (layerParts, list) => {
    const data = { P, groups, layers: T.layers.map((layer) => ({ name: layer.name, parts: [], rest: [] })), parts };
    const scene = TURN.prepare(data, { unit: T.scale });
    const frames = scene.runs.map((run) => TURN.frame(run));
    return { scene, frames, data };
  };
  void sweep;
  const routes = T.routes.map((route) => ({ points: route.points.flat().map(r4), r: route.r, spacing: route.spacing, wide: route.wide, hue: route.hue, rings: route.rings }));
  const scratch = { P, groups, layers: T.layers.map((layer) => ({ name: layer.name, parts: [], rest: [] })), parts, routes };
  let scene = TURN.prepare(scratch, { unit: T.scale });
  const slotNeed = parts.map(() => ({ bevel: 0, ribs: 0 }));
  const sampleCams = new Map();
  const camsFor = (group) => {
    const key = group ? group.index : -1;
    if (!sampleCams.has(key)) {
      const list = groupStates(T, group, states, { tumble: true }).map((values) => TURN.cams(scene, values));
      sampleCams.set(key, list);
    }
    return sampleCams.get(key);
  };
  T.parts.forEach((part, index) => {
    const run = scene.runs[index];
    const needs = part.kind === "item" ? false : part.kind === "prism" ? Boolean(part.spec.bevel) : part.kind === "tube" ? Boolean(part.spec.rings) : Boolean(part.details.ribs);
    if (!needs) return;
    const g = part.group ? part.group.index : -1;
    const wide = { ...run, slots: { bevel: 64, ribs: 64 }, role: { ...run.role } };
    for (let j = 0; j < 64; j++) {
      wide.role[`v${j}`] = 1000 + j;
      wide.role[`r${j}`] = 2000 + j;
    }
    const f = TURN.frame(run);
    f.d = new Array(3000).fill("");
    f.o = new Float64Array(3000);
    f.q = new Int32Array(3000);
    f.w = new Float64Array(3000);
    f.t = new Array(3000).fill("");
    f.cx = new Float64Array(3000);
    f.cy = new Float64Array(3000);
    let most = 0;
    for (const cams of camsFor(part.group)) {
      TURN.emit(wide, g >= 0 ? cams[g] : cams.world, f);
      most = Math.max(most, f.levels ?? 0);
    }
    slotNeed[index][part.kind === "prism" ? "bevel" : "ribs"] = Math.max(1, most);
  });
  parts.forEach((data, index) => {
    data.spec = slotNeed[index].bevel || slotNeed[index].ribs ? { ...data.spec, slots: slotNeed[index] } : { ...data.spec };
  });
  scene = TURN.prepare({ ...scratch, parts }, { unit: T.scale });
  const centre = (list) => mul3(list.reduce((s, p) => add3(s, p), [0, 0, 0]), 1 / Math.max(1, list.length));
  const spheres = T.parts.map((part) => {
    const c = centre(part.samples);
    let r = 0;
    for (const p of part.samples) r = Math.max(r, G.len3(sub3(p, c)));
    return { c, r };
  });
  const planes = {};
  const dynamic = {};
  const free = {};
  let stateCamList = null;
  const stateCams = () => stateCamList ?? (stateCamList = states.map((values) => TURN.cams(scene, values)));
  const boxCache = new Map();
  const boxesOf = (part, list, key) => {
    const id = `${part.id}|${key}`;
    if (boxCache.has(id)) return boxCache.get(id);
    const local = thinCloud(part.samples, 40);
    const out = new Float64Array(4 * list.length + 1);
    const steps = [];
    list.forEach((cams, at) => {
      const c = part.group ? cams[part.group.index] : cams.world;
      let x0 = Infinity;
      let y0 = Infinity;
      let x1 = -Infinity;
      let y1 = -Infinity;
      for (const q of local) {
        const x = c.ox + c.m[0] * q[0] + c.m[1] * q[1] + c.m[2] * q[2];
        const y = c.oy + c.m[3] * q[0] + c.m[4] * q[1] + c.m[5] * q[2];
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
      if (at) steps.push(Math.max(Math.abs(x0 - out[4 * at - 4]), Math.abs(y0 - out[4 * at - 3]), Math.abs(x1 - out[4 * at - 2]), Math.abs(y1 - out[4 * at - 1])));
      out.set([x0, y0, x1, y1], 4 * at);
    });
    steps.sort((x, y) => x - y);
    out[4 * list.length] = steps.length ? steps[Math.floor(steps.length * 0.95)] : 0;
    boxCache.set(id, out);
    return out;
  };
  const boxesMeet = (a, b, list, key) => {
    const A = boxesOf(a, list, key);
    const B = boxesOf(b, list, key);
    const pad = 1.5 + A[4 * list.length] + B[4 * list.length];
    for (let i = 0; i < list.length; i++) {
      const o = 4 * i;
      if (A[o] < B[o + 2] + pad && B[o] < A[o + 2] + pad && A[o + 1] < B[o + 3] + pad && B[o + 1] < A[o + 3] + pad) return true;
    }
    return false;
  };
  let framedCount = 0;
  const framedAll = {};
  const groupPairsAll = {};
  const ancestors = (g) => {
    const out = [];
    for (let x = g; x; x = x.parent) out.push(x);
    return out;
  };
  const commonOf = (g1, g2) => {
    const list = ancestors(g1);
    for (let x = g2; x; x = x.parent) if (list.includes(x)) return x;
    return null;
  };
  const sampleStates = () => {
    const all = stateCams();
    const count = Math.min(all.length, 24);
    return Array.from({ length: count }, (_, i) => all[Math.floor((i * all.length) / count)]);
  };
  const toFrame = (dir, from, frame) => {
    if (from === frame) return dir;
    const cams = sampleStates()[0] ?? TURN.cams(scene, {});
    const Rf = from ? cams[from.index].R : [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const w = [Rf[0] * dir[0] + Rf[1] * dir[1] + Rf[2] * dir[2], Rf[3] * dir[0] + Rf[4] * dir[1] + Rf[5] * dir[2], Rf[6] * dir[0] + Rf[7] * dir[1] + Rf[8] * dir[2]];
    const Rc = frame ? cams[frame.index].R : [1, 0, 0, 0, 1, 0, 0, 0, 1];
    return [Rc[0] * w[0] + Rc[3] * w[1] + Rc[6] * w[2], Rc[1] * w[0] + Rc[4] * w[1] + Rc[7] * w[2], Rc[2] * w[0] + Rc[5] * w[1] + Rc[8] * w[2]];
  };
  const cloudCache = new Map();
  const framedCloud = (part, frame) => {
    const key = `${part.id}|${frame ? frame.index : -1}`;
    if (cloudCache.has(key)) return cloudCache.get(key);
    const local = thinCloud(part.samples, 32);
    const out = [];
    for (const cams of sampleStates()) {
      const c = part.group ? cams[part.group.index] : cams.world;
      const f = frame ? cams[frame.index] : cams.world;
      for (const q of local) {
        const p = [c.R[0] * q[0] + c.R[1] * q[1] + c.R[2] * q[2] + c.t[0] - f.t[0], c.R[3] * q[0] + c.R[4] * q[1] + c.R[5] * q[2] + c.t[1] - f.t[1], c.R[6] * q[0] + c.R[7] * q[1] + c.R[8] * q[2] + c.t[2] - f.t[2]];
        out.push([f.R[0] * p[0] + f.R[3] * p[1] + f.R[6] * p[2], f.R[1] * p[0] + f.R[4] * p[1] + f.R[7] * p[2], f.R[2] * p[0] + f.R[5] * p[1] + f.R[8] * p[2]]);
      }
    }
    cloudCache.set(key, out);
    return out;
  };
  const skip = {};
  const problems = [];
  const margins = [];
  const planesByName = new Map();
  const layerData = [];
  for (const layer of T.layers) {
    const members = T.parts.filter((part) => part.layer === layer);
    const plan = [];
    const dyn = [];
    const framed = [];
    const skipped = [];
    const pairsOfGroups = [];
    const settled = new Set();
    const byGroup = new Map();
    for (const part of members) {
      if (part.group && part.group.loose) continue;
      const key = part.group ? part.group.index : -1;
      if (!byGroup.has(key)) byGroup.set(key, []);
      byGroup.get(key).push(part);
    }
    const keys = [...byGroup.keys()];
    if (states.length)
      for (let x = 0; x < keys.length; x++)
        for (let y = x + 1; y < keys.length; y++) {
          const A = byGroup.get(keys[x]);
          const B = byGroup.get(keys[y]);
          const ga = keys[x] >= 0 ? T.groups[keys[x]] : null;
          const gb = keys[y] >= 0 ? T.groups[keys[y]] : null;
          const common = commonOf(ga, gb);
          const ca = A.flatMap((part) => framedCloud(part, common).filter((_, i) => fullClouds || i % 2 === 0));
          const cb = B.flatMap((part) => framedCloud(part, common).filter((_, i) => fullClouds || i % 2 === 0));
          const hit = quickSeparate(ca, cb, [[0, 0, 1], ...A.flatMap((part) => part.hints).slice(0, 6).map((h) => toFrame(h, ga, common)), ...B.flatMap((part) => part.hints).slice(0, 6).map((h) => toFrame(h, gb, common))]);
          if (hit.margin < 0.1) continue;
          pairsOfGroups.push(keys[x], keys[y], common ? common.index : -1, hit.n[0], hit.n[1], hit.n[2]);
          settled.add(`${keys[x]}|${keys[y]}`);
          settled.add(`${keys[y]}|${keys[x]}`);
        }
    groupPairsAll[layer.name] = pairsOfGroups;
    for (let x = 0; x < members.length; x++)
      for (let y = x + 1; y < members.length; y++) {
        const a = members[x];
        const b = members[y];
        if (a.route && a.route === b.route && Math.abs(a.chunk - b.chunk) <= 1) {
          skipped.push(a.id, b.id);
          continue;
        }
        const sa = spheres[a.id];
        const sb = spheres[b.id];
        if (a.group === b.group) {
          const g = a.group ? a.group.index : -1;
          const reach = G.len3(sub3(sb.c, sa.c)) + sa.r + sb.r;
          let near = false;
          void reach;
          if (a.group && a.group.loose) near = G.len3(sub3(sb.c, sa.c)) < sa.r + sb.r + 1;
          else near = boxesMeet(a, b, camsFor(a.group), `g${g}`);
          if (!near) continue;
          const hit = separate(a.samples, b.samples, [[0, 0, 1], ...a.hints, ...b.hints]);
          margins.push({ a: a.name, b: b.name, margin: hit.margin });
          const o = a.group ? a.group.origin : [0, 0, 0];
          planesByName.set(`${a.name}|${b.name}`, { n: hit.n, d: hit.d + dot3(hit.n, o) });
          if (hit.margin < -0.08) {
            problems.push(`interlock ${a.name} × ${b.name} ${hit.margin.toFixed(2)}: split, flats or a flush contact`);
            if (strict) continue;
          }
          plan.push(a.id, b.id, hit.n[0], hit.n[1], hit.n[2]);
        } else if ((a.group && a.group.loose) || (b.group && b.group.loose)) {
          continue;
        } else if (settled.has(`${a.group ? a.group.index : -1}|${b.group ? b.group.index : -1}`)) {
          continue;
        } else {
          const near = !states.length || boxesMeet(a, b, stateCams(), "states");
          if (!near) continue;
          const common = commonOf(a.group, b.group);
          const ca = framedCloud(a, common);
          const cb = framedCloud(b, common);
          const hit = quickSeparate(ca, cb, [[0, 0, 1], ...a.hints, ...b.hints].map((h) => toFrame(h, a.group, common)).concat([[0, 0, 1], ...b.hints.map((h) => toFrame(h, b.group, common))]));
          if (hit.margin >= 0.1) {
            framed.push(a.id, b.id, common ? common.index : -1, hit.n[0], hit.n[1], hit.n[2]);
            framedCount++;
          } else dyn.push(a.id, b.id);
        }
      }
    planes[layer.name] = plan;
    dynamic[layer.name] = dyn;
    framedAll[layer.name] = framed;
    free[layer.name] = members.filter((part) => part.group && part.group.loose).map((part) => part.id);
    skip[layer.name] = skipped;
    layerData.push({ name: layer.name, parts: members.map((part) => part.id), rest: [] });
  }
  if (problems.length && strict) throw new Error(`turning: ${problems.length} pair${problems.length === 1 ? "" : "s"} with no separating plane\n  ${problems.join("\n  ")}`);
  const layered = proveLayers(T, statics, states, scene);
  if (layered.problems.length) throw new Error(`turning: ${layered.problems.length} layer problem${layered.problems.length === 1 ? "" : "s"}\n  ${layered.problems.join("\n  ")}`);
  const values = Object.fromEntries(T.groups.map((g) => [g.name, g.kind === "free" ? null : 0]));
  const data = { P, groups, layers: layerData, parts, routes, planes: packPairs(planes, 5, false), framed: packPairs(framedAll, 6, true), groupPairs: pack(groupPairsAll), dynamic, free, skip, values };
  const clouds = T.parts.map((part) => thinCloud(part.samples, 64));
  data.parts.forEach((p, index) => {
    if (p.kind === "fixed") p.cloud = clouds[index].flat().map(r4);
  });
  const packed = { ...pack({ ...data, parts: [] }), parts: pack3(data.parts) };
  scene = TURN.prepare(packed, { unit: T.scale });
  const cams = TURN.cams(scene, values);
  const frames = scene.runs.map((run) => TURN.emit(run, run.group >= 0 ? cams[run.group] : cams.world, TURN.frame(run)));
  const V = G.viewOf(P);
  scene.layers.forEach((layer, index) => {
    const provisional = Int32Array.from(Array.from(layer.parts).sort((i, j) => dot3(spheres[i].c, V) - dot3(spheres[j].c, V)));
    const order = TURN.order(layer, frames, cams, provisional, {}, scene);
    packed.layers[index].rest = Array.from(order);
    layer.rest = order;
  });
  const markup = {};
  scene.layers.forEach((layer, index) => {
    markup[layer.name] = `<g class="iso-parts" data-layer="${esc(layer.name)}">${packed.layers[index].rest.map((i) => partMarkup(T.parts[i], scene.runs[i], frames[i])).join("")}</g>`;
  });
  const report = {
    parts: T.parts.length,
    groups: T.groups.length,
    layers: T.layers.map((layer) => ({ name: layer.name, parts: packed.layers[layer.index].parts.length, planar: planes[layer.name].length / 5, framed: framedAll[layer.name].length / 6, groupPairs: groupPairsAll[layer.name].length / 6, dynamic: dynamic[layer.name].length / 2, skip: skipped(skip[layer.name]) })),
    tight: margins.filter((m) => m.margin < 0).sort((a, b) => a.margin - b.margin),
    margins,
    proofs: layered.proofs,
    seconds: (Date.now() - started) / 1000,
  };
  const out = { svg: markup, data: packed, css: TURN_CSS, report, scene, frames, planesByName };
  out.layerSvg = (name, { width, height, label = "" }) => `<svg xmlns="http://www.w3.org/2000/svg" class="iso-svg iso-live iso-layer" data-live="${esc(name)}" viewBox="0 0 ${width} ${height}" aria-hidden="true"${label ? ` aria-label="${esc(label)}"` : ""}>${markup[name]}</svg>`;
  if (verify) out.verify = verifyData(T, statics);
  T.built = out;
  T.data = packed;
  return out;
}

const skipped = (list) => list.length / 2;

function attrs(pairs) {
  return Object.entries(pairs)
    .filter(([, v]) => v !== undefined && v !== null && v !== false)
    .map(([key, v]) => ` ${key}="${esc(v)}"`)
    .join("");
}

function roleMarkup(part, run, f, r) {
  const role = run.roles[r];
  if (role.type === "path") return `<path${attrs({ class: role.cls, "data-shade": role.shade === undefined ? undefined : String(role.shade), "data-tone": role.tone, "data-r": role.name, d: f.d[r] })}/>`;
  if (role.type === "face") {
    const q = f.q[r];
    const band = Math.min(3, Math.floor(q / TURN.q));
    const mix = (q - TURN.q * band) / TURN.q;
    const style = [mix ? `--f:${mix}` : "", f.w[r] >= 0 ? `stroke-width:${f.w[r]}` : ""].filter(Boolean).join(";");
    return `<path${attrs({ class: "iso-tn", "data-r": role.name, "data-b": band ? String(band) : undefined, style: style || undefined, d: f.d[r] })}/>`;
  }
  if (role.type === "slot") return `<path${attrs({ class: role.cls, "data-tone": role.tone, "data-r": role.name, opacity: String(Math.round(f.o[r] * 1000) / 1000), d: f.d[r] })}/>`;
  if (role.type === "xform") return `<path${attrs({ class: role.cls, "data-r": role.name, transform: f.t[r], d: part.planPath })}/>`;
  if (role.type === "plane") {
    const index = Number(role.name.slice(5));
    return `<g${attrs({ "data-r": role.name, transform: f.t[r], opacity: String(f.o[r]) })}>${part.planes[index].svg}</g>`;
  }
  return "";
}

function dotsMarkup(part, run, f) {
  let out = "";
  let open = null;
  run.roles.forEach((role, r) => {
    if (role.type !== "dot") return;
    if (open !== role.tone) {
      if (open !== null) out += "</g>";
      out += `<g class="iso-dots" data-tone="${role.tone}">`;
      open = role.tone;
    }
    out += `<circle data-r="${role.name}" cx="${Math.round(f.cx[r] * 100) / 100}" cy="${Math.round(f.cy[r] * 100) / 100}" r="${role.size}" fill-opacity="${f.o[r]}"/>`;
  });
  return open !== null ? out + "</g>" : out;
}

function partMarkup(part, run, f) {
  if (part.kind === "item") return `<g class="iso-item" data-p="${run.index}" data-name="${esc(part.name)}"></g>`;
  const head = (cls, extra = {}) => `<g${attrs({ class: cls, "data-p": String(run.index), "data-name": part.name, "data-mat": part.material ?? undefined, "data-lit": part.lit ? "" : undefined, ...extra })}>`;
  const inner = run.roles.map((role, r) => (role.type === "dot" || role.type === "self" ? "" : roleMarkup(part, run, f, r))).join("");
  if (part.kind === "tube") return head("tb", { "data-hue": part.hue ?? undefined }) + inner + "</g>";
  if (part.kind === "fixed") {
    const at = part.svg.lastIndexOf('<path class="iso-line iso-edge"');
    const body = at >= 0 ? part.svg.slice(0, at) + inner + part.svg.slice(at) : part.svg + inner;
    return head("iso-part", { transform: f.t[run.role.move] || undefined }) + body + dotsMarkup(part, run, f) + "</g>";
  }
  return head("iso-solid", { "data-seamless": part.kind === "round" || part.kind === "ball" || part.kind === "lathe" ? "" : undefined }) + inner + dotsMarkup(part, run, f) + "</g>";
}

function verifyData(T, statics = []) {
  const order = T.order.length ? T.order : T.layers.map((layer) => layer.name);
  const live = new Set(T.layers.map((layer) => layer.name));
  const staticLayers = order.filter((name) => !live.has(name));
  const liveShapes = new Set(T.parts.flatMap((part) => part.shapes));
  return {
    parts: T.parts.map((part) => ({ name: part.name, group: part.group ? part.group.index : -1, layer: part.layer.index, shapes: part.shapes.map((shape) => localShape(shape, part.group)) })),
    statics: statics.filter((item) => item.shapes && item.shapes.length && staticLayers.includes(item.layer)).map((item) => ({ name: item.name, rank: staticLayers.indexOf(item.layer), shapes: item.shapes.filter((shape) => !liveShapes.has(shape)).map((shape) => localShape(shape, null)) })),
  };
}

function localShape(shape, group) {
  const o = group ? group.origin : [0, 0, 0];
  const out = { kind: shape.kind };
  if (shape.F) out.F = { o: sub3(shape.F.o, o), a: shape.F.a, u: shape.F.u, v: shape.F.v };
  for (const key of ["poly", "s0", "s1", "r", "radii", "flats"]) if (shape[key] !== undefined) out[key] = shape[key];
  if (shape.o && shape.kind !== "body" && shape.kind !== "prism") out.o = sub3(shape.o, o);
  if (shape.points) out.points = shape.points.map((p) => sub3(p, o));
  if (shape.axes) out.axes = shape.axes;
  if (shape.half) out.half = shape.half;
  return pack(out);
}

export { pack, sampleShape };
