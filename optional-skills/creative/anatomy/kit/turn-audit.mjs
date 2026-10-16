import { TURN } from "./turn.mjs";
import * as A from "./audit.mjs";
import * as G from "./lathe.mjs";
import * as k from "./iso-kit.mjs";
import { separate, sampleShape } from "./turn-build.mjs";

const TAU = Math.PI * 2;
const EYE = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const f2 = (v) => (Math.round(v * 100) / 100).toFixed(2);
const f3 = (v) => (Math.round(v * 1000) / 1000).toFixed(3);

const apply = (R, v) => [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
const back = (R, v) => [R[0] * v[0] + R[3] * v[1] + R[6] * v[2], R[1] * v[0] + R[4] * v[1] + R[7] * v[2], R[2] * v[0] + R[5] * v[1] + R[8] * v[2]];

const place = (pose, p) => {
  const q = apply(pose.R, [p[0] - pose.origin[0], p[1] - pose.origin[1], p[2] - pose.origin[2]]);
  return [q[0] + pose.t[0], q[1] + pose.t[1], q[2] + pose.t[2]];
};

const unplace = (pose, p) => {
  const q = back(pose.R, [p[0] - pose.t[0], p[1] - pose.t[1], p[2] - pose.t[2]]);
  return [q[0] + pose.origin[0], q[1] + pose.origin[1], q[2] + pose.origin[2]];
};

export function posed(shape, pose) {
  if (!pose) return shape;
  const out = { ...shape, _b: undefined, _f: undefined };
  if (shape.F) out.F = { o: place(pose, shape.F.o), a: apply(pose.R, shape.F.a), u: shape.F.u ? apply(pose.R, shape.F.u) : undefined, v: shape.F.v ? apply(pose.R, shape.F.v) : undefined };
  if (shape.kind === "ball" || shape.kind === "box") out.o = place(pose, shape.o);
  if (shape.flats) out.flats = shape.flats.map(([nx, ny, nz, d]) => [...apply(pose.R, [nx, ny, nz]), d]);
  if (shape.axes) out.axes = shape.axes.map((axis) => apply(pose.R, axis));
  if (shape.points) out.points = shape.points.map((p) => place(pose, p));
  if (shape.tips) out.tips = shape.tips.map((tip) => tip && place(pose, tip));
  if (shape.tipAxes) out.tipAxes = shape.tipAxes.map((axis) => axis && apply(pose.R, axis));
  if (shape.cut) out.cut = Array.isArray(shape.cut) ? shape.cut.map((one) => posed(one, pose)) : posed(shape.cut, pose);
  return out;
}

const IDENTITY = { R: EYE, t: [0, 0, 0], origin: [0, 0, 0] };
const poseFor = (cams, group) => (group && group !== "static" ? { R: cams[group.index].R, t: cams[group.index].t, origin: group.origin } : IDENTITY);

export function eventsOf(T, layer) {
  const data = T.data;
  const name = typeof layer === "string" ? layer : layer.name;
  const prepared = TURN.prepare(data, { unit: 1 }).layers.find((l) => l.name === name);
  const plan = prepared ? prepared.planar : [];
  const a = ((T.P.azimuth ?? 45) * Math.PI) / 180;
  const e = ((T.P.elevation ?? 30) * Math.PI) / 180;
  const out = new Set();
  for (let i = 0; i < plan.length; i += 5) {
    const part = T.parts[plan[i]];
    const g = part.group;
    if (!g || g.kind !== "turn" || g.parent) continue;
    const n = [plan[i + 2], plan[i + 3], plan[i + 4]];
    const flat = Math.hypot(n[0], n[1]);
    if (flat < 1e-9) continue;
    const c = (-n[2] * Math.sin(e)) / (flat * Math.cos(e));
    if (c <= -1 || c >= 1) continue;
    const phi = Math.atan2(n[1], n[0]);
    for (const b of [phi + Math.acos(c), phi - Math.acos(c)]) {
      const theta = ((((a - b) * 180) / Math.PI) % 360 + 360) % 360;
      out.add(Math.round(theta * 1e6) / 1e6);
    }
  }
  return [...out].sort((x, y) => x - y);
}

function polygonArea(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    s += a[0] * b[1] - b[0] * a[1];
  }
  return s / 2;
}

function clipArea(subject, clip) {
  if (subject.length < 3 || clip.length < 3) return 0;
  const sign = polygonArea(clip) >= 0 ? 1 : -1;
  let out = subject;
  for (let i = 0; i < clip.length && out.length; i++) {
    const a = clip[i];
    const b = clip[(i + 1) % clip.length];
    const side = (p) => sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]));
    const input = out;
    out = [];
    for (let j = 0; j < input.length; j++) {
      const p = input[j];
      const q = input[(j + 1) % input.length];
      const sp = side(p);
      const sq = side(q);
      if (sp >= 0) out.push(p);
      if ((sp >= 0) !== (sq >= 0)) {
        const t = sp / (sp - sq);
        out.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
      }
    }
  }
  return out.length >= 3 ? Math.abs(polygonArea(out)) : 0;
}

const pathPolys = (d) =>
  [...(d ?? "").matchAll(/M([^M]*)/g)].map(([, body]) => {
    const nums = (body.match(/-?\d+(?:\.\d+)?(?:e-?\d+)?/g) ?? []).map(Number);
    const pts = [];
    for (let i = 0; i + 1 < nums.length; i += 2) pts.push([nums[i], nums[i + 1]]);
    return pts;
  });

function convexHull(points) {
  const sorted = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (sorted.length < 3) return sorted;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function footprints(run, f) {
  if (run.kind === "tube") return f.quadList.map(([buf, n]) => convexHull(Array.from({ length: n }, (_, i) => [buf[2 * i], buf[2 * i + 1]])));
  if (run.kind === "fixed") return [Array.from({ length: f.hullN }, (_, i) => [f.hull[2 * i], f.hull[2 * i + 1]])];
  const polys = pathPolys(f.d[run.role.fill]);
  return polys.length ? [convexHull(polys.flat())] : [];
}

function overlapArea(runA, fa, runB, fb) {
  if (fa.box[2] < fb.box[0] || fb.box[2] < fa.box[0] || fa.box[3] < fb.box[1] || fb.box[3] < fa.box[1]) return 0;
  let best = 0;
  for (const a of footprints(runA, fa)) for (const b of footprints(runB, fb)) best = Math.max(best, clipArea(a, b));
  return best;
}

function hausdorffish(A, B) {
  if (!A.length || !B.length) return 0;
  const seg = (p, a, b) => {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const l = dx * dx + dy * dy;
    const t = l ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l)) : 0;
    return Math.hypot(p[0] - a[0] - dx * t, p[1] - a[1] - dy * t);
  };
  const one = (P, Q) => {
    let worst = 0;
    const m = Q.length;
    const n = P.length;
    for (let i = 0; i < n; i++) {
      const centre = Math.round((i / n) * m);
      let best = Infinity;
      for (let w = -10; w <= 10; w++) {
        const j = (((centre + w) % m) + m) % m;
        best = Math.min(best, seg(P[i], Q[j], Q[(j + 1) % m]));
      }
      if (best > 0.02) for (let j = 0; j < m; j++) best = Math.min(best, seg(P[i], Q[j], Q[(j + 1) % m]));
      worst = Math.max(worst, best);
    }
    return worst;
  };
  return Math.max(one(A, B), one(B, A));
}

const areaOfPath = (d) => pathPolys(d).reduce((sum, poly) => sum + Math.abs(polygonArea(poly)), 0);

function flatPolys(d) {
  const out = [];
  for (const [, body] of (d ?? "").matchAll(/M([^M]*)/g)) {
    const nums = (body.match(/-?\d+(?:\.\d+)?(?:e-?\d+)?/g) ?? []).map(Number);
    if (nums.length >= 6) out.push(nums);
  }
  return out;
}

const jitter = (a, b) => {
  const t = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453;
  return t - Math.floor(t);
};

function toneRaster(layers, box, scale) {
  const x0 = box[0];
  const y0 = box[1];
  const w = Math.max(1, Math.ceil((box[2] - x0) * scale));
  const h = Math.max(1, Math.ceil((box[3] - y0) * scale));
  const out = new Float32Array(w * h).fill(-1);
  const owner = new Int16Array(w * h).fill(-1);
  for (const { polys, v, id } of layers) {
    if (!polys.length) continue;
    let ymin = Infinity;
    let ymax = -Infinity;
    for (const p of polys) for (let i = 1; i < p.length; i += 2) {
      if (p[i] < ymin) ymin = p[i];
      if (p[i] > ymax) ymax = p[i];
    }
    const r0 = Math.max(0, Math.floor((ymin - y0) * scale));
    const r1 = Math.min(h - 1, Math.ceil((ymax - y0) * scale));
    const hits = [];
    for (let row = r0; row <= r1; row++)
      for (let sub = 0; sub < 8; sub++) {
      const y = y0 + (row + (sub + 0.5) / 8) / scale;
      hits.length = 0;
      for (const p of polys) {
        const n = p.length;
        for (let i = 0; i < n; i += 2) {
          const ax = p[i];
          const ay = p[i + 1];
          const bx = p[(i + 2) % n];
          const by = p[(i + 3) % n];
          if ((ay <= y) === (by <= y)) continue;
          hits.push([ax + ((y - ay) * (bx - ax)) / (by - ay), by > ay ? 1 : -1]);
        }
      }
      if (hits.length < 2) continue;
      hits.sort((a, b) => a[0] - b[0]);
      let winding = 0;
      for (let k = 0; k < hits.length - 1; k++) {
        winding += hits[k][1];
        if (!winding) continue;
        const xa = hits[k][0];
        const xb = hits[k + 1][0];
        const c0 = Math.max(0, Math.floor((xa - x0) * scale));
        const c1 = Math.min(w - 1, Math.floor((xb - x0) * scale));
        for (let c = c0; c <= c1; c++) {
          if (Math.floor(jitter(c, row) * 8) !== sub) continue;
          if (c === c0 || c === c1) {
            const x = x0 + (c + jitter(row, c)) / scale;
            if (x < xa || x >= xb) continue;
          }
          out[row * w + c] = v;
          owner[row * w + c] = id;
        }
      }
    }
  }
  return { out, owner };
}

function toneLayers(run, f) {
  const layers = [];
  run.roles.forEach((role, r) => {
    const d = f.d[r];
    if (!d) return;
    if (role.type === "face") layers.push({ polys: flatPolys(d), v: f.q[r] / TURN.q, id: r });
    else if (role.type === "path" && role.name === "fill") layers.push({ polys: flatPolys(d), v: 2, id: r });
    else if (role.type === "path" && /^s[0-4]$/.test(role.name)) layers.push({ polys: flatPolys(d), v: Number(role.name.slice(1)), id: r });
    else if (role.name === "body") layers.push({ polys: flatPolys(d), v: 2, id: r });
    else if (role.name === "shine") layers.push({ polys: flatPolys(d), v: 4, id: r });
    else if (role.name === "shade") layers.push({ polys: flatPolys(d), v: 0, id: r });
  });
  return layers;
}

function visibleChange(run, frames, scale = 4) {
  const box = [Infinity, Infinity, -Infinity, -Infinity];
  for (const f of frames) {
    box[0] = Math.min(box[0], f.box[0] - 1);
    box[1] = Math.min(box[1], f.box[1] - 1);
    box[2] = Math.max(box[2], f.box[2] + 1);
    box[3] = Math.max(box[3], f.box[3] + 1);
  }
  const rasters = frames.map((f) => toneRaster(toneLayers(run, f), box, scale));
  const steps = [];
  for (let i = 1; i < rasters.length; i++) {
    const a = rasters[i - 1];
    const b = rasters[i];
    let sum = 0;
    for (let j = 0; j < a.out.length; j++) {
      if (a.owner[j] === b.owner[j]) continue;
      const va = a.out[j];
      const vb = b.out[j];
      if (va < 0 || vb < 0) sum += 2;
      else {
        const d = Math.abs(va - vb);
        if (d > 1 / 16 + 1e-6) sum += d;
      }
    }
    steps.push(sum / (scale * scale));
  }
  return steps;
}

function surfacePoints(shape, spacing = 0.6) {
  return sampleShape(shape, Math.max(16, Math.round(TAU / Math.max(0.05, spacing / Math.max(1, shape.r ?? 3)))));
}

function sphereOf(points) {
  const c = [0, 0, 0];
  for (const p of points) for (let i = 0; i < 3; i++) c[i] += p[i] / Math.max(1, points.length);
  let r = 0;
  for (const p of points) r = Math.max(r, Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]));
  return { c, r };
}

function familiesOf(T, { sweep, clips }) {
  const groups = T.groups;
  const out = [];
  const roots = groups.filter((g) => g.kind === "turn" && !g.parent);
  if (sweep) out.push({ name: "sweep", frame: 0.3, periodic: true, unit: "°", ...sweep });
  else if (roots.length) out.push({ name: "sweep", at: (x) => Object.fromEntries(roots.map((g) => [g.name, x])), from: 0, to: 360, frame: 0.3, periodic: true, unit: "°" });
  for (const clip of clips ?? []) out.push({ from: 0, frame: 1 / 60, unit: "s", ...clip });
  return out;
}

function samplesOf(family, frame) {
  const xs = [];
  const count = Math.max(1, Math.round((family.to - family.from) / frame));
  for (let i = 0; i < count + (family.periodic ? 0 : 1); i++) xs.push(family.from + i * frame);
  return xs;
}

const ORBIT_LINES = { separation: null, unseparated: "order", cycles: "order", forced: "order", swaps: "order", layers: "depth", rigid: "rigid", swept: "swept", posed: "posed", folds: "routes", self: "routes", crowding: "routes", endon: "routes", coverage: "coverage", depth: "depth", continuity: "continuity" };
const ORBIT_CHECKS = ["order", "rigid", "swept", "posed", "routes", "coverage", "continuity", "depth"];

export function orbit(T, R, P, options = {}) {
  const started = Date.now();
  const { sweep = null, clips = [], poses = null, random = 2000, step = 0.05, crowd = [], cover = [], fittings = [], depthAngles = 36, depthEvents = 36, quick = false, log = null, routeStates = 240, posedStep = null, only = null } = options;
  const say = (text) => log && log(text);
  const unknown = (only ?? []).filter((name) => !ORBIT_CHECKS.includes(name));
  if (unknown.length) throw new Error(`orbit: unknown check ${unknown.join(", ")} in only (the checks are ${ORBIT_CHECKS.join(", ")})`);
  const wants = (name) => !only || only.includes(name);
  const data = T.data;
  const scene = TURN.prepare(data, { unit: 1 });
  const runs = scene.runs;
  const groups = T.groups;
  const E = ((P.elevation ?? 30) * Math.PI) / 180;
  const result = { separation: [], unseparated: [], cycles: [], forced: [], swaps: [], layers: [], rigid: [], swept: [], posed: [], folds: [], self: [], crowding: [], endon: [], coverage: [], depth: [], continuity: [], notes: [], stats: {}, quick, skipped: Object.keys(ORBIT_LINES).filter((line) => ORBIT_LINES[line] && !wants(ORBIT_LINES[line])) };
  const families = familiesOf(T, { sweep, clips });
  const extra = typeof poses === "function" ? poses() : Array.isArray(poses) ? poses : [];
  const frameOf = (family) => (quick ? family.frame * (family.periodic ? Math.max(1, 1 / family.frame) : 3) : family.frame);
  const label = (family, x) => `${family.name} ${x.toFixed(3)}${family.unit ?? ""}`;
  const emitAll = (values, frames = runs.map((run) => TURN.frame(run))) => {
    const cams = TURN.cams(scene, values);
    for (const run of runs) TURN.emit(run, run.group >= 0 ? cams[run.group] : cams.world, frames[run.index]);
    return { cams, frames };
  };
  say("separation");
  const allMargins = T.built?.report?.margins ?? [];
  for (const m of allMargins) if (m.margin < -0.08) result.separation.push({ a: m.a, b: m.b, margin: m.margin });
  result.stats.planes = allMargins.length;
  result.stats.tightest = allMargins.length ? Math.min(...allMargins.map((m) => m.margin)) : 0;
  const relationOf = (layer, i, j, cams) => {
    const plan = layer.planar;
    for (let p = 0; p < plan.length; p += 5)
      if ((plan[p] === i && plan[p + 1] === j) || (plan[p] === j && plan[p + 1] === i)) {
        const g = runs[plan[p]].group;
        const V = g >= 0 ? cams[g].V : cams.world.V;
        const s = plan[p + 2] * V[0] + plan[p + 3] * V[1] + plan[p + 4] * V[2];
        return plan[p] === i ? s : -s;
      }
    const a = runs[i];
    const b = runs[j];
    const ca = a.group >= 0 ? cams[a.group] : cams.world;
    const cb = b.group >= 0 ? cams[b.group] : cams.world;
    const hit = TURN.gjk(a.cloud, ca, b.cloud, cb, null, 32, TURN.hintsOf(a, ca, b, cb, cams.world));
    const V = cams.world.V;
    return hit.n[0] * V[0] + hit.n[1] * V[1] + hit.n[2] * V[2];
  };
  const pairsOf = (layer) => {
    if (layer._pairs) return layer._pairs;
    const list = [];
    const seen = new Set();
    for (let p = 0; p < layer.planar.length; p += 5) list.push([layer.planar[p], layer.planar[p + 1]]);
    for (let p = 0; p < layer.dynamic.length; p += 2) list.push([layer.dynamic[p], layer.dynamic[p + 1]]);
    for (let p = 0; p < (layer.framed ? layer.framed.length : 0); p += 6) list.push([layer.framed[p], layer.framed[p + 1]]);
    for (const gp of layer.groupPairs ?? []) for (const i of gp.a) for (const j of gp.b) list.push([i, j]);
    for (const [i, j] of list) seen.add(i * 65536 + j);
    for (const i of layer.free) for (const j of layer.parts) {
      if (i === j || runs[i].group === runs[j].group) continue;
      const key = Math.min(i, j) * 65536 + Math.max(i, j);
      if (seen.has(key)) continue;
      seen.add(key);
      list.push([Math.min(i, j), Math.max(i, j)]);
    }
    layer._pairs = list;
    return list;
  };
  const checkState = (values, tag, previous, record) => {
    const { cams, frames } = emitAll(values);
    const orders = [];
    scene.layers.forEach((layer, index) => {
      const stats = {};
      const order = TURN.order(layer, frames, cams, previous ? previous[index] : null, stats, scene);
      orders.push(order);
      if (stats.forced) result.forced.push({ layer: layer.name, state: tag, forced: stats.forced });
      const framed = layer.framed ?? [];
      for (let p = 0; p < framed.length; p += 6) {
        const i = framed[p];
        const j = framed[p + 1];
        if (TURN.disjoint(frames[i], frames[j], 0.1)) continue;
        const g = framed[p + 2];
        const R = g >= 0 ? cams[g].R : EYE;
        const n = [R[0] * framed[p + 3] + R[1] * framed[p + 4] + R[2] * framed[p + 5], R[3] * framed[p + 3] + R[4] * framed[p + 4] + R[5] * framed[p + 5], R[6] * framed[p + 3] + R[7] * framed[p + 4] + R[8] * framed[p + 5]];
        const extent = (run) => {
          const c = run.group >= 0 ? cams[run.group] : cams.world;
          let lo = Infinity;
          let hi = -Infinity;
          for (let q = 0; q < run.cloud.length; q += 3) {
            const x = run.cloud[q];
            const y = run.cloud[q + 1];
            const z = run.cloud[q + 2];
            const v = n[0] * (c.R[0] * x + c.R[1] * y + c.R[2] * z + c.t[0]) + n[1] * (c.R[3] * x + c.R[4] * y + c.R[5] * z + c.t[1]) + n[2] * (c.R[6] * x + c.R[7] * y + c.R[8] * z + c.t[2]);
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
          return [lo, hi];
        };
        const margin = extent(runs[j])[0] - extent(runs[i])[1];
        if (margin < -0.08) result.unseparated.push({ layer: layer.name, state: tag, a: runs[i].name, b: runs[j].name, margin, framed: true });
      }
      for (const gp of layer.groupPairs ?? []) {
        const R = gp.frame >= 0 ? cams[gp.frame].R : EYE;
        const nw = [R[0] * gp.n[0] + R[1] * gp.n[1] + R[2] * gp.n[2], R[3] * gp.n[0] + R[4] * gp.n[1] + R[5] * gp.n[2], R[6] * gp.n[0] + R[7] * gp.n[1] + R[8] * gp.n[2]];
        const along = (list, high) => {
          let best = high ? -Infinity : Infinity;
          for (const index of list) {
            const run = runs[index];
            const c = run.group >= 0 ? cams[run.group] : cams.world;
            for (let q = 0; q < run.cloud.length; q += 3) {
              const x = run.cloud[q];
              const y = run.cloud[q + 1];
              const z = run.cloud[q + 2];
              const v = nw[0] * (c.R[0] * x + c.R[1] * y + c.R[2] * z + c.t[0]) + nw[1] * (c.R[3] * x + c.R[4] * y + c.R[5] * z + c.t[1]) + nw[2] * (c.R[6] * x + c.R[7] * y + c.R[8] * z + c.t[2]);
              best = high ? Math.max(best, v) : Math.min(best, v);
            }
          }
          return best;
        };
        const margin = along(gp.b, false) - along(gp.a, true);
        if (margin < -0.08) result.unseparated.push({ layer: layer.name, state: tag, a: runs[gp.a[0]].name, b: runs[gp.b[0]].name, margin, framed: true });
      }
      const n = layer.parts.length;
      const local = layer.local;
      const adj = Array.from({ length: n }, () => []);
      const indeg = new Int32Array(n);
      for (const [i, j] of pairsOf(layer)) {
        if (TURN.disjoint(frames[i], frames[j], 0.1)) continue;
        const ri = runs[i];
        const rj = runs[j];
        if (stats.unseparated && ri.group !== rj.group) {
          const ca = ri.group >= 0 ? cams[ri.group] : cams.world;
          const cb = rj.group >= 0 ? cams[rj.group] : cams.world;
          const hit = TURN.gjk(ri.cloud, ca, rj.cloud, cb, null, 32, TURN.hintsOf(ri, ca, rj, cb, cams.world));
          if (hit.margin < -0.08) result.unseparated.push({ layer: layer.name, state: tag, a: ri.name, b: rj.name, margin: hit.margin });
        }
        const s = relationOf(layer, i, j, cams);
        if (s < 1e-7 && s > -1e-7) continue;
        const from = s > 0 ? local.get(i) : local.get(j);
        const to = s > 0 ? local.get(j) : local.get(i);
        adj[from].push(to);
        indeg[to]++;
      }
      const queue = [];
      for (let v = 0; v < n; v++) if (!indeg[v]) queue.push(v);
      let seen = 0;
      while (queue.length) {
        const v = queue.pop();
        seen++;
        for (const w of adj[v]) if (--indeg[w] === 0) queue.push(w);
      }
      if (seen < n) {
        const stuck = [];
        for (let v = 0; v < n; v++) if (indeg[v] > 0) stuck.push(runs[layer.parts[v]].name);
        let split = null;
        for (const m of T.built?.report?.margins ?? []) if (stuck.includes(m.a) && stuck.includes(m.b) && (!split || m.margin > split.margin)) split = m;
        const plane = split && T.built.planesByName ? T.built.planesByName.get(`${split.a}|${split.b}`) : null;
        result.cycles.push({ layer: layer.name, state: tag, parts: stuck.slice(0, 8), split: plane ? { part: split.b, a: split.a, b: split.b, n: plane.n, d: plane.d } : null });
      }
    });
    if (record) record(cams, frames, orders);
    return { cams, frames, orders };
  };
  const swapsOf = (a, b) => {
    const list = [];
    scene.layers.forEach((layer, index) => {
      const pa = a.orders[index];
      const pb = b.orders[index];
      const rankA = new Map();
      pa.forEach((g, at) => rankA.set(g, at));
      const rankB = new Map();
      pb.forEach((g, at) => rankB.set(g, at));
      const moved = new Set();
      for (let at = 0; at < pb.length; at++) if (pa[at] !== pb[at]) moved.add(pb[at]);
      const moving = [...moved];
      for (let x = 0; x < moving.length; x++)
        for (const j of layer.parts) {
          const i = moving[x];
          if (i === j || (moved.has(j) && j < i)) continue;
          if ((rankA.get(i) < rankA.get(j)) === (rankB.get(i) < rankB.get(j))) continue;
          if (T.parts[i].route && T.parts[i].route === T.parts[j].route && Math.abs(T.parts[i].chunk - T.parts[j].chunk) <= 1) continue;
          const area = Math.min(overlapArea(runs[i], a.frames[i], runs[j], a.frames[j]), overlapArea(runs[i], b.frames[i], runs[j], b.frames[j]));
          list.push({ layer: layer.name, i, j, area });
        }
    });
    return list;
  };
  const clone = (frames) => frames.map((f) => ({ ...f, d: f.d.slice(), box: Float64Array.from(f.box), hull: Float64Array.from(f.hull), quadList: f.quadList, fill: undefined }));
  const flush = new Set();
  for (const m of T.built?.report?.margins ?? []) if (Math.abs(m.margin) < 0.05) flush.add(`${m.a}|${m.b}`);
  const flushOf = (a, b) => flush.has(`${a}|${b}`) || flush.has(`${b}|${a}`);
  const changes = [];
  let orderStates = 0;
  for (const family of wants("order") ? families : []) {
    const xs = samplesOf(family, frameOf(family));
    say(`order scan ${family.name}: ${xs.length} states`);
    let previous = null;
    const local = [];
    for (const x of xs) {
      const values = family.at(x);
      const tag = label(family, x);
      const state = checkState(values, tag, previous ? previous.orders : null);
      orderStates++;
      const snapshot = { orders: state.orders.map((o) => Int32Array.from(o)), frames: clone(state.frames), x };
      if (previous) {
        const swaps = swapsOf(previous, snapshot);
        for (const swap of swaps) {
          if (swap.area <= 0.25) continue;
          const layerIndex = scene.layers.findIndex((l) => l.name === swap.layer);
          const layer = scene.layers[layerIndex];
          const prior = previous.orders[layerIndex];
          const probe = (value) => {
            const { cams, frames } = emitAll(family.at(value));
            const order = TURN.order(layer, frames, cams, prior, {}, scene);
            const at = new Map();
            order.forEach((g, index) => at.set(g, index));
            return { first: at.get(swap.i) < at.get(swap.j), frames };
          };
          const start = probe(previous.x).first;
          let lo = previous.x;
          let hi = x;
          let frames = null;
          for (let it = 0; it < 16; it++) {
            const mid = (lo + hi) / 2;
            const next = probe(mid);
            if (next.first === start) lo = mid;
            else {
              hi = mid;
              frames = next.frames;
            }
          }
          if (!frames) frames = probe(hi).frames;
          const area = overlapArea(runs[swap.i], frames[swap.i], runs[swap.j], frames[swap.j]);
          const built = flushOf(runs[swap.i].name, runs[swap.j].name);
          if (area > (built ? 0.5 : 0.25)) result.swaps.push({ layer: swap.layer, a: runs[swap.i].name, b: runs[swap.j].name, area, at: `${family.name} ${f3(hi)}` });
        }
        if (swaps.length) local.push({ family, x, count: swaps.length, pairs: swaps });
      }
      previous = snapshot;
    }
    changes.push(...local);
    const share = Math.round((quick ? Math.min(200, random) : random) / families.length);
    let seed = 12345 + families.indexOf(family);
    const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
    const events = local.flatMap((c) => [c.x - (family.delta ?? (family.periodic ? 0.001 : family.frame / 50)), c.x + (family.delta ?? (family.periodic ? 0.001 : family.frame / 50))]);
    say(`order scan ${family.name}: ${share} random states, ${events.length} event states`);
    for (let i = 0; i < share; i++) {
      const x = family.from + rand() * (family.to - family.from);
      checkState(family.at(x), `random ${label(family, x)}`, null);
    }
    for (const x of events.slice(0, quick ? 100 : 4000)) checkState(family.at(x), `event ${label(family, x)}`, null);
    orderStates += share + Math.min(events.length, quick ? 100 : 4000);
  }
  for (const values of wants("order") ? extra.slice(0, quick ? 100 : extra.length) : []) checkState(values, `pose ${JSON.stringify(values).slice(0, 60)}`, null);
  result.stats.orderStates = orderStates + extra.length;
  result.stats.changes = changes.length;
  result.changes = changes.sort((a, b) => b.count - a.count).slice(0, 10).map((c) => ({ family: c.family.name, x: c.x, count: c.count, pairs: c.pairs.slice(0, 4).map((p) => `${runs[p.i].name} / ${runs[p.j].name}`), at: c.family.at }));
  say("rigid");
  const mounts = A.mountsOf(R.tubes, R.solids);
  if (wants("rigid")) {
  for (const c of A.clearances(R.tubes, R.solids, { mounts })) result.rigid.push({ kind: "clearance", text: `${c.kind} ${c.tube} × ${c.other} clearance ${f2(c.clearance)} < ${f2(c.need)}` });
  for (const e of A.terminals(R.tubes, R.solids, { mounts })) result.rigid.push({ kind: "terminal", text: `${e.tube} ${e.end} ${Number.isFinite(e.d) ? f2(e.d) : "∞"} from ${e.nearest}` });
  const held = A.supports(R.solids, R.tubes);
  for (const name of held.loose) result.rigid.push({ kind: "support", text: `loose ${name}` });
  for (const name of held.hanging ?? []) result.rigid.push({ kind: "support", text: `hanging ${name}` });
  for (const d of held.sunk) result.rigid.push({ kind: "support", text: `sunk ${d.a} × ${d.b} ${f2(d.depth)} deep` });
  for (const c of A.solidClearances(R.solids)) result.rigid.push({ kind: "solid", text: `${c.tube} × ${c.other} sunk ${f2(-c.clearance)}` });
  }
  say("swept");
  const liveShapes = [];
  for (const part of T.parts) for (const shape of part.shapes) liveShapes.push({ shape, part, group: part.group });
  const liveSet = new Set(liveShapes.map((entry) => entry.shape));
  const staticShapes = [];
  for (const item of R.items) for (const shape of item.shapes ?? []) if (!liveSet.has(shape)) staticShapes.push({ shape, item });
  const pointsCache = new Map();
  const pointsOf = (shape) => {
    if (!pointsCache.has(shape)) pointsCache.set(shape, surfacePoints(shape));
    return pointsCache.get(shape);
  };
  for (const g of wants("swept") ? groups : []) {
    if (g.kind !== "turn" || g.parent) continue;
    const mine = liveShapes.filter((entry) => entry.group === g);
    const slices = new Map();
    for (const { shape } of mine)
      for (const p of pointsOf(shape)) {
        const band = Math.floor(p[2] * 2);
        const r = Math.hypot(p[0] - g.origin[0], p[1] - g.origin[1]);
        slices.set(band, Math.max(slices.get(band) ?? 0, r));
      }
    const flagged = [];
    for (const { shape, item } of staticShapes) {
      let worst = Infinity;
      for (const p of pointsOf(shape)) {
        const r = Math.hypot(p[0] - g.origin[0], p[1] - g.origin[1]);
        for (const band of [Math.floor(p[2] * 2), Math.floor(p[2] * 2 - 0.2), Math.floor(p[2] * 2 + 0.2)]) {
          const reach = slices.get(band);
          if (reach !== undefined) worst = Math.min(worst, r - reach);
        }
      }
      if (worst < -0.1) flagged.push({ shape, item });
    }
    for (const { shape, item } of flagged) {
      let deepest = 0;
      let where = 0;
      const count = quick ? 90 : 720;
      for (let i = 0; i < count; i++) {
        const angle = (i * 360) / count;
        const cams = TURN.cams(scene, { [g.name]: angle });
        const pose = poseFor(cams, g);
        const b = A.boundsOf(shape);
        for (const { shape: other } of mine)
          for (const p of pointsOf(other)) {
            const q = place(pose, p);
            if (q[0] < b.min[0] - 0.5 || q[0] > b.max[0] + 0.5 || q[1] < b.min[1] - 0.5 || q[1] > b.max[1] + 0.5 || q[2] < b.min[2] - 0.5 || q[2] > b.max[2] + 0.5) continue;
            const d = A.sdf(shape, q);
            if (d < deepest) {
              deepest = d;
              where = angle;
            }
          }
      }
      if (deepest < -0.1) result.swept.push({ text: `${item.name} × ${g.name} sweep: ${f2(-deepest)} deep at ${f2(where)}°` });
    }
  }
  say("posed");
  const posedStates = [];
  for (const family of families) {
    const frame = posedStep ?? (family.periodic ? (quick ? 4 : 0.5) : family.frame * (quick ? 4 : 1));
    for (const x of samplesOf(family, frame)) posedStates.push({ values: family.at(x), tag: label(family, x) });
  }
  for (const values of extra.slice(0, quick ? 40 : extra.length)) posedStates.push({ values, tag: "pose" });
  const entries = [
    ...liveShapes.map((entry) => ({ ...entry, kind: "live", sphere: sphereOf(pointsOf(entry.shape)) })),
    ...staticShapes.map((entry) => ({ shape: entry.shape, part: { name: entry.item.name }, group: "static", item: entry.item, kind: "static", sphere: sphereOf(pointsOf(entry.shape)) })),
  ];
  const posedSeen = new Set();
  const movingGroups = groups.filter((g) => !(g.kind === "turn" && !g.parent));
  const needsStatic = movingGroups.length > 0;
  for (const { values, tag } of wants("posed") ? posedStates : []) {
    const cams = TURN.cams(scene, values);
    const placed = entries.map((entry) => {
      if (entry.kind === "static") return { entry, c: entry.sphere.c, pose: null };
      const pose = poseFor(cams, entry.group);
      return { entry, c: entry.group ? place(pose, entry.sphere.c) : entry.sphere.c, pose: entry.group ? pose : null };
    });
    placed.sort((a, b) => a.c[0] - a.entry.sphere.r - (b.c[0] - b.entry.sphere.r));
    for (let i = 0; i < placed.length; i++) {
      const A0 = placed[i];
      const reachA = A0.c[0] + A0.entry.sphere.r + 0.5;
      for (let j = i + 1; j < placed.length; j++) {
        const B0 = placed[j];
        if (B0.c[0] - B0.entry.sphere.r > reachA) break;
        const a = A0.entry;
        const b = B0.entry;
        if (a.group === b.group) continue;
        if (a.kind === "static" && b.kind === "static") continue;
        if ((a.kind === "static" || b.kind === "static") && !needsStatic) continue;
        if (a.kind === "static" || b.kind === "static") {
          const live = a.kind === "static" ? b : a;
          if (!live.group || (live.group.kind === "turn" && !live.group.parent)) continue;
        }
        if (a.shape.kind === "tube" && b.shape.kind === "tube") continue;
        if (Math.hypot(A0.c[0] - B0.c[0], A0.c[1] - B0.c[1], A0.c[2] - B0.c[2]) > a.sphere.r + b.sphere.r + 0.5) continue;
        const key = `${a.part.name}|${b.part.name}`;
        if (posedSeen.has(key)) continue;
        const pa = A0.pose ?? IDENTITY;
        const pb = B0.pose ?? IDENTITY;
        let deepest = 0;
        for (const p of pointsOf(a.shape)) {
          const d = A.sdf(b.shape, unplace(pb, place(pa, p)));
          if (d < deepest) deepest = d;
        }
        for (const p of pointsOf(b.shape)) {
          const d = A.sdf(a.shape, unplace(pa, place(pb, p)));
          if (d < deepest) deepest = d;
        }
        const limit = a.shape.kind === "tube" || b.shape.kind === "tube" ? -0.1 : -0.5;
        if (deepest < limit) {
          posedSeen.add(key);
          result.posed.push({ text: `${a.part.name} × ${b.part.name} through ${f2(-deepest)} at ${tag}` });
        }
      }
    }
  }
  result.stats.posedStates = posedStates.length;
  say("folds, self, crowding, endon");
  const routes = T.routes ?? [];
  const turnGroups = groups.filter((g) => g.kind === "turn" && !g.parent);
  const tubesAll = R.tubes;
  const linkMap = A.linksOf(tubesAll, mounts);
  const routeSample = [];
  {
    const all = posedStates.filter((s) => s.tag !== "pose");
    const stride = Math.max(1, Math.floor(all.length / (quick ? 48 : routeStates)));
    for (let i = 0; i < all.length; i += stride) routeSample.push(all[i]);
    for (const values of extra.slice(0, 24)) routeSample.push({ values, tag: "pose" });
  }
  const foldSeen = new Set();
  const crowdSeen = new Set();
  const lo = (E * 180) / Math.PI;
  for (const route of wants("routes") ? routes : []) {
    const g = route.group;
    const root = g && g.kind === "turn" && !g.parent;
    const records = R.routes.filter((r) => r.name === route.name);
    const groupParts = T.parts.filter((part) => part.group === g);
    const itemsAt = (pose) => groupParts.map((part) => ({ name: part.name, route: part.route, shapes: pose ? part.shapes.map((shape) => posed(shape, pose)) : part.shapes }));
    const views = [];
    if (!g) views.push({ P, pose: null, tag: "rest" });
    else if (root) {
      const count = quick ? 72 : 720;
      for (let i = 0; i < count; i++) {
        const x = (i * 360) / count;
        views.push({ P: { ...P, azimuth: (P.azimuth ?? 45) - x }, pose: null, tag: `${f2(x)}°` });
      }
    } else for (const { values, tag } of routeSample) views.push({ P, pose: poseFor(TURN.cams(scene, values), g), tag });
    for (const view of views) {
      const posedRecords = view.pose ? records.map((r) => ({ ...r, points: r.points.map((p) => place(view.pose, p)) })) : records;
      const items = itemsAt(view.pose);
      for (const fold of A.folds(posedRecords, view.P, { items })) {
        const key = `${fold.name}|${fold.kind}`;
        if (foldSeen.has(key)) continue;
        foldSeen.add(key);
        result.folds.push({ text: `${fold.name} ${fold.kind} at ${view.tag}: turns ${fold.flatTurn.toFixed(0)}° on screen${fold.kind === "pinch" ? ` round a ${f2(fold.radius)} px bend` : ` for ${fold.realTurn.toFixed(0)}° in 3D`}` });
      }
      for (const s of A.selfOverlaps(items.filter((item) => item.route === route.name), view.P)) {
        const key = `self|${s.name}`;
        if (foldSeen.has(key)) continue;
        foldSeen.add(key);
        result.self.push({ text: `${s.name} folds back at ${view.tag}: radius ${f2(s.radius)} px under its half-width ${f2(s.width)} px` });
      }
    }
    const dense = route.world;
    const total = G.pathLength(dense);
    const stations = G.resample(dense, 0.25);
    const rotations = !g || root || g.upright ? [null] : routeSample.map(({ values }) => poseFor(TURN.cams(scene, values), g).R);
    const bad = [];
    let walked = 0;
    for (let i = 1; i < stations.length - 1; i++) {
      walked += G.len3(G.sub3(stations[i], stations[i - 1]));
      if (route.gaps.some(([a, b]) => walked >= a - 1e-6 && walked <= b + 1e-6)) continue;
      if (fittings.some(([name, a, b]) => name === route.name && walked >= a && walked <= b)) continue;
      const t0 = G.unit3(G.sub3(stations[i + 1], stations[i - 1]));
      for (const Rm of rotations) {
        const t = Rm ? apply(Rm, t0) : t0;
        const phi = (Math.asin(Math.max(-1, Math.min(1, t[2]))) * 180) / Math.PI;
        if ((phi > lo - 15 && phi < lo + 15) || (phi > -lo - 15 && phi < -lo + 15)) {
          bad.push({ at: walked, phi });
          break;
        }
      }
    }
    if (bad.length) result.endon.push({ text: `${route.name} points within 15° of the view along ${f2(bad[0].at)}–${f2(bad[bad.length - 1].at)} of ${f2(total)} (slope ${bad[0].phi.toFixed(1)}°): bend flat or inside a fitting` });
  }
  const crowdViews = [];
  for (const g of turnGroups) {
    const count = quick ? 24 : 72;
    for (let i = 0; i < count; i++) crowdViews.push({ group: g, P: { ...P, azimuth: (P.azimuth ?? 45) - (i * 360) / count }, tag: `${f2((i * 360) / count)}°`, values: null });
  }
  const others = [...new Set(routes.map((route) => route.group).filter((g) => g && !(g.kind === "turn" && !g.parent)))];
  for (const g of others) for (const { values, tag } of routeSample.slice(0, quick ? 24 : 72)) crowdViews.push({ group: g, P, tag, values });
  for (const view of wants("routes") ? crowdViews : []) {
    const g = view.group;
    const pose = view.values ? poseFor(TURN.cams(scene, view.values), g) : null;
    const mineRoutes = R.routes.filter((r) => routes.some((route) => route.name === r.name && route.group === g)).map((r) => (pose ? { ...r, points: r.points.map((p) => place(pose, p)) } : r));
    if (!mineRoutes.length) continue;
    const groupParts = T.parts.filter((part) => part.group === g);
    const items = groupParts.map((part) => ({ name: part.name, route: part.route, shapes: pose ? part.shapes.map((shape) => posed(shape, pose)) : part.shapes }));
    const rods = A.slenderOf(items.flatMap((item) => item.shapes).filter((shape) => shape.kind !== "tube"));
    for (const c of A.crowding(mineRoutes, view.P, { links: linkMap, items, allow: crowd, rods })) {
      const key = [c.a, c.b, c.where].join("|");
      if (crowdSeen.has(key)) continue;
      crowdSeen.add(key);
      result.crowding.push({ text: `${c.a} ${c.where} × ${c.b} at ${view.tag}: outlines ${c.gap < 0 ? `overlap ${f2(-c.gap)}` : `${f2(c.gap)} apart`} px` });
    }
  }
  say("coverage");
  const rest = emitAll(Object.fromEntries(groups.map((g) => [g.name, g.kind === "free" ? null : 0])));
  const coverItems = T.parts.map((part) => {
    const run = runs[part.id];
    const f = rest.frames[part.id];
    const outline = run.kind === "fixed" ? part.svg : run.kind === "tube" ? f.d[run.role.body] : f.d[run.role.fill];
    return { name: part.name, svg: `<path d="${outline}"/>`, shapes: part.shapes };
  });
  for (const c of wants("coverage") ? A.coverage(coverItems, P, { allow: cover }) : []) result.coverage.push({ text: `${c.name} its shapes cover ${Math.round(c.share * 100)}% of its drawing` });
  say("continuity");
  if (wants("continuity")) {
    TURN.exact(true);
    try {
      continuity(T, scene, runs, groups, { step, quick, result, say, families, frameOf });
    } finally {
      TURN.exact(false);
    }
  }
  say("depth oracle");
  const depthStates = [];
  const first = families[0];
  if (first) for (let i = 0; i < (quick ? 6 : depthAngles); i++) {
    const x = first.from + ((first.to - first.from) * i) / (quick ? 6 : depthAngles);
    depthStates.push({ values: first.at(x), tag: label(first, x) });
  }
  for (const family of families.slice(1)) for (let i = 0; i <= (quick ? 3 : 12); i++) {
    const x = family.from + ((family.to - family.from) * i) / (quick ? 3 : 12);
    depthStates.push({ values: family.at(x), tag: label(family, x) });
  }
  for (const change of result.changes.slice(0, quick ? 4 : depthEvents)) depthStates.push({ values: change.at(change.x), tag: `${change.family} ${f3(change.x)}` });
  let checked = 0;
  for (const { values, tag } of wants("depth") ? depthStates : []) {
    const state = checkState(values, `depth ${tag}`, null);
    scene.layers.forEach((layer, index) => {
      const items = [];
      for (const global of layer.parts) {
        const part = T.parts[global];
        const pose = poseFor(state.cams, part.group);
        items.push({ name: part.name, global, shapes: part.shapes.map((shape) => posed(shape, pose)), route: part.route, chunk: part.chunk });
      }
      const ranks = new Map((T.order.length ? T.order : T.layers.map((l) => l.name)).map((name, at) => [name, at]));
      for (const item of R.items) if (item.layer && ranks.has(item.layer) && item.shapes && item.shapes.length) items.push({ name: item.name, static: ranks.get(item.layer) < ranks.get(layer.name) ? "below" : "above", shapes: item.shapes });
      const overlaps = A.overlapsOf(items, P, { step: quick ? 0.5 : 0.25, skip: (a, b) => A.sameRun(a, b) || (a.static && b.static) });
      for (const pair of overlaps.pairs) {
        const a = items[pair.a];
        const b = items[pair.b];
        if (!a.static === !b.static) continue;
        const sideIndex = a.static ? 0 : 1;
        const fixed = a.static ? a : b;
        const moving = a.static ? b : a;
        const staticFront = pair.front[sideIndex].area >= overlaps.least;
        const liveFront = pair.front[1 - sideIndex].area >= overlaps.least;
        if (fixed.static === "below" && staticFront) result.layers.push({ text: `${fixed.name} (under ${layer.name}) is in front of ${moving.name} at ${tag}: ${f2(pair.front[sideIndex].area)} px²` });
        if (fixed.static === "above" && liveFront) result.layers.push({ text: `${fixed.name} (over ${layer.name}) is behind ${moving.name} at ${tag}: ${f2(pair.front[1 - sideIndex].area)} px²` });
      }
      const rank = new Map();
      state.orders[index].forEach((global, at) => rank.set(global, at));
      for (const pair of overlaps.pairs) {
        if (items[pair.a].static || items[pair.b].static) continue;
        checked++;
        const aFront = pair.front[0].area >= overlaps.least;
        const bFront = pair.front[1].area >= overlaps.least;
        if (aFront === bFront) continue;
        const front = items[aFront ? pair.a : pair.b];
        const backItem = items[aFront ? pair.b : pair.a];
        if (rank.get(front.global) < rank.get(backItem.global)) result.depth.push({ text: `${layer.name} at ${tag}: ${front.name} is in front of ${backItem.name} but drawn first (${f2(pair.front[aFront ? 0 : 1].area)} px²)` });
      }
    });
  }
  result.stats.depthStates = depthStates.length;
  result.stats.depthPairs = checked;
  result.stats.seconds = (Date.now() - started) / 1000;
  return result;
}

function continuity(T, scene, runs, groups, { step, quick, result, say, families = [], frameOf = (f) => f.frame }) {
  const stepSize = quick ? step * 10 : step;
  const measures = [];
  const worst = { q: 0, birth: 0, dot: 0, fade: 0, motion: 0, merge: 0 };
  const failures = [];
  const k = scene.P.scale * Math.sqrt(1.6);
  const sweeps = [];
  for (const g of groups) {
    if (g.loose) continue;
    if (g.kind === "turn") sweeps.push({ group: g, from: 0, to: 360, unit: "°" });
    else if (g.kind === "hinge") sweeps.push({ group: g, from: g.range[0], to: g.range[1], unit: "°" });
    else sweeps.push({ group: g, from: 0, to: g.travel, unit: "" });
  }
  for (const sweep of sweeps) {
    const g = sweep.group;
    const parts = runs.filter((run) => run.group === g.index);
    if (!parts.length) continue;
    const span = sweep.to - sweep.from;
    const delta = g.kind === "slide" ? Math.max(0.01, span / 400) : stepSize;
    const count = Math.max(1, Math.round(span / delta));
    const reach = parts.map((run) => {
      let r = 0;
      const part = T.parts[run.index];
      for (const p of part.samples) r = Math.max(r, g.kind === "slide" ? 0 : Math.hypot(p[0], p[1]));
      return r;
    });
    const valuesAt = (x) => ({ [g.name]: x });
    const emitPart = (run, x, f) => {
      const cams = TURN.cams(scene, valuesAt(x));
      TURN.emit(run, cams[g.index], f);
      return f;
    };
    say(`continuity ${g.name}: ${count} steps of ${delta}${sweep.unit} over ${parts.length} parts`);
    const snap = (run, f) => {
      const roles = run.roles;
      const out = { faces: [], sheets: [], dots: [], planes: [], fade: {}, outline: null, levels: f.levels ?? 0 };
      roles.forEach((role, r) => {
        if (role.type === "face") out.faces.push([f.d[r] ? areaOfPath(f.d[r]) : 0, f.q[r], Boolean(f.d[r])]);
        else if (role.type === "path" && /^s[0-4]$/.test(role.name)) out.sheets.push([areaOfPath(f.d[r]), (f.d[r].match(/M/g) ?? []).length]);
        else if (role.type === "dot") out.dots.push(f.o[r]);
        else if (role.type === "plane") out.planes.push(f.o[r]);
      });
      for (const [prefix, set] of Object.entries(f.fade ?? {})) {
        const map = new Map();
        set.key.forEach((key, i) => map.set(key, Math.round(set.alpha[i] * TURN.levels) / TURN.levels));
        out.fade[prefix] = map;
      }
      const edge = run.role.fill !== undefined ? f.d[run.role.fill] : run.role.body !== undefined ? f.d[run.role.body] : "";
      out.outline = run.kind === "fixed" ? [Array.from({ length: f.hullN }, (_, i) => [f.hull[2 * i], f.hull[2 * i + 1]])] : pathPolys(edge);
      out.slots = run.slots;
      return out;
    };
    const compare = (run, a, b, dx, reachPx) => {
      const m = { q: 0, birth: 0, dot: 0, fade: 0, motion: 0, merge: 0 };
      for (let i = 0; i < Math.min(a.faces.length, b.faces.length); i++) {
        const [areaA, qa, onA] = a.faces[i];
        const [areaB, qb, onB] = b.faces[i];
        if (onA && onB) m.q = Math.max(m.q, (Math.abs(qa - qb) * 16) / TURN.q);
        if (onA !== onB) m.birth = Math.max(m.birth, onA ? areaA : areaB);
        m.area = Math.max(m.area ?? 0, Math.abs(areaA - areaB));
      }
      for (let i = 0; i < Math.min(a.sheets.length, b.sheets.length); i++) {
        const [areaA, countA] = a.sheets[i];
        const [areaB, countB] = b.sheets[i];
        if (!countA !== !countB) m.birth = Math.max(m.birth, Math.abs(areaA - areaB));
        m.area = Math.max(m.area ?? 0, Math.abs(areaA - areaB));
      }
      for (let i = 0; i < Math.min(a.dots.length, b.dots.length); i++) m.dot = Math.max(m.dot, Math.abs(a.dots[i] - b.dots[i]));
      for (let i = 0; i < Math.min(a.planes.length, b.planes.length); i++) m.dot = Math.max(m.dot, Math.abs(a.planes[i] - b.planes[i]));
      for (const prefix of new Set([...Object.keys(a.fade), ...Object.keys(b.fade)])) {
        const A0 = a.fade[prefix] ?? new Map();
        const B0 = b.fade[prefix] ?? new Map();
        for (const key of new Set([...A0.keys(), ...B0.keys()])) m.fade = Math.max(m.fade, Math.abs((A0.get(key) ?? 0) - (B0.get(key) ?? 0)));
      }
      const slots = Math.max(run.slots.bevel ?? 0, run.slots.ribs ?? 0);
      if (slots && (a.levels > slots || b.levels > slots)) m.merge = Math.max(a.levels, b.levels) - slots;
      if (a.outline.length && b.outline.length && a.outline[0].length && b.outline[0].length) m.motion = Math.max(0, hausdorffish(a.outline[0], b.outline[0]) - (1.5 * reachPx * dx + 0.05));
      return m;
    };
    const frames = parts.map((run) => TURN.frame(run));
    let previous = parts.map((run, i) => snap(run, emitPart(run, sweep.from, frames[i])));
    for (let c = 1; c <= count; c++) {
      const x = sweep.from + c * delta;
      parts.forEach((run, i) => {
        const current = snap(run, emitPart(run, x, frames[i]));
        const reachPx = g.kind === "slide" ? k : (reach[i] * k * Math.PI) / 180;
        const m = compare(run, previous[i], current, g.kind === "slide" ? delta : delta, reachPx);
        for (const key of Object.keys(worst)) worst[key] = Math.max(worst[key], m[key] ?? 0);
        const scale = delta / step;
        if (m.q > Math.ceil(scale) || m.birth > 2 * scale || m.dot > 0.0105 * scale || m.fade > Math.ceil(scale) / 32 + 0.005 || m.merge > 0 || m.motion > 0) failures.push({ part: run.name, group: g.name, at: x, m });
        measures.push({ run, x: x - delta, dx: delta, value: m.birth + m.motion * 4 + (m.area ?? 0), motion: m.motion, birth: m.birth, valuesAt, reachPx });
        previous[i] = current;
      });
    }
  }
  measures.sort((a, b) => b.value - a.value);
  let halvingFails = 0;
  let halvingChecked = 0;
  const halving = (run, at, x, dx) => {
    const emitAt = (value) => {
      const f = TURN.frame(run);
      const cams = TURN.cams(scene, at(value));
      TURN.emit(run, run.group >= 0 ? cams[run.group] : cams.world, f);
      return f;
    };
    const frames = Array.from({ length: 9 }, (_, j) => emitAt(x + (dx * j) / 8));
    const measure = (scale) => {
      const eighth = Math.max(...visibleChange(run, frames, scale));
      const quarter = Math.max(...visibleChange(run, [0, 2, 4, 6, 8].map((j) => frames[j]), scale));
      const half = Math.max(...visibleChange(run, [frames[0], frames[4], frames[8]], scale));
      const full = visibleChange(run, [frames[0], frames[8]], scale)[0];
      return { full, half, quarter, eighth, fails: full >= 2 && half > 0.6 * full && quarter > 0.6 * half && eighth > 0.6 * quarter && eighth >= 2 };
    };
    const coarse = measure(4);
    return coarse.fails ? measure(16) : coarse;
  };
  const seen = new Set();
  for (const entry of measures.slice(0, quick ? 40 : 400)) {
    if (!(entry.value > 1e-6)) break;
    const key = `${entry.run.index}|${entry.x.toFixed(4)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    halvingChecked++;
    const g = groups[entry.run.group];
    const h = halving(entry.run, entry.valuesAt, entry.x, entry.dx);
    if (!h.fails) continue;
    halvingFails++;
    failures.push({ part: entry.run.name, group: g.name, at: entry.x, m: { halving: `${f2(h.full)} tone·px² step stays ${f2(h.half)}, ${f2(h.quarter)} and ${f2(h.eighth)} at a half, a quarter and an eighth of the step` } });
  }
  const areaRoles = (run) => run.roles.map((role, r) => (role.type === "face" || /^s[0-4]$/.test(role.name) || (run.kind === "tube" && role.name === "body") ? r : -1)).filter((r) => r >= 0);
  const clipSteps = [];
  for (const family of families) {
    if (family.periodic) continue;
    const xs = samplesOf(family, frameOf(family));
    say(`continuity ${family.name}: ${xs.length} frames over ${runs.length} parts`);
    const frames = runs.map((run) => TURN.frame(run));
    let previous = null;
    for (const x of xs) {
      const cams = TURN.cams(scene, family.at(x));
      const current = runs.map((run, index) => {
        TURN.emit(run, run.group >= 0 ? cams[run.group] : cams.world, frames[index]);
        return areaRoles(run).map((r) => areaOfPath(frames[index].d[r]));
      });
      if (previous)
        runs.forEach((run, index) => {
          let most = 0;
          let role = -1;
          current[index].forEach((v, k) => {
            const d = Math.abs(v - previous[index][k]);
            if (d > most) {
              most = d;
              role = k;
            }
          });
          if (most >= 0.5) clipSteps.push({ family, run, x: x - frameOf(family), dx: frameOf(family), role, value: most });
        });
      previous = current;
    }
  }
  clipSteps.sort((a, b) => b.value - a.value);
  let clipChecked = 0;
  for (const entry of clipSteps.slice(0, quick ? 60 : 400)) {
    clipChecked++;
    const h = halving(entry.run, entry.family.at, entry.x, entry.dx);
    if (!h.fails) continue;
    halvingFails++;
    failures.push({ part: entry.run.name, group: entry.family.name, at: entry.x, m: { halving: `${f2(h.full)} tone·px² step stays ${f2(h.half)}, ${f2(h.quarter)} and ${f2(h.eighth)} at a half, a quarter and an eighth of the frame` } });
  }
  result.continuity = failures;
  result.stats.continuity = { worst, halvingChecked: halvingChecked + clipChecked, halvingFails, measures: measures.length + clipSteps.length };
}

export function orbitReport(result) {
  const lines = [];
  const skipped = (name, items) => (result.skipped ?? []).includes(name) && !items.length;
  const list = (name, items, text) => {
    if (skipped(name, items)) return lines.push(`${name}: skipped`);
    lines.push(`${name}: ${items.length}`);
    for (const item of items.slice(0, 40)) lines.push(`  ${text(item)}`);
  };
  list("separation", result.separation, (m) => `interlock ${m.a} × ${m.b} ${f2(m.margin)}: split, flats or a flush contact`);
  list("unseparated", result.unseparated, (m) => `${m.state}  ${m.a} × ${m.b} margin ${f2(m.margin)} while their hulls overlap${m.framed ? " (framed plane: the build's poses() missed this motion; add it to poses())" : ""}`);
  const cycleGroups = new Map();
  for (const c of result.cycles) {
    const key = `${c.layer}|${c.parts.join(", ")}`;
    if (!cycleGroups.has(key)) cycleGroups.set(key, { ...c, first: c.state, last: c.state, count: 0 });
    const g = cycleGroups.get(key);
    g.last = c.state;
    g.count++;
  }
  lines.push(skipped("cycles", result.cycles) ? "cycles: skipped" : `cycles: ${result.cycles.length}`);
  for (const g of [...cycleGroups.values()].slice(0, 20)) lines.push(`  ${g.layer}: ${g.parts.join(", ")} in ${g.count} state${g.count === 1 ? "" : "s"} from ${g.first} to ${g.last}: ${g.split ? `split ${g.split.part} on n = (${g.split.n.map((v) => v.toFixed(3)).join(", ")}), d = ${g.split.d.toFixed(3)} (the plane of the cycle edge with the largest margin, ${g.split.a} × ${g.split.b})` : "split one of them (T.split) on the plane of the cycle edge with the largest margin"}`);
  list("forced", result.forced, (c) => `${c.state}  ${c.layer}: ${c.forced} forced pick${c.forced === 1 ? "" : "s"}`);
  list("swaps", result.swaps, (s) => `${s.layer}: ${s.a} / ${s.b} swap at ${s.at} while overlapping ${f2(s.area)} px²`);
  list("layers", result.layers, (s) => s.text);
  list("rigid", result.rigid, (s) => `${s.kind}  ${s.text}`);
  list("swept", result.swept, (s) => s.text);
  list("posed", result.posed, (s) => s.text);
  list("folds", result.folds, (s) => s.text);
  list("self", result.self, (s) => s.text);
  list("crowding", result.crowding, (s) => s.text);
  list("endon", result.endon, (s) => s.text);
  list("coverage", result.coverage, (s) => s.text);
  list("depth", result.depth, (s) => s.text);
  list("continuity", result.continuity, (c) => `${c.part} (${c.group}) at ${f3(c.at)}: ${Object.entries(c.m).filter(([, v]) => typeof v === "string" || v > 0).map(([key, v]) => `${key} ${typeof v === "string" ? v : f3(v)}`).join(", ")}`);
  const s = result.stats;
  const w = s.continuity?.worst ?? {};
  if (!skipped("continuity", result.continuity)) lines.push(`worst tone step ${w.q ?? 0} · worst birth ${f2(w.birth ?? 0)} px² · worst dot α step ${f3(w.dot ?? 0)} · worst faded α step ${f3(w.fade ?? 0)} · merges ${w.merge ?? 0} · outline motion over bound ${f3(w.motion ?? 0)} px · halving ${s.continuity?.halvingFails ?? 0} of ${s.continuity?.halvingChecked ?? 0}`);
  lines.push(`order changes: ${s.changes ?? 0} states with swaps · ${s.orderStates ?? 0} order states · ${s.posedStates ?? 0} posed states · ${s.planes ?? 0} planes (tightest margin ${f3(s.tightest ?? 0)}) · depth oracle ${s.depthStates ?? 0} states, ${s.depthPairs ?? 0} overlapping pairs`);
  for (const change of result.changes ?? []) lines.push(`  ${change.family} at ${f3(change.x)}: ${change.count} pair${change.count === 1 ? "" : "s"} reorder (${change.pairs.join("; ")})`);
  lines.push(`turn audit · ${f2(s.seconds ?? 0)} s`);
  return lines.join("\n");
}

export const orbitFailures = (result) => ["separation", "unseparated", "cycles", "forced", "swaps", "layers", "rigid", "swept", "posed", "folds", "self", "crowding", "endon", "coverage", "depth", "continuity"].reduce((sum, key) => sum + result[key].length, 0);

export function orbitOrExit(T, R, P, { flag = "--audit", argv = process.argv, ...options } = {}) {
  if (!argv.includes(flag)) return null;
  const result = orbit(T, R, P, options);
  console.log(orbitReport(result));
  const count = orbitFailures(result);
  const skipped = result.skipped.length;
  const partial = skipped ? `${skipped} line${skipped === 1 ? "" : "s"} skipped by only` : "";
  const undersampled = result.quick ? "quick run, undersampled: the full run is the proof" : "";
  const notes = [partial, undersampled].filter(Boolean).join("; ");
  if (count) console.log(`turn audit failed: ${count} problem${count === 1 ? "" : "s"}${notes ? ` (${notes})` : ""}`);
  else if (notes) console.log(`turn audit ${skipped ? "partial" : "quick"}: no problems in what ran (${notes})`);
  else console.log("turn audit passed");
  if (count) process.exitCode = 1;
  return result;
}
