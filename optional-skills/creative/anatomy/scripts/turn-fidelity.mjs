import * as k from "../kit/iso-kit.mjs";
import * as G from "../kit/lathe.mjs";
import { tubePieces } from "../kit/tube.mjs";
import { TURN } from "../kit/turn.mjs";

const rootOf = (group) => {
  let root = group;
  while (root && root.parent) root = root.parent;
  return root && root.kind === "turn" ? root : null;
};
const restValues = (T, theta) => Object.fromEntries(T.groups.map((g) => [g.name, g.kind === "turn" && !g.parent ? theta : g.kind === "free" ? null : 0]));

export function identity(T, P, angles, { plant = null } = {}) {
  const scene = TURN.prepare(T.data, { unit: 1 });
const subpaths = (d) => {
    const out = [];
    for (const [, body] of (d ?? "").matchAll(/M([^M]*)/g)) {
      const nums = (body.match(/-?\d+(?:\.\d+)?(?:e-?\d+)?/g) ?? []).map(Number);
      const pts = [];
      for (let i = 0; i + 1 < nums.length; i += 2) pts.push([nums[i], nums[i + 1]]);
      out.push({ pts, closed: /Z\s*$/.test(body) });
    }
    return out;
  };
  const segDist = (p, a, b) => {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const l = dx * dx + dy * dy;
    const t = l ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l)) : 0;
    return Math.hypot(p[0] - a[0] - dx * t, p[1] - a[1] - dy * t);
  };
  const toPoly = (pts, closed) => (pts, closed);
  void toPoly;
  const distToPoly = (p, poly, closed = true) => {
    let best = Infinity;
    const n = poly.length;
    for (let i = 0; i < (closed ? n : n - 1); i++) best = Math.min(best, segDist(p, poly[i], poly[(i + 1) % n]));
    if (n === 1) best = Math.hypot(p[0] - poly[0][0], p[1] - poly[0][1]);
    return best;
  };
  const oneWay = (A, B) => {
    let worst = 0;
    for (const p of A) {
      let near = null;
      let gap = Infinity;
      for (const q of B) {
        const g = Math.hypot(p[0] - q[0], p[1] - q[1]);
        if (g < gap) {
          gap = g;
          near = q;
        }
      }
      worst = Math.max(worst, gap < 0.15 ? Math.max(Math.abs(p[0] - near[0]), Math.abs(p[1] - near[1])) : distToPoly(p, B));
    }
    return worst;
  };
  const hausdorff = (A, B) => Math.max(oneWay(A, B), oneWay(B, A));
  const applyMatrix = (pts, t) => {
    const v = (t.match(/-?\d+(?:\.\d+)?(?:e-?\d+)?/g) ?? []).map(Number);
    return pts.map(([x, y]) => [v[0] * x + v[2] * y + v[4], v[1] * x + v[3] * y + v[5]]);
  };
  const levelOf = (score, bands) => {
    let u = 0;
    for (const t of bands) {
      const x = Math.min(1, Math.max(0, (score - (t - 0.035)) / 0.07));
      u += x * x * (3 - 2 * x);
    }
    return u;
  };


const results = { outline: 0, top: 0, faces: 0, compared: 0, skipped: 0, missing: 0, crease: 0 };
const worstAt = {};
for (const theta of angles) {
  const cams = TURN.cams(scene, restValues(T, theta));
  const Pp = { ...P, azimuth: P.azimuth - theta };
  for (const part of T.parts) {
    if (part.kind !== "prism" || !rootOf(part.group) || part.group.kind === "free") continue;
    const planted = plant && part.name === plant;
    const run = scene.runs[part.id];
    const cam = part.group ? cams[part.group.index] : cams.world;
    const f = TURN.emit(run, cam, TURN.frame(run));
    let root = part.group;
    while (root && root.parent) root = root.parent;
    const pivot = root ? [root.origin[0], root.origin[1], 0] : [0, 0, 0];
    const a = k.iso(pivot, P);
    const b = k.iso(pivot, Pp);
    const shift = [a[0] - b[0], a[1] - b[1]];
    const sh = (pts) => pts.map(([x, y]) => [x + shift[0], y + shift[1]]);
    const { F, ring, s0, s1 } = part.world;
    const up = run.up;
    const Pk = planted ? { ...Pp, azimuth: Pp.azimuth - 2 } : Pp;
    const kit = up ? k.extrude(ring, s0, s1 - s0, Pk) : G.prismOf(F, ring, s0, s1, Pk);
    const mine = subpaths(f.d[run.role.edge])[0].pts;
    const theirs = sh(subpaths(kit.fill)[0].pts);
    const ho = hausdorff(mine, theirs);
    results.outline = Math.max(results.outline, ho);
    if (ho > (worstAt[part.name]?.outline ?? -1)) worstAt[part.name] = { ...(worstAt[part.name] ?? {}), outline: ho, at: theta };
    if (up) {
      const topMine = applyMatrix(subpaths(part.planPath)[0].pts, f.t[run.role.top]);
      const topKit = sh(subpaths(kit.top)[0].pts);
      results.top = Math.max(results.top, hausdorff(topMine, topKit));
    }
    const kitFaces = kit.shades.map((d) => subpaths(d)).concat(up ? [] : [subpaths(kit.top)]);
    const faceCount = up ? run.n : run.n + 2;
    for (let i = 0; i < faceCount; i++) {
      const role = run.role[`f${i}`];
      const d = f.d[role];
      if (!d) continue;
      const q = f.q[role];
      if (q % TURN.q !== 0) {
        results.skipped++;
        continue;
      }
      const tone = q / TURN.q;
      const pts = subpaths(d)[0].pts;
      let best = Infinity;
      for (const sp of kitFaces[tone] ?? []) best = Math.min(best, hausdorff(pts, sh(sp.pts)));
      if (!Number.isFinite(best)) {
        results.missing++;
        continue;
      }
      results.compared++;
      results.faces = Math.max(results.faces, best);
    }
  }
}

  const limit = 0.06;
  const pass = results.outline <= limit && results.top <= limit && results.faces <= limit && results.missing === 0;
  const worst = Object.entries(worstAt).filter(([, w]) => w.outline > limit).map(([name, w]) => `${name} outline ${w.outline.toFixed(3)} at ${w.at}°`);
  return { pass, text: `prisms at ${angles.length} angles: outline ${results.outline.toFixed(4)} px · top ${results.top.toFixed(4)} px · faces ${results.faces.toFixed(4)} px over ${results.compared} faces (${results.skipped} in crossfade windows, ${results.missing} with no kit face of that tone)`, worst };
}

export function rounds(T, P, angles, { only = null } = {}) {
  const scene = TURN.prepare(T.data, { unit: 1 });
  const CELL = 0.1;
const subpaths = (d) => {
    const out = [];
    for (const [, body] of (d ?? "").matchAll(/M([^M]*)/g)) {
      const nums = (body.match(/-?\d+(?:\.\d+)?(?:e-?\d+)?/g) ?? []).map(Number);
      const pts = [];
      for (let i = 0; i + 1 < nums.length; i += 2) pts.push([nums[i], nums[i + 1]]);
      out.push(pts);
    }
    return out;
  };
  const segDist = (p, a, b) => {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const l = dx * dx + dy * dy;
    const t = l ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l)) : 0;
    return Math.hypot(p[0] - a[0] - dx * t, p[1] - a[1] - dy * t);
  };
  const distToPoly = (p, poly) => {
    let best = Infinity;
    for (let i = 0; i < poly.length; i++) best = Math.min(best, segDist(p, poly[i], poly[(i + 1) % poly.length]));
    return best;
  };
  const hausdorff = (A, B) => {
    let worst = 0;
    for (const p of A) worst = Math.max(worst, distToPoly(p, B));
    for (const p of B) worst = Math.max(worst, distToPoly(p, A));
    return worst;
  };
  function raster(layers, box, grow = 0) {
    const [x0, y0, x1, y1] = box;
    const w = Math.ceil((x1 - x0) / CELL) + 1;
    const h = Math.ceil((y1 - y0) / CELL) + 1;
    const out = new Int8Array(w * h).fill(-1);
    for (const { tone, d } of layers) {
      if (!d) continue;
      const mask = new Uint8Array(w * h);
      const polys = subpaths(d);
      for (let row = 0; row < h; row++) {
        const y = y0 + row * CELL + CELL / 2;
        const hits = [];
        for (const poly of polys)
          for (let i = 0; i < poly.length; i++) {
            const a = poly[i];
            const b = poly[(i + 1) % poly.length];
            if ((a[1] <= y && b[1] > y) || (b[1] <= y && a[1] > y)) {
              const x = a[0] + ((y - a[1]) / (b[1] - a[1])) * (b[0] - a[0]);
              hits.push([x, b[1] > a[1] ? 1 : -1]);
            }
          }
        hits.sort((p, q) => p[0] - q[0]);
        let wind = 0;
        for (let i = 0; i < hits.length - 1; i++) {
          wind += hits[i][1];
          if (!wind) continue;
          const from = Math.max(0, Math.ceil((hits[i][0] - x0 - CELL / 2) / CELL));
          const to = Math.min(w - 1, Math.floor((hits[i + 1][0] - x0 - CELL / 2) / CELL));
          for (let c = from; c <= to; c++) mask[row * w + c] = 1;
        }
      }
      const reach = Math.round(grow / CELL);
      for (let row = 0; row < h; row++)
        for (let c = 0; c < w; c++) {
          let on = mask[row * w + c];
          for (let dy = -reach; dy <= reach && !on; dy++)
            for (let dx = -reach; dx <= reach && !on; dx++) {
              if (dx * dx + dy * dy > reach * reach) continue;
              const rr = row + dy;
              const cc = c + dx;
              if (rr >= 0 && cc >= 0 && rr < h && cc < w && mask[rr * w + cc]) on = 1;
            }
          if (on) out[row * w + c] = tone;
        }
    }
    return { out, w, h };
  }

const stats = { outline: 0, outlineFine: 0, bands: 0, cells: 0, mismatched: 0, parts: 0 };
const worst = [];
for (const theta of angles) {
  const cams = TURN.cams(scene, restValues(T, theta));
  const Pp = { ...P, azimuth: P.azimuth - theta };
  const Pf = { ...Pp, origin: [Pp.origin[0] * 10, Pp.origin[1] * 10], scale: Pp.scale * 10 };
  for (const part of T.parts) {
    if ((part.kind !== "round" && part.kind !== "lathe") || (only && !only.test(part.name)) || !rootOf(part.group) || part.group.kind === "free") continue;
    const run = scene.runs[part.id];
    const cam = part.group ? cams[part.group.index] : cams.world;
    const f = TURN.emit(run, cam, TURN.frame(run));
    let root = part.group;
    while (root && root.parent) root = root.parent;
    const pivot = root ? [root.origin[0], root.origin[1], 0] : [0, 0, 0];
    const a = k.iso(pivot, P);
    const b = k.iso(pivot, Pp);
    const shift = [a[0] - b[0], a[1] - b[1]];
    let F, s0, s1, r0, r1, ends, profile;
    if (part.kind === "lathe") {
      const { curve, knots } = part.world;
      F = part.world.F;
      s0 = curve.s0;
      s1 = curve.s1;
      r0 = curve.radius(s0);
      r1 = curve.radius(s1);
      ends = ["flat", "flat"];
      profile = [[s0, r0], ...knots.slice(1, -1).map((sv) => [sv, curve.radius(sv), 1]), [s1, r1]];
    } else ({ F, s0, s1, r0, r1, ends, profile } = part.world);
    const meridian = [[profile[0][0], 0], ...profile.filter((p) => p[1] > 1e-9), [profile[profile.length - 1][0], 0]];
    const domed = ends.includes("dome");
    const radius = (s) => {
      if (ends[0] === "dome" && s < s0) return Math.sqrt(Math.max(0, r0 * r0 - (s - s0) ** 2));
      if (ends[1] === "dome" && s > s1) return Math.sqrt(Math.max(0, r1 * r1 - (s - s1) ** 2));
      return r0 + ((r1 - r0) * (Math.max(s0, Math.min(s1, s)) - s0)) / (s1 - s0);
    };
    const slope = (s) => {
      if (ends[0] === "dome" && s < s0) return -(s - s0) / Math.max(1e-4, radius(s));
      if (ends[1] === "dome" && s > s1) return -(s - s1) / Math.max(1e-4, radius(s));
      return (r1 - r0) / (s1 - s0);
    };
    const lo = ends[0] === "dome" ? s0 - r0 : s0;
    const hi = ends[1] === "dome" ? s1 + r1 : s1;
    const knots = [];
    for (let i = 0; i <= 64; i++) knots.push(lo + ((hi - lo) * i) / 64);
    const analytic = [[lo, 0], ...knots.slice(1, -1).map((sv) => [sv, radius(sv), 1]), [hi, 0]];
    if (ends[0] !== "dome") analytic.splice(1, 0, [lo, r0]);
    if (ends[1] !== "dome") analytic.splice(analytic.length - 1, 0, [hi, r1]);
    const lathed = part.kind === "lathe" ? { slope: part.world.curve.slope, radius: part.world.curve.radius } : null;
    const kit = lathed ? G.lathe(meridian, F, Pp, { smooth: lathed }) : domed ? G.lathe(analytic, F, Pp, { smooth: { slope, radius } }) : G.lathe(meridian, F, Pp, { smooth: true });
    const kitFine = lathed ? G.lathe(meridian, F, Pf, { smooth: lathed }) : domed ? G.lathe(analytic, F, Pf, { smooth: { slope, radius } }) : G.lathe(meridian, F, Pf, { smooth: true });
    const sh = (pts) => pts.map(([x, y]) => [x + shift[0], y + shift[1]]);
    const mine = subpaths(f.d[run.role.edge])[0];
    const theirs = sh(subpaths(kit.fill)[0]);
    const fineRef = subpaths(kitFine.fill)[0].map(([x, y]) => [x / 10 + shift[0], y / 10 + shift[1]]);
    const ho = hausdorff(mine, theirs);
    const hf = hausdorff(mine, fineRef);
    stats.outline = Math.max(stats.outline, ho);
    stats.outlineFine = Math.max(stats.outlineFine, hf);
    stats.parts++;
    const ours = [0, 1, 2, 3, 4].map((t) => ({ tone: t, d: f.d[run.role[`s${t}`]] }));
    for (const end of [0, 1]) {
      const role = run.role[`k${end}`];
      if (role === undefined || !f.d[role]) continue;
      const q = f.q[role];
      ours.push({ tone: q % TURN.q === 0 ? q / TURN.q : 9, d: f.d[role] });
    }
    const theirsLayers = [...kitFine.shades.map((d, t) => ({ tone: t, d })), { tone: 4, d: kitFine.top }].map(({ tone, d }) => ({ tone, d: subpaths(d).map((poly) => "M" + poly.map(([x, y]) => `${x / 10 + shift[0]} ${y / 10 + shift[1]}`).join("L") + "Z").join("") }));
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (const [x, y] of mine) {
      x0 = Math.min(x0, x);
      y0 = Math.min(y0, y);
      x1 = Math.max(x1, x);
      y1 = Math.max(y1, y);
    }
    const box = [x0 - 1, y0 - 1, x1 + 1, y1 + 1];
    const A = raster(ours, box, 0.2);
    const B = raster(theirsLayers, box, 0.2);
    const inside = raster([{ tone: 1, d: f.d[run.role.edge] }], box).out;
    const capCells = raster([0, 1].map((end) => ({ tone: 1, d: run.role[`k${end}`] !== undefined ? f.d[run.role[`k${end}`]] : "" })), box, 0.6).out;
    const capExempt = part.kind === "lathe";
    const insideKit = raster([{ tone: 1, d: "M" + fineRef.map(([x, y]) => `${x} ${y}`).join("L") + "Z" }], box).out;
    let worstGap = 0;
    const reach = Math.ceil(0.5 / CELL) + 2;
    for (let row = 0; row < A.h; row++)
      for (let c = 0; c < A.w; c++) {
        const i = row * A.w + c;
        if (inside[i] < 0 || insideKit[i] < 0) continue;
        const ta = A.out[i];
        const tb = B.out[i];
        if (ta === 9 || ta < 0 || tb < 0) continue;
        if (capExempt && capCells[i] >= 0) continue;
        stats.cells++;
        if (ta === tb) continue;
        let edge = false;
        for (let dy = -3; dy <= 3 && !edge; dy++) for (let dx = -3; dx <= 3; dx++) if (inside[(row + dy) * A.w + c + dx] < 0) edge = true;
        if (edge) continue;
        stats.mismatched++;
        let near = Infinity;
        for (let dy = -reach; dy <= reach; dy++)
          for (let dx = -reach; dx <= reach; dx++) {
            const rr = row + dy;
            const cc = c + dx;
            if (rr < 0 || cc < 0 || rr >= A.h || cc >= A.w) continue;
            if (B.out[rr * A.w + cc] === ta) near = Math.min(near, Math.hypot(dx, dy) * CELL);
          }
        worstGap = Math.max(worstGap, near);
      }
    stats.bands = Math.max(stats.bands, worstGap);
    if (worstGap > 0.5 || hf > 0.05) worst.push({ part: part.name, theta, band: worstGap, outline: hf });
  }
}

  return { pass: stats.outlineFine <= 0.05 && stats.bands <= 0.5, text: `rounds: ${stats.parts} frames at ${angles.length} angles: outline vs kit ${stats.outline.toFixed(4)} px (kit rounds to 0.1), vs kit at 10x ${stats.outlineFine.toFixed(4)} px · band boundaries ≤ ${stats.bands.toFixed(2)} px apart (${stats.mismatched} of ${stats.cells} interior cells differ)`, worst: worst.sort((a, b) => b.band - a.band || b.outline - a.outline).slice(0, 12).map((w) => `${w.part} at ${w.theta}°: outline ${w.outline.toFixed(3)} px, band ${w.band.toFixed(2)} px`) };
}

export function tubes(T, P, angles) {
  const scene = TURN.prepare(T.data, { unit: 1 });
const subpaths = (d) => [...(d ?? "").matchAll(/M([^M]*)/g)].map(([, body]) => {
    const nums = (body.match(/-?\d+(?:\.\d+)?(?:e-?\d+)?/g) ?? []).map(Number);
    const pts = [];
    for (let i = 0; i + 1 < nums.length; i += 2) pts.push([nums[i], nums[i + 1]]);
    return pts;
  });
  const segDist = (p, a, b) => {
    const dx = b[0] - a[0], dy = b[1] - a[1], l = dx * dx + dy * dy;
    const t = l ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l)) : 0;
    return Math.hypot(p[0] - a[0] - dx * t, p[1] - a[1] - dy * t);
  };
  const compare = (mine, theirs) => {
    const A = subpaths(mine);
    const B = subpaths(theirs);
    if (A.length !== B.length) return { error: Infinity, why: `subpaths ${A.length} vs ${B.length}` };
    let worst = 0;
    for (let i = 0; i < A.length; i++) {
      if (A[i].length === B[i].length) for (let j = 0; j < A[i].length; j++) worst = Math.max(worst, Math.abs(A[i][j][0] - B[i][j][0]), Math.abs(A[i][j][1] - B[i][j][1]));
      else for (const p of A[i]) worst = Math.max(worst, Math.min(...B[i].slice(1).map((q, j) => segDist(p, B[i][j], q))));
    }
    return { error: worst };
  };

const stats = { body: 0, shine: 0, shade: 0, edges: 0, rings: 0, chunks: 0 };
const bad = [];
for (const theta of angles) {
  const cams = TURN.cams(scene, restValues(T, theta));
  const Pp = { ...P, azimuth: P.azimuth - theta };
  const Pf = { ...Pp, origin: [Pp.origin[0] * 10, Pp.origin[1] * 10], scale: Pp.scale * 10 };
  for (const route of T.routes) {
    if (route.group && (!rootOf(route.group) || route.group.kind === "free")) continue;
    const root = rootOf(route.group);
    const pivot = root ? [root.origin[0], root.origin[1], 0] : [0, 0, 0];
    const a = k.iso(pivot, P);
    const b = k.iso(pivot, Pp);
    const shift = [a[0] - b[0], a[1] - b[1]];
    const kit = tubePieces(route.world, route.r, Pf, { maxLength: route.chunk, spacing: route.spacing, gaps: route.gaps, breaks: route.breaks, caps: route.caps, stripes: route.wide, rings: route.rings });
    const scaled = (d) => subpaths(d).map((pts) => "M" + pts.map(([x, y]) => `${x / 10 + shift[0]} ${y / 10 + shift[1]}`).join("L")).join("");
    const chunks = T.parts.filter((part) => part.route === route.name);
    if (kit.length !== chunks.length) {
      bad.push(`${route.name}: ${chunks.length} chunks vs kit ${kit.length}`);
      continue;
    }
    chunks.forEach((part, index) => {
      const run = scene.runs[part.id];
      const f = TURN.emit(run, route.group ? cams[route.group.index] : cams.world, TURN.frame(run));
      const piece = kit[index];
      stats.chunks++;
      for (const key of ["body", "shine", "shade", "edges"]) {
        if (run.role[key] === undefined) continue;
        const { error, why } = compare(f.d[run.role[key]], scaled(piece[key]));
        stats[key] = Math.max(stats[key], error);
        if (error > 0.05) bad.push(`${part.name} ${key} at ${theta}°: ${why ?? error.toFixed(3)}`);
      }
      if (route.rings) {
        const mine = run.roles.filter((role) => role.name.startsWith("r")).map((role) => f.d[run.role[role.name]]).join("");
        const theirs = scaled(piece.rings.filter((ring) => ring.alpha * 32 >= 0.5).map((ring) => ring.d).join(""));
        const A = subpaths(mine);
        const B = subpaths(theirs);
        let worst = A.length === B.length ? 0 : Infinity;
        for (const seg of A) {
          let best = Infinity;
          for (const other of B) best = Math.min(best, Math.max(...seg.map((p, j) => Math.hypot(p[0] - other[j][0], p[1] - other[j][1]))));
          worst = Math.max(worst, best);
        }
        stats.rings = Math.max(stats.rings, worst);
        if (worst > 0.05) bad.push(`${part.name} rings at ${theta}°: ${A.length} vs ${B.length} segments, worst ${worst.toFixed(3)}`);
      }
    });
  }
}

  return { pass: !bad.length, text: `tubes: ${stats.chunks} chunk frames at ${angles.length} angles against tubePieces at 10x: body ${stats.body.toFixed(4)} · shine ${stats.shine.toFixed(4)} · shade ${stats.shade.toFixed(4)} · edges ${stats.edges.toFixed(4)} · rings ${stats.rings.toFixed(4)} px`, worst: bad.slice(0, 20) };
}
