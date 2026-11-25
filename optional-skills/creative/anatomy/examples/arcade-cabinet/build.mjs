import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../kit/iso-kit.mjs";
import * as G from "../../kit/lathe.mjs";
import { tubePieces, tubeSvg } from "../../kit/tube.mjs";
import * as A from "../../kit/audit.mjs";
import { kitScript } from "../../scripts/inline-kit.mjs";
import { makeParts } from "./parts.mjs";
import { SCREEN } from "./screen.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const THEME = process.argv.includes("--light") ? "light" : "dark";

const W = 600;
const H = 760;
const AZIMUTH = 146;
const ELEVATION = 24;

const PLATE = { x: -98, y: -18, w: 184, d: 236, r: 10 };
const PLATE_H = 10;
const FOOT_H = 5;
const XI = 60;
const XW = 64;
const TOPZ = 358;

const P = k.fitProjection(
  [...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0), [-XW, 0, TOPZ], [XW, 0, TOPZ], [XW, 126, TOPZ], [-XW, 126, TOPZ], [-XW, 178, 184], [XW, 178, 184]],
  W,
  H,
  { pad: 30, azimuth: AZIMUTH, elevation: ELEVATION },
);
const V = G.viewOf(P);
const KS = G.scaleOf(P);
const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const Ln = (d, style) => k.lineSvg(d, style);
const Dots = (points, style) => k.dotsSvg(points, style);
const Faded = (list, style) => k.fadedSvg(list, style);
const path3 = (points, closed = false) => k.pathOf(points.map(at), closed);
const f2 = (v) => Math.round(v * 100) / 100;

const R = A.recorder(P);
const record = R.solid;
function put(name, ref, svg, shapes = [], { bias = 0, ...rest } = {}) {
  return R.put({ name, at: ref, bias, svg, shapes, piece: "cab", ...rest });
}

const signed = (poly) => poly.reduce((a, [x, y], i) => {
  const [x2, y2] = poly[(i + 1) % poly.length];
  return a + x * y2 - x2 * y;
}, 0);
const screenOriented = (pts) => (signed(pts) < 0 ? [...pts].reverse() : pts);
function runsOf(flags, want) {
  const count = flags.length;
  const runs = [];
  const start = flags.findIndex((flag, index) => flag !== flags[(index + count - 1) % count]);
  if (start < 0) return flags[0] === want ? [Array.from({ length: count + 1 }, (_, index) => index % count)] : [];
  let run = null;
  for (let step = 0; step < count; step++) {
    const index = (start + step) % count;
    if (flags[index] === want) {
      if (!run) run = [index];
      run.push((index + 1) % count);
    } else if (run) {
      runs.push(run);
      run = null;
    }
  }
  if (run) runs.push(run);
  return runs;
}

function prismPaths(F, poly, s0, s1) {
  const ring = signed(poly) > 0 ? poly : [...poly].reverse();
  const n = ring.length;
  const W3 = (s, [x, y]) => [F.o[0] + F.a[0] * s + F.u[0] * x + F.v[0] * y, F.o[1] + F.a[1] * s + F.u[1] * x + F.v[1] * y, F.o[2] + F.a[2] * s + F.u[2] * x + F.v[2] * y];
  const capNear = G.dot3(F.a, V) > 0;
  const [nearS, farS] = capNear ? [s1, s0] : [s0, s1];
  const near = ring.map((p) => at(W3(nearS, p)));
  const far = ring.map((p) => at(W3(farS, p)));
  const normals = ring.map(([x0, y0], i) => {
    const [x1, y1] = ring[(i + 1) % n];
    return G.unit3(G.add3(G.mul3(F.u, y1 - y0), G.mul3(F.v, -(x1 - x0))));
  });
  const facing = normals.map((m) => G.dot3(m, V) > 1e-9);
  const outline = [];
  const crease = [];
  for (const run of runsOf(facing, false)) outline.push(k.pathOf(run.map((i) => near[i])));
  for (const run of runsOf(facing, true)) {
    crease.push(k.pathOf(run.map((i) => near[i])));
    outline.push(k.pathOf(run.map((i) => far[i])));
  }
  for (let i = 0; i < n; i++) if (facing[(i + n - 1) % n] !== facing[i]) outline.push(k.pathOf([near[i], far[i]]));
  const parts = [k.pathOf(screenOriented(near), true), k.pathOf(screenOriented(far), true)];
  const buckets = [[], [], [], [], []];
  for (let i = 0; i < n; i++) {
    if (!facing[i]) continue;
    const quad = screenOriented([near[i], near[(i + 1) % n], far[(i + 1) % n], far[i]]);
    parts.push(k.pathOf(quad, true));
    buckets[G.toneOf(normals[i], P)].push(k.pathOf(quad, true));
  }
  const capTone = G.toneOf(capNear ? F.a : G.mul3(F.a, -1), P);
  buckets[capTone].push(k.pathOf(screenOriented(near), true));
  return { fill: parts.join(""), outline: outline.join(""), crease: crease.join(""), top: buckets[4].join(""), shades: buckets.slice(0, 4).map((b) => b.join("")) };
}

const XF = G.frameOf([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]);
function board(name, poly, x0, x1, style = {}, { extra = "", inner = "", bias = 0, holes = [], ref } = {}) {
  const shape = record(A.prism(XF, poly, x0, x1, { name }));
  const cy = poly.reduce((a, p) => a + p[0], 0) / poly.length;
  const cz = poly.reduce((a, p) => a + p[1], 0) / poly.length;
  const dots = holes.length ? Dots(holes.map(([y, z]) => at([x0, y, z])), { size: 0.5, tone: "lo" }) : "";
  put(name, ref ?? [(x0 + x1) / 2, cy, cz], S(prismPaths(XF, poly, x0, x1), { tone: "mid", ...style, inner }) + dots + extra, [shape], { bias });
  return shape;
}
function prism(name, F, poly, s0, s1, style = {}, { extra = "", inner = "", bias = 0, ref } = {}) {
  const shape = record(A.prism(F, poly, s0, s1, { name }));
  const cx = poly.reduce((a, p) => a + p[0], 0) / poly.length;
  const cy = poly.reduce((a, p) => a + p[1], 0) / poly.length;
  const centre = G.add3(G.add3(G.add3(F.o, G.mul3(F.a, (s0 + s1) / 2)), G.mul3(F.u, cx)), G.mul3(F.v, cy));
  put(name, ref ?? centre, S(prismPaths(F, poly, s0, s1), { tone: "mid", ...style, inner }) + extra, [shape], { bias });
  return shape;
}
const rect = (u0, u1, v0, v1) => [[u0, v0], [u1, v0], [u1, v1], [u0, v1]];
function box(name, [x0, y0, z0], [x1, y1, z1], style = {}, { r = 0.4, steps = 2, bevel = 0, extra = "", inner = "", bias = 0 } = {}) {
  const plan = { x: x0, y: y0, w: x1 - x0, d: y1 - y0, r };
  const shape = record(A.slab(plan, z0, z1 - z0, { name }, steps));
  put(name, [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2], S(k.slabOf(plan, z0, z1 - z0, P, steps, bevel), { tone: "mid", ...style, inner }) + extra, [shape], { bias });
  return shape;
}
function turned(name, F, profile, style = {}, { extra = "", inner = "", bias = 0, smooth, steps, rims, bevel = 0, owner } = {}) {
  const opts = { bevel, ...(smooth !== undefined ? { smooth } : {}), ...(steps ? { steps } : {}), ...(rims ? { rims } : {}) };
  const paths = G.solidOf(profile, F, P, opts);
  const mid = (profile[0][0] + profile[profile.length - 1][0]) / 2;
  const shape = record(A.solid(F, profile, { name, owner, radius: smooth && smooth.radius ? smooth.radius : null }));
  put(name, G.add3(F.o, G.mul3(F.a, mid)), S(paths, { tone: "mid", ...style, inner }) + extra, [shape], { bias });
  return shape;
}
function disc(name, F, s0, s1, r, style = {}, options = {}) {
  if (options.owner === undefined && name.startsWith("crt.")) options = { ...options, owner: "crt" };
  return turned(name, F, [[s0, r], [s1, r]], style, options);
}
function ball(name, c, r, style = {}, { bias = 0, extra = "" } = {}) {
  const shape = record(A.ball(c, r, { name }));
  put(name, c, S(G.sphereOf(c, r, P, { steps: 32 }), { tone: "mid", ...style }) + extra, [shape], { bias });
  return shape;
}
function planeMatrix(o, ex, ey) {
  const p0 = at(o);
  const px = at(G.add3(o, ex));
  const py = at(G.add3(o, ey));
  const v = (x) => x.toFixed(4);
  return `matrix(${v(px[0] - p0[0])} ${v(px[1] - p0[1])} ${v(py[0] - p0[0])} ${v(py[1] - p0[1])} ${p0[0].toFixed(2)} ${p0[1].toFixed(2)})`;
}

const CABLES = [];
function cable(name, points, r = 0.5, { tone = "mid", bend = 3, ends = [true, true], shells = [false, false], rings, maxLength = 16, bias = 0.05, limp = true, touch, bundle } = {}) {
  const route = G.fillet(points, Math.max(bend, r * 2.6), 10, { P, tube: r });
  const total = G.pathLength(route);
  const SHELL = Math.max(1.6, r * 2.4);
  const gaps = [...(shells[0] ? [[0, SHELL]] : []), ...(shells[1] ? [[total - SHELL, total]] : [])];
  R.route(name, route, r, { gaps, limp, touch, bundle, rims: shells.map((on) => (on ? r * 1.9 : r)), piece: "cab" });
  CABLES.push(name);
  tubePieces(route, r, P, { maxLength, spacing: Math.max(0.6, Math.min(1.6, r)), gaps, ...(rings ? { rings } : {}) }).forEach((piece, index) =>
    put(`${name}:${index}`, piece.mid, tubeSvg(piece, { tone }), [A.tube(piece.points, piece.radii)], { bias, route: name, chunk: index, attrs: ` data-route="${name}" data-chunk="${index}"` }),
  );
  [0, 1].forEach((index) => {
    const end = index ? route[route.length - 1] : route[0];
    if (shells[index]) {
      const tangent = G.tangentAlong(route, index ? total : 0);
      const F = G.frameAlong(end, tangent);
      const [s0, s1] = index ? [-SHELL, 0] : [0, SHELL];
      const shape = record(A.disc(F, s0, s1, r * 1.9, { name: `${name}.shell${index}`, owner: name }));
      put(`${name}.shell${index}`, G.add3(end, G.mul3(tangent, (s0 + s1) / 2)), S(G.disc(s0, s1, r * 1.9, F, P, { steps: 20, bevel: 0.2 }), { tone: "mid" }), [shape], { bias: bias + 0.1 });
    } else if (ends[index]) {
      put(`${name}.end${index}`, end, Dots([at(end)], { size: Math.max(0.5, r * KS * 0.9), tone: "hi" }), [A.ball(end, Math.max(0.5, r * KS * 0.9) / KS, { name: `${name}.end${index}`, owner: name })], { bias: bias + 0.15 });
    }
  });
  return route;
}

const back = [];
back.push(`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
for (const [x, y] of byDepth(k.corners(PLATE, 22))) {
  back.push(S(k.cylinder(x, y, 7, -PLATE_H - FOOT_H, 1.6, P, 28), { tone: "lo" }));
  back.push(S(k.cylinder(x, y, 5.4, -PLATE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: "mid" }));
}
back.push(S(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 8, 1.6), { tone: "mid", inner: Ln(k.sideSeam(PLATE, -4, P, 8), { tone: "faint" }) }));
back.push(Ln(k.planOutline(k.insetPlan(PLATE, 5), 0, P), { tone: "lo" }));
const plateScrews = k.corners(PLATE, 10);
back.push(Ln(plateScrews.map(([x, y]) => k.ring(x, y, 2.2, 0, P, 16)).join(""), { tone: "lo" }));
back.push(Dots(plateScrews.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
record(A.slab(PLATE, -PLATE_H, PLATE_H, { name: "stand.plate", ground: true }, 8));
const STRIP = { x: PLATE.x + 14, y: PLATE.y + PLATE.d - 13, w: PLATE.w - 28, d: 6, r: 1 };
back.push(S(k.slabOf(STRIP, 0, 0.6, P, 3), { tone: "lo", crease: "none" }));
const ticks = k.topTicks(STRIP.x + 4, STRIP.x + STRIP.w - 4, 4, 5, STRIP.y, 0.6, [1.8, 3.4], P);
back.push(Ln(ticks.minor, { tone: "lo" }), Ln(ticks.major, { tone: "mid" }));
const LABEL = { x: 48, y: PLATE.y + PLATE.d - 34, w: 30, d: 13, r: 1.2 };
back.push(S(k.slabOf(LABEL, 0, 0.8, P, 3), { tone: "lo", crease: "none" }));
back.push(Ln(k.planOutline(k.insetPlan(LABEL, 1.4), 0.8, P, 3), { tone: "faint" }));
back.push(Ln([20, 14, 17].map((w, i) => k.lineOnTop([LABEL.x + 4, LABEL.y + 3.4 + i * 3], [LABEL.x + 4 + w, LABEL.y + 3.4 + i * 3], 0.8, P)).join(""), { tone: "lo", free: true }));
back.push(Dots([at([LABEL.x + LABEL.w - 2.6, LABEL.y + 2.4, 0.8]), at([LABEL.x + LABEL.w - 2.6, LABEL.y + LABEL.d - 2.4, 0.8])], { size: 0.45 }));

const TRAY = { x: -74, y: 188 };
const STACK = 7;
back.push(`<path class="iso-halo" d="${k.haloOf({ x: TRAY.x - 9, y: TRAY.y - 9, w: 18, d: 18, r: 9 }, 0, 2, P)}"/>`);
const COIN_STEP = 0.42;
const stackTop = (n) => 1.6 + n * COIN_STEP;

const BEZEL_T = [110, 306];
const BEZEL_B = [136, 195];
const WALL = [[0, 4], [156, 4], [156, 168], [178, 170], [178, 184], BEZEL_B, BEZEL_T, [126, 314], [126, 358], [0, 358]];
const bd = G.unit3([0, BEZEL_B[0] - BEZEL_T[0], BEZEL_B[1] - BEZEL_T[1]]);
const NB = [0, -bd[2], bd[1]];
const C3 = [0, (BEZEL_T[0] + BEZEL_B[0]) / 2, (BEZEL_T[1] + BEZEL_B[1]) / 2];
const CPD = G.unit3([0, 178 - 136, 184 - 195]);
const CPN = [0, -CPD[2], CPD[1]];
const cpTop = (y) => 195 + ((y - 136) * (184 - 195)) / (178 - 136);

for (const [x, y] of byDepth([[-54, 12], [54, 12], [-54, 144], [54, 144]])) {
  const F = G.frameOf([x, y, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]);
  const shapes = [record(A.disc(F, 0, 1.2, 3.4, { name: `leveler.pad` })), record(A.disc(F, 1.2, 2.4, 2.4, { name: `leveler.nut` })), record(A.disc(F, 2.4, 4, 1, { name: `leveler.stem` }))];
  put(`leveler`, [x, y, 2], S(G.disc(0, 1.2, 3.4, F, P, { steps: 32 }), { tone: "lo" }) + S(G.disc(1.2, 2.4, 2.4, F, P, { steps: 6 }), { tone: "mid" }) + S(G.disc(2.4, 4, 1, F, P, { steps: 16 }), { tone: "mid" }), shapes);
}

const wallPaths = prismPaths(XF, WALL, XI, XW);
const ringW = signed(WALL) > 0 ? WALL : [...WALL].reverse();
const tFacing = ringW.map(([y0, z0], i) => {
  const [y1, z1] = ringW[(i + 1) % ringW.length];
  return G.dot3(G.unit3([0, z1 - z0, -(y1 - y0)]), V) > 1e-9;
});
const bead = runsOf(tFacing, true).map((run) => path3(run.map((i) => [62, ...ringW[i]]))).join("");
{
  const shape = record(A.prism(XF, WALL, XI, XW, { name: "wall" }));
  put("wall", [XW, 80, 180], S(wallPaths, { tone: "mid", inner: Ln(bead, { tone: "lo" }) }), [shape], { bias: -400 });
}
box("cleat.floor", [56, 8, 8], [60, 148, 12], { tone: "lo" });
box("cleat.back", [56, 4, 12], [60, 8, 350], { tone: "lo" });
box("cleat.top", [56, 8, 350], [60, 122, 354], { tone: "lo" });
box("cleat.front", [56, 148, 12], [60, 152, 164], { tone: "lo" });

board("floor", [[4, 4], [152, 4], [152, 8], [4, 8]], -XI, XI, { tone: "lo" }, { holes: [[24, 6], [130, 6]] });
board("back", [[0, 4], [4, 4], [4, 354], [0, 354]], -XI, XI, {}, { holes: [[2, 40], [2, 180], [2, 320]], bias: -300 });
board("top", [[0, 354], [126, 354], [126, 358], [0, 358]], -XI, XI, {}, {
  holes: [[30, 356], [100, 356]],
  extra: Ln(Array.from({ length: 9 }, (_, i) => k.planOutline({ x: -36 + i * 8, y: 14, w: 3, d: 30, r: 1.5 }, 358, P, 3)).join(""), { tone: "faint" }),
});
board("front", [[152, 4], [156, 4], [156, 168], [152, 168]], -XI, XI, {}, { holes: [[154, 20], [154, 150]] });
box("kick", [-XI, 156, 6], [XI, 156.8, 24], { tone: "lo" }, { r: 0.3, extra: Dots([-54, -18, 18, 54].flatMap((x) => [at([x, 156.8, 9]), at([x, 156.8, 21])]), { size: 0.45, tone: "hi" }) });
board("marquee.shelf", [[98, 318], [126, 318], [126, 322], [98, 322]], -XI, XI, { tone: "lo" });
const SPK = [[110, 306], [126, 314], [124.2, 317.6], [108.2, 309.6]];
board("speaker.board", SPK, -XI, XI);
board("cp.top", [[136, 195], [178, 184], [177.24, 181.1], [135.24, 192.1]], -XI, XI, { tone: "mid" }, { holes: [[157, 188]] });
board("cp.front", [[174, 170], [178, 170], [178, 180.89], [174, 181.94]], -XI, XI);
board("cp.bottom", [[152, 168], [174, 170], [174, 174], [152, 172]], -XI, XI, { tone: "lo" });
{
  const m = planeMatrix([0, 125.02, 338], [1, 0, 0], [0, 0, -1]);
  const sc = 1.7;
  const w = (7 * 8 - 3) * sc;
  const art = `<g class="ac-marquee" transform="${m}"><path class="ac-marquee-art" d="${SCREEN.text("ANATOMY", -w / 2, -6, sc)}"/><rect class="ac-marquee-frame" x="-56" y="-13.6" width="112" height="27.2" rx="2"/></g>`;
  board("marquee.plexi", [[123, 322], [125, 322], [125, 354], [123, 354]], -XI, XI, { tone: "faint" }, { extra: art });
}
const TUBE_F = G.frameOf([-46, 110, 338], [1, 0, 0], [0, 1, 0], [0, 0, 1]);
turned("marquee.tube", TUBE_F, [[0, 3.6], [92, 3.6]], { tone: "hi" }, { inner: Ln(G.arcOf(TUBE_F, 4, 3.6, P) + G.arcOf(TUBE_F, 88, 3.6, P), { tone: "faint" }), extra: "" });
box("marquee.socket", [-51, 106, 333], [-46, 114, 342], { tone: "mid" });
box("marquee.socket", [46, 106, 333], [51, 114, 342], { tone: "mid" });
box("marquee.bracket", [-51, 106, 342], [-49, 114, 354], { tone: "lo" });
box("marquee.bracket", [49, 106, 342], [51, 114, 354], { tone: "lo" });
box("marquee.ballast", [8, 99, 322], [40, 105, 327], { tone: "mid" }, { inner: Ln(k.sideSeam({ x: 8, y: 99, w: 32, d: 6, r: 0.4 }, 324.5, P, 2), { tone: "faint" }), extra: Dots([at([10, 105, 327]), at([38, 105, 327])], { size: 0.45 }) });
turned("marquee.starter", G.frameOf([44, 103, 322], [0, 0, 1], [1, 0, 0], [0, 1, 0]), [[0, 1.8], [5, 1.8], [5.6, 1.3, 1], [6, 0.6, 1]], { tone: "mid" });

const SPK_N = G.unit3([0, -8, 16]);
const SPK_C = [22, 115, 313.0];
const SPK_F = G.frameAlong(SPK_C, SPK_N);
turned("speaker.basket", SPK_F, [[0, 5.6], [0.8, 5.6], [0.8, 4.8], [2.4, 2.8, 1], [2.8, 2.4]], { tone: "mid" });
disc("speaker.magnet", SPK_F, 2.8, 3.8, 2.6, { tone: "lo" });

const MON_A = G.mul3(NB, -1);
const CF = G.add3(C3, G.mul3(NB, -5));
const MON = G.frameOf(CF, MON_A, [1, 0, 0], bd);
const BEZEL_OPEN = { u: 29.5, w: 37.5, r: 4 };
const monPoint = (u, w, s) => G.add3(G.add3(G.add3(CF, G.mul3([1, 0, 0], u)), G.mul3(bd, w)), G.mul3(MON_A, s));
const funnel = (s) => 3 + 31 * Math.pow(Math.max(0, (64 - s) / 55), 1.4);
const funnelSlope = (s) => (-31 * 1.4 * Math.pow(Math.max(1e-6, (64 - s) / 55), 0.4)) / 55;
prism("crt.face", MON, rect(-34, 34, -42, 42).map((p) => p), 0, 3, { tone: "hi" });
const BAND = [[-35, -43], [35, -43], [35, 43], [-35, 43]];
prism("crt.band", MON, BAND, 3, 9, { tone: "mid" }, { inner: "" });
for (const [su, sv] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
  const ear = [[su * 35, sv * 37], [su * 41, sv * 37], [su * 41, sv * 43], [su * 35, sv * 43]];
  prism("crt.ear", MON, ear, 5, 6, { tone: "mid" }, { extra: Dots([at(monPoint(su * 38.6, sv * 40, 5))], { size: 0.5, tone: "hi" }) });
}
const FUNNEL = G.bandsOf([[9, funnel(9)], [62, funnel(62)]], [48], MON, P, { radius: funnel, slope: funnelSlope, rows: 6 });
const funnelFront = FUNNEL.bands[0];
{
  const shape = record(A.solid(MON, funnelFront.profile, { name: "crt.funnel", radius: funnel }));
  const sides = FUNNEL.sidesOf(9, 48, { ends: false });
  put("crt.funnel", G.add3(CF, G.mul3(MON_A, 26)), S(funnelFront.paths, { tone: "hi" }) + Ln(sides.d, { tone: "hi" }), [shape]);
}
turned("crt.yoke", MON, [[48, 15.5], [52, 15.5], [60, 11, 1], [64, 8]], { tone: "mid" }, { inner: Ln(G.arcOf(MON, 52, 15.5, P), { tone: "faint" }) });
for (const [s0, s1] of [[64, 69], [70.4, 72.4], [73.8, 88.4]]) disc("crt.neck", MON, s0, s1, 3, { tone: "hi" });
disc("crt.ring", MON, 69, 70.4, 4.2, { tone: "mid" });
disc("crt.ring", MON, 72.4, 73.8, 4.2, { tone: "mid" });
prism("crt.neckboard", MON, rect(-7, 7, -7, 7), 90, 91, { tone: "mid" }, { extra: "" });
disc("crt.socket", MON, 88.4, 90, 4.4, { tone: "lo" });
const ANODE_ANGLE = -2.35;
const anodeAt = G.pointOf(MON, 30, funnel(30), ANODE_ANGLE);
const anodeN = G.unit3(G.sub3(G.radialOf(MON, ANODE_ANGLE), G.mul3(MON.a, funnelSlope(30))));
const ANODE_F = G.frameAlong(G.add3(anodeAt, G.mul3(anodeN, -0.4)), anodeN);
turned("crt.anode", ANODE_F, [[0, 3.6], [0.8, 3.4, 1], [1.8, 2.4, 1], [2.4, 1.2, 1], [2.6, 0.6]], { tone: "lo" });

for (const su of [-1, 1]) prism("mon.angle", MON, [[su * 41, -46], [su * 44, -46], [su * 44, 46], [su * 41, 46]].map(([u, w]) => [u, w]), 4, 8, { tone: "mid" });
for (const su of [-1, 1]) prism("mon.rail", MON, [[su * 36, 46], [su * 44, 46], [su * 44, 48], [su * 36, 48]], 4, 66, { tone: "mid" }, { extra: Dots([at(monPoint(su * 40, 46, 20)), at(monPoint(su * 40, 46, 50))], { size: 0.45 }) });
prism("mon.crossbar", MON, [[-36, 46], [36, 46], [36, 48], [-36, 48]], 62, 66, { tone: "mid" });
prism("mon.shelf", MON, [[-XI, 48], [XI, 48], [XI, 52], [-XI, 52]], -2, 70, { tone: "lo" });
prism("mon.cleat", MON, [[56, 52], [XI, 52], [XI, 56], [56, 56]], 2, 66, { tone: "lo" });
board("mon.post", [[130, 174], [134, 174], [134, 194.6], [130, 195.6]], -XI, XI, { tone: "lo" });

const CH = G.frameOf(monPoint(-44, 0, 0), [-1, 0, 0], MON_A, bd);
const chPoint = (s, w, out = 0) => G.add3(G.add3(G.add3(monPoint(-44, 0, 0), G.mul3(MON_A, s)), G.mul3(bd, w)), [-out, 0, 0]);
prism("chassis.board", CH, [[22, 14], [64, 14], [64, 46], [22, 46]], 0, 1, { tone: "lo" });
prism("chassis.flyback", CH, [[46, 18], [56, 18], [56, 29], [46, 29]], 1, 7, { tone: "mid" }, { inner: "" });
disc("chassis.flyback.coil", G.frameAlong(chPoint(51, 23.5, 7), [-1, 0, 0]), 0, 2.6, 4, { tone: "hi" });
for (const [s, w] of [[48, 31], [53, 31]]) disc("chassis.pot", G.frameAlong(chPoint(s, w, 1), [-1, 0, 0]), 0, 1.6, 1.1, { tone: "mid" }, { extra: Ln(k.pathOf([at(chPoint(s - 0.8, w, 2.7)), at(chPoint(s + 0.8, w, 2.7))]), { tone: "hi" }) });
prism("chassis.heatsink", CH, [[26, 17], [38, 17], [38, 27], [26, 27]], 1, 6, { tone: "mid" }, {
  extra: Ln([28, 30, 32, 34, 36].map((s) => k.pathOf([at(chPoint(s, 17, 6)), at(chPoint(s, 27, 6))])).join(""), { tone: "lo" }),
});
for (const [s, w, r, h] of [[30, 34, 1.8, 5], [35, 38, 1.4, 4], [47, 40, 1.6, 4.4], [42, 42, 1.2, 3.4]]) disc("chassis.cap", G.frameAlong(chPoint(s, w, 1), [-1, 0, 0]), 0, h, r, { tone: "hi" }, { inner: "" });
prism("chassis.plug.video", CH, [[56, 36], [62, 36], [62, 42], [56, 42]], 1, 3, { tone: "mid" });
prism("chassis.plug.neck", CH, [[57, 18], [62, 18], [62, 23], [57, 23]], 1, 3, { tone: "mid" });

const JOY = { x: -49, y: 160 };
const J_O = [JOY.x, JOY.y, cpTop(JOY.y)];
const CPF = (o) => G.frameOf(o, CPN, [1, 0, 0], CPD);
const cpPoint = (x, y, s = 0) => G.add3([x, y, cpTop(y)], G.mul3(CPN, s));
const overlay = [[-56, 139], [56, 139], [56, 175], [-56, 175]].map(([x, y]) => cpPoint(x, y, 0.05));
const STICK_DATA = {
  o: J_O,
  nc: CPN,
  dc: CPD,
  pivot: -4,
  switches: [
    { id: "left", c: G.add3(cpPoint(JOY.x - 4.6, JOY.y), G.mul3(CPN, -10.4)), out: [-1, 0, 0] },
    { id: "right", c: G.add3(cpPoint(JOY.x + 4.6, JOY.y), G.mul3(CPN, -10.4)), out: [1, 0, 0] },
    { id: "up", c: G.add3(G.add3(cpPoint(JOY.x, JOY.y), G.mul3(CPD, -4.6)), G.mul3(CPN, -10.4)), out: G.mul3(CPD, -1) },
    { id: "down", c: G.add3(G.add3(cpPoint(JOY.x, JOY.y), G.mul3(CPD, 4.6)), G.mul3(CPN, -10.4)), out: CPD },
  ],
};
const PARTS = makeParts(k, G, P, { stick: STICK_DATA });
const JF = CPF(J_O);
const plateShape = record(A.prism(JF, rect(-9, 9, -9, 9), -3.6, -3, { name: "stick.plate" }));
put("stick.plate", G.add3(J_O, G.mul3(CPN, -3.3)), S(prismPaths(JF, rect(-9, 9, -9, 9), -3.6, -3), { tone: "mid" }), [plateShape]);
const boltHeads = [[-7, -7], [7, -7], [-7, 7], [7, 7]].map(([u, v]) => at(G.add3(G.add3(J_O, G.mul3([1, 0, 0], u)), G.mul3(CPD, v))));
for (const sw of STICK_DATA.switches) {
  const side = G.cross3(CPN, sw.out);
  const F = G.frameOf(G.add3(sw.c, G.mul3(sw.out, 0.65)), CPN, sw.out, side);
  prism("stick.bracket", F, [[0, -2.2], [0.5, -2.2], [0.5, 2.2], [0, 2.2]], -1, 6.8, { tone: "lo" });
}
disc("stick.bushing", JF, -5.2, -3.6, 2.2, { tone: "lo" }, { owner: "stick" });
const PIV = G.frameAlong(G.add3(J_O, G.mul3(CPN, -4)), CPN);
const stickLowerShapes = [
  record(A.disc(PIV, -8.6, -4.6, 2.4, { name: "stick.actuator", owner: "stick" })),
  record(A.disc(PIV, -9.4, -0.6, 0.85, { name: "stick.shaft.lower", owner: "stick" })),
  record(A.disc(PIV, -10, -9.4, 1.5, { name: "stick.clip", owner: "stick" })),
  ...STICK_DATA.switches.map((sw) => record(A.prism(G.frameOf(sw.c, CPN, sw.out, G.cross3(CPN, sw.out)), rect(-0.65, 0.65, -2, 2), -1, 1, { name: `stick.switch.${sw.id}` }))),
  ...STICK_DATA.switches.flatMap((sw) => [-1.1, 0, 1.1].map((o) => record(A.prism(G.frameOf(G.add3(G.add3(sw.c, G.mul3(CPN, -1)), G.mul3(G.cross3(CPN, sw.out), o)), CPN, sw.out, G.cross3(CPN, sw.out)), rect(-0.3, 0.3, -0.35, 0.35), -1, 0, { name: `stick.tab.${sw.id}`, owner: `stick.switch.${sw.id}` })))),
];
put("stick.lower", G.add3(J_O, G.mul3(CPN, -9)), `<g data-part="stick-lower">${PARTS.stickLower(0, 0)}</g>`, stickLowerShapes);
put("stick.bolts", J_O, Dots(boltHeads, { size: 0.55, tone: "mid" }), [[-7, -7], [7, -7], [-7, 7], [7, 7]].map(([u, v]) => record(A.ball(G.add3(G.add3(J_O, G.mul3([1, 0, 0], u)), G.mul3(CPD, v)), 0.4, { name: "stick.bolt" }))), { bias: 0.1 });
put("stick.upper", G.add3(J_O, G.mul3(CPN, 8)), `<g data-part="stick-upper">${PARTS.stickUpper(0, 0)}</g>`, [record(A.ball(G.add3(J_O, G.mul3(CPN, 12.4)), 3.5, { name: "stick.ball", owner: "stick" })), record(A.disc(G.frameAlong(J_O, CPN), 0.4, 8.95, 0.85, { name: "stick.shaft", owner: "stick" })), record(A.disc(G.frameAlong(J_O, CPN), 0, 0.4, 4.4, { name: "stick.washer", owner: "stick" }))], { bias: 2 });

const BUTTONS = [
  { id: "fire", x: -18, y: 158, lamp: false },
  { id: "jump", x: -6, y: 162, lamp: false },
  { id: "p1", x: 18, y: 146, lamp: true },
  { id: "p2", x: 32, y: 146, lamp: true },
];
for (const b of BUTTONS) {
  const o = cpPoint(b.x, b.y);
  const F = CPF(o);
  const r = b.lamp ? 2.2 : 2.6;
  const bez = record(A.disc(F, 0, 0.7, r + 1, { name: `button.${b.id}.bezel` }));
  put(`button.${b.id}.bezel`, o, S(G.disc(0, 0.7, r + 1, F, P, { steps: 36 }), { tone: "mid" }), [bez], { bias: 0.5 });
  const plunger = record(A.disc(F, 0.7, 2.2, r, { name: `button.${b.id}` }));
  const cap = S(G.disc(0.7, 2.2, r, F, P, { steps: 32 }), { tone: "hi" }) + Ln(G.circleOf(F, 2.2, r * 0.68, P, 24), { tone: "lo" });
  put(`button.${b.id}`, G.add3(o, G.mul3(CPN, 1.5)), `<g data-button="${b.id}"${b.lamp ? ' class="ac-lamp"' : ""}>${cap}</g>`, [plunger], { bias: 0.8 });
  disc(`button.${b.id}.housing`, F, -8, -4.6, 2.3, { tone: "mid" });
  disc(`button.${b.id}.nut`, F, -4.6, -3, 3.6, { tone: "lo" }, { steps: 6 });
  const SW = G.frameOf(G.add3(o, G.mul3(CPN, -8)), CPN, [1, 0, 0], CPD);
  prism(`button.${b.id}.switch`, SW, rect(-2, 2, -1.3, 1.3), -4.2, 0, { tone: "mid" });
}

const DOOR = { x0: -57, x1: -3, z0: 38, z1: 120 };
const DOOR_CX = (DOOR.x0 + DOOR.x1) / 2;
const FY = 156;
box("door.frame", [DOOR.x0, FY, DOOR.z0], [DOOR.x1, FY + 1.4, DOOR.z1], { tone: "lo" }, { r: 1.2, steps: 3 });
box("door.plate", [DOOR.x0 + 2, FY + 1.4, DOOR.z0 + 2], [DOOR.x1 - 2, FY + 2.4, DOOR.z1 - 2], { tone: "mid" }, { r: 1, steps: 3, bevel: 0.4 });
const ENTRIES = [-42, -18];
const SLOT_Z = [100.5, 107.5];
for (const cx of ENTRIES) {
  const front = FY + 3.8;
  box("door.entry", [cx - 8, FY + 2.4, 88], [cx + 8, front, 114], { tone: "mid" }, { r: 1.2, steps: 3, bevel: 0.3 });
  const slot = [[cx - 0.5, SLOT_Z[0]], [cx + 0.5, SLOT_Z[0]], [cx + 0.5, SLOT_Z[1]], [cx - 0.5, SLOT_Z[1]]];
  const lamp = `<g class="ac-lamp" data-lamp="reject">${S(k.slabOf({ x: cx - 5, y: front, w: 10, d: 1.2, r: 0.6 }, 90, 8, P, 2, 0.3), { tone: "hi" })}</g>`;
  put("door.slot", [cx, front + 0.1, 104], `<path class="ac-slot" d="${path3(slot.map(([x, z]) => [x, front + 0.02, z]), true)}"/>` + lamp, [record(A.box([cx - 5, front, 90], [cx + 5, front + 1.2, 98], { name: "door.reject" }))], { bias: 1 });
}
disc("door.lock", G.frameAlong([DOOR_CX, FY + 2.4, 72], [0, 1, 0]), 0, 1.6, 2.4, { tone: "hi" }, { extra: Ln(path3([[DOOR_CX, FY + 4.02, 70.8], [DOOR_CX, FY + 4.02, 73.2]]), { tone: "lo", free: true }) });
box("door.return", [DOOR_CX - 9, FY + 2.4, 46], [DOOR_CX + 9, FY + 4.4, 60], { tone: "mid" }, { r: 1.4, steps: 3 });
put("door.return.mouth", [DOOR_CX, FY + 4.5, 52], `<path class="ac-slot" d="${path3([[DOOR_CX - 6, FY + 4.42, 49], [DOOR_CX + 6, FY + 4.42, 49], [DOOR_CX + 6, FY + 4.42, 55], [DOOR_CX - 6, FY + 4.42, 55]], true)}"/>`, [record(A.box([DOOR_CX - 6, FY + 4.4, 49], [DOOR_CX + 6, FY + 4.5, 55], { name: "door.return.mouth" }))], { bias: 1 });
for (const z of [48, 110]) disc("door.hinge", G.frameAlong([DOOR.x1 + 0.9, FY + 0.9, z], [0, 0, 1]), -3, 3, 0.9, { tone: "mid" });
box("door.price", [DOOR_CX - 6, FY + 2.4, 80], [DOOR_CX + 6, FY + 2.9, 85], { tone: "lo" }, { extra: Ln(path3([[DOOR_CX - 4, FY + 2.92, 82.5], [DOOR_CX + 2, FY + 2.92, 82.5]]), { tone: "lo", free: true }) });

box("door.back", [DOOR.x0, 150, DOOR.z0], [DOOR.x1, 152, DOOR.z1], { tone: "lo" });
const MECH = (cx) => ({ x0: cx - 6.5, x1: cx + 6.5, y0: 141, y1: 150, z0: 84, z1: 106 });
for (const cx of ENTRIES) {
  const m = MECH(cx);
  const side = m.x0;
  const extra = [
    Ln(path3([[side, m.y0 + 2, m.z0 + 4], [side, m.y1 - 2, m.z0 + 4], [side, m.y1 - 2, m.z1 - 4], [side, m.y0 + 2, m.z1 - 4]], true), { tone: "faint" }),
    Dots([at([side, m.y0 + 2.4, m.z1 - 2.4]), at([side, m.y1 - 2.4, m.z1 - 2.4]), at([side, m.y0 + 2.4, m.z0 + 2.4]), at([side, m.y1 - 2.4, m.z0 + 2.4])], { size: 0.45 }),
  ].join("");
  box(`mech`, [m.x0, m.y0, m.z0], [m.x1, m.y1, m.z1], { tone: "mid" }, { r: 0.6, steps: 2, bevel: 0.4, extra });
  disc("mech.magnet", G.frameAlong([side, 145.5, 96], [-1, 0, 0]), 0, 1.4, 2.4, { tone: "hi" });
  disc("mech.pivot", G.frameAlong([side, 146, 89], [-1, 0, 0]), 0, 0.8, 0.9, { tone: "lo" });
  box("mech.lever", [cx - 1, 141.5, 106], [cx + 1, 150, 107.6], { tone: "lo" });
  disc("door.socket", G.frameAlong([cx, 150, 112], [0, -1, 0]), 0, 3.4, 1.6, { tone: "mid" });
}

const CHUTE = { x0: -50, x1: -10, y0: 148, y1: 104, z0: 82, slope: Math.tan((30 * Math.PI) / 180) };
const chuteZ = (y) => CHUTE.z0 - CHUTE.slope * (CHUTE.y0 - y);
const CH_T = 0.8;
const cs = Math.cos((30 * Math.PI) / 180);
const sn = Math.sin((30 * Math.PI) / 180);
const chuteFloor = [[CHUTE.y0, chuteZ(CHUTE.y0)], [CHUTE.y1, chuteZ(CHUTE.y1)], [CHUTE.y1 + CH_T * sn, chuteZ(CHUTE.y1) - CH_T * cs], [CHUTE.y0 + CH_T * sn, chuteZ(CHUTE.y0) - CH_T * cs]];
board("chute.floor", chuteFloor, CHUTE.x0, CHUTE.x1, { tone: "mid" });
const wallOf = (h) => [[CHUTE.y0, chuteZ(CHUTE.y0)], [CHUTE.y0, chuteZ(CHUTE.y0) + h], [CHUTE.y1, chuteZ(CHUTE.y1) + h], [CHUTE.y1, chuteZ(CHUTE.y1)]];
board("chute.far", wallOf(6), CHUTE.x1, CHUTE.x1 + 0.7, { tone: "mid" });
board("chute.lip", wallOf(1.2), CHUTE.x0 - 0.7, CHUTE.x0, { tone: "hi" });
box("chute.tab", [CHUTE.x0 + 2, CHUTE.y0, chuteZ(CHUTE.y0) - 2.6], [CHUTE.x1 - 2, 150, chuteZ(CHUTE.y0) - CH_T * cs], { tone: "lo" });
for (const x of [CHUTE.x0 + 1, CHUTE.x1 - 3]) box("chute.leg", [x, 109, 8], [x + 1.4, 111, chuteZ(109 - CH_T * sn) - CH_T * cs], { tone: "lo" });

const SW_Y = 124;
const SWITCH = { x0: -54.4, x1: -50.7, y0: SW_Y - 2.6, y1: SW_Y + 2.6, z0: chuteZ(SW_Y) - 2.4, z1: chuteZ(SW_Y) + 3 };
box("coin.switch", [SWITCH.x0, SWITCH.y0, SWITCH.z0], [SWITCH.x1, SWITCH.y1, SWITCH.z1], { tone: "mid" }, { extra: Dots([at([SWITCH.x0, SW_Y - 1.4, SWITCH.z0 + 1]), at([SWITCH.x0, SW_Y + 1.4, SWITCH.z1 - 1])], { size: 0.4 }) });
box("coin.switch.bracket", [SWITCH.x1, SW_Y - 2.2, SWITCH.z0], [CHUTE.x0 - 0.7, SW_Y + 2.2, SWITCH.z0 + 1], { tone: "lo" });
const WIRE = { from: [SWITCH.x1, SW_Y, chuteZ(SW_Y) + 1.4], length: 13 };

const BOX = { x0: -58, x1: -16, y0: 62, y1: 104, z0: 8, z1: 30 };
const T = 0.7;
box("cash.floor", [BOX.x0, BOX.y0, BOX.z0], [BOX.x1, BOX.y1, BOX.z0 + T], { tone: "lo" });
box("cash.back", [BOX.x0, BOX.y0, BOX.z0 + T], [BOX.x1, BOX.y0 + T, BOX.z1], { tone: "mid" });
box("cash.far", [BOX.x1 - T, BOX.y0 + T, BOX.z0 + T], [BOX.x1, BOX.y1, BOX.z1], { tone: "mid" });
box("cash.near", [BOX.x0, BOX.y0 + T, BOX.z0 + T], [BOX.x0 + T, BOX.y1, BOX.z1], { tone: "mid" });
box("cash.front", [BOX.x0 + T, BOX.y1 - T, BOX.z0 + T], [BOX.x1 - T, BOX.y1, BOX.z1], { tone: "mid" });
box("cash.handle", [BOX.x0 - 1.6, 78, 22], [BOX.x0, 88, 24], { tone: "hi" });

const METER = { x0: -58, x1: -52, y0: 108, y1: 118, z0: 12, z1: 20 };
box("meter.foot", [-58, 106, 8], [-50, 120, 12], { tone: "lo" });
box("meter", [METER.x0, METER.y0, METER.z0], [METER.x1, METER.y1, METER.z1], { tone: "mid" }, { r: 0.6 });
const METER_FACE = { o: [METER.x0 - 0.02, 109.6, 18.4], ex: [0, 1, 0], ey: [0, 0, -1] };
{
  const m = planeMatrix(METER_FACE.o, METER_FACE.ex, METER_FACE.ey);
  put("meter.face", [METER.x0, 113, 16], `<g transform="${m}"><rect class="ac-window" x="0" y="0" width="6.8" height="4.8" rx="0.4"/><text class="ac-digits" data-part="meter" x="0.35" y="3.55" font-size="3" textLength="6.1" lengthAdjust="spacingAndGlyphs">004127</text></g>`, [record(A.box([METER.x0 - 0.05, 109.6, 13.6], [METER.x0, 116.4, 18.4], { name: "meter.face" }))], { bias: 0.3 });
}

const PCB = { y: 8, x0: -50, x1: 10, z0: 116, z1: 170 };
const px = (v) => PCB.x0 + v;
const pz = (v) => PCB.z0 + v;
const FACE_Y = PCB.y;
const BF = G.frameOf([0, FACE_Y, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
for (const [x, z] of [[3, 3], [57, 3], [3, 51], [57, 51]]) disc("pcb.standoff", G.frameAlong([px(x), 4, pz(z)], [0, 1, 0]), 0, 3.2, 1, { tone: "lo" });
box("pcb.board", [PCB.x0, FACE_Y - 0.8, PCB.z0], [PCB.x1, FACE_Y, PCB.z1], { tone: "lo" }, { r: 0.6 });
const chip = (name, x, z, len, wid, tall = 1.2) => {
  const y1 = FACE_Y + tall;
  const pins = [];
  for (let i = 0.6; i < len - 0.3; i += 1.27) pins.push(at([px(x + i), FACE_Y, pz(z) - 0.25]), at([px(x + i), FACE_Y, pz(z + wid) + 0.25]));
  box(name, [px(x), FACE_Y, pz(z)], [px(x + len), y1, pz(z + wid)], { tone: "hi" }, { r: 0.15, steps: 1, extra: Dots(pins, { size: 0.22, tone: "lo" }) + Dots([at([px(x + 0.8), y1, pz(z + wid / 2)])], { size: 0.35, tone: "lo" }) });
};
chip("pcb.cpu", 4, 40, 10.4, 3);
for (let i = 0; i < 4; i++) chip("pcb.rom", 18 + i * 8.6, 45, 7.2, 3);
for (let i = 0; i < 3; i++) chip("pcb.ram", 4 + i * 6, 33, 4.6, 1.6);
for (let r = 0; r < 3; r++) for (let c = 0; c < 6; c++) chip("pcb.logic", 24 + c * 5.6, 12 + r * 7.4, 4, 1.6);
chip("pcb.sound", 4, 22, 5.4, 3);
chip("pcb.video", 4, 12, 7.2, 3, 1.6);
box("pcb.crystal", [px(17), FACE_Y, pz(41)], [px(20), FACE_Y + 1.4, pz(42.6)], { tone: "hi" }, { r: 0.6, steps: 3 });
box("pcb.dips", [px(16), FACE_Y, pz(33)], [px(24), FACE_Y + 0.8, pz(35)], { tone: "mid" }, { extra: Dots(Array.from({ length: 8 }, (_, i) => at([px(16.5 + i), FACE_Y + 0.8, pz(34)])), { size: 0.25, tone: "hi" }) });
prism("pcb.heatsink", BF, rect(px(4), px(12), pz(3), pz(8)), 0, 4.2, { tone: "mid" }, { extra: Ln([5.6, 7.2, 8.8, 10.4].map((x) => path3([[px(x), FACE_Y + 4.2, pz(3)], [px(x), FACE_Y + 4.2, pz(8)]])).join(""), { tone: "lo" }) });
for (const [x, z, r, h] of [[50, 37.4, 1.6, 3.6], [56, 37.4, 1.2, 2.8], [16, 26, 1, 2.4], [18, 16, 1, 2.4]]) disc("pcb.cap", G.frameAlong([px(x), FACE_Y, pz(z)], [0, 1, 0]), 0, h, r, { tone: "hi" });
put("pcb.led", [px(54), FACE_Y + 0.4, pz(48)], `<g class="ac-led" data-led="pcb">${Dots([at([px(54), FACE_Y + 0.4, pz(48)])], { size: 0.9, tone: "hi" })}</g>`, [record(A.ball([px(54), FACE_Y + 0.4, pz(48)], 0.6, { name: "pcb.led" }))], { bias: 0.4 });
{
  const traces = [];
  const pads = [];
  const lane = (x0, z0, x1, z1, via) => {
    const pts = via ? [[x0, z0], [via, z0], [via, z1], [x1, z1]] : [[x0, z0], [x1, z0], [x1, z1]];
    traces.push(path3(pts.map(([x, z]) => [px(x), FACE_Y + 0.02, pz(z)])));
    pads.push(at([px(x0), FACE_Y + 0.02, pz(z0)]), at([px(x1), FACE_Y + 0.02, pz(z1)]));
  };
  lane(15, 41.4, 18, 39, 16.6);
  lane(26.6, 45, 29, 37);
  lane(35.2, 45, 37.6, 37);
  lane(9, 39, 9, 35);
  lane(15, 34.6, 24, 27.6, 20);
  lane(9.6, 25, 24, 23.2, 16);
  lane(11.4, 15, 24, 11.4, 14);
  lane(34, 9.4, 44, 4, 39);
  lane(50, 18, 54, 26);
  traces.push(path3([[px(58), FACE_Y + 0.02, pz(2)], [px(58), FACE_Y + 0.02, pz(52)]]));
  pads.push(at([px(58), FACE_Y + 0.02, pz(2)]), at([px(58), FACE_Y + 0.02, pz(52)]));
  put("pcb.traces", [px(30), FACE_Y + 0.05, pz(27)], Ln(traces.join(""), { tone: "faint" }) + Dots(pads, { size: 0.3, tone: "lo" }), [record(A.box([PCB.x0, FACE_Y, PCB.z0], [PCB.x1, FACE_Y + 0.05, PCB.z1], { name: "pcb.traces" }))], { bias: 0.05 });
}
const EDGE = { x0: PCB.x1, x1: PCB.x1 + 4.6, y0: 4.6, y1: 11, z0: 124, z1: 162 };
box("pcb.edge", [EDGE.x0, EDGE.y0, EDGE.z0], [EDGE.x1, EDGE.y1, EDGE.z1], { tone: "mid" }, { r: 0.5, extra: Ln(path3([[EDGE.x0 + 1, EDGE.y1, EDGE.z0], [EDGE.x0 + 1, EDGE.y1, EDGE.z1]]), { tone: "lo" }) });
const plugAt = (z) => [EDGE.x1, 7.8, z];
box("pcb.vplug", [px(6), 5.4, PCB.z1], [px(12), 10.4, PCB.z1 + 2.4], { tone: "mid" });

const FAN = { x: -18, z: 262, r: 12 };
const FANF = G.frameOf([FAN.x, 4, FAN.z], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
prism("fan.frame", FANF, rect(-13, 13, -13, 13), 0, 3.4, { tone: "mid" }, {
  extra: Dots([[-11, -11], [11, -11], [-11, 11], [11, 11]].map(([u, v]) => at([FAN.x + u, 7.4, FAN.z + v])), { size: 0.5, tone: "hi" }),
});
{
  const m = planeMatrix([FAN.x, 7.42, FAN.z], [1, 0, 0], [0, 0, -1]);
  const blades = Array.from({ length: 7 }, (_, i) => {
    const a = (i / 7) * Math.PI * 2;
    const c = Math.cos(a);
    const s = Math.sin(a);
    const pt = (r, da) => [r * Math.cos(a + da), r * Math.sin(a + da)].map((v) => f2(v)).join(" ");
    return `M${pt(3.6, 0)}Q${pt(8, 0.25)} ${pt(11, 0.12)}L${pt(11, 0.62)}Q${pt(7.6, 0.72)} ${pt(3.6, 0.7)}Z`;
  }).join("");
  const guard = [11.6, 8, 4.6].map((r) => `M${r} 0A${r} ${r} 0 1 1 ${-r} 0A${r} ${r} 0 1 1 ${r} 0Z`).join("");
  const spokes = [0, 1, 2, 3].map((i) => `M${f2(3 * Math.cos(i * Math.PI / 2 + 0.4))} ${f2(3 * Math.sin(i * Math.PI / 2 + 0.4))}L${f2(11.6 * Math.cos(i * Math.PI / 2 + 0.4))} ${f2(11.6 * Math.sin(i * Math.PI / 2 + 0.4))}`).join("");
  put("fan.rotor", [FAN.x, 7.5, FAN.z], `<g transform="${m}"><circle class="ac-fan-hole" r="12"/><g data-part="fan"><path class="ac-blade" d="${blades}"/></g><circle class="ac-fan-hub" r="3.4"/><path class="ac-guard" d="${guard}${spokes}"/></g>`, [record(A.disc(FANF, 3.4, 3.5, 12, { name: "fan.rotor" }))], { bias: 0.2 });
}

box("psu.base", [-48, 10, 8], [-20, 40, 9.4], { tone: "lo" }, { extra: Dots([at([-46, 12, 9.4]), at([-22, 12, 9.4]), at([-46, 38, 9.4]), at([-22, 38, 9.4])], { size: 0.45 }) });
box("xfmr.core", [-46, 16, 9.4], [-22, 34, 30], { tone: "mid" }, {
  inner: Ln([12, 15, 18, 21, 24, 27].map((z) => k.sideSeam({ x: -46, y: 16, w: 24, d: 18, r: 0.4 }, z, P, 2)).join(""), { tone: "faint" }),
});
box("xfmr.coil", [-42, 34, 13], [-26, 37, 27], { tone: "hi" }, { r: 1.4, steps: 3 });
box("xfmr.coil.back", [-42, 13, 13], [-26, 16, 27], { tone: "hi" }, { r: 1.4, steps: 3 });
box("xfmr.terminals", [-40, 22, 30], [-28, 28, 32], { tone: "lo" }, { extra: Dots([-38, -35, -32, -29].map((x) => at([x, 25, 32])), { size: 0.45, tone: "hi" }) });
box("psu.case", [-12, 10, 8], [30, 40, 22], { tone: "mid" }, { bevel: 0.4, extra: Dots(k.dotGrid(-8, 13, 14, 8, 2.6, 22, P), { size: 0.35, tone: "lo" }) });
box("psu.strip", [-8, 40, 9], [26, 43, 14], { tone: "lo" }, { extra: Dots([-5, -1, 3, 7, 11, 15, 19, 23].map((x) => at([x, 43, 12.4])), { size: 0.45, tone: "hi" }) });
const PW = { x: 58.6, y0: 14, y1: 58, z0: 28, z1: 70 };
box("pw.plate", [PW.x, PW.y0, PW.z0], [XI, PW.y1, PW.z1], { tone: "lo" }, { extra: Dots([[PW.y0 + 2, PW.z0 + 2], [PW.y1 - 2, PW.z0 + 2], [PW.y0 + 2, PW.z1 - 2], [PW.y1 - 2, PW.z1 - 2]].map(([y, z]) => at([PW.x, y, z])), { size: 0.45, tone: "hi" }) });
box("relay", [PW.x - 7, 18, 52], [PW.x, 30, 64], { tone: "mid" }, { r: 0.6, steps: 2, extra: `<g class="ac-led" data-led="relay">${Dots([at([PW.x - 7, 27.4, 61])], { size: 0.7, tone: "hi" })}</g>` });
for (const z of [56, 62]) {
  box("fuse.holder", [PW.x - 2.6, 36, z - 1.6], [PW.x, 50, z + 1.6], { tone: "mid" }, { r: 1, steps: 3 });
  disc("fuse.cap", G.frameAlong([PW.x - 1.3, 50, z], [0, 1, 0]), 0, 2, 1.3, { tone: "hi" });
}
box("pw.strip", [PW.x - 3, 18, 34], [PW.x, 54, 39], { tone: "mid" }, { extra: Dots([21, 25, 29, 33, 37, 41, 45, 49].map((y) => at([PW.x - 3, y, 36.5])), { size: 0.45, tone: "hi" }) });
box("pw.filter", [PW.x - 6, 18, 42], [PW.x, 34, 49], { tone: "hi" }, { r: 0.8, steps: 2 });
const SV = { x: 58.6, y0: 16, y1: 52, z0: 98, z1: 120 };
box("service.plate", [SV.x, SV.y0, SV.z0], [XI, SV.y1, SV.z1], { tone: "lo" }, { extra: Dots([[SV.y0 + 2, SV.z0 + 2], [SV.y1 - 2, SV.z0 + 2], [SV.y0 + 2, SV.z1 - 2], [SV.y1 - 2, SV.z1 - 2]].map(([y, z]) => at([SV.x, y, z])), { size: 0.45, tone: "hi" }) });
box("service.toggle", [SV.x - 3, 20, 108], [SV.x, 26, 114], { tone: "mid" }, { r: 0.5, extra: Ln(path3([[SV.x - 3, 23, 111], [SV.x - 6, 23, 113.4]]), { tone: "hi", free: "end" }) });
disc("service.button.ring", G.frameAlong([SV.x, 33, 111], [-1, 0, 0]), 0, 1, 3, { tone: "mid" });
disc("service.button", G.frameAlong([SV.x - 1, 33, 111], [-1, 0, 0]), 0, 1.6, 2.1, { tone: "hi" });
disc("service.pot", G.frameAlong([SV.x, 44, 111], [-1, 0, 0]), 0, 1.6, 2.6, { tone: "mid" });
{
  const KF = G.frameAlong([SV.x - 1.6, 44, 111], [-1, 0, 0]);
  disc("service.knob", KF, 0, 2.6, 2.2, { tone: "hi" }, { extra: Faded(G.ribsOf(KF, 0, 2.6, 2.2, 24, P, { fade: [0.1, 0.55], seams: true }), { tone: "lo" }) + Ln(path3([[SV.x - 4.25, 44, 111], [SV.x - 4.25, 45.6, 112.2]]), { tone: "hi", free: "end" }) });
}
disc("cord.grip", G.frameAlong([-30, 4, 22], [0, 1, 0]), 0, 2, 2.6, { tone: "mid" });
box("outlet", [-40, 4, 330], [-10, 8, 338], { tone: "mid" }, { extra: Dots([-34, -25, -16].map((x) => at([x, 8, 334])), { size: 0.5, tone: "lo" }) });
put("back.vents", [20, 4.1, 300], Ln(Array.from({ length: 5 }, (_, i) => path3([[8 + i * 9, 4.05, 290], [13 + i * 9, 4.05, 290], [13 + i * 9, 4.05, 314], [8 + i * 9, 4.05, 314]], true)).join(""), { tone: "faint" }), [record(A.box([8, 4, 290], [50, 4.1, 314], { name: "back.vents" }))], { bias: 0.1 });

const PLUG = { x0: -55, x1: -43, y0: 147.2, y1: 150.4, z0: 173.2, z1: 176.8 };
box("loom.plug", [PLUG.x0, PLUG.y0, PLUG.z0], [PLUG.x1, PLUG.y1, PLUG.z1], { tone: "mid" }, { r: 0.4 });
const plugTop = 192.1 - (148 - 135.24) * ((192.1 - 181.1) / (177.24 - 135.24));
box("loom.plug.bracket", [-51, 148, PLUG.z1], [-47.6, 149, plugTop], { tone: "lo" });
cable("loom", [[PLUG.x1, 148.8, 175], [46, 148.8, 175], [52, 146, 168], [52, 146, 146], [52, 20, 146], [28, 7.8, 146], plugAt(146)], 1.3, { tone: "mid", bend: 6, shells: [false, true], rings: { pitch: 1.1, twist: 0.8, cross: true, fade: [0.1, 0.55] } });
const LANES = { up: [-49, 175.8], left: [-53.6, 175], right: [-44.4, 175], down: [-50.8, 174.2] };
STICK_DATA.switches.forEach((sw) => {
  const tab = G.add3(sw.c, G.mul3(CPN, -2));
  const [x, z] = LANES[sw.id];
  const route = sw.id === "down" ? [tab, [tab[0], tab[1], z], [x, tab[1] - 3, z], [x, PLUG.y1, z]] : [tab, [tab[0], tab[1], z], [x, PLUG.y1, z]];
  cable(`wire.stick.${sw.id}`, route, 0.25, { tone: "lo", bend: 1.2 });
});
const SWX = (SWITCH.x0 + SWITCH.x1) / 2;
cable("wire.coin", [[SWX + 0.9, SW_Y, SWITCH.z0], [SWX + 0.9, SW_Y, 12.6], [50, SW_Y, 12.6], [50, 66, 12.6], [50, 66, 128], [50, 20, 128], [28, 7.8, 128], plugAt(128)], 0.5, { tone: "mid", bend: 3, shells: [false, true] });
cable("wire.meter", [[METER.x0 + 3, METER.y0 + 5, METER.z1], [METER.x0 + 3, METER.y0 + 5, 40], [SWX - 0.9, SW_Y - 1, 52], [SWX - 0.9, SW_Y, SWITCH.z0]], 0.4, { tone: "lo", bend: 3 });
cable("cable.power", [[-1, 43, 12.4], [-1, 50, 12.4], [44, 50, 12.4], [54, 62, 12.4], [54, 62, 136], [54, 20, 136], [28, 7.8, 136], plugAt(136)], 0.7, { tone: "mid", bend: 5, shells: [false, true] });
cable("wire.service", [[SV.x, 30, SV.z0 + 1.6], [56.6, 30, SV.z0 + 1.6], [56.6, 13, SV.z0 + 1.6], [56.6, 13, 141.4], [30, 7.8, 141.4], plugAt(141.4)], 0.4, { tone: "lo", bend: 3, shells: [false, true] });
cable("cable.mains", [[PW.x - 6, 26, 45.5], [46, 26, 45.5], [46, 26, 15], [30, 26, 15]], 0.6, { tone: "lo", bend: 4 });
cable("cord.ac", [[-30, 6, 22], [-30, 13, 22]], 0.9, { tone: "mid", bend: 3 });

const SPK_BACK = G.add3(SPK_C, G.mul3(SPK_N, 3.8));
cable("cable.speaker", [SPK_BACK, G.add3(SPK_BACK, G.mul3(SPK_N, 0.6)), [22, 104, 316.6], [22, 7.8, 316.6], [22, 7.8, 158], plugAt(158)], 0.4, { tone: "lo", bend: 3 });
const VPLUG = chPoint(59, 39, 3);
cable("cable.video", [[px(9), 7.9, PCB.z1 + 2.4], [px(9), 7.9, 184], [-52, 36, 192], [-52, VPLUG[1], VPLUG[2]], VPLUG], 0.6, { tone: "mid", bend: 5 });
const flyTop = chPoint(51, 18, 4);
const capTop = G.add3(anodeAt, G.mul3(anodeN, 2.2));
cable("lead.anode", [flyTop, G.add3(flyTop, G.mul3(bd, -6)), G.add3(capTop, G.mul3(anodeN, 6)), capTop], 0.8, { tone: "hi", bend: 5 });
const NPLUG = chPoint(59.5, 20.5, 3);
cable("cable.neck", [monPoint(-4, 4, 91), monPoint(-4, 4, 94), monPoint(-30, 16, 94), G.add3(NPLUG, [-5, 0, 0]), NPLUG], 0.45, { tone: "lo", bend: 3 });

const OPEN = BEZEL_OPEN;
const FACE_OFF = -5;
const bezelPt = (u, w, off = 0) => G.add3(G.add3(G.add3(C3, [u, 0, 0]), G.mul3(bd, w)), G.mul3(NB, off));
const bezelMatrix = (off) => planeMatrix(bezelPt(0, 0, off), [1, 0, 0], bd);
const roundRect = (u0, u1, w0, w1, r, steps = 5) => k.roundedPlan({ x: u0, y: w0, w: u1 - u0, d: w1 - w0, r }, steps);
const openRing = roundRect(-OPEN.u, OPEN.u, -OPEN.w, OPEN.w, OPEN.r);
const flat = (ring) => `M${ring.map(([u, w]) => `${f2(u)} ${f2(w)}`).join("L")}Z`;
const HALF_LEN = 57;
const bezelFace = `M-${XI} ${-HALF_LEN}H${XI}V${HALF_LEN}H-${XI}Z` + flat([...openRing].reverse());
const recess = [];
{
  const ring = openRing;
  const n = ring.length;
  for (let i = 0; i < n; i++) {
    const [a, b] = [ring[i], ring[(i + 1) % n]];
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const inward = G.unit3(G.add3(G.mul3([1, 0, 0], -mid[0]), G.mul3(bd, -mid[1])));
    const along = G.unit3(G.add3(G.mul3([1, 0, 0], b[0] - a[0]), G.mul3(bd, b[1] - a[1])));
    let normal = G.cross3(along, NB);
    if (G.dot3(normal, inward) < 0) normal = G.mul3(normal, -1);
    if (G.dot3(normal, V) <= 0) continue;
    const quad = [bezelPt(a[0], a[1], -1), bezelPt(b[0], b[1], -1), bezelPt(b[0], b[1], FACE_OFF), bezelPt(a[0], a[1], FACE_OFF)];
    recess.push({ tone: G.toneOf(normal, P), d: path3(quad, true) });
  }
}
const openClip = path3(openRing.map(([u, w]) => bezelPt(u, w, -1)), true);
const faceClip = path3(roundRect(-33, 33, -41, 41, 7).map(([u, w]) => bezelPt(u, w, FACE_OFF)), true);
const screenSvg = SCREEN.svg();
const bezelItem = [
  `<clipPath id="ac-open"><path d="${openClip}"/></clipPath>`,
  `<clipPath id="ac-face"><path d="${faceClip}"/></clipPath>`,
  `<g clip-path="url(#ac-open)">`,
  `<path class="ac-glass" data-part="glass" d="${faceClip}"/>`,
  `<g clip-path="url(#ac-face)"><g transform="${bezelMatrix(FACE_OFF)}"><g data-part="screen" class="ac-screen">${screenSvg}</g></g></g>`,
  ...recess.map((q) => `<path class="iso-shade" data-shade="${Math.min(3, q.tone)}" d="${q.d}"/>`),
  `</g>`,
  `<g transform="${bezelMatrix(-1)}"><path class="ac-bezel" d="${bezelFace}"/><path class="iso-line" data-tone="lo" d="${flat(openRing)}"/>`,
  `<path class="iso-line" data-tone="faint" d="${flat(roundRect(-OPEN.u - 3, OPEN.u + 3, -OPEN.w - 3, OPEN.w + 3, OPEN.r + 3))}"/>`,
  `<path class="iso-line" data-tone="lo" data-free="" d="M-18 46H18M-12 49.6H12M-15 53.2H15"/></g>`,
].join("");
const bezelSection = [[0, -HALF_LEN], [0, HALF_LEN], [-1.6, HALF_LEN], [-1.6, -HALF_LEN]];
const BZF = G.frameOf(bezelPt(0, 0, 0), [1, 0, 0], NB, bd);
{
  const shape = record(A.prism(BZF, [[-1.6, -HALF_LEN], [0, -HALF_LEN], [0, HALF_LEN], [-1.6, HALF_LEN]], -XI, XI, { name: "bezel" }));
  const glassEdge = S(prismPaths(BZF, bezelSection, -XI, XI), { tone: "lo", flat: true });
  put("bezel", bezelPt(0, 0, 0), `<g class="ac-bezel-solid">${glassEdge}</g>${bezelItem}<path class="ac-glare" d="${path3([bezelPt(-24, -50, 0.05), bezelPt(-6, -50, 0.05), bezelPt(26, 50, 0.05), bezelPt(8, 50, 0.05)], true)}"/>`, [shape], { bias: 1 });
}
A.settle(R, P);
R.items.sort((a, b) => a.key - b.key);

const indexOf = (name) => R.items.findIndex((item) => item.name === name);
const slots = new Map();
slots.set(Math.max(indexOf("chute.floor"), indexOf("chute.far")), `<g data-part="coin-chute"></g><g data-part="wire">${Ln(path3([WIRE.from, G.add3(WIRE.from, [WIRE.length, 0, 0])]), { tone: "hi", free: "end" })}</g>`);
slots.set(Math.max(indexOf("cash.floor"), indexOf("cash.back"), indexOf("cash.far")), `<g data-part="pile"></g><g data-part="coin-fall"></g>`);
const groups = R.items.map((item, index) => `<g class="it" data-name="${item.name}"${item.attrs ?? ""}>${item.svg}</g>${slots.get(index) ?? ""}`);

const trayF = G.frameOf([TRAY.x, TRAY.y, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]);
const tray = S(G.lathe([[0, 0], [0, 9], [1.6, 9], [1.6, 7.6], [0.8, 7.2]], trayF, P, { steps: 48 }), { tone: "lo" });
const stack = Array.from({ length: STACK }, (_, i) => `<g data-stack="${i}">${PARTS.coin([TRAY.x, TRAY.y, stackTop(i) + 0.2], [0, 0, 1], 0, { lit: false })}</g>`).join("");

const LABEL_TEXT =
  "An upright arcade cabinet on a bench with its left side panel off: a CRT turned on its side behind the tilted glass, a marquee lamp and speaker above, a control panel with a joystick and four buttons whose microswitches hang below it, a coin door with two coin mechs, a coin switch, a meter and a cash box, the game board on the far wall, and the power supply on the floor. A stack of quarters sits on a tray on the bench.";
const svg = k.figureSvg({
  width: W,
  height: H,
  label: LABEL_TEXT,
  body: [...back, `<g data-part="tray">${tray}${stack}</g>`, `<g class="ac-cab">${groups.join("")}</g>`, `<g data-part="coin-fly"></g>`, `<g data-part="pulse"></g>`],
});

const DATA = {
  P,
  V,
  stick: STICK_DATA,
  tray: { x: TRAY.x, y: TRAY.y, top: stackTop(STACK - 1) + 0.2, step: COIN_STEP, count: STACK },
  slot: { x: ENTRIES[0], y: FY + 3.8, z: (SLOT_Z[0] + SLOT_Z[1]) / 2 },
  mech: MECH(ENTRIES[0]),
  chute: { ...CHUTE, slope: CHUTE.slope },
  wire: WIRE,
  box: BOX,
  meterFace: METER_FACE,
  screen: { off: FACE_OFF },
  routes: Object.fromEntries(["wire.coin", "cable.video", "cable.power", "loom"].map((name) => [name, R.routes.find((r) => r.name === name).points.map((p) => at(p).map(f2))])),
  meterMatrix: planeMatrix(METER_FACE.o, METER_FACE.ex, METER_FACE.ey),
};

const CSS = readFileSync(join(HERE, "arcade-cabinet.css"), "utf8");
const latheSource = readFileSync(join(HERE, "..", "..", "kit", "lathe.mjs"), "utf8")
  .replace(/^import .*$/m, "")
  .replace(/^export /gm, "");
const latheScript = `const G = (() => {${latheSource}\nreturn { viewOf, scaleOf, frameAlong, frameOf, pointOf, disc, sphereOf, prismOf, circleOf, arcOf, dot3, add3, sub3, mul3, unit3, cross3, len3, lightOf, toneOf, radialOf };})();\n`;
const partsScript = readFileSync(join(HERE, "parts.mjs"), "utf8").replace(/^export /gm, "");
const screenScript = readFileSync(join(HERE, "screen.mjs"), "utf8").replace(/^export /gm, "");
const LIVE = readFileSync(join(HERE, "live.js"), "utf8").replace("__DATA__", JSON.stringify(DATA));

const body =
  `<style>${CSS}</style>` +
  k.plateHtml({
    fig: "Fig 4",
    title: "Upright cabinet, side off",
    hint: "C or click the quarters · ← → · space",
    readout: "off · 0 credits",
    keys: [
      { mark: "lit", label: "The picture: drawn in the plane of the CRT behind the tilted glass" },
      { mark: "raised", label: "Coin path: slot, mech, switch wire, meter, cash box" },
      { mark: "edge", label: "Harness from the joystick, buttons and coin switch to the board's edge connector" },
    ],
    caption:
      "An upright cabinet in the style of 1981, 178 cm tall, with the left side panel off. Drop a quarter (24.26 mm) in the left slot: the mech checks it and lets it fall 27 cm past the coin-switch wire into the open cash box. The switch pulses the meter and the board's coin line, and on this machine it also latches the power relay, so the first coin wakes the cabinet: the marquee strikes, the CRT's heater warms for about two seconds, the board runs its RAM and ROM self-test and the attract mode starts. The monitor lies on its side for a vertical game, 224 × 288 pixels on a 15.734 kHz raster, so its scan lines run up and down the glass and it collapses to a vertical line when switched off. Move the stick to see the actuator close a microswitch under the panel.",
    body: `<div class="iso-stage ac-stage" tabindex="0" role="group" aria-label="Arcade cabinet. Press C or click the quarters to insert a coin; left and right arrows move the joystick; space presses the fire button; Escape switches the cabinet off.">${svg}<span class="ac-sr" role="status" aria-live="polite" data-status></span></div>`,
  });

const page = k.pageHtml({ title: "Arcade cabinet", theme: THEME, body, script: kitScript() + latheScript + partsScript + screenScript + LIVE, width: 650 });
const name = THEME === "light" ? "arcade-cabinet-light.html" : "arcade-cabinet.html";
writeFileSync(join(HERE, name), page);
console.log("wrote", name, `${(page.length / 1024).toFixed(0)} KB`, `${R.items.length} items`, `scale ${KS.toFixed(3)}`, `${CABLES.length} cables`);

A.auditOrExit(R, P, {});
