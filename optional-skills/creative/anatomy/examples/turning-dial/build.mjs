import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../kit/iso-kit.mjs";
import * as G from "../../kit/lathe.mjs";
import * as A from "../../kit/audit.mjs";
import { turning } from "../../kit/turn-build.mjs";
import * as TA from "../../kit/turn-audit.mjs";
import { turnScript, glScript } from "../../scripts/inline-kit.mjs";
import { glCamera } from "../../kit/gl.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const theme = argv.includes("--light") ? "light" : "dark";

const WIDTH = 640;
const HEIGHT = 470;
const AZIMUTH = 40;
const ELEVATION = 31;

const BASE = { x: -82, y: -78, w: 164, d: 156, r: 8 };
const BASE_H = 12;
const FOOT_H = 5;
const PLATE = { x: -72, y: -68, w: 144, d: 136, r: 6 };
const PLATE_Z0 = 8;
const PLATE_Z1 = 10;
const DIAL_R = 50;
const SKIRT_TOP = 14.4;
const FACE_Z = 17.6;
const FACE_R = 46;
const HUB_R = 12;
const HUB_TOP = 19.4;
const PALM = { x: -18.5, y: -16.5, w: 30, d: 33, r: 3.2 };
const PALM_TOP = HUB_TOP + 8;
const COVER = { x: -15.5, y: -13.2, w: 26.6, d: 26.4, r: 2.6 };
const COVER_TOP = PALM_TOP + 1.2;
const INDEX_AT = ((AZIMUTH + 180) * Math.PI) / 180;

export const P = k.fitProjection([...k.boxCorners(BASE, -BASE_H - FOOT_H, 0), [0, 0, 40]], WIDTH, HEIGHT, { pad: 22, azimuth: AZIMUTH, elevation: ELEVATION });
const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const VERT = (x, y) => G.frameOf([x, y, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));

export const R = A.recorder(P);
const put = (item) => R.put({ ...item, shapes: item.shapes.map((shape, index) => R.solid(Object.assign(shape, { name: shape.name ?? (index ? `${item.name}#${index + 1}` : item.name) }))) });

function base() {
  for (const [x, y] of k.corners(BASE, 22)) {
    const svg = S(k.cylinder(x, y, 7, -BASE_H - FOOT_H, 1.6, P, 28), { tone: "lo" }) + S(k.cylinder(x, y, 5.4, -BASE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: "mid" });
    put({ name: `stand.foot.${x}.${y}`, layer: "back", at: [x, y, -BASE_H - FOOT_H], svg, shapes: [A.cylinder(x, y, 7, -BASE_H - FOOT_H, 1.6, { ground: true }), A.cylinder(x, y, 5.4, -BASE_H - FOOT_H + 1.6, FOOT_H - 1.6)] });
  }
  const screws = k.corners(BASE, 7);
  const svg = S(k.slabOf(BASE, -BASE_H, BASE_H, P, 8, 1.6), { tone: "mid", inner: L(k.sideSeam(BASE, -BASE_H + 4, P, 8), { tone: "faint" }) }) + L(screws.map(([x, y]) => k.ring(x, y, 2.1, 0, P, 16)).join(""), { tone: "lo" }) + D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 });
  put({ name: "stand.base", layer: "back", at: [0, 0, -BASE_H], bias: 4, svg, shapes: [A.slab(BASE, -BASE_H, BASE_H, { ground: true })] });
}

const WHEEL_AT = [40 * Math.cos((AZIMUTH * Math.PI) / 180), 40 * Math.sin((AZIMUTH * Math.PI) / 180)];
const WHEEL_R = 32;
const PINION_R = 8;
const LOCK_Z0 = 3.2;
const LOCK_Z1 = 7.2;
const WINDOW = { x: 42.4, y: 35.6, w: 12.6, d: 10.4, r: 2.2 };

function backPosts() {
  const out = [];
  const V = [Math.cos((AZIMUTH * Math.PI) / 180), Math.sin((AZIMUTH * Math.PI) / 180)];
  for (const [x, y] of byDepth(k.corners(PLATE, 12))) {
    const front = x * V[0] + y * V[1] > 0;
    const svg = S(k.cylinder(x, y, 3.4, 0, PLATE_Z0, P, 28), { tone: "mid", crease: "none" });
    put({ name: `stand.post.${x}.${y}`, layer: front ? "mid" : "back", at: [x, y, 0], svg, shapes: [A.cylinder(x, y, 3.4, 0, PLATE_Z0)] });
  }
  for (const [name, [x, y], r] of [["stand.washer.dial", [0, 0], 4.2], ["stand.washer.wheel", WHEEL_AT, 6.4]]) {
    const svg = S(k.cylinder(x, y, r, 0, LOCK_Z0, P, 32), { tone: "lo" });
    put({ name, layer: "back", at: [x, y, 0], svg, shapes: [A.cylinder(x, y, r, 0, LOCK_Z0)] });
  }
}

function windowed() {
  const slab = k.slabOf(PLATE, PLATE_Z0, PLATE_Z1 - PLATE_Z0, P, 8, 1);
  const ring = k.roundedPlan(WINDOW, 4);
  const hole = k.onTop(ring, PLATE_Z1, P, true);
  const V = [Math.cos((AZIMUTH * Math.PI) / 180), Math.sin((AZIMUTH * Math.PI) / 180)];
  const lx = -Math.sin((AZIMUTH * Math.PI) / 180);
  const ly = Math.cos((AZIMUTH * Math.PI) / 180);
  const n = ring.length;
  const walls = [[], [], [], []];
  const bottoms = [];
  for (let i = 0; i < n; i++) {
    const a = ring[i];
    const b = ring[(i + 1) % n];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
    const nx = -(b[1] - a[1]) / len;
    const ny = (b[0] - a[0]) / len;
    if (nx * V[0] + ny * V[1] <= 1e-9) continue;
    const shade = Math.min(3, Math.max(0, Math.floor(((nx * lx + ny * ly + 1) / 2) * 4)));
    walls[shade].push(k.pathOf([at([a[0], a[1], PLATE_Z1]), at([b[0], b[1], PLATE_Z1]), at([b[0], b[1], PLATE_Z0]), at([a[0], a[1], PLATE_Z0])], true));
    bottoms.push([a, b]);
  }
  const clip = `<clipPath id="td-window"><path d="${hole}"/></clipPath>`;
  const wallSvg = walls.map((list, shade) => (list.length ? `<path class="iso-shade" data-shade="${shade}" d="${list.join("")}"/>` : "")).join("");
  const under = bottoms.map(([a, b]) => k.pathOf([at([a[0], a[1], PLATE_Z0]), at([b[0], b[1], PLATE_Z0])])).join("");
  const screws = byDepth(k.corners(PLATE, 12));
  const inner = [
    `<g clip-path="url(#td-window)">${wallSvg}${L(under, { tone: "lo" })}</g>`,
    L(k.planOutline(k.insetPlan(PLATE, 4), PLATE_Z1, P), { tone: "faint" }),
    L(screws.map(([x, y]) => k.ring(x, y, 2.4, PLATE_Z1, P, 18)).join(""), { tone: "lo" }),
    L(k.planOutline(k.insetPlan(WINDOW, -1.4), PLATE_Z1, P, 4), { tone: "faint" }),
  ].join("");
  const body = `<defs>${clip}</defs><g class="iso-solid"><path class="iso-fill" fill-rule="evenodd" d="${slab.fill}${hole}"/>${slab.shades.map((d, shade) => (d ? `<path class="iso-shade" data-shade="${shade}" d="${d}"/>` : "")).join("")}<path class="iso-top" fill-rule="evenodd" d="${slab.top}${hole}"/><path class="iso-line iso-crease" data-tone="faint" d="${slab.crease}"/><path class="iso-line iso-bevel" data-tone="faint" d="${slab.bevel}"/>${inner}<path class="iso-line iso-edge" data-tone="mid" d="${slab.outline}${hole}"/></g>`;
  const dots = D(screws.map(([x, y]) => at([x, y, PLATE_Z1])), { size: 0.5 });
  return `<path class="iso-halo" d="${k.haloOf(k.insetPlan(PLATE, 4), PLATE_Z0, 0.1, P)}" style="filter:blur(5px)"/>${body}${dots}`;
}

function midStatic() {
  const svg = windowed();
  const corners = k.roundedPlan(PLATE, 8);
  void corners;
  put({ name: "stand.plate", layer: "mid", at: [0, 0, PLATE_Z0], svg, shapes: [A.slab(PLATE, PLATE_Z0, PLATE_Z1 - PLATE_Z0)] });
  const c = Math.cos(INDEX_AT);
  const s = Math.sin(INDEX_AT);
  const wedge = [[DIAL_R + 1.2, -0.7], [DIAL_R + 1.2, 0.7], [DIAL_R + 7.4, 2.6], [DIAL_R + 7.4, -2.6]].map(([r, t]) => [c * r - s * t, s * r + c * t]);
  const wedgeSvg = S(k.extrude(wedge, PLATE_Z1, 3, P, { bevel: 0.35 }), { tone: "hi", lit: true });
  put({ name: "stand.index", layer: "mid", at: [c * 48, s * 48, PLATE_Z1], svg: `<g data-mat="brass">${wedgeSvg}</g>`, shapes: [A.extruded(wedge, PLATE_Z1, 3)] });
}


export const T = turning(P, { recorder: R });
const dial = T.group("dial", { turn: { pivot: [0, 0] } });
const wheel = T.group("wheel", { turn: { pivot: WHEEL_AT } });
const PAWL_AT = [0, DIAL_R + 8.3];
const pawl = T.group("pawl", { hinge: { point: [PAWL_AT[0], PAWL_AT[1], 0], axis: [0, 0, 1], range: [-4, 4] } });
const lock = T.layer("lock");
const face = T.layer("dial");
const hand = T.layer("hand");
T.stack(["back", "lock", "mid", "dial", "hand", "front"]);
export const follow = (values) => ({ ...values, wheel: -(values.dial ?? 0) / 4, pawl: 4 * Math.sin(((values.dial ?? 0) * Math.PI) / 18), "f2.mcp": -22 * (values.lift ?? 0), "f2.pip": -14 * (values.lift ?? 0), "f2.dip": -9 * (values.lift ?? 0) });
export const tapAt = (time) => {
  const phase = ((time % 4) + 4) % 4;
  const up = Math.min(1, Math.max(0, (phase - 0.4) / 0.9));
  const down = Math.min(1, Math.max(0, (phase - 2.2) / 1.1));
  const ease = (x) => x * x * (3 - 2 * x);
  return ease(up) * (1 - ease(down));
};
export const breakAt = (time) => {
  if (!(time > 0)) return null;
  const flight = 0.79888;
  const c = 1.0;
  let phi = 0;
  let psi = 0;
  let y = 0;
  let z = 0;
  if (time < flight) {
    const s = time / flight;
    phi = 2 * Math.PI * s * s * (3 - 2 * s);
    y = 16 * time;
    z = 25 * time - 49 * time * time;
  } else {
    const k = 1 - Math.exp(-(time - flight) / 0.25);
    phi = 2 * Math.PI;
    y = 16 * flight + 1.5 * k;
    z = -11.3;
    psi = (40 * Math.PI * k) / 180;
  }
  const cp = Math.cos(phi);
  const sp = Math.sin(phi);
  const cz = Math.cos(psi);
  const sz = Math.sin(psi);
  const R = [cz, -sz * cp, sz * sp, sz, cz * cp, -cz * sp, 0, sp, cp];
  return { R, t: [-R[2] * c, y - R[5] * c, z + c - R[8] * c] };
};
export const BREAK_END = 2.2;
export const poses = () => {
  const out = Array.from({ length: 1440 }, (_, deg) => follow({ dial: deg, lift: tapAt(deg / 30) }));
  for (let start = 0; start < 360; start += 45) for (let f = 0; f <= 66; f++) out.push(follow({ dial: start + f * 0.6, lift: tapAt(f / 30), pod: breakAt(f / 30) }));
  return out;
};

const DIAL_PROFILE = [[PLATE_Z1, DIAL_R], [SKIRT_TOP, DIAL_R], [FACE_Z, FACE_R]];
const DIAL_F = VERT(0, 0);
const tick = (angle, from, to) => `M${(Math.sin(angle) * from).toFixed(3)} ${(-Math.cos(angle) * from).toFixed(3)}L${(Math.sin(angle) * to).toFixed(3)} ${(-Math.cos(angle) * to).toFixed(3)}`;
const ringPath = (r, steps) => Array.from({ length: steps }, (_, j) => `${j ? "L" : "M"}${(r * Math.cos((j / steps) * Math.PI * 2)).toFixed(3)} ${(r * Math.sin((j / steps) * Math.PI * 2)).toFixed(3)}`).join("") + "Z";
const faceMarkup = () => {
  const rim = FACE_R - 0.8;
  const minor = [];
  const major = [];
  for (let i = 0; i < 100; i++) {
    const angle = (i / 100) * Math.PI * 2;
    (i % 10 ? minor : major).push(tick(angle, rim - (i % 10 ? 1.4 : 2.6), rim));
  }
  const numerals = Array.from({ length: 10 }, (_, i) => {
    const angle = (i / 10) * Math.PI * 2;
    const r = rim - 5.4;
    return `<text class="iso-face-text" data-tone="lo" font-size="3.4" text-anchor="middle" transform="translate(${(Math.sin(angle) * r).toFixed(3)} ${(-Math.cos(angle) * r).toFixed(3)}) rotate(${(i * 36).toFixed(1)}) translate(0 1.2)">${i * 10}</text>`;
  }).join("");
  return `<path class="iso-line" data-tone="lo" d="${ringPath(rim, 160)}"/><path class="iso-line" data-tone="lo" d="${minor.join("")}"/><path class="iso-line" data-tone="mid" d="${major.join("")}"/>${numerals}<path class="iso-line" data-tone="faint" d="${ringPath(33.5, 140)}"/>`;
};
const dialBody = T.round({ name: "dial.body", group: dial, layer: face, F: DIAL_F, s0: PLATE_Z1, s1: SKIRT_TOP, r: DIAL_R, tone: "hi", crease: "lo", material: "steel", details: { ribs: { s0: PLATE_Z1 + 0.9, s1: SKIRT_TOP - 0.7, count: 150, fade: [0.08, 0.5], seams: true, tone: "lo" } } });
const dialBevel = T.round({ name: "dial.bevel", group: dial, layer: face, F: DIAL_F, s0: SKIRT_TOP, s1: FACE_Z, r: [DIAL_R, FACE_R], tone: "hi", crease: "lo", material: "steel" });
T.plane({ part: dialBevel, o: [0, 0, FACE_Z], u: [1, 0, 0], v: [0, 1, 0], svg: faceMarkup(), fade: [0.04, 0.3] });
void dialBody;
T.round({ name: "dial.hub", group: dial, layer: hand, F: DIAL_F, s0: FACE_Z, s1: HUB_TOP, r: HUB_R, tone: "mid", material: "chrome", details: { seams: [FACE_Z + 0.9] } });

const wheelFace = () => {
  const minor = [];
  const major = [];
  for (let i = 0; i < 100; i++) {
    const angle = (i / 100) * Math.PI * 2;
    (i % 10 ? minor : major).push(tick(angle, WHEEL_R - 3.2 - (i % 10 ? 1.2 : 2.4), WHEEL_R - 3.2));
  }
  const numerals = Array.from({ length: 10 }, (_, i) => {
    const angle = (i / 10) * Math.PI * 2;
    const r = WHEEL_R - 9.4;
    return `<text class="iso-face-text" data-tone="mid" font-size="4.2" text-anchor="middle" transform="translate(${(Math.sin(angle) * r).toFixed(3)} ${(-Math.cos(angle) * r).toFixed(3)}) rotate(${(i * 36).toFixed(1)}) translate(0 1.5)">${i}</text>`;
  }).join("");
  return `<path class="iso-line" data-tone="lo" d="${ringPath(WHEEL_R - 3.2, 180)}"/><path class="iso-line" data-tone="lo" d="${minor.join("")}"/><path class="iso-line" data-tone="mid" d="${major.join("")}"/>${numerals}`;
};
T.round({ name: "lock.pinion", group: dial, layer: lock, F: VERT(0, 0), s0: LOCK_Z0, s1: LOCK_Z1, r: PINION_R, tone: "mid", material: "brass", details: { ribs: { s0: LOCK_Z0 + 0.5, s1: LOCK_Z1 - 0.5, count: 48, seams: true, tone: "lo" } } });
const wheelDisc = T.round({ name: "lock.wheel", group: wheel, layer: lock, F: VERT(WHEEL_AT[0], WHEEL_AT[1]), s0: LOCK_Z0, s1: LOCK_Z1, r: WHEEL_R, tone: "hi", crease: "lo", material: "brass", details: { ribs: { s0: LOCK_Z0 + 0.5, s1: LOCK_Z1 - 0.5, count: 192, seams: true, tone: "lo" } } });
T.plane({ part: wheelDisc, o: [WHEEL_AT[0], WHEEL_AT[1], LOCK_Z1], u: [1, 0, 0], v: [0, 1, 0], svg: wheelFace() });
T.round({ name: "lock.wheel.hub", group: wheel, layer: lock, F: VERT(WHEEL_AT[0], WHEEL_AT[1]), s0: LOCK_Z1, s1: LOCK_Z1 + 0.5, r: 6, tone: "mid", material: "steel", details: { bolts: [{ s: LOCK_Z1 + 0.5, r: 3.8, count: 4, size: 0.5, phase: 0.4 }] } });

T.round({ name: "pawl.post", layer: face, F: VERT(PAWL_AT[0], PAWL_AT[1]), s0: PLATE_Z1, s1: PLATE_Z1 + 1, r: 2.4, tone: "mid", material: "gunmetal" });
const pawlRing = [];
for (let j = 0; j <= 10; j++) pawlRing.push([PAWL_AT[0] + 2.2 * Math.cos((j / 10) * Math.PI), PAWL_AT[1] + 2.2 * Math.sin((j / 10) * Math.PI)]);
pawlRing.push([PAWL_AT[0] - 0.6, DIAL_R + 0.4], [PAWL_AT[0], DIAL_R + 0.15], [PAWL_AT[0] + 0.6, DIAL_R + 0.4]);
T.prism({ name: "pawl.arm", group: pawl, layer: face, F: { o: [0, 0, 0], a: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] }, polygon: pawlRing, s0: PLATE_Z1 + 1, s1: PLATE_Z1 + 2.6, tone: "hi", crease: "lo", bevel: 0.3, material: "chrome" });
T.round({ name: "pawl.pin", group: pawl, layer: face, F: VERT(PAWL_AT[0], PAWL_AT[1]), s0: PLATE_Z1 + 2.6, s1: PLATE_Z1 + 3.3, r: 1.1, tone: "mid", material: "steel", details: { bolts: [{ s: PLATE_Z1 + 3.3, r: 0, count: 1, size: 0.45 }] } });

const H = { group: dial, layer: hand };
const plan = (name, rect, z, h, extra = {}) => T.prism({ name, ...H, plan: rect, z, h, steps: extra.steps ?? 2, tone: "mid", ...extra });
const along = (name, F, s0, s1, r, extra = {}) => T.round({ name, ...H, F, s0, s1, r, tone: "mid", ...extra });
const frameOf = (o, a, u) => {
  const axis = G.unit3(a);
  const uu = u ? G.unit3(G.sub3(u, G.mul3(axis, G.dot3(u, axis)))) : G.frameAlong(o, axis).u;
  return { o, a: axis, u: uu, v: G.cross3(axis, uu) };
};

const sideScrews = [];
for (const [y, ny] of [[PALM.y, -1], [PALM.y + PALM.d, 1]]) for (const x of [-11, 3]) sideScrews.push({ at: [HUB_TOP + 5.2, x, y], normal: [0, ny, 0], size: 0.45, tone: "mid" });
for (const y of [-7, 7]) sideScrews.push({ at: [HUB_TOP + 5.2, PALM.x, y], normal: [-1, 0, 0], size: 0.45, tone: "mid" });
T.prism({ name: "hand.palm", ...H, plan: PALM, z: HUB_TOP, h: PALM_TOP - HUB_TOP, steps: 4, bevel: 0.9, tone: "hi", crease: "lo", material: "gold", details: { seams: [HUB_TOP + 2.6], dots: sideScrews, rings: sideScrews.map((d) => ({ at: d.at, normal: d.normal, r: 0.95, tone: "lo" })) } });
const coverScrews = [[-13.7, -11.4], [-13.7, 11.4], [9.3, -11.4], [9.3, 11.4]].map(([x, y]) => ({ at: [COVER_TOP, x, y], normal: [0, 0, 1], size: 0.48, tone: "mid" }));
T.prism({ name: "hand.cover", ...H, plan: COVER, z: PALM_TOP, h: COVER_TOP - PALM_TOP, steps: 3, bevel: 0.45, tone: "mid", material: "gold", details: { dots: coverScrews, rings: coverScrews.map((d) => ({ at: d.at, normal: [0, 0, 1], r: 1.1, tone: "lo" })) } });
const CONN = { x: -15.1, y: -10.4, w: 4.4, d: 20.8, r: 0.9 };
plan("hand.connector", CONN, COVER_TOP, 5.2, { bevel: 0.35, material: "gunmetal", details: { seams: [COVER_TOP + 1.2] } });
const CABLE_Z = COVER_TOP + 1.9;
const PORT_X = CONN.x + CONN.w;
const PORTS = { red: -8.6, green: -5.4, blue: 8.6 };
const CLAMP_X = -2.3;
const CLAMP_R = 1.45;
for (const [colour, y] of Object.entries(PORTS)) along(`hand.port.${colour}`, frameOf([PORT_X, y, CABLE_Z], [1, 0, 0], [0, 0, 1]), 0, 0.7, 1.15, { owner: `cable.${colour}`, material: "brass" });
const CLAMP_T = 1.1;
for (const colour of Object.keys(PORTS)) along(`hand.clamp.${colour}`, frameOf([CLAMP_X, PORTS[colour], CABLE_Z], [1, 0, 0], [0, 0, 1]), -CLAMP_T / 2, CLAMP_T / 2, CLAMP_R, { owner: `cable.${colour}`, material: "brass" });
plan("hand.clamp.left", { x: CLAMP_X - 1.3, y: -10.4, w: 2.6, d: 6.6, r: 0.6 }, COVER_TOP, CABLE_Z - CLAMP_R - COVER_TOP, { material: "gunmetal" });
plan("hand.clamp.right", { x: CLAMP_X - 1.3, y: 6.9, w: 2.6, d: 3.4, r: 0.6 }, COVER_TOP, CABLE_Z - CLAMP_R - COVER_TOP, { material: "gunmetal" });
const TERM = { x: 7.3, y: PORTS.green - 1.6, w: 2.6, d: 3.2, r: 0.5 };
plan("hand.terminal", TERM, COVER_TOP, CABLE_Z + 1.8 - COVER_TOP, { bevel: 0.3, material: "gunmetal" });

const PIST_Y = 0;
const BODY_R = 2.4;
const SADDLE_H = 0.9;
const PIST_Z = COVER_TOP + SADDLE_H + BODY_R;
const PIST_F = frameOf([0, PIST_Y, PIST_Z], [1, 0, 0], [0, 0, 1]);
const BODY = [-6.1, 2.9];
for (const [index, x] of [-5.1, 0.1].entries()) plan(`hand.saddle.${index}`, { x, y: PIST_Y - 1.9, w: 2.4, d: 3.8, r: 0.5 }, COVER_TOP, SADDLE_H, { material: "gunmetal" });
along("hand.piston.body", PIST_F, BODY[0], BODY[1], BODY_R, { tone: "hi", crease: "lo", material: "gunmetal", details: { seams: [BODY[0] + 1.6, BODY[1] - 1.6] } });
const GLAND = [BODY[1], BODY[1] + 1.4];
const glandRadius = (sv) => 2.3 + 0.55 * Math.sqrt(Math.max(0, 1 - ((sv - GLAND[0]) / (GLAND[1] - GLAND[0])) ** 2));
const glandSlope = (sv) => {
  const x = Math.min(0.999, (sv - GLAND[0]) / (GLAND[1] - GLAND[0]));
  return (-0.55 * x) / ((GLAND[1] - GLAND[0]) * Math.sqrt(1 - x * x));
};
T.lathe({ name: "hand.piston.gland", ...H, F: PIST_F, profile: [[GLAND[0], glandRadius(GLAND[0])], [GLAND[1], glandRadius(GLAND[1])]], radius: glandRadius, slope: glandSlope, tone: "mid", material: "brass", details: { bolts: [{ s: GLAND[1], r: 1.75, count: 6, phase: 0.3, size: 0.34, tone: "mid" }] } });
const ROD_END = 8.9;
along("hand.piston.rod", PIST_F, GLAND[1], ROD_END, 0.95, { tone: "hi", material: "chrome" });
plan("hand.clevis", { x: ROD_END, y: PIST_Y - 1.9, w: 2.4, d: 3.8, r: 0.5 }, COVER_TOP, PIST_Z + 2.4 - COVER_TOP, { bevel: 0.3, material: "steel", details: { dots: [-1, 1].map((side) => ({ at: [PIST_Z, ROD_END + 1.2, PIST_Y + side * 1.9], normal: [0, side, 0], size: 0.42 })) } });
const BALL_R = 2.2;
const BALL = [BODY[0] - BALL_R, PIST_Y, PIST_Z];
T.ball({ name: "hand.ball", ...H, c: BALL, r: BALL_R, tone: "hi", material: "chrome" });
along("hand.ball.post", VERT(BALL[0], BALL[1]), COVER_TOP, BALL[2] - BALL_R, 1.3, { material: "steel" });

const FINGERS = [
  { id: 1, y: -9.2, psi: -13, port: true, cable: "red" },
  { id: 2, y: 0, psi: 0, guide: true, flex: true },
  { id: 3, y: 9.2, psi: 13, port: true, cable: "blue" },
];
const D2R = Math.PI / 180;
const FZ = HUB_TOP + 4;
const RL = 2.35;
const LENGTHS = [7.4, 5.8, 3.6];
const GAP = 0.6;
const DISC_T = 1.2;
const PITCH = 8 * D2R;
const JUNCTION_H = 6;
export const TIPS = [];
export const LINKS = [];
export const LED = [];
const FINGER_PORTS = {};
for (const finger of FINGERS) {
  const psi = finger.psi * D2R;
  const d1 = [Math.cos(psi), Math.sin(psi), 0];
  const w = [-Math.sin(psi), Math.cos(psi), 0];
  const pitch = (a) => [Math.cos(a) * Math.cos(psi), Math.cos(a) * Math.sin(psi), -Math.sin(a)];
  const B = [PALM.x + PALM.w, finger.y, FZ];
  const R0 = 3.7;
  const e0 = R0 + Math.abs(Math.tan(psi)) * (RL + DISC_T);
  const C1 = G.add3(B, G.mul3(d1, e0));
  const E1 = G.add3(C1, G.mul3(d1, LENGTHS[0]));
  const d2 = pitch(PITCH);
  const C2 = G.add3(E1, G.mul3(G.unit3(G.add3(d1, d2)), 2 * RL + GAP));
  const J1 = G.mul3(G.add3(E1, C2), 0.5);
  const E2 = G.add3(C2, G.mul3(d2, LENGTHS[1]));
  const tipOf = (a3) => {
    const d3 = pitch(a3);
    const C3 = G.add3(E2, G.mul3(G.unit3(G.add3(d2, d3)), 2 * RL + GAP));
    return { d3, C3, E3: G.add3(C3, G.mul3(d3, LENGTHS[2])), J2: G.mul3(G.add3(E2, C3), 0.5) };
  };
  let lo = 0;
  let hi = 70 * D2R;
  for (let it = 0; it < 60; it++) {
    const mid = (lo + hi) / 2;
    if (tipOf(mid).E3[2] - RL > FACE_Z) lo = mid;
    else hi = mid;
  }
  const { C3, E3, J2 } = tipOf(hi);
  TIPS.push(E3);
  const tag = `hand.f${finger.id}`;
  const g1 = finger.flex ? T.group(`f${finger.id}.mcp`, { parent: dial, hinge: { point: C1, axis: w, range: [-22, 0] } }) : dial;
  const g2 = finger.flex ? T.group(`f${finger.id}.pip`, { parent: g1, hinge: { point: J1, axis: w, range: [-14, 0] } }) : dial;
  const g3 = finger.flex ? T.group(`f${finger.id}.dip`, { parent: g2, hinge: { point: J2, axis: w, range: [-9, 0] } }) : dial;
  if (finger.flex) LINKS.push({ tip: E3, dip: J2, groups: [g1.name, g2.name, g3.name], d3: G.unit3(G.sub3(E3, C3)), w });
  const link = (name, C, E, seams, group) => T.round({ name: `${tag}.${name}`, group, layer: hand, F: frameOf(C, G.sub3(E, C), [0, 0, 1]), s0: 0, s1: G.len3(G.sub3(E, C)), r: RL, ends: ["dome", "dome"], tone: "hi", crease: "lo", material: "gold", details: { seams } });
  link("p1", C1, E1, [1.2, LENGTHS[0] - 1.2], g1);
  link("p2", C2, E2, [1.0, LENGTHS[1] - 1.0], g2);
  const tipLink = link("p3", C3, E3, [0.9], g3);
  if (finger.flex) {
    const d3 = G.unit3(G.sub3(E3, C3));
    const led = G.add3(E3, G.mul3(d3, RL));
    LED.push({ at: led, normal: d3, part: tipLink.name, group: g3.name });
    T.billboard({ part: tipLink, at: led, normal: d3, fade: [0.02, 0.3], svg: `<g class="td-led"><circle r="1.15" class="td-led-halo"/><circle r="0.62" class="td-led-body"/><circle r="0.26" class="td-led-core"/></g>` });
  }
  const discs = (name, J, rDisc, group) => {
    for (const side of [-1, 1]) {
      const centre = G.add3(J, G.mul3(w, side * (RL + DISC_T / 2)));
      const F = frameOf(centre, G.mul3(w, side), [0, 0, 1]);
      T.round({ name: `${tag}.${name}${side > 0 ? "+" : "-"}`, group, layer: hand, F, s0: -DISC_T / 2, s1: DISC_T / 2, r: rDisc, tone: "mid", material: "chrome", details: { dots: [{ at: [DISC_T / 2, 0, 0], normal: G.mul3(w, side), size: 0.42 }], rings: [{ at: [DISC_T / 2, 0, 0], normal: G.mul3(w, side), r: 0.85, tone: "lo" }] } });
    }
  };
  discs("k0", C1, R0, dial);
  discs("k1", J1, 3.5, g1);
  discs("k2", J2, 3.3, g2);
  if (finger.port) {
    const M = G.mul3(G.add3(C1, E1), 0.5);
    void M;
    const ring = [[-2.1, -1.7], [2.1, -1.7], [2.1, 1.7], [-2.1, 1.7]].map(([a, b]) => [M[0] + d1[0] * a + w[0] * b, M[1] + d1[1] * a + w[1] * b]);
    const z0 = M[2] + RL;
    T.prism({ name: `${tag}.junction`, ...H, F: { o: [0, 0, 0], a: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] }, polygon: ring, s0: z0, s1: z0 + JUNCTION_H, tone: "mid", bevel: 0.3, material: "gunmetal", details: { seams: [z0 + 1] } });
    const back = [M[0] - d1[0] * 2.1, M[1] - d1[1] * 2.1, z0 + JUNCTION_H * 0.62];
    along(`${tag}.port`, frameOf(back, G.mul3(d1, -1), [0, 0, 1]), 0, 0.6, 0.95, { owner: `cable.${finger.cable}`, material: "brass" });
    FINGER_PORTS[finger.id] = { face: back, d1 };
  }
  if (finger.guide) {
    const n = G.unit3(G.cross3(d2, w));
    const up = n[2] > 0 ? n : G.mul3(n, -1);
    T.prism({ name: `${tag}.guide`, group: g2, layer: hand, F: { o: C2, a: d2, u: w, v: up }, polygon: [[-1.5, RL], [1.5, RL], [1.15, RL + 0.9], [-1.15, RL + 0.9]], s0: 1.1, s1: LENGTHS[1] - 1.1, tone: "mid", bevel: 0.25, material: "gunmetal" });
  }
}

const CABLE_R = 0.7;
const BEND = 8 * CABLE_R;
const portStart = (colour) => [PORT_X, PORTS[colour], CABLE_Z];
const cable = (colour, waypoints, ends = [0.7, 0]) => {
  const points = G.fillet(waypoints, BEND, 12);
  const at = (x) => {
    let walked = 0;
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1];
      const b = points[i];
      const length = G.len3(G.sub3(b, a));
      if ((a[0] - x) * (b[0] - x) <= 0 && Math.abs(b[0] - a[0]) > 1e-9) return walked + (length * (x - a[0])) / (b[0] - a[0]);
      walked += length;
    }
    return null;
  };
  const hit = at(CLAMP_X);
  const total = G.pathLength(points);
  const gaps = [[0, ends[0]], ...(hit === null ? [] : [[hit - CLAMP_T / 2, hit + CLAMP_T / 2]]), ...(ends[1] ? [[total - ends[1], total]] : [])];
  T.tube({ name: `cable.${colour}`, ...H, route: points, r: CABLE_R, chunk: 6, gaps, hue: colour, tone: "hi", material: "rubber" });
};
const approach = (port, length, rise) => G.add3(G.add3(port.face, G.mul3(port.d1, -length)), [0, 0, rise]);
cable("red", [portStart("red"), [-0.4, PORTS.red, CABLE_Z], approach(FINGER_PORTS[1], 7.6, 0.3), FINGER_PORTS[1].face], [0.7, 0.6]);
cable("green", [portStart("green"), [TERM.x, PORTS.green, CABLE_Z]]);
cable("blue", [portStart("blue"), [-0.4, PORTS.blue, CABLE_Z], approach(FINGER_PORTS[3], 7.6, 0.3), FINGER_PORTS[3].face], [0.7, 0.6]);
const HOSE_R = 0.9;
const HOSE_Y = 4.6;
const HOSE_Z = COVER_TOP + 3.3;
const VALVE = { x: 3.4, y: HOSE_Y - 1.8, w: 3.4, d: 3.6, r: 0.6 };
plan("hand.valve", VALVE, COVER_TOP, HOSE_Z + 1.9 - COVER_TOP, { bevel: 0.3, material: "gunmetal", details: { seams: [COVER_TOP + 1.2], dots: [[VALVE.x + 0.8, VALVE.y + 0.8], [VALVE.x + VALVE.w - 0.8, VALVE.y + VALVE.d - 0.8]].map(([x, y]) => ({ at: [HOSE_Z + 1.9, x, y], normal: [0, 0, 1], size: 0.4 })) } });
const FERRULE = 1.3;
const hoseFrom = [PORT_X, HOSE_Y, HOSE_Z];
const hoseTo = [VALVE.x, HOSE_Y, HOSE_Z];
along("hand.hose.ferrule.0", frameOf(hoseFrom, [1, 0, 0], [0, 0, 1]), 0, FERRULE, 1.25, { owner: "hand.hose", material: "brass", details: { seams: [0.45] } });
along("hand.hose.ferrule.1", frameOf(hoseTo, [-1, 0, 0], [0, 0, 1]), 0, FERRULE, 1.25, { owner: "hand.hose", material: "brass", details: { seams: [0.45] } });
const hoseLength = G.len3(G.sub3(hoseTo, hoseFrom));
T.tube({ name: "hand.hose", ...H, route: [hoseFrom, hoseTo], r: HOSE_R, chunk: 6, gaps: [[0, FERRULE], [hoseLength - FERRULE, hoseLength]], rings: { pitch: 1.5, twist: 1.6, cross: true, fade: [0.1, 0.55], tone: "lo" }, tone: "hi", material: "steel" });

const KNUCKLE_R = 3.2;
const KNUCKLE_D = 2.6;
const KNUCKLE = [-4, PALM.y - KNUCKLE_D, FZ];
T.ball({ name: "hand.thumb.knuckle", ...H, c: KNUCKLE, r: KNUCKLE_R, flats: [{ n: [0, 1, 0], d: KNUCKLE_D }, { n: [0, -1, 0], d: KNUCKLE_D }], tone: "hi", material: "chrome" });
const THUMB_F = frameOf([KNUCKLE[0], KNUCKLE[1] - KNUCKLE_D, KNUCKLE[2]], [0, -1, 0], [0, 0, 1]);
T.round({ name: "hand.thumb.cone", ...H, F: THUMB_F, s0: 0, s1: 3, r: [1.8, 1.25], tone: "mid", material: "steel" });
const THUMB_R = 1.9;
T.round({ name: "hand.thumb.link", ...H, F: frameOf(G.pointOf(THUMB_F, 3 + THUMB_R, 0, 0), [0, -1, 0], [0, 0, 1]), s0: 0, s1: 6.5, r: THUMB_R, ends: ["dome", "dome"], tone: "hi", crease: "lo", material: "gold", details: { seams: [1.2, 5.3] } });

const POD = [-7, 11.4];
const POD_Z = COVER_TOP + 0.3;
T.round({ name: "hand.pod.socket", ...H, F: VERT(POD[0], POD[1]), s0: COVER_TOP, s1: POD_Z, r: 1.7, tone: "mid", material: "gunmetal" });
const podGroup = T.group("pod", { parent: dial, free: { origin: [POD[0], POD[1], POD_Z] } });
const hex = Array.from({ length: 6 }, (_, j) => [1.35 * Math.cos(((2 * j + 1) * Math.PI) / 6), 1.35 * Math.sin(((2 * j + 1) * Math.PI) / 6)]);
T.prism({ name: "hand.pod.nut", group: podGroup, layer: hand, F: { o: [POD[0], POD[1], 0], a: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] }, polygon: hex, s0: POD_Z, s1: POD_Z + 1.4, tone: "hi", crease: "lo", bevel: 0.2, material: "brass" });
T.round({ name: "hand.pod.cap", group: podGroup, layer: hand, F: VERT(POD[0], POD[1]), s0: POD_Z + 1.4, s1: POD_Z + 2.0, r: 0.9, tone: "mid", material: "chrome", details: { bolts: [{ s: POD_Z + 2.0, r: 0, count: 1, size: 0.4 }] } });
export const POD_AT = [POD[0], POD[1], POD_Z];

base();
backPosts();
midStatic();
const verify = argv.includes("--verify");
const out = T.build({ statics: R.items, poses, verify });
A.settle(R, P);
const layerOf = (name) => R.items.filter((item) => item.layer === name).sort((a, b) => a.key - b.key).map((item) => item.svg);
const BACK = [`<path class="iso-halo" d="${k.haloOf(BASE, -BASE_H - FOOT_H, BASE_H, P)}"/>`, ...layerOf("back")];
const MID = layerOf("mid");
const FRONT = layerOf("front");
if (argv.includes("--audit")) {
  A.auditOrExit(R, P);
  TA.orbitOrExit(T, R, P, {
    sweep: { at: (x) => follow({ dial: x }), from: 0, to: 1440, frame: 0.3 },
    clips: [
      { name: "tap", at: (t) => follow({ dial: 40 + 12 * t, lift: tapAt(t) }), from: 0, to: 4, frame: 1 / 60 },
      { name: "break", at: (t) => follow({ dial: 100 + 12 * t, lift: 0.3, pod: breakAt(t) }), from: 0, to: BREAK_END, frame: 1 / 60 },
    ],
    crowd: [["cable.red", "hand.f1.p2"], ["cable.red", "hand.f2.p1"], ["cable.red", "hand.f2.p2"], ["cable.red", "hand.thumb.link"], ["cable.blue", "hand.f2.p1"], ["cable.blue", "hand.f3.p2"]], quick: argv.includes("--quick"), only: argv.find((a) => a.startsWith("--only="))?.slice(7).split(",") ?? null, log: argv.includes("--verbose") ? (text) => console.log(`  … ${text}`) : null });
}

const LABEL = "A combination dial on a raised door plate. A small robot hand lies on the dial's hub and turns with it: a palm with a cover plate, a connector block, a piston on two saddles with a ball joint, three jointed fingers whose tips rest on the dial face, and a thumb on a ball knuckle.";
const backSvg = k.figureSvg({ width: WIDTH, height: HEIGHT, label: LABEL, body: BACK });
const staticSvg = (list) => `<svg xmlns="http://www.w3.org/2000/svg" class="iso-svg iso-layer" viewBox="0 0 ${WIDTH} ${HEIGHT}" aria-hidden="true">${list.join("")}</svg>`;
const frontSvg = staticSvg(FRONT);
const canvas = (name) => `<canvas class="iso-layer td-gl" data-canvas="${name}" aria-hidden="true"></canvas>`;
const stage = `<div class="iso-stage iso-turn td-stage" tabindex="0" role="slider" aria-label="Combination dial: drag around the dial or use the arrow keys to turn it, press B to break the pod off" aria-valuemin="0" aria-valuemax="99" aria-valuenow="0" aria-valuetext="dial at 0">${backSvg}${out.layerSvg("lock", { width: WIDTH, height: HEIGHT })}${staticSvg(MID)}${out.layerSvg("dial", { width: WIDTH, height: HEIGHT })}${canvas("back")}${out.layerSvg("hand", { width: WIDTH, height: HEIGHT })}${canvas("front")}${frontSvg}</div>`;

const DATA = { turn: out.data, index: INDEX_AT, pod: POD_AT, breakEnd: BREAK_END, links: LINKS, gl: { W: WIDTH, H: HEIGHT, camera: glCamera(P), led: LED[0], faceZ: FACE_Z, faceR: FACE_R, hubR: HUB_R, socket: [POD_AT[0], POD_AT[1], POD_AT[2]] } };
const FOLLOW = `const follow = ${follow.toString()};\nconst tapAt = ${tapAt.toString()};\nconst breakAt = ${breakAt.toString()};`;
const CSS = out.css + "\n" + readFileSync(join(HERE, "dial.css"), "utf8");
const LIVE = readFileSync(join(HERE, "live.js"), "utf8").replace("__DATA__", JSON.stringify(DATA)).replace("const follow = null;", FOLLOW);
const body = k.plateHtml({
  fig: "Fig 3D",
  title: "Turning dial",
  hint: "Drag the dial · arrow keys · B breaks the pod",
  readout: "dial 0 · 0.0°",
  keys: [
    { mark: "raised", label: "The dial and the hand on it: one rigid group" },
    { mark: "lit", label: "The index" },
  ],
  caption: "The optional 3D mode: every part on the dial is re-projected and re-shaded each frame by the kit's own rules, and painted in an order solved from separating planes.",
  body: `<style>${CSS}</style>${stage}`,
});
const page = k.pageHtml({ title: "Turning dial", theme, body, script: turnScript() + glScript() + (verify ? `window.__isoVerify = ${JSON.stringify(out.verify)};\n` : "") + LIVE, width: 680 });
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const name = `turning-dial${verify ? "-verify" : ""}${theme === "light" ? "-light" : ""}.html`;
  writeFileSync(join(HERE, name), page);
  console.log(`wrote ${name} · ${(page.length / 1024).toFixed(0)} KB · ${out.report.parts} parts · ${out.report.layers.map((l) => `${l.name}: ${l.planar} planes, ${l.dynamic} dynamic`).join(" · ")} · build ${out.report.seconds.toFixed(1)} s`);
}

export { out };
