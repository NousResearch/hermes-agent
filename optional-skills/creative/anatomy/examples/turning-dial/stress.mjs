import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../kit/iso-kit.mjs";
import * as G from "../../kit/lathe.mjs";
import * as A from "../../kit/audit.mjs";
import { tubePieces, tubeSvg } from "../../kit/tube.mjs";
import { turning } from "../../kit/turn-build.mjs";
import * as TA from "../../kit/turn-audit.mjs";
import { turnScript } from "../../scripts/inline-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const theme = argv.includes("--light") ? "light" : "dark";
const WIDTH = 760;
const HEIGHT = 560;
const BENCH = { x: -120, y: -105, w: 240, d: 210, r: 6 };
const BENCH_TOP = -20;

export const P = k.fitProjection([...k.boxCorners(BENCH, BENCH_TOP - 6, BENCH_TOP), [0, 0, 30]], WIDTH, HEIGHT, { pad: 20, azimuth: 40, elevation: 31 });
export const R = A.recorder(P);
const put = (item) => R.put({ ...item, shapes: item.shapes.map((shape, index) => R.solid(Object.assign(shape, { name: shape.name ?? (index ? `${item.name}#${index + 1}` : item.name) }))) });
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const VERT = (x, y) => ({ o: [x, y, 0], a: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] });
const mat = (name, svg) => `<g data-mat="${name}">${svg}</g>`;

function workshop() {
  put({ name: "stand.bench", layer: "back", at: [0, 0, BENCH_TOP - 6], bias: 8, svg: S(k.slabOf(BENCH, BENCH_TOP - 6, 6, P, 8, 1.2), { tone: "lo", inner: L(k.sideSeam(BENCH, BENCH_TOP - 2.4, P, 8), { tone: "faint" }) }), shapes: [A.slab(BENCH, BENCH_TOP - 6, 6, { ground: true })] });
  put({ name: "stand.base.flange", layer: "back", at: [0, 0, BENCH_TOP], svg: mat("gunmetal", S(k.cylinder(0, 0, 48, BENCH_TOP, 2, P, 96), { tone: "mid" })), shapes: [A.cylinder(0, 0, 48, BENCH_TOP, 2)] });
  put({ name: "stand.base", layer: "back", at: [0, 0, BENCH_TOP + 2], svg: mat("gunmetal", S(k.cylinder(0, 0, 44, BENCH_TOP + 2, 14, P, 96), { tone: "mid", inner: L(G.arcOf(VERT(0, 0), BENCH_TOP + 9, 44, P, { least: 0 }), { tone: "faint" }) })), shapes: [A.cylinder(0, 0, 44, BENCH_TOP + 2, 14)] });
  const unit = { x: -104, y: -62, w: 34, d: 26, r: 2 };
  const vents = Array.from({ length: 9 }, (_, i) => k.pathOf([k.iso([unit.x + 4 + i * 3.2, unit.y, BENCH_TOP + 5], P), k.iso([unit.x + 4 + i * 3.2, unit.y, BENCH_TOP + 15], P)])).join("");
  put({ name: "stand.power", layer: "back", at: [unit.x + 17, unit.y + 13, BENCH_TOP], svg: mat("gunmetal", S(k.slabOf(unit, BENCH_TOP, 20, P, 4, 0.8), { tone: "hi", inner: L(vents, { tone: "faint" }) })), shapes: [A.slab(unit, BENCH_TOP, 20)] });
  for (const [i, x] of [-98, -90, -82].entries()) put({ name: `stand.power.knob.${i}`, layer: "back", at: [x, unit.y + unit.d + 0.6, BENCH_TOP + 22], svg: mat("brass", S(k.cylinder(x, unit.y + 8 + i * 5, 1.6, BENCH_TOP + 20, 1.6, P, 24), { tone: "hi" })), shapes: [A.cylinder(x, unit.y + 8 + i * 5, 1.6, BENCH_TOP + 20, 1.6)] });
  const rack = { x: -60, y: -100, w: 70, d: 10, r: 1 };
  put({ name: "stand.rack", layer: "back", at: [rack.x + 35, rack.y + 5, BENCH_TOP], svg: mat("steel", S(k.slabOf(rack, BENCH_TOP, 26, P, 2, 0.6), { tone: "mid" })), shapes: [A.slab(rack, BENCH_TOP, 26)] });
  for (let i = 0; i < 7; i++) {
    const x = rack.x + 6 + i * 9.4;
    const ring = [[x - 1, rack.y + rack.d], [x + 1, rack.y + rack.d], [x + 1, rack.y + rack.d + 1.4], [x - 1, rack.y + rack.d + 1.4]];
    put({ name: `stand.rack.tool.${i}`, layer: "back", at: [x, rack.y + rack.d, BENCH_TOP + 8], svg: mat(i % 2 ? "chrome" : "steel", S(k.extrude(ring, BENCH_TOP + 6 + (i % 3) * 2, 16, P), { tone: "hi" })), shapes: [A.extruded(ring, BENCH_TOP + 6 + (i % 3) * 2, 16)] });
  }
  const reel = [-88, 40];
  put({ name: "stand.reel", layer: "back", at: [reel[0], reel[1], BENCH_TOP], svg: mat("copper", S(k.cylinder(reel[0], reel[1], 9, BENCH_TOP, 4, P, 64), { tone: "hi" })), shapes: [A.cylinder(reel[0], reel[1], 9, BENCH_TOP, 4)] });
  const tray = { x: 62, y: 46, w: 36, d: 22, r: 2 };
  put({ name: "stand.tray", layer: "back", at: [tray.x + 18, tray.y + 11, BENCH_TOP], svg: mat("steel", S(k.slabOf(tray, BENCH_TOP, 3, P, 4, 0.5), { tone: "mid" })), shapes: [A.slab(tray, BENCH_TOP, 3)] });
  const wrench = [[70, -70], [96, -64], [97, -61], [71, -67]];
  put({ name: "stand.wrench", layer: "back", at: [84, -66, BENCH_TOP], svg: mat("chrome", S(k.extrude(wrench, BENCH_TOP, 0.9, P), { tone: "hi" })), shapes: [A.extruded(wrench, BENCH_TOP, 0.9)] });
  const cables = [
    { name: "stand.feed", r: 1.1, route: G.fillet([[unit.x + unit.w, unit.y + 8, BENCH_TOP + 1.1], [-60, unit.y + 8, BENCH_TOP + 1.1], [-46, -30, BENCH_TOP + 1.1], [-44 * Math.cos(0.6), -44 * Math.sin(0.6), BENCH_TOP + 1.1]], 9, 12), hue: "red" },
    { name: "stand.feed.2", r: 0.8, route: G.fillet([[unit.x + unit.w, unit.y + 16, BENCH_TOP + 0.8], [-62, unit.y + 16, BENCH_TOP + 0.8], [-50, -24, BENCH_TOP + 0.8], [-44 * Math.cos(0.5), -44 * Math.sin(0.5), BENCH_TOP + 0.8]], 7, 12), hue: "blue" },
    { name: "stand.loose", r: 0.7, route: G.fillet([[reel[0] + 9, reel[1], BENCH_TOP + 0.7], [-66, 52, BENCH_TOP + 0.7], [-60, 76, BENCH_TOP + 0.7], [-30, 84, BENCH_TOP + 0.7]], 6, 12), hue: "green" },
  ];
  for (const c of cables) {
    R.route(c.name, c.route, c.r, { owner: c.name, limp: true, ends: [true, c.name !== "stand.loose"], free: [false, c.name === "stand.loose"] });
    tubePieces(c.route, c.r, P, { maxLength: 30, spacing: 0.9 }).forEach((piece, index) => put({ name: `${c.name}:${index}`, layer: "back", route: c.name, chunk: index, at: piece.mid, svg: `<g data-mat="rubber">${tubeSvg(piece, { tone: "hi" }).replace('class="tb"', `class="tb" data-hue="${c.hue}"`)}</g>`, shapes: [A.tube(piece.points, piece.radii, { name: `${c.name}:${index}`, owner: c.name })] }));
  }
}

export const T = turning(P, { recorder: R });
const table = T.group("table", { turn: { pivot: [0, 0] } });
const W0 = [-30, 0, 12.5];
const wrist = T.group("wrist", { parent: table, hinge: { point: W0, axis: [0, 1, 0], range: [-3, 3] } });
const thumbBase = T.group("thumb", { parent: wrist, hinge: { point: [-8, -16, 0], axis: [0, 0, 1], range: [-10, 10] } });
const hand = T.layer("hand");
T.stack(["back", "hand", "front"]);

const groupsOf = [];
const H = (group) => ({ group, layer: hand });
const prismAt = (name, group, rect, z0, z1, extra = {}) => T.prism({ name, ...H(group), plan: rect, z: z0, h: z1 - z0, steps: extra.steps ?? 2, tone: extra.tone ?? "mid", ...extra });
const frameOf = (o, a, u) => {
  const axis = G.unit3(a);
  const uu = G.unit3(G.sub3(u, G.mul3(axis, G.dot3(u, axis))));
  return { o, a: axis, u: uu, v: G.cross3(axis, uu) };
};

T.round({ name: "table.platter", ...H(table), F: VERT(0, 0), s0: -4, s1: 0, r: 50, tone: "hi", crease: "lo", material: "steel", details: { ribs: { s0: -3.4, s1: -0.6, count: 160, seams: true, tone: "lo" } } });
T.round({ name: "table.hub", ...H(table), F: VERT(0, 0), s0: 0, s1: 1, r: 8, tone: "mid", material: "chrome" });
T.round({ name: "table.foot", ...H(table), F: VERT(W0[0], W0[1]), s0: 0, s1: 1.2, r: 8, tone: "mid", material: "gunmetal" });
T.round({ name: "table.column", ...H(table), F: VERT(W0[0], W0[1]), s0: 1.2, s1: 9.5, r: 4, tone: "hi", material: "chrome" });
for (let i = 0; i < 8; i++) {
  const t = (i / 8) * Math.PI * 2 + 0.2;
  T.round({ name: `table.bolt.${i}`, ...H(table), F: VERT(42 * Math.cos(t), 42 * Math.sin(t)), s0: 0, s1: 0.6, r: 1.1, tone: "mid", material: "brass" });
}
prismAt("table.term.a", table, { x: -11.5, y: -4.6, w: 2, d: 9.2, r: 0.4 }, 0, 1.3, { material: "gunmetal", bevel: 0.2 });
prismAt("table.term.b", table, { x: -21.5, y: -4.6, w: 2, d: 9.2, r: 0.4 }, 0, 1.3, { material: "gunmetal", bevel: 0.2 });
const tableCables = [];
for (let i = 0; i < 7; i++) {
  const y = -3.6 + i * 1.2;
  tableCables.push(T.tube({ name: `table.cable.${i}`, ...H(table), route: [[-11.5, y, 0.5], [-19.5, y, 0.5]], r: 0.42, chunk: 6, hue: ["red", "green", "blue"][i % 3], material: "rubber", bundle: "table.loom" }));
}

T.round({ name: "wrist.barrel", ...H(wrist), F: frameOf([W0[0], -13.2, W0[2]], [0, 1, 0], [0, 0, 1]), s0: 0, s1: 26.4, r: 2.9, tone: "hi", material: "chrome" });
for (const side of [-1, 1]) T.round({ name: `wrist.cap.${side}`, ...H(wrist), F: frameOf([W0[0], side * 13.2, W0[2]], [0, side, 0], [0, 0, 1]), s0: 0, s1: 0.8, r: 3.4, tone: "mid", material: "gunmetal" });
const PALM = { x: -27, y: -13, w: 36, d: 26, r: 2 };
const PALM_TOP = 14.5;
prismAt("palm.plate", wrist, PALM, 11, PALM_TOP, { steps: 4, bevel: 0.6, tone: "hi", material: "gold", details: { seams: [12.6] } });
prismAt("palm.manifold", wrist, { x: -26.8, y: -12, w: 3, d: 24, r: 0.6 }, PALM_TOP, 17, { material: "gunmetal", bevel: 0.25 });
for (let i = 0; i < 6; i++) prismAt(`palm.valve.${i}`, wrist, { x: -26.4, y: -11 + i * 4, w: 2.2, d: 2.4, r: 0.4 }, 17, 18.2, { material: "brass" });
prismAt("palm.knuckles", wrist, { x: 4, y: -13, w: 5, d: 26, r: 0.8 }, PALM_TOP, 18, { material: "gunmetal", bevel: 0.3 });
for (let i = 0; i < 10; i++) T.round({ name: `palm.bolt.${i}`, ...H(wrist), F: VERT(i % 2 ? 7.6 : 5.4, -11 + Math.floor(i / 2) * 5.5), s0: 18, s1: 18.4, r: 0.45, tone: "mid", material: "steel" });
const FINGER_Y = [-9.6, -3.2, 3.2, 9.6];
for (const [i, yk] of FINGER_Y.entries()) {
  const z = 16.5;
  const F = frameOf([0, yk, z], [1, 0, 0], [0, 0, 1]);
  for (const x of [-20, -12]) prismAt(`palm.ram${i}.saddle.${x}`, wrist, { x, y: yk - 1.2, w: 2, d: 2.4, r: 0.4 }, PALM_TOP, 15.1, { material: "gunmetal" });
  T.round({ name: `palm.ram${i}.cap`, ...H(wrist), F, s0: -22.6, s1: -22, r: 1.55, tone: "mid", material: "steel" });
  T.round({ name: `palm.ram${i}.body`, ...H(wrist), F, s0: -22, s1: -8, r: 1.4, tone: "hi", material: i % 2 ? "copper" : "gunmetal", details: { seams: [-20.5, -9.5] } });
  T.round({ name: `palm.ram${i}.gland`, ...H(wrist), F, s0: -8, s1: -6.6, r: 1.6, tone: "mid", material: "brass" });
  T.round({ name: `palm.ram${i}.rod`, ...H(wrist), F, s0: -6.6, s1: 1, r: 0.6, tone: "hi", material: "chrome" });
  prismAt(`palm.ram${i}.clevis`, wrist, { x: 1, y: yk - 1, w: 2, d: 2, r: 0.3 }, 15.1, 17.9, { material: "steel" });
}
const lanes = [[-12.5, -11.6], [-7.5, -6.4, -5.3], [-1.1, 0, 1.1], [5.3, 6.4, 7.5], [11.6, 12.5]];
const palmCables = [];
lanes.forEach((lane, l) =>
  lane.forEach((y, j) => {
    const r = Math.abs(y) > 11 ? 0.4 : 0.45;
    palmCables.push(T.tube({ name: `palm.cable.${l}.${j}`, ...H(wrist), route: [[-23.8, y, PALM_TOP + r + 0.05], [4, y, PALM_TOP + r + 0.05]], r, chunk: 6, hue: ["red", "green", "blue"][(l + j) % 3], material: "rubber", bundle: `palm.lane.${l}` }));
  }),
);

const LINK = [10, 8, 6];
const RL = 1.5;
function digit(tag, J0, d, w, parent, ranges) {
  const up = G.cross3(d, w);
  const at = (base, along, side, height) => G.add3(base, G.add3(G.mul3(d, along), G.add3(G.mul3(w, side), G.mul3(up, height))));
  const joints = [J0];
  for (const length of LINK) joints.push(G.add3(joints[joints.length - 1], G.mul3(d, length)));
  const groups = [];
  let g = parent;
  for (let i = 0; i < 3; i++) {
    g = T.group(`${tag}.j${i}`, { parent: g, hinge: { point: joints[i], axis: w, range: ranges[i] } });
    groups.push(g);
  }
  groupsOf.push(groups.map((group) => group.name));
  const owners = [parent, groups[0], groups[1]];
  const rect = (group, name, a0, a1, s0, s1, h0, h1, extra = {}) => {
    const F = { o: joints[0], a: up, u: d, v: w };
    void F;
    const o = at(joints[0], 0, 0, 0);
    T.prism({ name, ...H(group), F: { o, a: up, u: d, v: w }, polygon: [[a0, s0], [a1, s0], [a1, s1], [a0, s1]], s0: h0, s1: h1, tone: extra.tone ?? "mid", bevel: extra.bevel ?? 0, material: extra.material ?? "gunmetal" });
  };
  for (let i = 0; i < 3; i++) {
    const J = joints[i];
    const group = groups[i];
    const s = LINK.slice(0, i).reduce((sum, v) => sum + v, 0);
    const e = s + LINK[i];
    for (const side of [-1, 1]) T.round({ name: `${tag}.k${i}${side > 0 ? "+" : "-"}`, ...H(owners[i]), F: frameOf(G.add3(J, G.mul3(w, side * RL)), G.mul3(w, side), up), s0: 0, s1: 1, r: 2.2, tone: "mid", material: "chrome", details: { dots: [{ at: [1, 0, 0], normal: G.mul3(w, side), size: 0.45 }] } });
    T.round({ name: `${tag}.p${i}`, ...H(group), F: frameOf(at(J0, s + 1.8, 0, 0), d, up), s0: 0, s1: LINK[i] - 3.6, r: RL, ends: ["dome", "dome"], tone: "hi", crease: "lo", material: "gold", details: { seams: [0.8, LINK[i] - 4.4] } });
    const a0 = s + 2.4;
    const a1 = e - 2.4;
    for (const side of [-1, 1]) {
      rect(group, `${tag}.p${i}.plate${side > 0 ? "+" : "-"}`, a0, a1, side > 0 ? RL : -RL - 0.6, side > 0 ? RL + 0.6 : -RL, -1.2, 1.2, { material: "gold", tone: "hi", bevel: 0.15 });
      const count = Math.max(1, Math.floor((a1 - a0 - 0.8) / 1.2));
      for (let b = 0; b < count; b++) {
        const along = a0 + 0.8 + ((a1 - a0 - 1.6) * (b + 0.5)) / count;
        T.round({ name: `${tag}.p${i}.boss${side > 0 ? "+" : "-"}${b}`, ...H(group), F: frameOf(at(J0, along, side * (RL + 0.6), 0), G.mul3(w, side), up), s0: 0, s1: 0.4, r: 0.4, tone: "mid", material: "steel" });
      }
    }
    rect(group, `${tag}.p${i}.spine`, a0, a1, -0.9, 0.9, RL, RL + 0.8, { material: "gunmetal", bevel: 0.15 });
    const blocks = Math.max(1, Math.floor((a1 - a0 - 1.2) / 1.6));
    for (let b = 0; b < blocks; b++) {
      const c = a0 + 0.6 + ((a1 - a0 - 1.2) * (b + 0.5)) / blocks;
      rect(group, `${tag}.p${i}.block${b}`, c - 0.5, c + 0.5, -0.6, 0.6, RL + 0.8, RL + 1.5, { material: b % 2 ? "brass" : "steel" });
    }
    rect(group, `${tag}.p${i}.pad`, s + 2.6, e - 2.6, -0.8, 0.8, -RL - 0.6, -RL, { material: "rubber" });
    const cableSides = i === 0 ? [-1, 1] : i === 1 ? [-1, 1] : [];
    for (const side of cableSides) {
      const y0 = side * 1.8;
      rect(group, `${tag}.p${i}.clip${side > 0 ? "+" : "-"}0`, a0, a0 + 0.6, side > 0 ? RL : -RL - 0.6, side > 0 ? RL + 0.6 : -RL, 1.2, 2.0, { material: "brass" });
      rect(group, `${tag}.p${i}.clip${side > 0 ? "+" : "-"}1`, a1 - 0.6, a1, side > 0 ? RL : -RL - 0.6, side > 0 ? RL + 0.6 : -RL, 1.2, 2.0, { material: "brass" });
      T.tube({ name: `${tag}.p${i}.cable${side > 0 ? "+" : "-"}`, ...H(group), route: [at(J0, a0 + 0.6, y0, 1.55), at(J0, a1 - 0.6, y0, 1.55)], r: 0.35, chunk: 6, hue: side > 0 ? "red" : "blue", material: "rubber" });
    }
  }
  return groups;
}

const FZ = 16.25;
for (const [i, yk] of FINGER_Y.entries()) digit(`f${i}`, [11.5, yk, FZ], [1, 0, 0], [0, 1, 0], wrist, [[-5, 5], [-5, 5], [-5, 5]]);
T.round({ name: "thumb.turret", ...H(thumbBase), F: VERT(-8, -16), s0: 11, s1: 16.5, r: 3, tone: "hi", material: "chrome" });
prismAt("thumb.arm", thumbBase, { x: -9.4, y: -21.6, w: 2.8, d: 2.55, r: 0.3 }, 12.5, 15, { material: "gunmetal" });
digit("thumb", [-8, -21.5, 13.75], [0, -1, 0], [1, 0, 0], thumbBase, [[-5, 5], [-5, 5], [-5, 5]]);

const flexAt = (time, phase) => 0.5 + 0.5 * Math.sin(time * 1.3 + phase);
export const follow = (values) => {
  const out = { ...values };
  const lift = values.lift ?? 0;
  const t = values.time ?? 0;
  out.wrist = 3 * Math.sin(t * 0.7) * lift;
  out.thumb = 10 * Math.sin(t * 0.5) * lift;
  for (let f = 0; f < 5; f++) {
    const tag = f < 4 ? `f${f}` : "thumb";
    const curl = (Math.sin(t * 1.3 + f * 0.9) * 0.5 + 0.5) * lift;
    out[`${tag}.j0`] = -5 + 10 * curl;
    out[`${tag}.j1`] = -5 + 10 * curl;
    out[`${tag}.j2`] = -5 + 10 * curl;
  }
  return out;
};
void flexAt;
export const poses = () => {
  const out = [];
  for (let i = 0; i < 720; i++) out.push(follow({ table: i * 0.5, lift: 1, time: i * 0.11 }));
  for (let i = 0; i < 72; i++) out.push(follow({ table: i * 5, lift: 0 }));
  return out;
};

workshop();
const started = Date.now();
const out = T.build({ statics: R.items, poses, verify: argv.includes("--verify") });
A.settle(R, P);
if (argv.includes("--audit")) {
  A.auditOrExit(R, P);
  TA.orbitOrExit(T, R, P, {
    sweep: { at: (x) => follow({ table: x, lift: 0 }), from: 0, to: 360, frame: 0.3 },
    clips: [{ name: "flex", at: (t) => follow({ table: 30 + 18 * t, lift: 1, time: t }), from: 0, to: 6, frame: 1 / 30 }],
    quick: argv.includes("--quick"),
    only: argv.find((a) => a.startsWith("--only="))?.slice(7).split(",") ?? null,
    log: argv.includes("--verbose") ? (text) => console.log(`  … ${text}`) : null,
  });
}
const layerOf = (name) => R.items.filter((item) => item.layer === name).sort((a, b) => a.key - b.key).map((item) => item.svg);
const BACK = [`<path class="iso-halo" d="${k.haloOf(BENCH, BENCH_TOP - 6, 6, P)}"/>`, ...layerOf("back")];
const backSvg = k.figureSvg({ width: WIDTH, height: HEIGHT, label: "A dense robot hand on a turntable in a workshop: a stress scene for the 3D mode.", body: BACK });
const staticSvg = (list) => `<svg xmlns="http://www.w3.org/2000/svg" class="iso-svg iso-layer" viewBox="0 0 ${WIDTH} ${HEIGHT}" aria-hidden="true">${list.join("")}</svg>`;
const stage = `<div class="iso-stage iso-turn ts-stage" tabindex="0" role="img" aria-label="A dense robot hand turning on a turntable">${backSvg}${out.layerSvg("hand", { width: WIDTH, height: HEIGHT })}${staticSvg(layerOf("front"))}</div>`;
const LIVE = `const DATA = ${JSON.stringify(out.data)};
const follow = ${follow.toString()};
const stage = document.querySelector(".ts-stage");
const still = matchMedia("(prefers-reduced-motion: reduce)");
let touring = !still.matches;
let angle = 0;
let time = 0;
let last = 0;
let frame = 0;
let visible = false;
let slow = 0;
let skip = false;
const controller = TURN.mount(stage, DATA, { follow, onHold: () => (touring = false) });
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  if (touring) {
    angle += 14 * dt;
    time += dt;
  }
  skip = !skip;
  if (!(slow > 14 && skip)) {
    const t0 = performance.now();
    controller.set({ table: angle, lift: 1, time });
    slow = slow * 0.9 + (performance.now() - t0) * 0.1;
    if (slow < 9) skip = false;
  }
  frame = visible && touring ? requestAnimationFrame(tick) : 0;
}
new IntersectionObserver((entries) => {
  visible = entries[entries.length - 1].isIntersecting;
  if (visible && touring && !frame) frame = requestAnimationFrame(tick);
}).observe(stage);
`;
const body = k.plateHtml({ fig: "Fig 3D·S", title: "Stress hand", hint: "Tour only", readout: `${out.report.parts} live parts`, caption: "A stress scene for the 3D mode: a dense hand of hinge chains on a turntable among static workshop clutter.", body: `<style>${out.css}\n.ts-stage{outline:none}</style>${stage}` });
const page = k.pageHtml({ title: "Stress hand", theme, body, script: turnScript() + (out.verify ? `window.__isoVerify = ${JSON.stringify(out.verify)};\n` : "") + LIVE, width: WIDTH + 40 });
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const name = `turning-stress${out.verify ? "-verify" : ""}${theme === "light" ? "-light" : ""}.html`;
  writeFileSync(join(HERE, name), page);
  const routes = T.routes.length;
  console.log(`wrote ${name} · ${(page.length / 1024).toFixed(0)} KB · ${out.report.parts} parts · ${routes} routes · ${T.groups.length} groups · ${out.report.layers.map((l) => `${l.name}: ${l.planar} planes, ${l.dynamic} dynamic`).join(" · ")} · ${out.report.proofs.length} layer proofs · data ${(JSON.stringify(out.data).length / 1024).toFixed(0)} KB · build ${((Date.now() - started) / 1000).toFixed(1)} s`);
}
export { out };
