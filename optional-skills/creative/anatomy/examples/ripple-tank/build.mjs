import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../kit/iso-kit.mjs";
import * as G from "../../kit/lathe.mjs";
import { glCamera } from "../../kit/gl.mjs";
import { tubePieces, tubeSvg } from "../../kit/tube.mjs";
import { kitScript, glScript } from "../../scripts/inline-kit.mjs";
import * as M from "./model.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const theme = process.argv.includes("--light") ? "light" : "dark";
const DEG = Math.PI / 180;

const W = 720;
const H = 730;
const AZIMUTH = 35;
const ELEVATION = 44;

const BASE_H = 16;
const FOOT_H = 7;
const Z_PAPER = 0.3;
const IN = { x: -150, y: -120, w: 300, d: 240, r: 0 };
const OUT = { x: -159, y: -129, w: 318, d: 258, r: 1.5 };
const FRAME = OUT.x + OUT.w - IN.x - IN.w;
const Z = { rail: 193, bed: 202, floor: 206, water: 212, rim: 218, top: 223 };
const LAMP = [0, 0, 612];
const LAMP_R = 1.5;
const SHADOW = (LAMP[2] - Z_PAPER) / (LAMP[2] - Z.bed);
const PAPER_HALF = [Math.ceil((OUT.w / 2) * SHADOW + 3), Math.ceil((OUT.d / 2) * SHADOW + 3)];
const PAPER = { x: -PAPER_HALF[0], y: -PAPER_HALF[1], w: PAPER_HALF[0] * 2, d: PAPER_HALF[1] * 2, r: 1.5 };
const MARGIN = 26;
const BASE = { x: PAPER.x - 58, y: PAPER.y - MARGIN, w: PAPER.w + 58 + MARGIN, d: PAPER.d + MARGIN * 2, r: 12 };
const BEAM_SPREAD = 0.94;
const XS = -70;
const BAR = { x: XS - 4, y: -66, w: 8, d: 132, r: 1 };
const BAR_Z = Z.top + 2;
const BAR_H = 5;
const BEAM = { x: XS - 6, y: OUT.y - 4, w: 12, d: OUT.d + 8, r: 1.2 };
const BEAM_Z = Z.top + 28;
const BEAM_H = 6;
const SPRING_Y = 56;
const BALL_R = 2.6;
const BALL_Z = Z.water + BALL_R - 0.9;
const CONTACT_R = Math.sqrt(BALL_R * BALL_R - (BALL_Z - Z.water) ** 2);
const MOTOR_Z = BAR_Z + BAR_H + 1.6 + 6;
const MOTOR_X = [XS - 10, XS + 8];
const ECCENTRIC_X = XS + 14.6;
const LEG_AT = [IN.w / 2 + 4.5, IN.d / 2 + 4.5];
const LEGS = [
  [LEG_AT[0], LEG_AT[1]],
  [-LEG_AT[0], LEG_AT[1]],
  [LEG_AT[0], -LEG_AT[1]],
  [-LEG_AT[0], -LEG_AT[1]],
];
const LEG_R = 4.2;
const LEG_TOP = Z.rail - 6;
const POST = [172, BASE.y + 12];
const POST_R = 4.5;
const ARM_Z = LAMP[2] + 30;
const POST_TOP = ARM_Z + 16;
const BOX = { x: BASE.x + 8, y: 46, w: 44, d: 80, r: 4 };
const BOX_Z = 2;
const BOX_TOP = BOX_Z + 24;
const KNOB = [BOX.x + 23, BOX.y + 56];
const METER = [BOX.x + 22, BOX.y + 22];
const SWITCH = [BOX.x + 36, BOX.y + 72];
const LED = [BOX.x + 8, BOX.y + 72];
const RANGE = M.RANGE;
const START = { frequency: 16, spacing: 50 };
const BEACH = 9;
const FOAM_PASS = 0.12;

const P = k.fitProjection(
  [...k.boxCorners(BASE, -BASE_H - FOOT_H, 0), [0, 0, ARM_Z + 10], [POST[0] - 6, POST[1] - 6, POST_TOP + 3], [POST[0] + 6, POST[1] + 6, POST_TOP + 3], ...k.boxCorners(BEAM, BEAM_Z, BEAM_Z + BEAM_H)],
  W,
  H,
  { pad: 60, azimuth: AZIMUTH, elevation: ELEVATION },
);
const V = G.viewOf(P);
const at = (point) => k.iso(point, P);
const S = (paths, style) => k.solidSvg(paths, style);
const Ln = (d, style) => k.lineSvg(d, style);
const Dots = (points, style) => k.dotsSvg(points, style);
const slab = (plan, z, h, steps = 3, bevel = 0.5) => k.slabOf(plan, z, h, P, steps, bevel);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
const UP = (o) => G.frameOf(o, [0, 0, 1], [1, 0, 0], [0, 1, 0]);
const glassLines = (paths) => Ln(paths.outline + paths.crease, { tone: "faint" });

function vcoil(cx, cy, z0, z1, r, turns) {
  const count = Math.round(turns * 24);
  const back = [];
  const front = [];
  let run = [];
  let near = null;
  for (let index = 0; index <= count; index++) {
    const share = index / count;
    const angle = share * turns * Math.PI * 2;
    const point = at([cx + r * Math.cos(angle), cy + r * Math.sin(angle), z0 + (z1 - z0) * share]);
    const facing = Math.cos(angle) * V[0] + Math.sin(angle) * V[1] > 0;
    if (near === null) near = facing;
    if (facing !== near) {
      run.push(point);
      (near ? front : back).push(k.pathOf(run));
      run = [point];
      near = facing;
      continue;
    }
    run.push(point);
  }
  if (run.length > 1) (near ? front : back).push(k.pathOf(run));
  return { back: back.join(""), front: front.join("") };
}

function sphere(centre, r, style) {
  const profile = [];
  for (let index = 0; index <= 12; index++) {
    const a = -Math.PI / 2 + (index / 12) * Math.PI;
    profile.push([r * Math.sin(a), r * Math.cos(a), 1]);
  }
  return S(G.lathe(profile, UP(centre), P, { steps: 28 }), style);
}

function screwX(centre, r, length, { tone = "mid", ribs = 14 } = {}) {
  const F = G.frameAlong(centre, [1, 0, 0]);
  return (
    S(G.disc(0, length * 0.4, r * 0.45, F, P), { tone }) +
    S(G.disc(length * 0.4, length, r, F, P), { tone }) +
    Ln(G.ribsOf(F, length * 0.4 + 0.3, length - 0.3, r, ribs, P), { tone: "lo" })
  );
}

function cable(points, { r = 1.1, bend = 7, maxLength = 18 } = {}) {
  return tubePieces(G.fillet(points, bend, 8), r, P, { maxLength, spacing: 1.2 });
}

const back = [];
back.push(`<path class="iso-halo" d="${k.haloOf(BASE, -BASE_H - FOOT_H, BASE_H, P)}"/>`);
const FEET = byDepth([...k.corners(BASE, 28), [BASE.x + BASE.w / 2, BASE.y + 28], [BASE.x + BASE.w / 2, BASE.y + BASE.d - 28], [BASE.x + 28, 0], [BASE.x + BASE.w - 28, 0]]);
for (const [x, y] of FEET) {
  back.push(S(k.cylinder(x, y, 10, -BASE_H - FOOT_H, 2, P, 28), { tone: "lo" }));
  back.push(S(k.cylinder(x, y, 8, -BASE_H - FOOT_H + 2, FOOT_H - 2, P, 28), { tone: "mid" }));
}
back.push(S(k.slabOf(BASE, -BASE_H, BASE_H, P, 8, 1.6), { tone: "mid" }));
back.push(Ln(k.planOutline(k.insetPlan(BASE, 6), 0, P), { tone: "lo" }));
const BASE_SCREWS = k.corners(BASE, 14);
back.push(Ln(BASE_SCREWS.map(([x, y]) => k.ring(x, y, 2.8, 0, P, 16)).join(""), { tone: "lo" }));
back.push(Dots(BASE_SCREWS.map(([x, y]) => at([x, y, 0])), { size: 0.55 }));
const SIDE = k.sideTicks(BASE.y + 26, BASE.y + BASE.d - 26, 10, 5, BASE.x + BASE.w, -2.6, [2.6, 5.2], P, "y");
back.push(Ln(SIDE.minor, { tone: "lo" }), Ln(SIDE.major, { tone: "mid" }));
const STRIP = { x: PAPER.x + 10, y: BASE.y + BASE.d - 18, w: PAPER.w - 20, d: 8, r: 1 };
back.push(S(k.slabOf(STRIP, 0, 0.8, P, 3), { tone: "lo", crease: "none" }));
const TICKS = k.topTicks(STRIP.x + 5, STRIP.x + STRIP.w - 5, 5, 5, STRIP.y + 0.9, 0.8, [2.2, 4.4], P);
back.push(Ln(TICKS.minor, { tone: "lo" }), Ln(TICKS.major, { tone: "mid" }));
back.push(Dots([at([STRIP.x + 2.4, STRIP.y + 4, 0.8]), at([STRIP.x + STRIP.w - 2.4, STRIP.y + 4, 0.8])], { size: 0.45 }));
const LABEL = { x: BASE.x + BASE.w - 20, y: 40, w: 13, d: 92, r: 1.5 };
back.push(S(k.slabOf(LABEL, 0, 1, P, 3), { tone: "mid" }));
back.push(Ln(k.planOutline(k.insetPlan(LABEL, 1.8), 1, P), { tone: "faint" }));
back.push(k.faceTextSvg(k.topMatrix([LABEL.x + 8.4, LABEL.y + LABEL.d - 7], 1, P, "y"), "RIPPLE TANK", { size: 4.6, tone: "lo" }));
back.push(Ln([0, 1].map((row) => k.onTop([[LABEL.x + 3.6 + row * 2.6, LABEL.y + 7], [LABEL.x + 3.6 + row * 2.6, LABEL.y + (row ? 21 : 28)]], 1, P)).join(""), { tone: "lo", free: true }));
back.push(Dots([at([LABEL.x + LABEL.w / 2, LABEL.y + 2.8, 1]), at([LABEL.x + LABEL.w / 2, LABEL.y + LABEL.d - 2.8, 1])], { size: 0.45 }));
back.push(S(k.slabOf(PAPER, 0, Z_PAPER, P, 3), { tone: "lo", flat: true, crease: "none", className: "rt-sheet" }));
for (const [sx, sy] of [
  [1, 1],
  [-1, 1],
  [1, -1],
  [-1, -1],
]) {
  const cx = (sx > 0 ? PAPER.x + PAPER.w : PAPER.x) - sx * 2;
  const cy = (sy > 0 ? PAPER.y + PAPER.d : PAPER.y) - sy * 2;
  const along = [sx / Math.SQRT2, -sy / Math.SQRT2];
  const across = [sx / Math.SQRT2, sy / Math.SQRT2];
  const ring = [
    [-10, -3.6],
    [10, -3.6],
    [10, 3.6],
    [-10, 3.6],
  ].map(([a, b]) => [cx + along[0] * a + across[0] * b, cy + along[1] * a + across[1] * b]);
  back.push(S(k.extrude(ring, Z_PAPER, 0.2, P), { tone: "faint", flat: true, crease: "none", className: "rt-tape" }));
}

const PATTERN_GEO = { xs: XS, inside: [IN.x, IN.y, IN.x + IN.w, IN.y + IN.d], lamp: LAMP, zWater: Z.water, zPaper: Z_PAPER };
const optics = { magnification: M.magnificationOf(LAMP[2], Z_PAPER, Z.water), caustic: M.causticOf(LAMP[2], Z_PAPER, Z.water, Z.floor, Z.bed) };
const START_PHASE = 0.9;
const startPattern = M.patternOf(M.stateOf(START.frequency, START.spacing, optics), PATTERN_GEO, at, START_PHASE);
const LIFT = (LAMP[2] - Z_PAPER) / (LAMP[2] - Z.water);
const SHADOW_IN = { x: IN.x * LIFT, y: IN.y * LIFT, w: IN.w * LIFT, d: IN.d * LIFT, r: 0 };
back.push(
  `<g class="rt-fallback">${Ln(k.planOutline(SHADOW_IN, Z_PAPER, P), { tone: "faint" })}<path data-part="crests" class="iso-line" data-tone="faint" d="${startPattern.crests}"/><path data-part="nodal" class="iso-line" data-tone="lo" d="${startPattern.nodal}"/></g>`,
);

const mid = [];
const ARM_U = G.unit3([LAMP[0] - POST[0], LAMP[1] - POST[1], 0]);
const ARM_N = [-ARM_U[1], ARM_U[0], 0];
const ARM_LENGTH = Math.hypot(LAMP[0] - POST[0], LAMP[1] - POST[1]);
const FACING = Math.sign(-(ARM_N[0] * V[0] + ARM_N[1] * V[1])) || 1;
const onArm = (a, b, z = 0) => [POST[0] + ARM_U[0] * a + ARM_N[0] * b, POST[1] + ARM_U[1] * a + ARM_N[1] * b, z];
const armRing = (a0, a1, b0, b1) =>
  [
    [a0, b0],
    [a1, b0],
    [a1, b1],
    [a0, b1],
  ].map(([a, b]) => onArm(a, b).slice(0, 2));
const cableFoot = onArm(-16, 0, 1.2);
const LAMP_CABLE_LOW = cable(
  [
    onArm(-6, 0, ARM_Z + 2),
    onArm(-11.5, 0, ARM_Z - 1),
    onArm(-7.6, 0, ARM_Z - 12),
    onArm(-7.6, 0, 12),
    onArm(-14, 0, 5),
    cableFoot,
    [cableFoot[0] + 1, BASE.y + 4, 1.2],
    [cableFoot[0] + 1.4, BASE.y - 2.6, -1.4],
    [cableFoot[0] + 1.6, BASE.y - 3, -12],
  ],
  { r: 1.1, bend: 3, maxLength: 8 },
);
for (const piece of LAMP_CABLE_LOW) {
  if (piece.mid[1] < BASE.y - 0.2) back.splice(1, 0, tubeSvg(piece, { tone: "mid" }));
  else mid.push(tubeSvg(piece, { tone: "mid" }));
}
const postSeg = (z0, z1) => S(k.cylinder(POST[0], POST[1], POST_R, z0, z1 - z0, P, 28), { tone: "mid", crease: "none" });
mid.push(S(k.cylinder(POST[0], POST[1], 9.6, 0, 3.6, P, 40, 0.8), { tone: "mid" }));
mid.push(Dots(G.dotsOf(UP([POST[0], POST[1], 0]), 3.6, 7.6, 6, P, { all: true }), { size: 0.45 }));
mid.push(S(k.cylinder(POST[0], POST[1], 6.4, 3.6, 4.4, P, 28), { tone: "mid" }));
mid.push(postSeg(8, ARM_Z - 6));
mid.push(Ln([140, 280, 420].map((z) => k.sideArc(POST[0], POST[1], POST_R, z, P, 16)).join(""), { tone: "faint" }));

mid.push(S(k.extrude(armRing(-8, 8, -8, 8), ARM_Z - 6, 16, P, { bevel: 0.6 }), { tone: "mid" }));
mid.push(Dots([onArm(-5, -5, ARM_Z + 10), onArm(5, 5, ARM_Z + 10)].map(at), { size: 0.45 }));
{
  const F = G.frameAlong(onArm(0, 8 * -FACING, ARM_Z + 2), G.mul3(ARM_N, -FACING));
  mid.push(S(G.disc(0, 2.8, 1.4, F, P), { tone: "mid" }) + S(G.disc(2.8, 7, 3, F, P), { tone: "mid" }) + Ln(G.ribsOf(F, 3.1, 6.7, 3, 16, P), { tone: "lo" }));
}
mid.push(postSeg(ARM_Z + 10, POST_TOP));
mid.push(S(k.cylinder(POST[0], POST[1], 5.2, POST_TOP, 2.4, P, 28, 0.4), { tone: "mid" }));
mid.push(S(k.cylinder(0, 0, 2.6, LAMP[2] + 20, ARM_Z - LAMP[2] - 20, P, 20), { tone: "mid", crease: "none" }));
mid.push(S(k.cylinder(0, 0, 5, LAMP[2] + 18.4, 2.2, P, 24), { tone: "mid" }));
const SHADE = [
  [LAMP[2], 13.4],
  [LAMP[2] + 12.6, 13.4],
  [LAMP[2] + 13.6, 12.6, 1],
  [LAMP[2] + 16.4, 9.4, 1],
  [LAMP[2] + 18.4, 5],
];
mid.push(
  S(G.solidOf(SHADE, UP([0, 0, 0]), P, { bevel: 0.5 }), { tone: "hi", crease: "lo" }) +
    Ln(G.arcOf(UP([0, 0, 0]), LAMP[2] + 3.4, 13.4, P), { tone: "faint" }) +
    k.fadedSvg(G.ribsOf(UP([0, 0, 0]), LAMP[2] + 5.6, LAMP[2] + 10.6, 13.4, 36, P, { fade: [0.12, 0.5] }), { tone: "lo" }) +
    Dots(G.dotsOf(UP([0, 0, 0]), LAMP[2] + 1.4, 13.4, 24, P, { fade: [0.04, 0.42] }), { size: 0.4, tone: "mid" }),
);
mid.push(S(k.extrude(armRing(8, ARM_LENGTH + 7, -3.5, 3.5), ARM_Z, 6, P, { bevel: 0.5 }), { tone: "mid" }));
mid.push(Dots([20, ARM_LENGTH / 2, ARM_LENGTH - 4].map((a) => at(onArm(a, 0, ARM_Z + 6))), { size: 0.45 }));
const LAMP_CABLE_HIGH = cable(
  [
    onArm(ARM_LENGTH - 3.6, -2.2 * FACING, LAMP[2] + 20.6),
    onArm(ARM_LENGTH - 6, -5.6 * FACING, ARM_Z + 2),
    onArm(ARM_LENGTH - 16, -1.6 * FACING, ARM_Z + 7.1),
    onArm(22, -1.6 * FACING, ARM_Z + 7.1),
    onArm(15, -5.6 * FACING, ARM_Z + 3),
    onArm(8.3, -5.6 * FACING, ARM_Z + 3),
  ],
  { r: 1.1, bend: 4, maxLength: 24 },
);
for (const piece of LAMP_CABLE_HIGH) mid.push(tubeSvg(piece, { tone: "mid" }));
mid.push(S(G.disc(0, 1.6, 2, G.frameAlong(onArm(ARM_LENGTH - 3.6, -2.2 * FACING, LAMP[2] + 20.6), [0, 0, -1]), P), { tone: "mid" }));

const PLUG = [BOX.x + BOX.w, BOX.y + 40, BOX_Z + 9];
const LEFT_LEG = [-LEG_AT[0], LEG_AT[1]];
const RIM_Y = IN.y + IN.d + 2;
const MOTOR_CABLE = cable(
  [
    [MOTOR_X[0] - 1.6, 1.6, MOTOR_Z + 2],
    [MOTOR_X[0] - 4.4, 2, MOTOR_Z + 4],
    [XS - 7.6, 6, BEAM_Z + 2],
    [XS - 7.6, IN.y + IN.d - 4, BEAM_Z + 2],
    [XS - 7.4, IN.y + IN.d + 3.4, BEAM_Z - 4],
    [XS - 7.4, IN.y + IN.d + 3.4, Z.top + 3],
    [XS - 12, RIM_Y, Z.top + 1.1],
    [OUT.x + 6, RIM_Y, Z.top + 1.1],
    [OUT.x - 2, OUT.y + OUT.d + 2, Z.top - 3],
    [OUT.x - 2, OUT.y + OUT.d + 2, Z.rail - 8],
    [LEFT_LEG[0] - 5.4, LEFT_LEG[1] + 1.4, Z.rail - 20],
    [LEFT_LEG[0] - 5.4, LEFT_LEG[1] + 1.4, 16],
    [LEFT_LEG[0] - 14, LEFT_LEG[1] - 4, 1.2],
    [PLUG[0] + 22, PLUG[1], 1.2],
    [PLUG[0] + 8, PLUG[1], 1.2],
    [PLUG[0] + 4, PLUG[1], PLUG[2]],
    [PLUG[0], PLUG[1], PLUG[2]],
  ],
  { r: 1.1, bend: 6, maxLength: 12 },
);
const motorCableBridge = [];
const motorCableRim = [];
for (const piece of MOTOR_CABLE) {
  if (piece.mid[2] > Z.top - 1 && piece.mid[1] < IN.y + IN.d - 10) motorCableBridge.push(tubeSvg(piece, { tone: "mid" }));
  else if (piece.mid[2] > Z.top - 1) motorCableRim.push(tubeSvg(piece, { tone: "mid" }));
  else mid.push(tubeSvg(piece, { tone: "mid" }));
}
for (const z of [Z.rail - 44, 118, 44]) {
  mid.push(S(slab({ x: LEFT_LEG[0] - 5.4, y: LEFT_LEG[1] - 0.6, w: 5.4, d: 2.4, r: 0.4 }, z + 0.3, 2.4, 2, 0.2), { tone: "mid" }));
  mid.push(S(k.cylinder(LEFT_LEG[0] - 5.4, LEFT_LEG[1] + 1.4, 2, z, 3, P, 14), { tone: "mid" }));
}

const box = [];
for (const [x, y] of byDepth(k.corners(BOX, 7))) box.push(S(k.cylinder(x, y, 3, 0, BOX_Z, P, 16), { tone: "lo" }));
box.push(S(slab(BOX, BOX_Z, BOX_TOP - BOX_Z, 6, 0.8), { tone: "mid" }));
box.push(Ln(k.planOutline(k.insetPlan(BOX, 3), BOX_TOP, P, 4), { tone: "faint" }));
box.push(Dots(k.corners(BOX, 5.4).map(([x, y]) => at([x, y, BOX_TOP])), { size: 0.45 }));
const slotOf = (x, y0, y1, z, half) => k.pathOf([...Array.from({ length: 9 }, (_, i) => [y1 - half + half * Math.sin((i / 8) * Math.PI), z + half * Math.cos((i / 8) * Math.PI)]), ...Array.from({ length: 9 }, (_, i) => [y0 + half - half * Math.sin((i / 8) * Math.PI), z - half * Math.cos((i / 8) * Math.PI)])].map(([y, zz]) => at([x, y, zz])), true);
box.push(Ln([BOX_Z + 7, BOX_Z + 12, BOX_Z + 17].map((z) => slotOf(BOX.x + BOX.w, BOX.y + 8, BOX.y + 24, z, 0.9)).join(""), { tone: "faint" }));
box.push(S(G.disc(0, 2.2, 2.6, G.frameAlong([PLUG[0], PLUG[1], PLUG[2]], [1, 0, 0]), P), { tone: "mid" }));
const FA = (f) => (-135 + (270 * (f - RANGE.frequency[0])) / (RANGE.frequency[1] - RANGE.frequency[0])) * DEG;
box.push(S(k.cylinder(KNOB[0], KNOB[1], 10, BOX_TOP, 1, P, 40), { tone: "mid" }));
const KNOB_TICKS = k.radialTicks(KNOB[0], KNOB[1], 9.6, 10, 5, BOX_TOP + 1, [1.2, 2.4], P, -135 * DEG, 270 * DEG);
box.push(Ln(KNOB_TICKS.minor, { tone: "lo" }), Ln(KNOB_TICKS.major, { tone: "mid" }));
box.push(S(k.cylinder(KNOB[0], KNOB[1], 6.6, BOX_TOP + 1, 7, P, 36), { tone: "hi", crease: "lo" }));
box.push(Ln(k.knurl(KNOB[0], KNOB[1], 6.6, BOX_TOP + 1, BOX_TOP + 8, 40, P), { tone: "lo" }));
box.push(Ln(k.ring(KNOB[0], KNOB[1], 5, BOX_TOP + 8, P, 32), { tone: "faint" }));
const knobAt = (f) => {
  const a = FA(f);
  return k.lineOnTop([KNOB[0] + Math.sin(a) * 1.4, KNOB[1] - Math.cos(a) * 1.4], [KNOB[0] + Math.sin(a) * 5.6, KNOB[1] - Math.cos(a) * 5.6], BOX_TOP + 8, P);
};
box.push(`<path data-part="pointer" class="iso-line" data-tone="hi" d="${knobAt(START.frequency)}" style="stroke-width:1"/>`);
box.push(S(k.cylinder(METER[0], METER[1], 13, BOX_TOP, 3.2, P, 56, 0.6), { tone: "hi", crease: "lo" }));
box.push(Ln(k.ring(METER[0], METER[1], 11.2, BOX_TOP + 3.2, P, 56), { tone: "lo" }));
const METER_TICKS = k.radialTicks(METER[0], METER[1], 10.4, 20, 5, BOX_TOP + 3.2, [1.1, 2.4], P, -135 * DEG, 270 * DEG);
box.push(Ln(METER_TICKS.minor, { tone: "lo" }), Ln(METER_TICKS.major, { tone: "mid" }));
const needleAt = (f) => {
  const a = FA(f);
  return k.lineOnTop([METER[0] - Math.sin(a) * 2.2, METER[1] + Math.cos(a) * 2.2], [METER[0] + Math.sin(a) * 9.6, METER[1] - Math.cos(a) * 9.6], BOX_TOP + 3.4, P);
};
box.push(`<path data-part="needle" class="iso-line" data-tone="hi" d="${needleAt(START.frequency)}" style="stroke-width:1"/>`);
box.push(Ln(k.ring(METER[0], METER[1], 1.4, BOX_TOP + 3.4, P, 14), { tone: "mid" }));
box.push(S(k.cylinder(SWITCH[0], SWITCH[1], 3, BOX_TOP, 2, P, 6), { tone: "mid" }));
box.push(Ln(k.segment([SWITCH[0], SWITCH[1], BOX_TOP + 2], [SWITCH[0] + 2.6, SWITCH[1] + 1.4, BOX_TOP + 8], P), { tone: "hi", className: "rt-lever" }));
box.push(Dots([at([SWITCH[0] + 2.6, SWITCH[1] + 1.4, BOX_TOP + 8])], { size: 1, tone: "hi" }));
box.push(S(k.cylinder(LED[0], LED[1], 2.4, BOX_TOP, 0.8, P, 16), { tone: "mid" }));
box.push(sphere([LED[0], LED[1], BOX_TOP + 0.8], 1.4, { tone: "hi" }));
mid.push(box.join(""));

for (const [x, y] of byDepth(LEGS)) {
  const parts = [];
  parts.push(S(k.cylinder(x, y, 7, Z_PAPER, 5.6, P, 32), { tone: "mid" }));
  parts.push(Ln(k.knurl(x, y, 7, Z_PAPER, Z_PAPER + 5.6, 32, P), { tone: "lo" }));
  parts.push(S(k.cylinder(x, y, 2.3, 5.9, 12, P, 20), { tone: "hi", crease: "none" }));
  parts.push(Ln([7.6, 9, 10.4, 11.8, 13.2, 14.6, 16, 17.4].map((z) => k.sideArc(x, y, 2.3, z, P, 12)).join(""), { tone: "lo" }));
  parts.push(S(k.cylinder(x, y, 4.8, 17.9, 3.2, P, 6), { tone: "mid" }));
  parts.push(S(k.cylinder(x, y, LEG_R, 21.1, LEG_TOP - 21.1, P, 24), { tone: "mid", crease: "none" }));
  parts.push(Ln([30, 104].map((z) => k.sideArc(x, y, LEG_R, z, P, 14)).join(""), { tone: "faint" }));
  parts.push(S(k.cylinder(x, y, 5, 100, 8, P, 24, 0.4), { tone: "mid" }));
  parts.push(Ln(k.knurl(x, y, 5, 100.8, 107.2, 24, P), { tone: "lo" }));
  const bracket = { x: x - 5, y: y - 5, w: 10, d: 10, r: 1 };
  parts.push(S(slab(bracket, LEG_TOP, Z.rail - LEG_TOP, 2, 0.4), { tone: "mid" }));
  parts.push(Dots([at([bracket.x + bracket.w, y, LEG_TOP + 3]), at([x, bracket.y + bracket.d, LEG_TOP + 3])], { size: 0.4 }));
  mid.push(parts.join(""));
}

const railMinusX = { x: OUT.x, y: OUT.y, w: FRAME, d: OUT.d, r: 1 };
const railPlusX = { x: IN.x + IN.w, y: OUT.y, w: FRAME, d: OUT.d, r: 1 };
const railMinusY = { x: IN.x, y: OUT.y, w: IN.w, d: FRAME, r: 0.3 };
const railPlusY = { x: IN.x, y: IN.y + IN.d, w: IN.w, d: FRAME, r: 0.3 };
mid.push(S(slab(railMinusX, Z.rail, Z.bed - Z.rail, 2, 0.4), { tone: "mid" }));
mid.push(S(slab(railMinusY, Z.rail, Z.bed - Z.rail, 2, 0.4), { tone: "mid" }));
const POST_PLAN = (x, y) => ({ x, y, w: FRAME, d: FRAME, r: 0.8 });
const cornerPost = (plan) => S(slab(plan, Z.bed, Z.top - Z.bed, 2, 0.4), { tone: "mid" }) + Dots([at([plan.x + plan.w / 2, plan.y + plan.d / 2, Z.top])], { size: 0.45 });
mid.push(cornerPost(POST_PLAN(OUT.x, OUT.y)));
mid.push(cornerPost(POST_PLAN(IN.x + IN.w, OUT.y)));
mid.push(cornerPost(POST_PLAN(OUT.x, IN.y + IN.d)));
mid.push(glassLines(slab({ x: IN.x - 4, y: IN.y, w: 4, d: IN.d, r: 0 }, Z.floor, Z.rim - Z.floor, 1, 0)));
mid.push(glassLines(slab({ x: IN.x, y: IN.y - 4, w: IN.w, d: 4, r: 0 }, Z.floor, Z.rim - Z.floor, 1, 0)));
{
  const FOOT = k.insetPlan(IN, BEACH);
  const top = (x, y) => at([x, y, Z.water - 0.4]);
  const foot = (x, y) => at([x, y, Z.floor]);
  const [x0, y0, x1, y1] = [IN.x, IN.y, IN.x + IN.w, IN.y + IN.d];
  const [u0, v0, u1, v1] = [FOOT.x, FOOT.y, FOOT.x + FOOT.w, FOOT.y + FOOT.d];
  const faces = [
    [[top(x0, y0), top(x1, y0), foot(u1, v0), foot(u0, v0)], 3],
    [[top(x0, y0), top(x0, y1), foot(u0, v1), foot(u0, v0)], 1],
    [[top(x1, y0), top(x1, y1), foot(u1, v1), foot(u1, v0)], 2],
    [[top(x0, y1), top(x1, y1), foot(u1, v1), foot(u0, v1)], 0],
  ];
  const shell = faces.map(([ring, shade]) => `<path class="iso-fill" d="${k.pathOf(ring, true)}"/><path class="iso-shade" data-shade="${shade}" d="${k.pathOf(ring, true)}"/>`).join("");
  const seams = k.pathOf([foot(u0, v0), foot(u1, v0), foot(u1, v1), foot(u0, v1)], true) + [[x0, y0, u0, v0], [x1, y0, u1, v0], [x1, y1, u1, v1], [x0, y1, u0, v1]].map(([a, b, c, d]) => k.pathOf([top(a, b), foot(c, d)])).join("");
  mid.push(`<g class="iso-solid rt-beach">${shell}</g>` + Ln(seams, { tone: "faint" }));
}
mid.push(S(k.slabOf(IN, Z.water - 0.05, 0.05, P, 1), { tone: "faint", flat: true, crease: "none", className: "rt-water" }));

const front = [];
const CAP = 5.6;
const rimMinusX = { x: IN.x - 4 - (CAP - 4) / 2, y: IN.y, w: CAP, d: IN.d, r: 0.6 };
const rimPlusX = { x: IN.x + IN.w - (CAP - 4) / 2, y: IN.y, w: CAP, d: IN.d, r: 0.6 };
const rimMinusY = { x: IN.x, y: IN.y - 4 - (CAP - 4) / 2, w: IN.w, d: CAP, r: 0.6 };
const rimPlusY = { x: IN.x, y: IN.y + IN.d - (CAP - 4) / 2, w: IN.w, d: CAP, r: 0.6 };
const rimSvg = (plan) => S(slab(plan, Z.rim, Z.top - Z.rim, 2, 0.3), { tone: "mid" });
front.push(rimSvg(rimMinusX), rimSvg(rimMinusY));
front.push(...motorCableBridge);

const pillar = (y0) => {
  const plan = { x: XS - 5, y: y0, w: 10, d: 6, r: 1 };
  return S(slab(plan, Z.top, BEAM_Z - Z.top, 3, 0.5), { tone: "mid" }) + screwX([XS + 5, y0 + 3, Z.top + 9], 2.4, 5.6);
};
const spring = (y) => {
  const coil = vcoil(XS, y, BAR_Z + BAR_H + 0.6, BEAM_Z - 0.6, 2.3, 8);
  return Ln(coil.back, { tone: "lo" }) + S(k.cylinder(XS, y, 1.4, BAR_Z + BAR_H, 0.8, P, 16), { tone: "mid" }) + Ln(coil.front, { tone: "hi" });
};
front.push(pillar(rimMinusY.y - 0.2));
front.push(spring(-SPRING_Y));

const BAR_TICKS = k.topTicks(-48, 48, 2, 5, BAR.x + 0.8, BAR_Z + BAR_H, [1.2, 2.4], P, "y");
front.push(
  `<g data-part="bar">` +
    S(slab(BAR, BAR_Z, BAR_H, 2, 0.4), { tone: "hi", crease: "lo" }) +
    Ln(BAR_TICKS.minor, { tone: "lo" }) +
    Ln(BAR_TICKS.major, { tone: "mid" }) +
    Ln(k.lineOnTop([BAR.x + 0.8, 0], [BAR.x + 3.8, 0], BAR_Z + BAR_H, P), { tone: "hi" }) +
    Dots([at([XS, BAR.y + 2.4, BAR_Z + BAR_H]), at([XS, BAR.y + BAR.d - 2.4, BAR_Z + BAR_H])], { size: 0.4 }) +
    `</g>`,
);

function dipper() {
  const parts = [];
  parts.push(sphere([XS, 0, BALL_Z], BALL_R, { tone: "hi", lit: true }));
  parts.push(S(k.cylinder(XS, 0, 0.8, BALL_Z + BALL_R * 0.7, BAR_Z - 1.6 - BALL_Z - BALL_R * 0.7, P, 12), { tone: "hi", lit: true, crease: "none" }));
  const clamp = { x: XS - 5.6, y: -3, w: 11.2, d: 6, r: 0.8 };
  parts.push(S(slab(clamp, BAR_Z - 1.6, BAR_H + 3.6, 2, 0.4), { tone: "mid" }));
  parts.push(Ln(k.lineOnTop([clamp.x + 0.8, 0], [clamp.x + 4.2, 0], BAR_Z + BAR_H + 2, P), { tone: "lit" }));
  parts.push(screwX([clamp.x + clamp.w, 0, BAR_Z + 2.4], 1.9, 4.6));
  return parts.join("");
}
const restFar = -START.spacing / 2;
const restNear = START.spacing / 2;
front.push(`<g data-part="dipper-far" transform="${k.translateAlong(P, [0, restFar, 0])}">${dipper()}</g>`);

const MF = G.frameAlong([MOTOR_X[0], 0, MOTOR_Z], [1, 0, 0]);
const motorLength = MOTOR_X[1] - MOTOR_X[0];
const saddle = { x: XS - 4, y: -4.6, w: 8, d: 9.2, r: 0.8 };
front.push(
  `<g data-part="motor">` +
    S(slab(saddle, BAR_Z + BAR_H, 3, 2, 0.3), { tone: "mid" }) +
    S(G.lathe([[-1.6, 0], [-1.6, 3.2], [-0.4, 4.6, 1], [0, 5.2], [0, 0]], MF, P, { steps: 32 }), { tone: "mid" }) +
    S(G.disc(0, motorLength, 6, MF, P, { bevel: 0.4 }), { tone: "mid" }) +
    k.fadedSvg(G.ribsOf(MF, 3, motorLength - 4, 6, 28, P, { fade: [0.08, 0.45] }), { tone: "lo" }) +
    Ln(G.arcOf(MF, 2, 6, P) + G.arcOf(MF, motorLength - 3, 6, P), { tone: "faint" }) +
    S(G.lathe([[motorLength, 0], [motorLength, 4.6], [motorLength + 1.6, 4.2, 1], [motorLength + 2.2, 2.4], [motorLength + 2.2, 0]], MF, P, { steps: 32 }), { tone: "mid" }) +
    S(G.disc(motorLength + 2.2, ECCENTRIC_X - MOTOR_X[0] - 2.4, 0.9, MF, P), { tone: "hi", crease: "none" }) +
    S(G.disc(ECCENTRIC_X - MOTOR_X[0] - 2.4, ECCENTRIC_X - MOTOR_X[0], 5.2, MF, P, { bevel: 0.4 }), { tone: "hi", crease: "lo" }) +
    Dots([at([ECCENTRIC_X, 0, MOTOR_Z])], { size: 0.5, tone: "hi" }) +
    `<path data-part="eccentric" class="iso-line" data-tone="hi" d=""/>` +
    S(slab({ x: MOTOR_X[0] + 1.6, y: -1.4, w: 2.4, d: 2.8, r: 0.4 }, MOTOR_Z + 5.6, 1.6, 2), { tone: "mid" }) +
    S(slab({ x: MOTOR_X[0] + 1.6, y: 2.4, w: 2.4, d: 2.8, r: 0.4 }, MOTOR_Z + 5.2, 1.6, 2), { tone: "mid" }) +
    `</g>`,
);
front.push(`<g data-part="dipper-near" transform="${k.translateAlong(P, [0, restNear, 0])}">${dipper()}</g>`);
front.push(spring(SPRING_Y));

front.push(S(slab(railPlusY, Z.rail, Z.bed - Z.rail, 2, 0.4), { tone: "mid" }));
front.push(S(slab(railPlusX, Z.rail, Z.bed - Z.rail, 2, 0.4), { tone: "mid" }) + Dots([-80, 0, 80].map((y) => at([railPlusX.x + FRAME, y, Z.rail + 4.5])), { size: 0.42 }));
front.push(glassLines(slab({ x: OUT.x + 2.5, y: OUT.y + 2.5, w: OUT.w - 5, d: OUT.d - 5, r: 0 }, Z.bed, Z.floor - Z.bed, 1, 0)));
front.push(Ln(k.segment([IN.x, IN.y, Z.water], [IN.x + IN.w, IN.y, Z.water], P) + k.segment([IN.x, IN.y, Z.water], [IN.x, IN.y + IN.d, Z.water], P), { tone: "faint" }));
front.push(Ln(k.segment([IN.x + IN.w, IN.y, Z.water], [IN.x + IN.w, IN.y + IN.d, Z.water], P) + k.segment([IN.x, IN.y + IN.d, Z.water], [IN.x + IN.w, IN.y + IN.d, Z.water], P), { tone: "lo" }));
front.push(glassLines(slab({ x: IN.x + IN.w, y: IN.y, w: 4, d: IN.d, r: 0 }, Z.floor, Z.rim - Z.floor, 1, 0)));
front.push(glassLines(slab({ x: IN.x, y: IN.y + IN.d, w: IN.w, d: 4, r: 0 }, Z.floor, Z.rim - Z.floor, 1, 0)));
front.push(rimSvg(rimPlusY), rimSvg(rimPlusX));
front.push(...motorCableRim);
front.push(cornerPost(POST_PLAN(IN.x + IN.w, IN.y + IN.d)));
front.push(pillar(rimPlusY.y - 0.2));
front.push(
  S(slab(BEAM, BEAM_Z, BEAM_H, 3, 0.6), { tone: "mid" }) +
    Dots([BEAM.y + 5, -SPRING_Y, SPRING_Y, BEAM.y + BEAM.d - 5].map((y) => at([XS, y, BEAM_Z + BEAM_H])), { size: 0.45 }) +
    Ln(k.planOutline(k.insetPlan(BEAM, 2), BEAM_Z + BEAM_H, P, 2), { tone: "faint" }),
);

const startState = M.stateOf(START.frequency, START.spacing, optics);
const drop = Z.water - Z_PAPER;

const DATA = {
  W,
  H,
  P,
  camera: glCamera(P),
  range: RANGE,
  start: START,
  optics,
  beat: M.BEAT,
  slope: M.SLOPE,
  histDt: 0.05,
  phase: START_PHASE,
  pattern: PATTERN_GEO,
  reach: M.REACH,
  xs: XS,
  inside: [IN.x, IN.y, IN.x + IN.w, IN.y + IN.d],
  outside: [OUT.x, OUT.y, OUT.x + OUT.w, OUT.y + OUT.d],
  paper: [PAPER.x, PAPER.y, PAPER.x + PAPER.w, PAPER.y + PAPER.d],
  base: [BASE.x, BASE.y, BASE.x + BASE.w, BASE.y + BASE.d],
  round: [PAPER.r, BASE.r],
  z: [Z_PAPER, Z.rail, Z.water, Z.bed],
  top: Z.top,
  bed: Z.bed,
  floor: Z.floor,
  rim: Z.rim,
  lamp: LAMP,
  lampR: LAMP_R,
  blur: (0.5 * LAMP_R * drop) / (LAMP[2] - Z_PAPER),
  spread: BEAM_SPREAD,
  foam: [(Z.floor + Z.water) / 2, BEACH / 2, FOAM_PASS],
  beach: BEACH,
  legs: [LEG_AT[0], LEG_AT[1], LEG_R, Z.rail],
  post: [POST[0], POST[1], POST_R, POST_TOP],
  bar: [BAR.x - 1.6, BAR.y, BAR.x + BAR.w + 1.6, BAR.y + BAR.d],
  beam: [BEAM.x, BEAM.y, BEAM.x + BEAM.w, BEAM.y + BEAM.d],
  motor: [MOTOR_X[0] - 1.6, -6, ECCENTRIC_X, 6],
  high: [BAR_Z + BAR_H / 2, BEAM_Z + BEAM_H / 2, MOTOR_Z],
  box: [BOX.x, BOX.y, BOX.x + BOX.w, BOX.y + BOX.d],
  boxTop: BOX_TOP,
  near: [IN.x + IN.w, IN.y + IN.d],
  contact: CONTACT_R + 0.35,
  eccentric: [ECCENTRIC_X, MOTOR_Z],
  knob: KNOB,
  meter: METER,
};

const LABEL_TEXT =
  "A ripple tank on a base plate: a shallow glass-bottomed tank of water standing on four tall legs with knurled levelling screws, high over a paper screen. A strobe lamp on an arm from a post behind the tank shines down through the water. A bridge across the tank carries a vibrating bar on two coil springs, shaken by a small motor with an eccentric; two point dippers on the bar touch the water. A control box beside the paper sets the frequency. The two sets of circular ripples interfere; the water's surface sheen moves with them, and the lamp projects the pattern of bright crest lines and grey nodal lines onto the paper far below.";

const backSvg = k.figureSvg({ width: W, height: H, label: LABEL_TEXT, body: back, className: "rt-back" });
const layerSvg = (name, body) => `<svg xmlns="http://www.w3.org/2000/svg" class="iso-svg rt-layer ${name}" viewBox="0 0 ${W} ${H}" aria-hidden="true">${body.join("")}</svg>`;
const stage = `<div class="rt-stage" tabindex="0" role="group" aria-roledescription="ripple tank control" aria-label="Ripple tank. Move the pointer across the tank to set the frequency and up or down to set the dipper spacing; or use the left and right arrow keys for frequency and up and down for spacing, with Shift for bigger steps."><span class="rt-said" aria-live="polite"></span>${backSvg}<canvas class="rt-layer rt-gl rt-paper-gl" aria-hidden="true"></canvas>${layerSvg("rt-mid", mid)}<canvas class="rt-layer rt-gl rt-water-gl" aria-hidden="true"></canvas>${layerSvg("rt-front", front)}</div>`;

const at16 = M.stateOf(16, 50, optics);
const at28 = M.stateOf(28, 50, optics);
const cm = (mm) => (mm / 10).toFixed(1);
const caption = `A ripple tank seen by strobe light. Two dippers on one vibrating bar touch 6 mm of water in phase; the lamp flashes ${M.BEAT} times a second slower than they dip, so each wave seems to crawl outward at ${M.BEAT} Hz, which is honest slow motion. Ripples this small are held flat by surface tension as well as gravity, ω² = (gk + σk³/ρ) tanh kh, so at 16 Hz the wavelength is ${(at16.lambdaMm / 10).toFixed(2)} cm and a crest travels ${(at16.speed * 100).toFixed(0)} cm/s. Wherever the paths from the two dippers differ by half a wavelength the waves cancel, so dippers d apart draw 2⌊d/λ + ½⌋ nodal lines: ${at16.nodal} at 5 cm. A crest one wavelength out stands only ${at16.amplitude.toFixed(3)} mm high (ka = ${M.SLOPE}), yet it is a weak lens. The lamp is ${cm(LAMP[2] - Z.water)} cm above the water and the paper ${cm(drop)} cm below it, so the shadow is ${optics.magnification.toFixed(2)} times life size and the paper sits ${at16.focus.toFixed(2)} of a crest's focal length down at 16 Hz, ${at28.focus.toFixed(2)} at 28 Hz. Near the dippers, where the two waves add, the light passes through focus and every crest draws a sharp bright line, and the plain grey bands that break them are the nodal lines. A sloped foam beach inside the walls soaks up each wave before it can reflect. The damping here is clean water's viscosity alone, so these ripples carry farther than in a real tank, where a surface film calms them sooner.`;

const body = `<div class="rt">${k.plateHtml({
  fig: "Fig 3",
  title: "Ripple tank",
  hint: "Move across for f · up and down for spacing",
  readout: M.readoutOf(startState),
  keys: [
    { mark: "lit", label: "Two dippers, in phase: the sources" },
    { mark: "raised", label: "Tank on legs, bridge and vibrating bar" },
    { mark: "flat", label: `The paper, ${cm(drop)} cm down: the pattern ${optics.magnification.toFixed(2)}× life size` },
  ],
  caption,
  body: stage,
})}</div>`;

const MODEL = readFileSync(join(HERE, "model.mjs"), "utf8").replace(/^export /gm, "");
const CSS = readFileSync(join(HERE, "ripple.css"), "utf8");
const LIVE = readFileSync(join(HERE, "live.js"), "utf8").replace("__DATA__", JSON.stringify(DATA));
const page = k.pageHtml({ title: "Ripple tank", theme, body: `<style>${CSS}</style>${body}`, script: kitScript() + glScript() + MODEL + LIVE, width: 760 });
const name = theme === "light" ? "ripple-tank-light.html" : "ripple-tank.html";
writeFileSync(join(HERE, name), page);
console.log("wrote", name, `${(page.length / 1024).toFixed(0)} KB`, `${back.length + mid.length + front.length} layers`, `M ${optics.magnification.toFixed(3)} caustic ${optics.caustic.toFixed(2)} mm x16 ${at16.focus.toFixed(3)} x28 ${at28.focus.toFixed(3)}`);
