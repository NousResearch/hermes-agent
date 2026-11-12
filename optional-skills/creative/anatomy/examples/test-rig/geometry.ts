import { circleRing, extrude, insetPlan, onTop, pathOf, planOutline, slabOf, topMatrix, type Flat, type Plan, type Plane, type Point3 } from "../../kit/iso-kit";
import {
  BAR,
  BAR_H,
  BASE,
  BASE_H,
  BUSH_D,
  BUSH_END,
  BUSH_W,
  CARRIAGE,
  CARRIAGE_H,
  CARRIAGE_TOP,
  CARRIAGE_Z,
  DIAL,
  DIAL_H,
  DIAL_R,
  DIAL_SPAN,
  DIAL_TOP,
  DIAL_Z,
  FOOT_H,
  FRAME_X,
  FRAME_Y,
  GLASS_TOP,
  HOUSING_H,
  HOUSING_R,
  HOUSING_Z,
  P,
  PAD_H,
  PAD_R,
  PLUNGER_Z,
  POST,
  POST_L,
  POST_R,
  PROBE_HOME,
  RAIL_D,
  RAIL_H,
  RAIL_Y,
  RAIL_Z,
  ROD_HALF,
  ROD_Y,
  ROD_Z,
  STEM,
  at,
} from "./view";

const solid = (plan: Plan, z: number, h: number, steps = 6, bevel = 0) => slabOf(plan, z, h, P, steps, bevel);
const cylinder = (cx: number, cy: number, r: number, z: number, h: number, steps = 32) => extrude(circleRing(cx, cy, r, steps), z, h, P, { convex: true });
const ring = (cx: number, cy: number, r: number, z: number, steps = 40) => onTop(circleRing(cx, cy, r, steps), z, P, true);
const segment = (a: Point3, b: Point3) => pathOf([at(a), at(b)]);
const join = (paths: string[]) => paths.join("");
const RADIANS = Math.PI / 180;

function knurl(cx: number, cy: number, r: number, z0: number, z1: number, count: number) {
  const lines: string[] = [];
  for (let index = 0; index < count; index++) {
    const angle = (index / count) * Math.PI * 2;
    const nx = Math.cos(angle);
    const ny = Math.sin(angle);
    if (nx * Math.cos(57 * RADIANS) + ny * Math.sin(57 * RADIANS) < 0.15) continue;
    lines.push(segment([cx + r * nx, cy + r * ny, z0 + 0.4], [cx + r * nx, cy + r * ny, z1 - 0.4]));
  }
  return join(lines);
}

function sideArc(cx: number, cy: number, r: number, z: number, steps = 24) {
  const points: Flat[] = [];
  for (let step = 0; step <= steps; step++) {
    const angle = (57 - 90 + (step / steps) * 180) * RADIANS;
    points.push(at([cx + r * Math.cos(angle), cy + r * Math.sin(angle), z]));
  }
  return pathOf(points);
}

function faceRing(x: number, cy: number, cz: number, r: number, steps = 20) {
  const points: Flat[] = [];
  for (let step = 0; step < steps; step++) {
    const angle = (step / steps) * Math.PI * 2;
    points.push(at([x, cy + r * Math.cos(angle), cz + r * Math.sin(angle)]));
  }
  return pathOf(points, true);
}

const FEET: Plane[] = [
  [BASE.x + 18, BASE.y + 16],
  [BASE.x + BASE.w - 18, BASE.y + 16],
  [BASE.x + 18, BASE.y + BASE.d - 16],
  [BASE.x + BASE.w - 18, BASE.y + BASE.d - 16],
];

const BASE_SCREWS: Plane[] = [
  [BASE.x + 8, BASE.y + 8],
  [BASE.x + BASE.w - 8, BASE.y + 8],
  [BASE.x + 8, BASE.y + BASE.d - 8],
  [BASE.x + BASE.w - 8, BASE.y + BASE.d - 8],
];

const PLATE: Plan = { x: BASE.x + 20, y: FRAME_Y + BAR + 6, w: 56, d: 11, r: 2 };

export const BASE_ART = {
  halo: solid(BASE, -BASE_H - FOOT_H, BASE_H).fill,
  feet: FEET.map(([x, y]) => [cylinder(x, y, 7, -BASE_H - FOOT_H, 1.6), cylinder(x, y, 5.5, -BASE_H - FOOT_H + 1.6, FOOT_H - 1.6)]),
  slab: solid(BASE, -BASE_H, BASE_H, 8, 1.6),
  groove: planOutline(insetPlan(BASE, 4.5), 0, P),
  screws: BASE_SCREWS.map(([x, y]) => at([x, y, 0])),
  screwRings: join(BASE_SCREWS.map(([x, y]) => ring(x, y, 2.2, 0, 16))),
  plate: solid(PLATE, 0, 0.8, 4),
  plateLine: planOutline(insetPlan(PLATE, 1.6), 0.8, P),
  plateScrews: [at([PLATE.x + 3.5, PLATE.y + PLATE.d / 2, 0.8]), at([PLATE.x + PLATE.w - 3.5, PLATE.y + PLATE.d / 2, 0.8])],
  plateRule: join([0, 1, 2].map((row) => onTop([[PLATE.x + 8, PLATE.y + 3.4 + row * 2.1], [PLATE.x + (row === 1 ? 34 : 44), PLATE.y + 3.4 + row * 2.1]], 0.8, P))),
};

const POST_PLANS: Plan[] = [POST_L, POST_R].map((x) => ({ x, y: RAIL_Y - POST / 2, w: POST, d: POST, r: 2 }));
const RAIL: Plan = { x: POST_L - 5, y: RAIL_Y - RAIL_D / 2, w: POST_R + POST - POST_L + 10, d: RAIL_D, r: 2 };
const RAIL_FACE = RAIL_Y + RAIL_D / 2;
const RAIL_TOP = RAIL_Z + RAIL_H;
const RULE_FROM = POST_L + POST + 4;
const RULE_TO = POST_R - 4;

function ruler() {
  const minor: string[] = [];
  const major: string[] = [];
  for (let x = Math.ceil(RULE_FROM / 5) * 5; x <= RULE_TO; x += 5) {
    const big = x % 25 === 0;
    (big ? major : minor).push(segment([x, RAIL_FACE, RAIL_TOP - 1.2], [x, RAIL_FACE, RAIL_TOP - (big ? 5 : 3)]));
  }
  return { minor: join(minor), major: join(major) };
}

const RULE = ruler();

export const RAIL_ART = {
  feet: POST_PLANS.map((plan) => solid(insetPlan(plan, -3), 0, 2, 4)),
  footScrews: POST_PLANS.flatMap((plan) => [at([plan.x - 1, plan.y + POST / 2, 2]), at([plan.x + POST + 1, plan.y + POST / 2, 2])]),
  posts: POST_PLANS.map((plan) => solid(plan, 2, RAIL_Z - 2, 4)),
  postLines: join(POST_PLANS.map((plan) => segment([plan.x + POST, plan.y + POST, 6], [plan.x + POST, plan.y + POST, RAIL_Z - 4]))),
  rail: solid(RAIL, RAIL_Z, RAIL_H, 4, 0.8),
  railScrews: POST_PLANS.map((plan) => at([plan.x + POST / 2, RAIL_Y, RAIL_TOP])),
  ruleMinor: RULE.minor,
  ruleMajor: RULE.major,
  ruleBase: segment([RULE_FROM, RAIL_FACE, RAIL_TOP - 1.2], [RULE_TO, RAIL_FACE, RAIL_TOP - 1.2]),
  stops: [RULE_FROM - 2, RULE_TO + 2].map((x) => solid({ x: x - 1.5, y: RAIL_Y - RAIL_D / 2 - 0.5, w: 3, d: RAIL_D + 1, r: 0.8 }, RAIL_TOP, 2.2, 3)),
};

export const RIDER = {
  body: solid({ x: -1.6, y: RAIL_Y - RAIL_D / 2 - 0.6, w: 3.2, d: RAIL_D + 1.2, r: 0.8 }, RAIL_TOP, 1.4, 3),
  tick: segment([0, RAIL_FACE + 0.6, RAIL_TOP + 1.4], [0, RAIL_FACE + 0.6, RAIL_TOP - 3]),
};

const H = PROBE_HOME;
const BLOCK: Plan = { x: H - 11, y: RAIL_Y - 7, w: 22, d: 14, r: 2.5 };
const BLOCK_Z = RAIL_TOP;
const BLOCK_H = 9;
const ARM_Z = BLOCK_Z + 3;
const ARM_H = 5;
const ARM: Plan = { x: H - 3.5, y: BLOCK.y + BLOCK.d - 1, w: 7, d: -(BLOCK.y + BLOCK.d - 1) + 2, r: 1.5 };

export const PROBE_ART = {
  block: solid(BLOCK, BLOCK_Z, BLOCK_H, 5, 0.8),
  blockLine: planOutline(insetPlan(BLOCK, 2), BLOCK_Z + BLOCK_H, P),
  blockScrews: [at([BLOCK.x + 3, BLOCK.y + 3, BLOCK_Z + BLOCK_H]), at([BLOCK.x + BLOCK.w - 3, BLOCK.y + 3, BLOCK_Z + BLOCK_H])],
  thumb: cylinder(H - 5, RAIL_Y - 1, 2.6, BLOCK_Z + BLOCK_H, 3.2, 20),
  thumbKnurl: knurl(H - 5, RAIL_Y - 1, 2.6, BLOCK_Z + BLOCK_H, BLOCK_Z + BLOCK_H + 3.2, 16),
  pointer: solid({ x: H - 2, y: BLOCK.y + BLOCK.d, w: 4, d: 1, r: 0.4 }, RAIL_TOP - 4, 4 + 2, 2),
  index: segment([H, BLOCK.y + BLOCK.d + 1, RAIL_TOP + 1.5], [H, BLOCK.y + BLOCK.d + 1, RAIL_TOP - 3.6]),
  arm: solid(ARM, ARM_Z, ARM_H, 4, 0.6),
  armLine: onTop([[H, ARM.y + 3], [H, -9]], ARM_Z + ARM_H, P),
  armScrews: [at([H, ARM.y + 2, ARM_Z + ARM_H]), at([H, -2, ARM_Z + ARM_H])],
  housing: cylinder(H, 0, HOUSING_R, HOUSING_Z, HOUSING_H - 2, 32),
  cap: cylinder(H, 0, HOUSING_R + 0.8, HOUSING_Z + HOUSING_H - 2, 2, 32),
  collar: cylinder(H, 0, HOUSING_R + 0.8, HOUSING_Z, 2.4, 32),
  housingSeam: join([HOUSING_Z + 6, HOUSING_Z + 11].map((z) => sideArc(H, 0, HOUSING_R, z))),
  lock: solid({ x: H - 1.6, y: HOUSING_R - 0.5, w: 3.2, d: 3.5, r: 0.8 }, HOUSING_Z + 8, 3.2, 2),
  lockHead: cylinder(H, HOUSING_R + 4.4, 2.2, HOUSING_Z + 7.4, 4.4, 16),
};

const PAD_Z = GLASS_TOP;
export const FINGER_ART = {
  stem: cylinder(H, 0, 1.5, PAD_Z + PAD_H + 2, STEM, 16),
  stemMarks: join([8, 11, 14].map((rise) => sideArc(H, 0, 1.5, PAD_Z + PAD_H + 2 + rise, 12))),
  collar: cylinder(H, 0, 2.6, PAD_Z + PAD_H, 2, 20),
  pad: cylinder(H, 0, PAD_R, PAD_Z, PAD_H, 32),
  padRing: ring(H, 0, PAD_R - 1.2, PAD_Z + PAD_H, 28),
};

const BARS: Plan[] = [
  { x: -FRAME_X - BAR, y: -FRAME_Y - BAR, w: (FRAME_X + BAR) * 2, d: BAR, r: 1.5 },
  { x: -FRAME_X - BAR, y: -FRAME_Y, w: BAR, d: FRAME_Y * 2, r: 0.6 },
  { x: FRAME_X, y: -FRAME_Y, w: BAR, d: FRAME_Y * 2, r: 0.6 },
  { x: -FRAME_X - BAR, y: FRAME_Y, w: (FRAME_X + BAR) * 2, d: BAR, r: 1.5 },
];

const barScrews = (plan: Plan): Flat[] =>
  plan.w > plan.d
    ? [-1, 1].map((side) => at([side * (FRAME_X + BAR / 2), plan.y + BAR / 2, BAR_H]))
    : [-1, 1].map((side) => at([plan.x + BAR / 2, side * (FRAME_Y - 6), BAR_H]));

const FLOOR: Plan = { x: -FRAME_X, y: -FRAME_Y, w: FRAME_X * 2, d: FRAME_Y * 2, r: 0 };

function floorGrid() {
  const lines: string[] = [];
  for (let x = -FRAME_X + 10; x < FRAME_X; x += 10) lines.push(onTop([[x, -FRAME_Y], [x, FRAME_Y]], 0.8, P));
  for (let y = -FRAME_Y + 7.4; y < FRAME_Y; y += 7.4) lines.push(onTop([[-FRAME_X, y], [FRAME_X, y]], 0.8, P));
  return join(lines);
}

function barTicks() {
  const minor: string[] = [];
  const major: string[] = [];
  const y = FRAME_Y + 1;
  for (let x = -40; x <= 40; x += 2) {
    const big = x % 10 === 0;
    (big ? major : minor).push(onTop([[x, y], [x, y + (big ? 3.2 : 1.8)]], BAR_H, P));
  }
  return { minor: join(minor), major: join(major) };
}

const TICKS = barTicks();

export const FRAME_ART = {
  back: solid(BARS[0]!, 0, BAR_H, 4, 0.8),
  left: solid(BARS[1]!, 0, BAR_H, 3, 0.8),
  right: solid(BARS[2]!, 0, BAR_H, 3, 0.8),
  front: solid(BARS[3]!, 0, BAR_H, 4, 0.8),
  backScrews: barScrews(BARS[0]!),
  leftScrews: barScrews(BARS[1]!),
  rightScrews: barScrews(BARS[2]!),
  frontScrews: barScrews(BARS[3]!),
  floor: solid(FLOOR, 0, 0.8, 2),
  grid: floorGrid(),
  ticksMinor: TICKS.minor,
  ticksMajor: TICKS.major,
  zero: onTop([[0, FRAME_Y + 0.6], [0, FRAME_Y + BAR - 0.6]], BAR_H, P),
  rodEnds: join([-ROD_Y, ROD_Y].map((y) => faceRing(FRAME_X + BAR, y, ROD_Z, 3, 16))),
  rodNuts: [-ROD_Y, ROD_Y].map((y) => at([FRAME_X + BAR, y, ROD_Z])),
  bore: faceRing(FRAME_X + BAR, 0, PLUNGER_Z, 2.2, 14),
  leftBores: join([-ROD_Y, ROD_Y].map((y) => faceRing(-FRAME_X, y, ROD_Z, 3, 16))),
};

export const RODS = [-ROD_Y, ROD_Y].map((y) => solid({ x: -FRAME_X, y: y - ROD_HALF, w: FRAME_X * 2, d: ROD_HALF * 2, r: ROD_HALF - 0.2 }, ROD_Z - ROD_HALF, ROD_HALF * 2, 3));

const SEAT: Plan = { x: -63, y: -18, w: 126, d: 36, r: 18 };
const BUSHES: Plan[] = [-1, 1].flatMap((side) =>
  [-ROD_Y, ROD_Y].map((y) => ({ x: side < 0 ? -BUSH_END : BUSH_END - BUSH_W, y: y - BUSH_D / 2, w: BUSH_W, d: BUSH_D, r: 2 })),
);

export const CARRIAGE_ART = {
  halo: solid(insetPlan(CARRIAGE, 3), 0.8, 0.1).fill,
  plate: solid(CARRIAGE, CARRIAGE_Z, CARRIAGE_H, 6, 0.8),
  seat: planOutline(SEAT, CARRIAGE_TOP, P, 10),
  seatInner: planOutline(insetPlan(SEAT, 2.4), CARRIAGE_TOP, P, 10),
  bushes: BUSHES.map((plan) => solid(plan, CARRIAGE_Z - 1, CARRIAGE_H + 4, 3, 0.6)),
  bushScrews: BUSHES.flatMap((plan) => [at([plan.x + 3, plan.y + plan.d / 2, CARRIAGE_TOP + 3]), at([plan.x + plan.w - 3, plan.y + plan.d / 2, CARRIAGE_TOP + 3])]),
  bushBores: join([-ROD_Y, ROD_Y].map((y) => faceRing(BUSH_END, y, ROD_Z, 2.8, 14))),
  contact: solid({ x: BUSH_END - 8, y: -4, w: 8, d: 8, r: 2 }, CARRIAGE_Z, CARRIAGE_H, 3, 0.4),
  rivets: [-56, -28, 0, 28, 56].map((x) => at([x, CARRIAGE.y + CARRIAGE.d - 2.4, CARRIAGE_TOP])),
};

export const PLUNGER_ART = {
  rod: solid({ x: BUSH_END, y: -1.1, w: 44, d: 2.2, r: 1 }, PLUNGER_Z - 1.1, 2.2, 3),
  collar: solid({ x: BUSH_END + 0.4, y: -2.2, w: 2.4, d: 4.4, r: 1 }, PLUNGER_Z - 2.2, 4.4, 3, 0.3),
};

const [DX, DY] = DIAL;
const STAND: Plan = { x: DX - 15, y: DY - 14, w: 30, d: 28, r: 3 };

function dialTicks() {
  const minor: string[] = [];
  const major: string[] = [];
  for (let step = 0; step < DIAL_SPAN; step++) {
    const angle = (step / DIAL_SPAN) * Math.PI * 2;
    const big = step % 5 === 0;
    const sx = Math.sin(angle);
    const sy = -Math.cos(angle);
    const outer = DIAL_R - 2.4;
    const inner = outer - (big ? 2.6 : 1.4);
    (big ? major : minor).push(onTop([[DX + sx * inner, DY + sy * inner], [DX + sx * outer, DY + sy * outer]], DIAL_TOP, P));
  }
  return { minor: join(minor), major: join(major) };
}

const DIAL_TICKS = dialTicks();
const GIVE_ANGLE = (14 / DIAL_SPAN) * Math.PI * 2;
const R_GIVE = DIAL_R - 1.4;

export const DIAL_ART = {
  stand: solid(STAND, 0, DIAL_Z, 4, 0.6),
  standScrews: [at([STAND.x + 3.4, STAND.y + 3.4, DIAL_Z]), at([STAND.x + STAND.w - 3.4, STAND.y + STAND.d - 3.4, DIAL_Z]), at([STAND.x + 3.4, STAND.y + STAND.d - 3.4, DIAL_Z]), at([STAND.x + STAND.w - 3.4, STAND.y + 3.4, DIAL_Z])],
  sleeve: solid({ x: FRAME_X + BAR, y: -2.6, w: DX - DIAL_R - FRAME_X - BAR + 0.6, d: 5.2, r: 1 }, PLUNGER_Z - 2.6, 5.2, 3),
  clamp: cylinder(FRAME_X + BAR + 3.2, 0, 1.6, PLUNGER_Z + 2.6, 2, 12),
  puck: cylinder(DX, DY, DIAL_R, DIAL_Z, DIAL_H, 56),
  bezel: ring(DX, DY, DIAL_R - 1, DIAL_TOP, 56),
  face: ring(DX, DY, DIAL_R - 2.4, DIAL_TOP, 56),
  knurl: knurl(DX, DY, DIAL_R, DIAL_Z + DIAL_H - 3, DIAL_Z + DIAL_H, 56),
  band: sideArc(DX, DY, DIAL_R, DIAL_Z + DIAL_H - 3, 40),
  minor: DIAL_TICKS.minor,
  major: DIAL_TICKS.major,
  give: [-1, 1].map((side) => at([DX + Math.sin(side * GIVE_ANGLE) * R_GIVE, DY - Math.cos(side * GIVE_ANGLE) * R_GIVE, DIAL_TOP])),
  crown: solid({ x: DX + DIAL_R - 0.6, y: -2.4, w: 4.4, d: 4.8, r: 1.2 }, DIAL_Z + 2, 4, 3),
  crownCap: solid({ x: DX + DIAL_R + 3.6, y: -3.2, w: 2.6, d: 6.4, r: 1.2 }, DIAL_Z + 1.2, 5.6, 3),
  lug: solid({ x: DX - 3, y: DY - DIAL_R - 3.4, w: 6, d: 4, r: 1.2 }, DIAL_Z + 1.5, 5, 3),
  label: topMatrix([DX, DY + 5.2], DIAL_TOP, P, "x"),
};

const REST_GLASS = { x: -60, y: -16, w: 120, d: 32, r: 16 };
export const REST_OUTLINE = planOutline(REST_GLASS, GLASS_TOP, P, 12);
