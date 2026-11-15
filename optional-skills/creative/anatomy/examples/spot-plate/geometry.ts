import { circleRing, extrude, insetPlan, onTop, pathOf, type Plan, type Plane, planOutline, type Point3, ringOf, type SolidPaths, topMatrix } from "../../kit/iso-kit";
import {
  BASE,
  BASE_H,
  BOTTLE,
  DATUM_Y,
  DECK,
  DECK_H,
  FOOT_H,
  GAPS,
  MERGE_MOST,
  P,
  PIVOT,
  SCALE_R,
  STACK_TOP,
  STOP_R,
  SWEEP,
  TRAY,
  TRAY_H,
  FLOOR,
  VIEW,
  WASHER_H,
  WELL_D,
  WELL_HALF,
  WELLS,
  BORDER,
  angleOf,
  at,
  rowX,
  rowY,
  ROW_RIM,
} from "./rig";

const RADIANS = Math.PI / 180;
const VIEW_ANGLE = Math.atan2(VIEW[1], VIEW[0]);

const line = (points: Point3[]) => pathOf(points.map(at));
const lines = (runs: Point3[][]) => runs.map(line).join("");

function clean(ring: Plane[]): Plane[] {
  return ring.filter(([x, y], index) => {
    const [nx, ny] = ring[(index + 1) % ring.length]!;
    return Math.hypot(nx - x, ny - y) > 1e-6;
  });
}

const solid = (plan: Plan, z: number, height: number, steps = 6, bevel = 0): SolidPaths =>
  extrude(clean(ringOf(plan, steps)), z, height, P, { convex: true, bevel });

const cyl = (x: number, y: number, r: number, z: number, h: number, steps = 28): SolidPaths => extrude(circleRing(x, y, r, steps), z, h, P);

const ring = (x: number, y: number, r: number, z: number, steps = 24) => onTop(circleRing(x, y, r, steps), z, P, true);

function knurl(x: number, y: number, r: number, z0: number, z1: number, count = 9) {
  const runs: Point3[][] = [];
  for (let index = 0; index < count; index++) {
    const angle = VIEW_ANGLE - Math.PI * 0.42 + (Math.PI * 0.84 * index) / (count - 1);
    const px = x + r * Math.cos(angle);
    const py = y + r * Math.sin(angle);
    runs.push([
      [px, py, z0 + 0.4],
      [px, py, z1 - 0.4],
    ]);
  }
  return lines(runs);
}

function normalsOf(points: Plane[]) {
  const count = points.length;
  let area = 0;
  for (let index = 0; index < count; index++) {
    const [ax, ay] = points[index]!;
    const [bx, by] = points[(index + 1) % count]!;
    area += ax * by - bx * ay;
  }
  const counter = area > 0;
  return points.map(([ax, ay], index) => {
    const [bx, by] = points[(index + 1) % count]!;
    const dx = bx - ax;
    const dy = by - ay;
    const length = Math.hypot(dx, dy) || 1;
    return (counter ? [dy / length, -dx / length] : [-dy / length, dx / length]) as Plane;
  });
}

function runOf(points: Plane[], want: boolean) {
  const count = points.length;
  const facing = normalsOf(points).map(([nx, ny]) => nx * VIEW[0] + ny * VIEW[1] > 1e-9);
  const start = facing.findIndex((flag, index) => flag === want && facing[(index + count - 1) % count] !== want);
  if (start < 0) return [] as Plane[];
  const run: Plane[] = [points[start]!];
  for (let step = 0; step < count && facing[(start + step) % count] === want; step++) run.push(points[(start + step + 1) % count]!);
  return run;
}

const visibleRun = (plan: Plan, z: number, steps = 8) => onTop(runOf(clean(ringOf(plan, steps)), true), z, P);

const corners = ({ x, y, w, d }: Plan, inset: number): Plane[] => [
  [x + inset, y + inset],
  [x + w - inset, y + inset],
  [x + w - inset, y + d - inset],
  [x + inset, y + d - inset],
];

const FEET = corners(BASE, 18);
const BASE_SCREWS = corners(BASE, 6);

export const BASE_ART = {
  halo: solid(insetPlan(BASE, 6), -BASE_H - FOOT_H, 0.4, 8).fill,
  feet: FEET.map(([x, y]) => [cyl(x, y, 3.6, -BASE_H - FOOT_H, FOOT_H - 1.2, 24), cyl(x, y, 4.8, -BASE_H - 1.2, 1.2, 28)]),
  slab: solid(BASE, -BASE_H, BASE_H, 10, 2.2),
  groove: visibleRun(BASE, -BASE_H * 0.45, 10),
  inlay: planOutline(insetPlan(BASE, 4), 0, P, 8),
  screws: BASE_SCREWS.map(([x, y]) => at([x, y, 0])),
  screwRings: BASE_SCREWS.map(([x, y]) => ring(x, y, 1.6, 0, 18)).join(""),
};

function wellOf(plan: Plan) {
  const points = clean(ringOf(plan, 12));
  const back = runOf(points, false);
  const wall = back.length > 1 ? pathOf([...back.map(([x, y]) => at([x, y, TRAY_H])), ...[...back].reverse().map(([x, y]) => at([x, y, FLOOR]))], true) : "";
  return {
    opening: onTop(points, TRAY_H, P, true),
    wall,
    foot: onTop(back, FLOOR, P),
  };
}

const RULE_X = WELL_HALF + 2.2;
const RULE_MINOR = 2.2;
const RULE_MAJOR = 4.6;
const RULE_REACH = MERGE_MOST;

function rulerOf(row: number) {
  const x0 = rowX(row) + RULE_X;
  const minor: Point3[][] = [];
  const major: Point3[][] = [];
  for (let step = 0; step <= RULE_REACH; step += 2) {
    const y = rowY(row) + DATUM_Y + step;
    (step % 8 === 0 ? major : minor).push([
      [x0, y, TRAY_H],
      [x0 + (step % 8 === 0 ? RULE_MAJOR : RULE_MINOR), y, TRAY_H],
    ]);
  }
  return { minor, major };
}

const RULERS = GAPS.map((_, row) => rulerOf(row));

const countY = (row: number) => rowY(row) + WELL_D + (row < 3 ? ROW_RIM : BORDER) / 2;

const TRAY_SCREWS = corners(TRAY, 7);

export const TRAY_ART = {
  screws: TRAY_SCREWS.map(([x, y]) => at([x, y, TRAY_H])),
  screwRings: TRAY_SCREWS.map(([x, y]) => ring(x, y, 1.7, TRAY_H, 18)).join(""),
  slab: solid(TRAY, 0, TRAY_H, 10, 2.4),
  groove: visibleRun(TRAY, TRAY_H * 0.42, 10),
  inlay: planOutline(insetPlan(TRAY, 4.2), TRAY_H, P, 10),
  wells: WELLS.map(wellOf),
  minor: lines(RULERS.flatMap((ruler) => ruler.minor)),
  major: lines(RULERS.flatMap((ruler) => ruler.major)),
  datum: lines(
    GAPS.map((_, row) => [
      [rowX(row) - WELL_HALF - 3.4, rowY(row) + DATUM_Y, TRAY_H],
      [rowX(row) - WELL_HALF - 0.8, rowY(row) + DATUM_Y, TRAY_H],
    ]),
  ),
  gapMarks: GAPS.map((gap, row) => at([rowX(row) + RULE_X + RULE_MAJOR + 2.4, rowY(row) + DATUM_Y + gap, TRAY_H])),
  counts: GAPS.flatMap((_, row) =>
    Array.from({ length: row + 1 }, (__, dot) => at([rowX(row) + (dot - row / 2) * 3.2, countY(row), TRAY_H])),
  ),
};

function sector(r0: number, r1: number, from: number, to: number, steps = 40): Plane[] {
  const out: Plane[] = [];
  const point = (r: number, degrees: number): Plane => [PIVOT[0] + r * Math.sin(degrees * RADIANS), PIVOT[1] - r * Math.cos(degrees * RADIANS)];
  for (let step = 0; step <= steps; step++) out.push(point(r1, from + ((to - from) * step) / steps));
  for (let step = steps; step >= 0; step--) out.push(point(r0, from + ((to - from) * step) / steps));
  return out;
}

const polar = (r: number, degrees: number, z: number): Point3 => [PIVOT[0] + r * Math.sin(degrees * RADIANS), PIVOT[1] - r * Math.cos(degrees * RADIANS), z];

const BAND_IN = SCALE_R - 2;
const BAND_OUT = SCALE_R + 10;
const BAND_FROM = -7;
const BAND_TO = SWEEP + 7;
const BAND_SPLIT = (Math.atan2(VIEW[1], VIEW[0]) * 180) / Math.PI;
const BAND_TOP = STACK_TOP;

function scaleTicks() {
  const minor: Point3[][] = [];
  const major: Point3[][] = [];
  for (let merge = 0; merge <= MERGE_MOST; merge += 2) {
    const angle = angleOf(merge);
    const long = merge % 8 === 0;
    (long ? major : minor).push([polar(BAND_IN + 1.2, angle, BAND_TOP), polar(BAND_IN + (long ? 6 : 3.6), angle, BAND_TOP)]);
  }
  return { minor: lines(minor), major: lines(major) };
}

const TICKS = scaleTicks();

const DECK_SCREWS = corners(DECK, 6);

const STOPS = GAPS.map((gap) => polar(STOP_R, angleOf(gap), DECK_H));

export const DECK_ART = {
  slab: solid(DECK, 0, DECK_H, 8, 1.6),
  inlay: planOutline(insetPlan(DECK, 3.4), DECK_H, P, 8),
  screws: DECK_SCREWS.map(([x, y]) => at([x, y, DECK_H])),
  screwRings: DECK_SCREWS.map(([x, y]) => ring(x, y, 1.5, DECK_H, 18)).join(""),
  guide: onTop(
    Array.from({ length: 61 }, (_, step) => {
      const [x, y] = polar(STOP_R + 3, (SWEEP * step) / 60, DECK_H);
      return [x, y] as Plane;
    }),
    DECK_H,
    P,
  ),
  stops: STOPS.map(([x, y]) => cyl(x, y, 1.3, DECK_H, 1.8, 16)),
  stopTops: STOPS.map(([x, y]) => at([x, y, DECK_H + 1.8])),
  bandBack: extrude(sector(BAND_IN, BAND_OUT, BAND_FROM, BAND_SPLIT, 24), DECK_H, BAND_TOP - DECK_H, P, { convex: false }),
  bandFront: extrude(sector(BAND_IN, BAND_OUT, BAND_SPLIT, BAND_TO, 64), DECK_H, BAND_TOP - DECK_H, P, { convex: false }),
  seam: at(polar((BAND_IN + BAND_OUT) / 2, BAND_SPLIT, BAND_TOP)),
  bandLine: onTop(sector(BAND_OUT - 1.6, BAND_OUT - 1.6, BAND_FROM + 1.2, BAND_TO - 1.2, 96).slice(0, 97), BAND_TOP, P),
  minor: TICKS.minor,
  major: TICKS.major,
  gapDots: GAPS.map((gap) => at(polar(BAND_OUT - 3.6, angleOf(gap), BAND_TOP))),
  ends: [0, SWEEP].map((angle) => at(polar(BAND_OUT - 3.6, angle, BAND_TOP))),
  label: topMatrix([PIVOT[0] - 13, PIVOT[1] + 150], DECK_H, P, "x"),
  washer: cyl(PIVOT[0], PIVOT[1], 9.4, DECK_H, WASHER_H, 32),
  cap: cyl(PIVOT[0], PIVOT[1], 8.2, STACK_TOP, 0.8, 32),
  nut: cyl(PIVOT[0], PIVOT[1], 5.8, STACK_TOP + 0.8, 4.2, 28),
  nutKnurl: knurl(PIVOT[0], PIVOT[1], 5.8, STACK_TOP + 0.8, STACK_TOP + 5, 13),
  nutRing: ring(PIVOT[0], PIVOT[1], 3.2, STACK_TOP + 5, 20),
  pin: at([PIVOT[0], PIVOT[1], STACK_TOP + 5]),
};

const [BX, BY] = BOTTLE;
const COASTER_TOP = DECK_H + 2;
const BODY_R = 19;
const BODY_TOP = COASTER_TOP + 32;
const SHOULDER_TOP = BODY_TOP + 2.2;
const NECK_TOP = SHOULDER_TOP + 2.2;
const COLLAR_TOP = NECK_TOP + 2.6;
const CAP_TOP = COLLAR_TOP + 11;
const BULB_TOP = CAP_TOP + 10;
const SHINE_ANGLE = VIEW_ANGLE + 0.95;
const shineAt = (r: number, z0: number, z1: number) =>
  line([
    [BX + r * Math.cos(SHINE_ANGLE), BY + r * Math.sin(SHINE_ANGLE), z0],
    [BX + r * Math.cos(SHINE_ANGLE), BY + r * Math.sin(SHINE_ANGLE), z1],
  ]);

export const BOTTLE_ART = {
  coaster: cyl(BX, BY, 25, DECK_H, COASTER_TOP - DECK_H, 44),
  coasterRing: ring(BX, BY, 22.6, COASTER_TOP, 40),
  coasterDots: [0, 1, 2, 3].map((index) => {
    const angle = Math.PI / 4 + (Math.PI / 2) * index;
    return at([BX + 23.8 * Math.cos(angle), BY + 23.8 * Math.sin(angle), COASTER_TOP]);
  }),
  body: cyl(BX, BY, BODY_R, COASTER_TOP, BODY_TOP - COASTER_TOP, 44),
  level: onTop(runOf(clean(circleRing(BX, BY, BODY_R, 44)), true), COASTER_TOP + 21, P),
  foot: onTop(runOf(clean(circleRing(BX, BY, BODY_R, 44)), true), COASTER_TOP + 2.4, P),
  shine: shineAt(BODY_R, COASTER_TOP + 5, BODY_TOP - 4),
  shoulder: cyl(BX, BY, 15.5, BODY_TOP, SHOULDER_TOP - BODY_TOP, 40),
  neck: cyl(BX, BY, 11, SHOULDER_TOP, NECK_TOP - SHOULDER_TOP, 36),
  collar: cyl(BX, BY, 9.4, NECK_TOP, COLLAR_TOP - NECK_TOP, 36),
  cap: cyl(BX, BY, 10, COLLAR_TOP, CAP_TOP - COLLAR_TOP, 36),
  capKnurl: knurl(BX, BY, 10, COLLAR_TOP + 1, CAP_TOP - 1, 15),
  capRing: ring(BX, BY, 7.2, CAP_TOP, 32),
  bulb: cyl(BX, BY, 6, CAP_TOP, BULB_TOP - CAP_TOP, 32),
  bulbShine: shineAt(6, CAP_TOP + 2, BULB_TOP - 2),
  bulbRing: ring(BX, BY, 3.4, BULB_TOP, 24),
  tip: at([BX, BY, BULB_TOP]),
};

export const LABEL = `A lab spot plate on a bench. The plate has six channels, each holding a pair of glass drops, R30 and R22, at gaps of ${GAPS.join(", ")} px, drawn to scale against a ruler on the rim. On the right, beside a dropper bottle, a fan of six feeler blades, one for each gap, turns on a pivot over a protractor marked 0 to ${MERGE_MOST} px; the top blade is merge. As it opens, each blade stops at its own gap, and the pair with that gap runs together into one piece of glass.`;
