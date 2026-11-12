import { cameraOf, type Flat, iso, type Plan, type Point3, type Projection } from "../../kit/iso-kit";

export const RADII = [30, 22] as const;
export const GAPS = [4, 10, 16, 24, 32, 44] as const;
export const MERGE = 20;
export const MERGE_LEAST = 0;
export const MERGE_MOST = 48;
export const MERGE_STEP = 1;

const UNIFORM = 2;
const SIDE = 0.35;
const FACING = 0.65;
const DEPTH = 0.25;

export const [RA, RB] = RADII;
export const ROWS = GAPS.length;
const FAR = GAPS[ROWS - 1];

export const spanOf = (gap: number) => RA + gap + RB;
export const reachOf = (merge: number) => merge * UNIFORM;
export const sideOf = (merge: number) => Math.round(reachOf(merge) * SIDE * 10) / 10;
export const seenGapOf = (gap: number, merge: number) => (gap >= 2 * merge ? gap : gap <= merge ? 0 : 2 * Math.sqrt(merge * (gap - merge)));
export const joinedAt = (merge: number) => GAPS.filter((gap) => gap < merge);

export type Field = (u: number, v: number) => number;

export function fieldOf(gap: number, merge: number): Field {
  const span = spanOf(gap);
  const most = reachOf(merge);
  return (u, v) => {
    const qu = u - span;
    const la = Math.sqrt(u * u + v * v);
    const lb = Math.sqrt(qu * qu + v * v);
    const a = la - RA;
    const b = lb - RB;
    const nearest = a < b ? a : b;
    if (most <= 0) return nearest;
    const spread = a < b ? b - a : a - b;
    if (spread >= most) return nearest;
    const raw = 0.5 - (0.5 * (u * qu + v * v)) / Math.max(la * lb, 0.0001);
    const facing = raw < 0 ? 0 : raw > 1 ? 1 : raw;
    const reach = most * (SIDE + FACING * facing);
    const h = Math.max(reach - spread, 0) / reach;
    return nearest - h * h * reach * DEPTH;
  };
}

function halfWidth(field: Field, u: number) {
  if (field(u, 0) >= 0) return 0;
  let low = 0;
  let high = RA + 12;
  for (let round = 0; round < 30; round++) {
    const mid = (low + high) / 2;
    if (field(u, mid) < 0) low = mid;
    else high = mid;
  }
  return low;
}

export function neckOf(gap: number, merge: number) {
  if (gap >= merge) return 0;
  const field = fieldOf(gap, merge);
  const from = RA - 8;
  const to = RA + gap + 8;
  const coarse = 48;
  let best = Infinity;
  let at = (from + to) / 2;
  for (let step = 0; step <= coarse; step++) {
    const u = from + ((to - from) * step) / coarse;
    const width = halfWidth(field, u);
    if (width > 0 && width < best) {
      best = width;
      at = u;
    }
  }
  const golden = (Math.sqrt(5) - 1) / 2;
  let low = Math.max(from, at - (to - from) / coarse);
  let high = Math.min(to, at + (to - from) / coarse);
  for (let round = 0; round < 24; round++) {
    const c = high - golden * (high - low);
    const d = low + golden * (high - low);
    if (halfWidth(field, c) < halfWidth(field, d)) high = d;
    else low = c;
  }
  return 2 * halfWidth(field, (low + high) / 2);
}

export const WIDTH = 600;
export const HEIGHT = 340;
const PAD_X = 80;
const PAD_Y = 34;
export const AZIMUTH = 50;

export const MARGIN = 6;
export const PITCH = 84;
export const WELL_HALF = RA + MARGIN;
export const BIG_Y = MARGIN + RA;
export const DATUM_Y = BIG_Y + RA;
export const WELL_D = DATUM_Y + FAR + 2 * RB + MARGIN;
export const BORDER = 14;

export const COLS = 3;
export const ROW_RIM = 16;
export const rowX = (row: number) => (row % COLS) * PITCH;
export const rowY = (row: number) => Math.floor(row / COLS) * (WELL_D + ROW_RIM);

export const WELLS: Plan[] = GAPS.map((_, row) => ({ x: rowX(row) - WELL_HALF, y: rowY(row), w: WELL_HALF * 2, d: WELL_D, r: WELL_HALF }));

export const TRAY: Plan = {
  x: -WELL_HALF - BORDER,
  y: -BORDER,
  w: rowX(COLS - 1) + (WELL_HALF + BORDER) * 2,
  d: rowY(ROWS - 1) + WELL_D + BORDER * 2,
  r: 16,
};
export const TRAY_H = 13;
export const WELL_SINK = 3;
export const FLOOR = TRAY_H - WELL_SINK;
export const THICK = 8;
export const GLASS_TOP = FLOOR + THICK;

export const DECK: Plan = { x: TRAY.x + TRAY.w + 10, y: TRAY.y, w: 172, d: TRAY.d, r: 12 };
export const DECK_H = 6;
export const PIVOT: [number, number] = [DECK.x + 22, DECK.y + DECK.d / 2 + 6];
export const BLADE_LONG = 104;
export const LEAD_LONG = 113;
export const STOP_R = 111;
export const SCALE_R = 120;
export const SWEEP = 180;
export const WASHER_H = 0.8;
export const STACK_Z = DECK_H + WASHER_H;
export const thicknessOf = (gap: number) => 0.45 + gap / 20;
export const LEAD_H = 1.1;

export const BLADE_Z: number[] = (() => {
  const out: number[] = [];
  let z = STACK_Z;
  for (const gap of GAPS) {
    out.push(z);
    z += thicknessOf(gap) + 0.12;
  }
  out.push(z);
  return out;
})();
export const STACK_TOP = BLADE_Z[ROWS]! + LEAD_H;

export const angleOf = (merge: number) => (Math.min(Math.max(merge, MERGE_LEAST), MERGE_MOST) / MERGE_MOST) * SWEEP;

export const BOTTLE: [number, number] = [DECK.x + DECK.w - 33, DECK.y + 35];
export const BOTTLE_TOP = DECK_H + 64;

export const BASE: Plan = { x: TRAY.x - 14, y: TRAY.y - 14, w: DECK.x + DECK.w + 14 - (TRAY.x - 14), d: TRAY.d + 28, r: 10 };
export const BASE_H = 10;
export const FOOT_H = 5;

const RADIANS = Math.PI / 180;

function fit(points: Point3[]): Projection {
  const unitView: Projection = { origin: [0, 0], scale: 1, azimuth: AZIMUTH };
  const flat = points.map((point) => iso(point, unitView));
  const xs = flat.map(([x]) => x);
  const ys = flat.map(([, y]) => y);
  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const up = Math.min(...ys);
  const down = Math.max(...ys);
  const scale = Math.min((WIDTH - PAD_X * 2) / (right - left), (HEIGHT - PAD_Y * 2) / (down - up));
  return { origin: [WIDTH / 2 - ((left + right) / 2) * scale, HEIGHT / 2 - ((up + down) / 2) * scale], scale, azimuth: AZIMUTH };
}

const box = ({ x, y, w, d }: Plan, z0: number, z1: number): Point3[] => {
  const out: Point3[] = [];
  for (const px of [x, x + w]) for (const py of [y, y + d]) for (const pz of [z0, z1]) out.push([px, py, pz]);
  return out;
};

export const P = fit([...box(BASE, -BASE_H - FOOT_H, 0), ...box(DECK, 0, STACK_TOP + 6), [BOTTLE[0], BOTTLE[1], BOTTLE_TOP]]);
export const at = (point: Point3): Flat => iso(point, P);

const CAMERA = cameraOf(P);
export const VIEW: [number, number] = [CAMERA.cosA, CAMERA.sinA];
export const LIGHT: [number, number] = [-CAMERA.sinA, CAMERA.cosA];
export const CLIMB = CAMERA.sinE / CAMERA.cosE;
export const K = CAMERA.k;
export const SIN_A = Math.sin(AZIMUTH * RADIANS);
export const COS_A = Math.cos(AZIMUTH * RADIANS);

const SLIDE_FROM = 0.16;
const SLIDE_TO = 0.84;

export function mergeAtX(x: number) {
  const share = (x / WIDTH - SLIDE_FROM) / (SLIDE_TO - SLIDE_FROM);
  return MERGE_LEAST + Math.min(1, Math.max(0, share)) * (MERGE_MOST - MERGE_LEAST);
}

export const roundMerge = (merge: number) => Math.min(MERGE_MOST, Math.max(MERGE_LEAST, Math.round(merge)));

const listOf = (values: readonly (number | string)[]) =>
  values.length < 2 ? values.join("") : `${values.slice(0, -1).join(", ")} and ${values[values.length - 1]}`;

export function readoutOf(merge: number) {
  const shown = roundMerge(merge);
  return `merge ${shown} px · ${joinedAt(shown).length} of ${ROWS} joined`;
}

export function joinedListOf(merge: number) {
  const joined = joinedAt(roundMerge(merge));
  if (!joined.length) return "none of the six";
  if (joined.length === ROWS) return "all six";
  return `${listOf(joined)} px`;
}

export function bladesOutOf(merge: number) {
  const out = joinedAt(roundMerge(merge)).length;
  if (!out) return "no blade is out";
  if (out === ROWS) return "all six blades are out";
  return out === 1 ? "one blade is out" : `${["", "", "two", "three", "four", "five"][out]} blades are out`;
}

export function spokenOf(merge: number) {
  const shown = roundMerge(merge);
  const joined = joinedAt(shown);
  const apart = GAPS.filter((gap) => gap >= shown);
  const head = `merge ${shown} px: ${joined.length} of ${ROWS} pairs joined`;
  if (!joined.length) return `${head}, all six apart`;
  if (!apart.length) return `${head}, every gap from ${GAPS[0]} to ${FAR} px`;
  return `${head}, gaps ${listOf(joined)} px; ${listOf(apart)} px ${apart.length > 1 ? "stay" : "stays"} apart`;
}

export const listText = listOf;
