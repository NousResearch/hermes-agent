import { iso, type Flat, type Plan, type Point3, type Projection } from "../../kit/iso-kit";

export const WIDTH = 600;
export const HEIGHT = 340;
const PAD = 34;
export const AZIMUTH = 57;

export const BASE: Plan = { x: -134, y: -82, w: 298, d: 150, r: 9 };
export const BASE_H = 9;
export const FOOT_H = 5;

export const FRAME_X = 101;
export const FRAME_Y = 37;
export const BAR = 9;
export const BAR_H = 12;

export const ROD_Y = 18;
export const ROD_Z = 6;
export const ROD_HALF = 2;
export const COIL_R = 4.2;
export const COIL_TURNS = 6;

export const CARRIAGE: Plan = { x: -72, y: -26, w: 144, d: 52, r: 4 };
export const CARRIAGE_Z = 3;
export const CARRIAGE_H = 5;
export const CARRIAGE_TOP = CARRIAGE_Z + CARRIAGE_H;
export const BUSH_END = 80;
export const BUSH_W = 14;
export const BUSH_D = 12;

export const GLASS_Z = CARRIAGE_TOP;
export const GLASS_H = 10;
export const GLASS_TOP = GLASS_Z + GLASS_H;

export const PLUNGER_Z = 6;
export const DIAL: Flat = [131, 0];
export const DIAL_R = 15;
export const DIAL_Z = 3;
export const DIAL_H = 8;
export const DIAL_TOP = DIAL_Z + DIAL_H;
export const DIAL_SPAN = 20;

export const RAIL_Y = -66;
export const RAIL_D = 8;
export const RAIL_Z = 50;
export const RAIL_H = 8;
export const POST_L = -124;
export const POST_R = 66;
export const POST = 12;

export const PROBE_HOME = 45;
export const PROBE_MIN = -98;
export const PROBE_MAX = 48;
export const HOUSING_R = 5.5;
export const HOUSING_Z = 44;
export const HOUSING_H = 20;
export const PAD_R = 4.4;
export const PAD_H = 3;
export const HOVER = 12;
export const STEM = 28;

export const GRAB_MOST = 52;
export const FINGER_MOST = 62;

function fit(points: Point3[]): Projection {
  const unit: Projection = { origin: [0, 0], scale: 1, azimuth: AZIMUTH };
  const flat = points.map((point) => iso(point, unit));
  const xs = flat.map(([x]) => x);
  const ys = flat.map(([, y]) => y);
  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const up = Math.min(...ys);
  const down = Math.max(...ys);
  const scale = Math.min((WIDTH - PAD * 2) / (right - left), (HEIGHT - PAD * 2) / (down - up));
  return { origin: [WIDTH / 2 - ((left + right) / 2) * scale, HEIGHT / 2 - ((up + down) / 2) * scale], scale, azimuth: AZIMUTH };
}

const corners = ({ x, y, w, d }: Plan, z0: number, z1: number): Point3[] => {
  const out: Point3[] = [];
  for (const px of [x, x + w]) for (const py of [y, y + d]) for (const z of [z0, z1]) out.push([px, py, z]);
  return out;
};

export const P = fit([
  ...corners(BASE, -BASE_H - FOOT_H, 0),
  ...corners({ x: POST_L, y: RAIL_Y - RAIL_D / 2, w: POST_R + POST - POST_L, d: RAIL_D, r: 0 }, RAIL_Z, RAIL_Z + RAIL_H + 6),
]);

export const at = (point: Point3) => iso(point, P);

const ORIGIN = at([0, 0, 0]);
const minus = (a: Flat, b: Flat): Flat => [a[0] - b[0], a[1] - b[1]];
export const EX = minus(at([1, 0, 0]), ORIGIN);
export const EZ = minus(at([0, 0, 1]), ORIGIN);

const fine = (value: number) => Math.round(value * 100) / 100;

export const shift = (x: number, z = 0) => `translate(${fine(EX[0] * x + EZ[0] * z)} ${fine(EX[1] * x + EZ[1] * z)})`;
