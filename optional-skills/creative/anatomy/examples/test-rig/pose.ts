import { insetPlan, onTop, pathOf, planOutline, slabOf, type Flat, type SolidPaths } from "../../kit/iso-kit";
import { carried, type Pose } from "./rig";
import { AZIMUTH, BUSH_END, COIL_R, COIL_TURNS, DIAL, DIAL_SPAN, DIAL_TOP, DIAL_R, FRAME_X, GLASS_H, GLASS_TOP, GLASS_Z, P, ROD_Y, ROD_Z, at, shift } from "./view";

export interface GlassPaths extends SolidPaths {
  rim: string;
  shine: string;
}

export interface Frame {
  glass: GlassPaths;
  coilsBack: string;
  coilsLeft: string;
  coilsRightBack: string;
  coilsRightFront: string;
  carriage: string;
  needle: string;
  grab: Flat;
  finger: Flat;
  pull: string;
  rider: string;
  probe: string;
  tip: string;
}

const RADIANS = Math.PI / 180;
const SIN_A = Math.sin(AZIMUTH * RADIANS);
const COS_E = Math.cos(30 * RADIANS);
const SIN_E = Math.sin(30 * RADIANS);
const PER_TURN = 16;
const tenth = (value: number) => Math.round(value * 10) / 10;

export function glassOf(pose: Pose): GlassPaths {
  const [hx, hy] = pose.half;
  const plan = { x: pose.mid - hx, y: -hy, w: hx * 2, d: hy * 2, r: hy };
  const paths = slabOf(plan, GLASS_Z, GLASS_H, P, 10, 1.2);
  const inner = insetPlan(plan, 3.2);
  const shine = [
    onTop([[pose.mid - hx * 0.62, -hy + 5.4], [pose.mid + hx * 0.18, -hy + 5.4]], GLASS_TOP, P),
    onTop([[pose.mid + hx * 0.3, -hy + 5.4], [pose.mid + hx * 0.42, -hy + 5.4]], GLASS_TOP, P),
  ].join("");
  return { ...paths, rim: planOutline(inner, GLASS_TOP, P, 10), shine };
}

function coil(from: number, to: number, y: number) {
  const count = COIL_TURNS * PER_TURN;
  const back: Flat[][] = [];
  const front: Flat[][] = [];
  let run: Flat[] = [];
  let facing: boolean | null = null;
  for (let index = 0; index <= count; index++) {
    const share = index / count;
    const angle = share * COIL_TURNS * Math.PI * 2;
    const cy = Math.cos(angle);
    const cz = Math.sin(angle);
    const point = at([from + (to - from) * share, y + COIL_R * cy, ROD_Z + COIL_R * cz]);
    const near = cy * SIN_A * COS_E + cz * SIN_E > 0;
    if (facing === null) facing = near;
    if (near !== facing) {
      run.push(point);
      (facing ? front : back).push(run);
      run = [point];
      facing = near;
      continue;
    }
    run.push(point);
  }
  if (run.length > 1) (facing ? front : back).push(run);
  return { back: back.map((points) => pathOf(points)).join(""), front: front.map((points) => pathOf(points)).join("") };
}

function needleOf(lean: number) {
  const angle = (-lean / DIAL_SPAN) * Math.PI * 2;
  const ux = Math.sin(angle);
  const uy = -Math.cos(angle);
  const [cx, cy] = DIAL;
  const point = (along: number, across: number) => at([cx + ux * along - uy * across, cy + uy * along + ux * across, DIAL_TOP + 0.1]);
  return pathOf([point(DIAL_R - 3.4, 0), point(0, 0.75), point(-3, 0.35), point(-3, -0.35), point(0, -0.75)], true);
}

export function frameOf(pose: Pose, lean: number, grab: number, finger: number, lift: number, home: number, held = false): Frame {
  const left = ROD_Y;
  const coils = [
    coil(-FRAME_X, -BUSH_END + lean, -left),
    coil(-FRAME_X, -BUSH_END + lean, left),
    coil(BUSH_END + lean, FRAME_X, -left),
    coil(BUSH_END + lean, FRAME_X, left),
  ];
  const grabX = carried(pose, grab);
  const grabAt = at([grabX, 0, GLASS_TOP]);
  const fingerAt = at([finger, 0, GLASS_TOP + lift]);
  return {
    glass: glassOf(pose),
    coilsBack: coils.map((one) => one.back).join(""),
    coilsLeft: coils[0]!.front + coils[1]!.front,
    coilsRightBack: coils[2]!.front,
    coilsRightFront: coils[3]!.front,
    carriage: shift(lean),
    needle: needleOf(lean),
    grab: [tenth(grabAt[0]), tenth(grabAt[1])],
    finger: [tenth(fingerAt[0]), tenth(fingerAt[1])],
    pull: !held || Math.abs(grabX - finger) < 0.6 ? "" : pathOf([grabAt, fingerAt]),
    rider: shift(grab),
    probe: shift(finger - home),
    tip: shift(finger - home, lift),
  };
}
