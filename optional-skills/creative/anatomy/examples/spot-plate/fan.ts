import { extrude, iso, pathOf, type Plane } from "../../kit/iso-kit";
import { BLADE_LONG, BLADE_Z, DECK_H, GAPS, LEAD_H, LEAD_LONG, P, PIVOT, ROWS, angleOf, thicknessOf } from "./rig";

const RADIANS = Math.PI / 180;
const BUTT = 7.2;
const TIP = 4.2;
const LEAD_TIP = 3.2;
const ARC = 10;

export interface BladePose {
  fill: string;
  shades: string[];
  top: string;
  crease: string;
  outline: string;
  line: string;
}

export interface FanPose {
  blades: BladePose[];
  halo: string;
  out: boolean[];
}

function outlineOf(length: number, tip: number): Plane[] {
  const out: Plane[] = [];
  for (let step = 0; step <= ARC; step++) {
    const angle = -90 + (180 * step) / ARC;
    out.push([length + tip * Math.cos(angle * RADIANS), tip * Math.sin(angle * RADIANS)]);
  }
  for (let step = 0; step <= ARC; step++) {
    const angle = 90 + (180 * step) / ARC;
    out.push([BUTT * Math.cos(angle * RADIANS), BUTT * Math.sin(angle * RADIANS)]);
  }
  return out;
}

const BLADE = outlineOf(BLADE_LONG, TIP);
const LEAD = outlineOf(LEAD_LONG, LEAD_TIP);

function place(shape: Plane[], degrees: number): Plane[] {
  const a = degrees * RADIANS;
  const dx = Math.sin(a);
  const dy = -Math.cos(a);
  const nx = Math.cos(a);
  const ny = Math.sin(a);
  return shape.map(([u, v]) => [PIVOT[0] + u * dx + v * nx, PIVOT[1] + u * dy + v * ny]);
}

function along(degrees: number, from: number, to: number, z: number) {
  const a = degrees * RADIANS;
  const dx = Math.sin(a);
  const dy = -Math.cos(a);
  return pathOf([iso([PIVOT[0] + from * dx, PIVOT[1] + from * dy, z], P), iso([PIVOT[0] + to * dx, PIVOT[1] + to * dy, z], P)]);
}

function bladeAt(shape: Plane[], degrees: number, z: number, height: number, length: number): BladePose {
  const ring = place(shape, degrees);
  const solid = extrude(ring, z, height, P, { convex: true });
  return {
    fill: solid.fill,
    shades: solid.shades ?? [],
    top: solid.top,
    crease: solid.crease,
    outline: solid.outline,
    line: along(degrees, 13, length - 6, z + height),
  };
}

export function fanOf(merge: number): FanPose {
  const lead = angleOf(merge);
  const blades: BladePose[] = [];
  const halo: string[] = [];
  const out: boolean[] = [];
  GAPS.forEach((gap, index) => {
    const angle = Math.min(lead, angleOf(gap));
    out.push(gap < merge);
    blades.push(bladeAt(BLADE, angle, BLADE_Z[index]!, thicknessOf(gap), BLADE_LONG));
    halo.push(pathOf(place(BLADE, angle).map(([x, y]) => iso([x, y, DECK_H], P)), true));
  });
  blades.push(bladeAt(LEAD, lead, BLADE_Z[ROWS]!, LEAD_H, LEAD_LONG));
  halo.push(pathOf(place(LEAD, lead).map(([x, y]) => iso([x, y, DECK_H], P)), true));
  return { blades, halo: halo.join(""), out };
}
