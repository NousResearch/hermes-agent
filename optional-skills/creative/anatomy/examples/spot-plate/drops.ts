import { type Flat, iso, pathOf, type Plane } from "../../kit/iso-kit";
import { BIG_Y, CLIMB, FLOOR, GAPS, LIGHT, P, RA, RB, THICK, VIEW, fieldOf, rowX, rowY, spanOf, type Field } from "./rig";

const STEP = 2;
const FINE = 0.5;
const FINE_HALF = 6;
const NUDGE = -0.05;
const KISS = 0.75;
const WINDOW = 4;
const BEND = 0.3;
const BEVEL = 2.2;
const TAPER = 10;
const SHADES = 4;
const STRIP = 8;
const SHINE_IN = 3.4;
const SHINE_FACING = 0.82;
const FADE_SHARE = 0.3;
const FADE_MOST = 9;
const FADE_STEPS = 6;

interface Grid {
  values: Float32Array;
  us: Float64Array;
  vs: Float64Array;
}

function ticksOf(centre: number, before: number, after: number) {
  const ticks: number[] = [];
  for (let k = -Math.ceil(before / STEP); k <= Math.ceil(after / STEP); k++) {
    if (Math.abs(k * STEP) > FINE_HALF) ticks.push(centre + k * STEP);
  }
  const fine = Math.round(FINE_HALF / FINE);
  for (let k = -fine; k <= fine; k++) ticks.push(centre + k * FINE);
  return Float64Array.from(ticks.sort((a, b) => a - b));
}

function sample(field: Field, gap: number): Grid {
  const middle = RA + gap / 2;
  const us = ticksOf(middle, middle + RA + 6, spanOf(gap) + RB + 6 - middle);
  const vs = ticksOf(0, RA + 6, RA + 6);
  const nx = us.length;
  const ny = vs.length;
  const values = new Float32Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    const mirror = ny - 1 - j;
    if (mirror < j) {
      values.copyWithin(j * nx, mirror * nx, mirror * nx + nx);
      continue;
    }
    const v = vs[j]!;
    for (let i = 0; i < nx; i++) values[j * nx + i] = field(us[i]!, v);
  }
  return { values, us, vs };
}

const SEGMENTS: Record<number, [number, number][]> = {
  1: [[3, 2]],
  2: [[2, 1]],
  3: [[3, 1]],
  4: [[0, 1]],
  6: [[0, 2]],
  7: [[3, 0]],
  8: [[3, 0]],
  9: [[0, 2]],
  11: [[0, 1]],
  12: [[3, 1]],
  13: [[2, 1]],
  14: [[3, 2]],
};

function rings({ values, us, vs }: Grid, level: number): Plane[][] {
  const nx = us.length;
  const ny = vs.length;
  const at = (i: number, j: number) => values[j * nx + i]!;
  const links = new Map<number, number[]>();
  const points = new Map<number, Plane>();
  const pointOf = (id: number) => {
    if (points.has(id)) return;
    const cell = id >> 1;
    const i = cell % nx;
    const j = (cell - i) / nx;
    const down = id & 1;
    const a = at(i, j);
    const b = down ? at(i, j + 1) : at(i + 1, j);
    const t = Math.min(Math.max((level - a) / (b - a), 0), 1);
    points.set(id, down ? [us[i]!, vs[j]! + (vs[j + 1]! - vs[j]!) * t] : [us[i]! + (us[i + 1]! - us[i]!) * t, vs[j]!]);
  };
  const link = (a: number, b: number) => {
    pointOf(a);
    pointOf(b);
    (links.get(a) ?? links.set(a, []).get(a)!).push(b);
    (links.get(b) ?? links.set(b, []).get(b)!).push(a);
  };
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const tl = at(i, j);
      const tr = at(i + 1, j);
      const br = at(i + 1, j + 1);
      const bl = at(i, j + 1);
      const code = (tl < level ? 8 : 0) | (tr < level ? 4 : 0) | (br < level ? 2 : 0) | (bl < level ? 1 : 0);
      if (code === 0 || code === 15) continue;
      const edges = [(j * nx + i) * 2, ((j * nx + i + 1) * 2) | 1, ((j + 1) * nx + i) * 2, ((j * nx + i) * 2) | 1];
      let pairs = SEGMENTS[code];
      if (!pairs) {
        const centre = (tl + tr + br + bl) / 4 < level;
        if (code === 5) pairs = centre ? [[3, 0], [2, 1]] : [[0, 1], [3, 2]];
        else pairs = centre ? [[0, 1], [3, 2]] : [[3, 0], [2, 1]];
      }
      for (const [a, b] of pairs) link(edges[a]!, edges[b]!);
    }
  }
  const seen = new Set<number>();
  const found: Plane[][] = [];
  for (const start of links.keys()) {
    if (seen.has(start)) continue;
    const ring: Plane[] = [];
    let previous = -1;
    let current = start;
    while (!seen.has(current)) {
      seen.add(current);
      ring.push(points.get(current)!);
      const next = links.get(current)!.find((candidate) => candidate !== previous && !seen.has(candidate));
      if (next === undefined) break;
      previous = current;
      current = next;
    }
    if (ring.length > 3) found.push(ring);
  }
  return found;
}

function simplify(points: Plane[], tolerance: number): Plane[] {
  if (points.length < 3) return points;
  const first = points[0]!;
  const last = points[points.length - 1]!;
  const dx = last[0] - first[0];
  const dy = last[1] - first[1];
  const length = Math.hypot(dx, dy) || 1;
  let worst = 0;
  let index = 0;
  for (let step = 1; step < points.length - 1; step++) {
    const point = points[step]!;
    const off = Math.abs((point[0] - first[0]) * dy - (point[1] - first[1]) * dx) / length;
    if (off > worst) {
      worst = off;
      index = step;
    }
  }
  if (worst <= tolerance) return [first, last];
  return [...simplify(points.slice(0, index + 1), tolerance).slice(0, -1), ...simplify(points.slice(index), tolerance)];
}

function simplifyRing(ring: Plane[], tolerance: number): Plane[] {
  let far = 0;
  let best = 0;
  for (let index = 1; index < ring.length; index++) {
    const distance = Math.hypot(ring[index]![0] - ring[0]![0], ring[index]![1] - ring[0]![1]);
    if (distance > best) {
      best = distance;
      far = index;
    }
  }
  const one = simplify(ring.slice(0, far + 1), tolerance);
  const two = simplify([...ring.slice(far), ring[0]!], tolerance);
  return [...one.slice(0, -1), ...two.slice(0, -1)];
}

function signedArea(points: Plane[] | Flat[]) {
  let area = 0;
  for (let index = 0; index < points.length; index++) {
    const [ax, ay] = points[index]!;
    const [bx, by] = points[(index + 1) % points.length]!;
    area += ax * by - bx * ay;
  }
  return area / 2;
}

const oriented = (points: Flat[]) => (signedArea(points) < 0 ? [...points].reverse() : points);

function runsOf(flags: boolean[], want: boolean) {
  const count = flags.length;
  const runs: number[][] = [];
  const start = flags.findIndex((flag, index) => flag !== flags[(index + count - 1) % count]);
  if (start < 0) return flags[0] === want ? [Array.from({ length: count + 1 }, (_, index) => index % count)] : [];
  let run: number[] | null = null;
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

export interface GlassShape {
  fill: string;
  shades: { d: string; shade: number }[];
  top: string;
  crease: string;
  bevel: string;
  outline: string;
  shine: string;
  depth: number;
}

const ease = (t: number) => t * t * (3 - 2 * t);

type Hidden = (x: number, y: number, z: number, edge: number) => boolean;

function hiddenBy(ring: Plane[], top: number): Hidden {
  const count = ring.length;
  const xs = new Float64Array(count);
  const ys = new Float64Array(count);
  for (let index = 0; index < count; index++) {
    xs[index] = ring[index]![0];
    ys[index] = ring[index]![1];
  }
  const inside = (x: number, y: number) => {
    let hit = false;
    for (let index = 0, previous = count - 1; index < count; previous = index++) {
      const ay = ys[index]!;
      const by = ys[previous]!;
      if (ay > y !== by > y && x < ((xs[previous]! - xs[index]!) * (y - ay)) / (by - ay) + xs[index]!) hit = !hit;
    }
    return hit;
  };
  // health: allow CC -- vendored upstream example (wheresryan22/anatomy), kept byte-close so it tracks upstream
  return (x, y, z, edge) => {
    const reach = (top - z) / CLIMB;
    if (reach <= 0) return false;
    if (inside(x + VIEW[0] * 0.05, y + VIEW[1] * 0.05)) return true;
    const fx = x + VIEW[0] * reach;
    const fy = y + VIEW[1] * reach;
    const left = Math.min(x, fx);
    const right = Math.max(x, fx);
    const low = Math.min(y, fy);
    const high = Math.max(y, fy);
    for (let index = 0; index < count; index++) {
      if (index === edge || (edge < 0 && (index === -edge - 1 || (index + 1) % count === -edge - 1))) continue;
      const next = index + 1 === count ? 0 : index + 1;
      const ax = xs[index]!;
      const ay = ys[index]!;
      const bx = xs[next]!;
      const by = ys[next]!;
      if ((ax < left && bx < left) || (ax > right && bx > right) || (ay < low && by < low) || (ay > high && by > high)) continue;
      const ex = bx - ax;
      const ey = by - ay;
      const cross = VIEW[0] * ey - VIEW[1] * ex;
      if (Math.abs(cross) < 1e-12) continue;
      const ox = ax - x;
      const oy = ay - y;
      const t = (ox * ey - oy * ex) / cross;
      const u = (ox * VIEW[1] - oy * VIEW[0]) / cross;
      if (t > 0.05 && t < reach && u >= 0 && u <= 1) return true;
    }
    return false;
  };
}

function visibleRuns(ring: Plane[], run: number[], z: number, hidden: Hidden): Flat[][] {
  const runs: Flat[][] = [];
  let line: Flat[] = [];
  const shown = run.map((index) => !hidden(ring[index]![0], ring[index]![1], z, -index - 1));
  const lift = ([x, y]: Plane) => iso([x, y, z], P);
  const cut = (edge: number, fromShown: boolean): Plane => {
    const a = ring[edge]!;
    const b = ring[(edge + 1) % ring.length]!;
    let low = 0;
    let high = 1;
    for (let round = 0; round < 14; round++) {
      const mid = (low + high) / 2;
      const seen = !hidden(a[0] + (b[0] - a[0]) * mid, a[1] + (b[1] - a[1]) * mid, z, edge);
      if (seen === fromShown) low = mid;
      else high = mid;
    }
    const t = (low + high) / 2;
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  };
  for (let index = 0; index < run.length; index++) {
    const point = ring[run[index]!]!;
    if (index > 0 && shown[index] !== shown[index - 1]) {
      const edge = cut(run[index - 1]!, shown[index - 1]!);
      if (shown[index - 1]) {
        line.push(lift(edge));
        if (line.length > 1) runs.push(line);
        line = [];
      } else {
        line = [lift(edge)];
      }
    }
    if (shown[index]) line.push(lift(point));
  }
  if (line.length > 1) runs.push(line);
  return runs;
}

// health: allow CC -- vendored upstream example (wheresryan22/anatomy), kept byte-close so it tracks upstream
function solidOf(ring: Plane[], z: number, height: number, occlude: boolean): GlassShape {
  const count = ring.length;
  const counter = signedArea(ring) > 0;
  const normals = ring.map(([ax, ay], index) => {
    const [bx, by] = ring[(index + 1) % count]!;
    const dx = bx - ax;
    const dy = by - ay;
    const length = Math.hypot(dx, dy) || 1;
    return (counter ? [dy / length, -dx / length] : [-dy / length, dx / length]) as Plane;
  });
  const facing = normals.map(([nx, ny]) => nx * VIEW[0] + ny * VIEW[1] > 1e-9);
  const top = ring.map(([x, y]) => iso([x, y, z + height], P));
  const bottom = ring.map(([x, y]) => iso([x, y, z], P));
  const hidden = occlude ? hiddenBy(ring, z + height) : null;

  const strips: { depth: number; shade: number; d: string }[] = [];
  let open: { from: number; to: number; shade: number } | null = null;
  const close = () => {
    if (!open) return;
    const upper: Flat[] = [];
    const lower: Flat[] = [];
    let depth = 0;
    for (let index = open.from; index <= open.to + 1; index++) {
      const point = index % count;
      upper.push(top[point]!);
      lower.push(bottom[point]!);
      depth += ring[point]![0] * VIEW[0] + ring[point]![1] * VIEW[1];
    }
    strips.push({ depth: depth / upper.length, shade: open.shade, d: pathOf(oriented([...upper, ...lower.reverse()]), true) });
    open = null;
  };
  for (let index = 0; index < count; index++) {
    if (!facing[index]) {
      close();
      continue;
    }
    const [nx, ny] = normals[index]!;
    const lit = (nx * LIGHT[0] + ny * LIGHT[1] + 1) / 2;
    const shade = Math.min(SHADES - 1, Math.max(0, Math.floor(lit * SHADES)));
    if (open && open.shade === shade && index - open.from < STRIP) open.to = index;
    else {
      close();
      open = { from: index, to: index, shade };
    }
  }
  close();
  strips.sort((a, b) => a.depth - b.depth);
  const shades: { d: string; shade: number }[] = [];
  for (const strip of strips) {
    const last = shades[shades.length - 1];
    if (last && last.shade === strip.shade) last.d += strip.d;
    else shades.push({ d: strip.d, shade: strip.shade });
  }

  const inward = (index: number, by: number): Plane => {
    const [ax, ay] = normals[(index + count - 1) % count]!;
    const [bx, by2] = normals[index]!;
    const mx = ax + bx;
    const my = ay + by2;
    const length = Math.hypot(mx, my) || 1;
    const [x, y] = ring[index]!;
    return [x - (mx / length) * by, y - (my / length) * by];
  };

  const crease: string[] = [];
  const outline: string[] = [];
  const bevel: string[] = [];
  for (const run of runsOf(facing, false)) outline.push(pathOf(run.map((index) => top[index]!)));
  for (const run of runsOf(facing, true)) {
    crease.push(pathOf(run.map((index) => top[index]!)));
    if (hidden) for (const seen of visibleRuns(ring, run, z, hidden)) outline.push(pathOf(seen));
    else outline.push(pathOf(run.map((index) => bottom[index]!)));
    const along = [0];
    for (let step = 1; step < run.length; step++) {
      const [ax, ay] = ring[run[step - 1]!]!;
      const [bx, by] = ring[run[step]!]!;
      along.push(along[step - 1]! + Math.hypot(bx - ax, by - ay));
    }
    const total = along[along.length - 1]!;
    bevel.push(
      pathOf(
        run.map((index, step) => {
          const near = Math.min(along[step]!, total - along[step]!);
          const [x, y] = inward(index, BEVEL * ease(Math.min(1, near / TAPER)));
          return iso([x, y, z + height], P);
        }),
      ),
    );
  }
  for (let index = 0; index < count; index++) {
    if (facing[(index + count - 1) % count] === facing[index]) continue;
    const [x, y] = ring[index]!;
    let from = z;
    if (hidden && hidden(x, y, z, -index - 1)) {
      let low = z;
      let high = z + height;
      for (let round = 0; round < 12; round++) {
        const mid = (low + high) / 2;
        if (hidden(x, y, mid, -index - 1)) low = mid;
        else high = mid;
      }
      from = high;
    }
    if (z + height - from > 0.05) outline.push(pathOf([top[index]!, iso([x, y, from], P)]));
  }

  const toward = normals.map(([nx, ny], index) => {
    const [px, py] = normals[(index + count - 1) % count]!;
    return ((nx + px) * LIGHT[0] + (ny + py) * LIGHT[1]) / 2 > SHINE_FACING;
  });
  const shine = runsOf(toward, true)
    .filter((run) => run.length > 3)
    .map((run) => pathOf(run.slice(1, -1).map((index) => iso([...inward(index, SHINE_IN), z + height] as [number, number, number], P))))
    .join("");

  const cx = ring.reduce((sum, [x]) => sum + x, 0) / count;
  const cy = ring.reduce((sum, [, y]) => sum + y, 0) / count;
  return {
    fill: [pathOf(oriented(top), true), pathOf(oriented(bottom), true)].join(""),
    shades,
    top: pathOf(oriented(top), true),
    crease: crease.join(""),
    bevel: bevel.join(""),
    outline: outline.join(""),
    shine,
    depth: cx * VIEW[0] + cy * VIEW[1],
  };
}

export interface Glint {
  d: string;
  alpha: number;
}

export interface RowShape {
  row: number;
  gap: number;
  joined: boolean;
  glass: GlassShape[];
  neck: Glint[];
}

const atTop = (points: Plane[]) => points.map(([x, y]) => iso([x, y, FLOOR + THICK], P));

function concaveRuns(ring: Plane[]): number[][] {
  const count = ring.length;
  const along = [0];
  for (let index = 0; index < count; index++) {
    const [ax, ay] = ring[index]!;
    const [bx, by] = ring[(index + 1) % count]!;
    along.push(along[index]! + Math.hypot(bx - ax, by - ay));
  }
  const total = along[count]!;
  const pointAt = (distance: number): Plane => {
    const s = ((distance % total) + total) % total;
    let low = 0;
    let high = count;
    while (high - low > 1) {
      const mid = (low + high) >> 1;
      if (along[mid]! <= s) low = mid;
      else high = mid;
    }
    const [ax, ay] = ring[low]!;
    const [bx, by] = ring[(low + 1) % count]!;
    const t = (s - along[low]!) / Math.max(along[low + 1]! - along[low]!, 1e-9);
    return [ax + (bx - ax) * t, ay + (by - ay) * t];
  };
  const turn = signedArea(ring) > 0 ? 1 : -1;
  const flags = ring.map(([x, y], index) => {
    const [ax, ay] = pointAt(along[index]! - WINDOW);
    const [bx, by] = pointAt(along[index]! + WINDOW);
    return turn * ((x - ax) * (by - y) - (y - ay) * (bx - x)) < -BEND;
  });
  return runsOf(flags, true).filter((run) => run.length > 2);
}

function glintsOf(points: Flat[]): Glint[] {
  const along = [0];
  for (let index = 1; index < points.length; index++) {
    along.push(along[index - 1]! + Math.hypot(points[index]![0] - points[index - 1]![0], points[index]![1] - points[index - 1]![1]));
  }
  const total = along[along.length - 1]!;
  if (total < 0.5) return [];
  const cut = (from: number, to: number): Flat[] => {
    const out: Flat[] = [];
    const place = (distance: number) => {
      let index = 1;
      while (index < along.length - 1 && along[index]! < distance) index++;
      const t = (distance - along[index - 1]!) / Math.max(along[index]! - along[index - 1]!, 1e-9);
      const [ax, ay] = points[index - 1]!;
      const [bx, by] = points[index]!;
      return [ax + (bx - ax) * t, ay + (by - ay) * t] as Flat;
    };
    out.push(place(from));
    for (let index = 0; index < points.length; index++) if (along[index]! > from && along[index]! < to) out.push(points[index]!);
    out.push(place(to));
    return out;
  };
  const fade = Math.min(total * FADE_SHARE, FADE_MOST);
  const glints: Glint[] = [];
  let before = 0;
  for (let layer = 0; layer < FADE_STEPS; layer++) {
    const now = layer === FADE_STEPS - 1 ? 1 : ease((layer + 1) / FADE_STEPS);
    const alpha = Math.round((1 - (1 - now) / (1 - before)) * 100) / 100;
    before = now;
    const inset = (fade * layer) / (FADE_STEPS - 1);
    glints.push({ d: pathOf(cut(inset, total - inset)), alpha });
  }
  return glints;
}

const cache = new Map<number, RowShape>();
const CACHE_MOST = 420;

export const keyOf = (merge: number) => Math.round(merge * 4);

export function rowOf(row: number, merge: number): RowShape {
  const gap = GAPS[row]!;
  const plain = gap >= 2 * merge;
  const key = row * 1000 + (plain ? 999 : keyOf(merge));
  const known = cache.get(key);
  if (known) return known;
  const exact = plain ? 0 : keyOf(merge) / 4;
  const x0 = rowX(row);
  const y0 = rowY(row) + BIG_Y;
  const toWorld = ([u, v]: Plane): Plane => [x0 + v, y0 + u];
  const field = fieldOf(gap, exact);
  const kiss = !plain && keyOf(merge) === gap * 4;
  const level = kiss ? field(RA + gap / 2, KISS) : NUDGE;
  const found = rings(sample(field, gap), level).map((ring) => simplifyRing(ring, 0.06));
  const joined = gap < exact && found.length === 1;
  const whole = found.length === 1;
  const glass = found.map((local) => solidOf(local.map(toWorld), FLOOR, THICK, whole)).sort((a, b) => a.depth - b.depth);
  const neck = joined ? concaveRuns(found[0]!.map(toWorld)).flatMap((run) => glintsOf(atTop(run.map((index) => toWorld(found[0]![index]!))))) : [];
  const shape: RowShape = { row, gap, joined, glass, neck };
  if (cache.size >= CACHE_MOST) cache.delete(cache.keys().next().value!);
  cache.set(key, shape);
  return shape;
}

export const rowsOf = (merge: number) => GAPS.map((_, row) => rowOf(row, merge));
