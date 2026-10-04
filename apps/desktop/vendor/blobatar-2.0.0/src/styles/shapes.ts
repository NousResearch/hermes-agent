/**
 * The private silhouette vocabulary.
 *
 * A shape is everything the layout needs to draw one silhouette and nothing
 * about when to draw it: how much of the frame its core body takes, how it
 * patches that body, what room it leaves the eyes, what it decorates with, and
 * which path primitive traces it. What it deliberately does *not* carry is its
 * threshold — how often it comes up is a property of the package major's band
 * table, not of the silhouette itself. See `blob.ts`.
 *
 * Shapes that are parameterizations of another share its implementation rather
 * than restating it: `boxy` is `round` with a squarer `n` and a tilt, `hexagon`
 * is `triangle` with six sides, `cloud` is `organic` with lobes. They share it
 * by naming the same function (`path: spline`), never by spreading the parent
 * value — `{ ...round, core: 0.86 }` reads better and defeats tree-shaking
 * completely, measured at ~920 B for a single shape. See ADR-0007.
 *
 * These are Blobatar 2's ten silhouettes. `test/geometry.test.ts` asserts the
 * eye cluster stays inside the body and the body inside the frame for all of
 * them.
 */
import { blobPath, box, polygon, taper } from "../shape";
import type { Traits } from "../traits";

export interface Body {
  cx: number; cy: number; rx: number; ry: number;
  n: number; rot: number; radii: number[];
  /** Polygon-only, set by the shapes that draw one. */
  sides?: number; round?: number;
}
export interface Ellipse { cx: number; cy: number; rx: number; ry: number }
export interface Deco {
  petals: { cx: number; cy: number; r: number }[];
  /** Extra outlines unioned with the core body, already traced. */
  extra: string[];
}

export interface Shape {
  name: string;
  /** How much of the frame the core body takes. */
  core: number;
  /** Patches the body before the face is measured. */
  body?(t: Traits, b: Body): void;
  /**
   * The region the eyes must fit inside. Omitted, it is the body itself —
   * which is what every silhouette that is convex around its own centre wants,
   * and half the roster is.
   */
  face?(b: Body): Ellipse;
  decorate?(t: Traits, b: Body, out: Deco): void;
  /**
   * The core path. Omitted, `superellipse` — free to default to, because the
   * eyes are superellipses too, so it is in every bundle already.
   */
  path?(b: Body): string;
}

// ---------------------------------------------------------------------------
// The shared implementations. Five path calls serve all ten shapes.

const poly = (b: Body) => polygon(b as Body & { sides: number; round: number });
const spline = (b: Body) => blobPath(b.cx, b.cy, b.rx, b.ry, b.radii, b.rot);

const shrunk = (k: number) => (b: Body): Ellipse => ({
  cx: b.cx, cy: b.cy, rx: b.rx * k, ry: b.ry * k,
});
const splineFace = (b: Body): Ellipse => shrunk(Math.min(...b.radii) * 0.95)(b);
const polyFace = (b: Body): Ellipse => shrunk(0.84)(b);

// ---------------------------------------------------------------------------

export const round: Shape = { name: "round", core: 1 };

export const organic: Shape = {
  name: "organic", core: 0.98, path: spline, face: splineFace,
};

/** `round`, squared off and tilted. Same path, different parameters. */
export const boxy: Shape = {
  name: "boxy", core: 0.86,
  body: (t, b) => {
    b.n = t.num("body.n", 3.4, 6);
    b.rot = t.num("body.rot", -20, 20);
  },
};

export const capsule: Shape = {
  name: "capsule", core: 1.02,
  body: (t, b) => { b.ry *= t.num("capsule.squat", 0.55, 0.68); },
  face: shrunk(0.94),
  decorate: (_t, b, out) => {
    for (const s of [-1, 1]) out.petals.push({ cx: b.cx + s * (b.rx - b.ry), cy: b.cy, r: b.ry });
  },
  path: b => box(b.cx, b.cy, b.rx - b.ry, b.ry),
};

export const nub: Shape = {
  name: "nub", core: 0.88,
  decorate: (t, b, out) => {
    const count = t.int("nub.n", 1, 2);
    for (let i = 0; i < count; i++) {
      const a = t.num(`nub.a${i}`, 0, 2 * Math.PI);
      out.petals.push({
        cx: b.cx + Math.cos(a) * b.rx * 0.88,
        cy: b.cy + Math.sin(a) * b.rx * 0.88,
        r: b.rx * t.num(`nub.r${i}`, 0.24, 0.4),
      });
    }
  },
};

/** `organic`, with lobes on the upper half. */
export const cloud: Shape = {
  name: "cloud", core: 0.78, face: splineFace, path: spline,
  decorate: (t, b, out) => {
    const count = t.int("cloud.n", 4, 6);
    for (let i = 0; i < count; i++) {
      const a = Math.PI + (Math.PI * (i + 0.5)) / count;
      out.petals.push({
        cx: b.cx + Math.cos(a) * b.rx * 0.8,
        cy: b.cy + Math.sin(a) * b.rx * 0.5,
        r: b.rx * t.num(`cloud.r${i}`, 0.44, 0.62),
      });
    }
  },
};

export const droplet: Shape = {
  name: "droplet", core: 0.78,
  // Shifted down by what the taper adds above, so the whole silhouette — head
  // and point together — sits centred in the frame rather than the head alone.
  // `n` is pinned to a true ellipse, which is the curve the taper is tangent to.
  body: (_t, b) => { b.cy += 0.22 * b.ry; b.n = 2; },
  face: b => ({ cx: b.cx, cy: b.cy + b.ry * 0.05, rx: b.rx * 0.88, ry: b.ry * 0.88 }),
  decorate: (t, b, out) => {
    out.extra.push(taper(b.cx, b.cy, b.rx, b.ry, t.num("droplet.tip", 1.4, 1.65)));
  },
};

export const hexagon: Shape = {
  name: "hexagon", core: 1.05, path: poly, face: polyFace,
  body: (t, b) => {
    b.sides = 6;
    b.rot = t.num("body.rot", -12, 12);
    b.round = t.num("poly.round", 0.24, 0.5);
  },
};

export const sun: Shape = {
  name: "sun", core: 0.7,
  decorate: (t, b, out) => {
    const count = t.int("sun.n", 6, 9);
    const dist = b.rx * t.num("sun.dist", 1.0, 1.08);
    const pr = b.rx * t.num("sun.r", 0.2, 0.26);
    const off = t.num("sun.rot", 0, 2 * Math.PI);
    for (let i = 0; i < count; i++) {
      const a = off + (2 * Math.PI * i) / count;
      out.petals.push({ cx: b.cx + Math.cos(a) * dist, cy: b.cy + Math.sin(a) * dist, r: pr });
    }
  },
};

/** `hexagon` with three sides, and a tighter tilt so it rests on its base. */
export const triangle: Shape = {
  name: "triangle", core: 1.15, path: poly,
  body: (t, b) => {
    b.sides = 3;
    b.rot = t.num("body.rot", -5, 5);
    b.round = t.num("poly.round", 0.24, 0.5);
  },
  face: b => ({ cx: b.cx, cy: b.cy + b.ry * 0.1, rx: b.rx * 0.54, ry: b.ry * 0.36 }),
};
