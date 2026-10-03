/**
 * The single primitive.
 *
 * |x/a|^n + |y/b|^n = 1 covers the whole part vocabulary: n=2 is an ellipse
 * (eyes, pupils), n≈4 a squircle (head, background), n→large a rectangle
 * (brows, mouth lines). One shape function, one continuous knob, so "head
 * shape" is a numeric trait rather than a set of hand-drawn alternatives.
 */

export interface Superellipse {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  /** Squareness. Useful range is roughly 1.6 (soft diamond) to 8 (near-rect). */
  n?: number;
  /** Degrees, clockwise. Baked into the coordinates so the SVG needs no transform. */
  rot?: number;
}

const r2 = (v: number) => {
  const s = Math.round(v * 100) / 100;
  return Object.is(s, -0) ? "0" : String(s);
};

/**
 * Approximates each quadrant with one cubic Bézier.
 *
 * The control offset is chosen so the curve passes exactly through the
 * superellipse's 45° point: B(0.5) = a(4+3k)/8 must equal a·2^(-1/n).
 * At n=2 this yields 0.5523 — the standard circle constant — which is a good
 * sign the derivation is right. Four segments instead of a 24-point sampled
 * polyline keeps each shape at ~130 bytes of path data.
 */
export function superellipse({ cx, cy, rx, ry, n = 4, rot = 0 }: Superellipse): string {
  // Above n≈5.55 the control offset exceeds the radius, and the curve bulges
  // outside the bounding box instead of squaring off — an inflated-looking
  // corner rather than a sharper one. Clamping k trades exactness at the 45°
  // point for a shape that always stays within its stated bounds; past that
  // point a superellipse is visually a rounded rect anyway.
  const k = Math.min(1, (8 * Math.pow(2, -1 / n) - 4) / 3);
  const a = rx;
  const b = ry;
  const ak = a * k;
  const bk = b * k;

  // Anchor, control, control — walking the four quadrants.
  const pts: [number, number][] = [
    [a, 0],
    [a, bk], [ak, b], [0, b],
    [-ak, b], [-a, bk], [-a, 0],
    [-a, -bk], [-ak, -b], [0, -b],
    [ak, -b], [a, -bk], [a, 0],
  ];

  const t = (rot * Math.PI) / 180;
  const cos = Math.cos(t);
  const sin = Math.sin(t);
  const at = (i: number) => {
    const [x, y] = pts[i]!;
    return `${r2(cx + x * cos - y * sin)} ${r2(cy + x * sin + y * cos)}`;
  };

  let d = `M${at(0)}`;
  for (let i = 1; i < 13; i += 3) d += `C${at(i)} ${at(i + 1)} ${at(i + 2)}`;
  return d + "Z";
}

/**
 * A quadratic arc, stroked — used only for smiles and frowns, where a closed
 * superellipse would need a boolean subtraction to get the same read.
 */
export function arc(cx: number, cy: number, w: number, depth: number): string {
  return `M${r2(cx - w)} ${r2(cy)}Q${r2(cx)} ${r2(cy + depth)} ${r2(cx + w)} ${r2(cy)}`;
}

/**
 * An organic closed curve: radii sampled around a circle, joined by a closed
 * Catmull-Rom spline converted to cubic Béziers.
 *
 * The superellipse handles everything symmetric; this handles everything that
 * needs to look hand-drawn. `radii` are multipliers of the base radius, one per
 * vertex, so a seed perturbing them by ±15% produces the lopsided pebble shapes
 * without any noise function — the vertex count alone controls how lumpy it is.
 *
 * Catmull-Rom rather than a Bézier fit because it interpolates its points
 * exactly, so the radii mean what they say and containment stays predictable.
 */
export function blobPath(
  cx: number,
  cy: number,
  rx: number,
  ry: number,
  radii: number[],
  rot = 0,
): string {
  const n = radii.length;
  const t0 = (rot * Math.PI) / 180;
  const p: [number, number][] = radii.map((m, i) => {
    const a = t0 + (2 * Math.PI * i) / n;
    return [cx + rx * m * Math.cos(a), cy + ry * m * Math.sin(a)];
  });

  const at = (i: number) => p[((i % n) + n) % n]!;
  let d = `M${r2(at(0)[0])} ${r2(at(0)[1])}`;

  for (let i = 0; i < n; i++) {
    const [x0, y0] = at(i - 1);
    const [x1, y1] = at(i);
    const [x2, y2] = at(i + 1);
    const [x3, y3] = at(i + 2);
    d +=
      `C${r2(x1 + (x2 - x0) / 6)} ${r2(y1 + (y2 - y0) / 6)}` +
      ` ${r2(x2 - (x3 - x1) / 6)} ${r2(y2 - (y3 - y1) / 6)}` +
      ` ${r2(x2)} ${r2(y2)}`;
  }

  return d + "Z";
}

export interface Polygon {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  /** How many sides. 3 is a triangle, 6 a hexagon. */
  sides: number;
  /**
   * Corner rounding, 0 (sharp) to 1 (every edge cut back to its own midpoint, so
   * the outline is all curve and no straight run).
   */
  round?: number;
  /** Degrees, clockwise. 0 puts a vertex at the top. */
  rot?: number;
}

/**
 * A regular polygon with rounded corners.
 *
 * The third primitive, and it is here because neither of the other two reaches
 * a flat-sided shape. `superellipse` interpolates between an ellipse and a
 * rectangle and has no odd-sided member at all — its n knob cannot produce a
 * triangle — and `blobPath` is a Catmull-Rom spline, which passes through its
 * vertices smoothly and so rounds a corner away rather than turning it.
 *
 * Corners are cut back along both adjoining edges by `round` and joined with a
 * quadratic through the vertex itself, which puts the whole outline inside the
 * polygon's convex hull for free: a quadratic never leaves the triangle of its
 * three points. At `round: 1` the cuts meet at the edge midpoints, the straight
 * runs vanish, and what is left is a smooth n-lobed shape rather than a polygon
 * — which is the top of the useful range, not a degenerate case.
 *
 * `rx` and `ry` are the circumradius on each axis, so the shape squashes with
 * the body like everything else here rather than staying stubbornly regular.
 */
export function polygon({ cx, cy, rx, ry, sides, round = 0.3, rot = 0 }: Polygon): string {
  // Halved because the cut is taken from both ends of every edge: at round = 1
  // each end reaches the midpoint and they meet exactly, so anything above that
  // would have the two cuts cross and the outline fold back on itself.
  const k = round > 0 ? (round < 1 ? round / 2 : 0.5) : 0;
  // −90° so a vertex sits at the top: a triangle points up and rests on a flat
  // edge, which is the orientation anybody who asks for a triangle means.
  const t0 = (rot * Math.PI) / 180 - Math.PI / 2;
  const v: [number, number][] = Array.from({ length: sides }, (_, i) => {
    const a = t0 + (2 * Math.PI * i) / sides;
    return [cx + rx * Math.cos(a), cy + ry * Math.sin(a)];
  });

  const at = (i: number) => v[((i % sides) + sides) % sides]!;
  /** The cut point on the edge leaving vertex `i` toward vertex `j`. */
  const cut = (i: number, j: number) => {
    const [x0, y0] = at(i);
    const [x1, y1] = at(j);
    return `${r2(x0 + (x1 - x0) * k)} ${r2(y0 + (y1 - y0) * k)}`;
  };

  let d = `M${cut(0, -1)}`;
  for (let i = 0; i < sides; i++) {
    const [x, y] = at(i);
    d += `Q${r2(x)} ${r2(y)} ${cut(i, i + 1)}`;
    // The straight run to the next corner's cut. Omitted when the cuts meet, so
    // a fully rounded polygon does not emit `sides` zero-length lines.
    if (k < 0.5) d += `L${cut(i + 1, i)}`;
  }
  return d + "Z";
}

/**
 * The straight run of a capsule, as a plain box.
 *
 * Drawn with the two cap circles the capsule already decorates with, the union
 * is an exact stadium: the box reaches full height everywhere, so each cap
 * meets it along its own diameter and there is no crease. A superellipse cannot
 * stand in — its corners round by a fraction of the whole radius, so it pinches
 * away from the caps and the join shows.
 */
export function box(cx: number, cy: number, rx: number, ry: number): string {
  const l = r2(cx - rx);
  const r = r2(cx + rx);
  return `M${l} ${r2(cy - ry)}H${r}V${r2(cy + ry)}H${l}Z`;
}

/**
 * The taper of a droplet: the two tangents from an apex to the body ellipse.
 *
 * Drawn with that ellipse, the union is a teardrop. A tangent meets the curve
 * without a corner, so the taper grows out of the head at every `tip` rather
 * than being stuck on — which is the whole reason this takes the body's radii
 * instead of drawing a cone of its own. `tip` is how far the apex sits above
 * the centre in units of `ry`, and so also how far down the sides the flanks
 * take hold: a taller apex has its tangent points further round.
 *
 * The point is eased with a quadratic through the apex, so the drawn tip stops
 * just short of it — no needle at small sizes, and `tip` bounds the silhouette
 * rather than touching it.
 */
export function taper(cx: number, cy: number, rx: number, ry: number, tip: number): string {
  // In the circle the ellipse is an affine image of, the tangent points sit at
  // angle acos(1/tip) from the apex direction. Affine maps preserve tangency,
  // so scaling those two points by rx and ry is exact, not an approximation.
  const t = Math.max(1.05, tip);
  const tx = rx * Math.sqrt(1 - 1 / (t * t));
  const ty = cy - ry / t;
  const apex = cy - t * ry;
  // How far up each flank the eased point takes over. Small, so the flanks stay
  // straight enough to read as a taper.
  const px = tx * 0.14;
  const py = ty + 0.86 * (apex - ty);
  return (
    `M${r2(cx - tx)} ${r2(ty)}` +
    `L${r2(cx - px)} ${r2(py)}` +
    `Q${r2(cx)} ${r2(apex)} ${r2(cx + px)} ${r2(py)}` +
    `L${r2(cx + tx)} ${r2(ty)}Z`
  );
}
