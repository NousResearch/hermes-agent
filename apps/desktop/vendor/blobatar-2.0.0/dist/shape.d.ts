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
/**
 * Approximates each quadrant with one cubic Bézier.
 *
 * The control offset is chosen so the curve passes exactly through the
 * superellipse's 45° point: B(0.5) = a(4+3k)/8 must equal a·2^(-1/n).
 * At n=2 this yields 0.5523 — the standard circle constant — which is a good
 * sign the derivation is right. Four segments instead of a 24-point sampled
 * polyline keeps each shape at ~130 bytes of path data.
 */
export declare function superellipse({ cx, cy, rx, ry, n, rot }: Superellipse): string;
/**
 * A quadratic arc, stroked — used only for smiles and frowns, where a closed
 * superellipse would need a boolean subtraction to get the same read.
 */
export declare function arc(cx: number, cy: number, w: number, depth: number): string;
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
export declare function blobPath(cx: number, cy: number, rx: number, ry: number, radii: number[], rot?: number): string;
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
export declare function polygon({ cx, cy, rx, ry, sides, round, rot }: Polygon): string;
/**
 * The straight run of a capsule, as a plain box.
 *
 * Drawn with the two cap circles the capsule already decorates with, the union
 * is an exact stadium: the box reaches full height everywhere, so each cap
 * meets it along its own diameter and there is no crease. A superellipse cannot
 * stand in — its corners round by a fraction of the whole radius, so it pinches
 * away from the caps and the join shows.
 */
export declare function box(cx: number, cy: number, rx: number, ry: number): string;
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
export declare function taper(cx: number, cy: number, rx: number, ry: number, tip: number): string;
//# sourceMappingURL=shape.d.ts.map