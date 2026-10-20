import type { Traits } from "../traits";
export interface Body {
    cx: number;
    cy: number;
    rx: number;
    ry: number;
    n: number;
    rot: number;
    radii: number[];
    /** Polygon-only, set by the shapes that draw one. */
    sides?: number;
    round?: number;
}
export interface Ellipse {
    cx: number;
    cy: number;
    rx: number;
    ry: number;
}
export interface Deco {
    petals: {
        cx: number;
        cy: number;
        r: number;
    }[];
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
export declare const round: Shape;
export declare const organic: Shape;
/** `round`, squared off and tilted. Same path, different parameters. */
export declare const boxy: Shape;
export declare const capsule: Shape;
export declare const nub: Shape;
/** `organic`, with lobes on the upper half. */
export declare const cloud: Shape;
export declare const droplet: Shape;
export declare const hexagon: Shape;
export declare const sun: Shape;
/** `hexagon` with three sides, and a tighter tilt so it rests on its base. */
export declare const triangle: Shape;
//# sourceMappingURL=shapes.d.ts.map