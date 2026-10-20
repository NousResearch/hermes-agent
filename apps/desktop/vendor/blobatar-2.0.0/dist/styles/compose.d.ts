/**
 * A Blobatar style composed from silhouette definitions.
 *
 * The band table chooses and weights silhouettes. Each silhouette owns its
 * geometry and safe face region; this module owns the shared body, eyes and SVG
 * serialization. The seam is private, so it can deepen when a future shape
 * proves the current definition insufficient without committing consumers to
 * its lifecycle.
 */
import type { Palette } from "../color";
import type { Traits } from "../traits";
import type { Body, Ellipse, Shape } from "./shapes";
export interface Eye {
    cx: number;
    cy: number;
    rx: number;
    ry: number;
    n: number;
    rot: number;
}
export type Fit = (t: Traits, b: Body, face: Ellipse) => Eye[];
/** Fits the eye cluster against the silhouette's face region on both axes. */
export declare const faceFit: Fit;
/** `[shape, upper edge of its band in [0, 1)]`, in order. */
export type Band = readonly [Shape, number];
export declare function compose(bands: Band[], fit: Fit): {
    layout: (t: Traits) => {
        shape: string;
        draw: ((b: Body) => string) | undefined;
        body: Body;
        face: Ellipse;
        petals: {
            cx: number;
            cy: number;
            r: number;
        }[];
        extra: string[];
        eyes: Eye[];
    };
    render: (l: ReturnType<(t: Traits) => {
        shape: string;
        draw: ((b: Body) => string) | undefined;
        body: Body;
        face: Ellipse;
        petals: {
            cx: number;
            cy: number;
            r: number;
        }[];
        extra: string[];
        eyes: Eye[];
    }>, p: Palette, mo?: boolean) => string;
    background: false;
};
/**
 * What a composed `layout` returns.
 *
 * `shape` is a `string` here rather than a union of names, because which names
 * are possible is a property of the band table and not of the composer. The
 * public `blob` entry narrows this to the package major's known vocabulary.
 */
export type Layout = ReturnType<ReturnType<typeof compose>["layout"]>;
//# sourceMappingURL=compose.d.ts.map