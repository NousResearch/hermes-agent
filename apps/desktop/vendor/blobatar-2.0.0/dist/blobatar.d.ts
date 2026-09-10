import { type Animate } from "./animate";
import type { Palette } from "./color";
import type { Expression } from "./expression";
import { type BlobatarOptions } from "./render";
export type { BlobatarOptions, Animate, Expression };
/**
 * Renders a deterministic blobatar as SVG markup.
 *
 * The same name always produces the same output within a major version. The
 * numeric ranges in `styles/compose.ts`, the bands in `styles/blob.ts`, and the
 * tone set are all part of that contract. Changing them requires a new major.
 */
export declare const blobatar: (name: string, opts?: BlobatarOptions) => string;
/**
 * The `<svg>` contents and its motion custom properties, separately.
 *
 * For renderers that own the outer element themselves — `blobatar/react` when
 * animating. Underscored because the shape of this object is not public API.
 */
export declare function _parts(name: string, opts?: BlobatarOptions): {
    cls: string | undefined;
    bg: import("./render").Backdrop | undefined;
    inner: string;
    vars: Record<string, string> | undefined;
};
/**
 * The numeric layout and resolved palette, before serialization.
 *
 * Kept separate from rendering so geometric invariants — features staying
 * inside the body, the body staying inside the frame — can be asserted directly
 * rather than by parsing path data back out of the markup. Underscored because
 * the shape of this object is not public API.
 */
export declare function _layout(name: string, opts?: BlobatarOptions): {
    shape: string;
    draw: ((b: import("./styles/shapes").Body) => string) | undefined;
    body: import("./styles/shapes").Body;
    face: import("./styles/shapes").Ellipse;
    petals: {
        cx: number;
        cy: number;
        r: number;
    }[];
    extra: string[];
    eyes: import("./styles/compose").Eye[];
    palette: Palette;
};
//# sourceMappingURL=blobatar.d.ts.map