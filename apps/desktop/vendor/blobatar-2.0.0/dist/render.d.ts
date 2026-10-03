import type { Animate } from "./animate";
import { type Palette } from "./color";
import type { Expression } from "./expression";
import { type TraitOverrides, type Traits } from "./traits";
export interface BlobatarOptions {
    /** Emits width/height attributes. Omit to let CSS size it (the viewBox always scales). */
    size?: number;
    /** Overrides the default backdrop. `false` renders transparent. */
    background?: boolean | "square" | "circle" | "squircle";
    /** Overrides specific palette entries. Overridden colors bypass the contrast guarantee. */
    palette?: Palette;
    /** Locks the hue in degrees, so the name drives shape only. */
    hue?: number;
    /** Locks the tone as a 0–1 position in the swatch set. */
    tone?: number;
    /**
     * Pins individual traits, so the name drives only what you leave out.
     *
     * Each value is the 0–1 position the hash would have produced for that key —
     * the same units the layout reads, so `{ "eye.gap": 1 }` is the top of
     * whatever range `eye.gap` is declared over rather than a measurement in
     * viewBox units. Values outside [0, 1) are clamped.
     *
     * ```ts
     * // Always a sun, always wide eyes — colour and everything else per name.
     * blobatar(user.email, { traits: { shape: 0.95, "eye.ratio": 0 } })
     * ```
     *
     * Pin every trait and the name stops mattering, which is how you build one
     * fixed blobatar: pass any constant string alongside a full map.
     *
     * The layout still runs in full, so the containment guarantees hold under any
     * combination — an eye cluster that would not fit is scaled down by `fit`
     * exactly as a hashed one is. That also means an extreme value can land short
     * of where you asked; `_layout` reports what it resolved to.
     *
     * Overlaps with `hue` and `tone`, which state the same two traits in friendlier
     * units. Those win: `hue` is degrees, `traits.hue` is a 0–1 position.
     */
    traits?: TraitOverrides;
    /** Applies NFC + trim + lowercase to the name. Default true. */
    normalize?: boolean;
    /** Enforces the minimum contrast ratios. Default true. */
    contrast?: boolean;
    /** Adds a <title> for screen readers. */
    title?: string;
    /**
     * Idle animation. Off by default.
     *
     * Requires `import "blobatar/motion.css"`, and requires the blobatar to be
     * inline SVG — content inside an `<img>` is an isolated document that hover
     * cannot reach. `blobatar/react` switches rendering mode for you; the string
     * API is already inline.
     *
     * **Honored by `blobatar/react` only, for now.** `blobatar()` returns static
     * markup regardless: a branch on `animate` inside it keeps the motion module
     * alive for every caller, animating or not, which measured at ~190 B. An
     * animated string API wants its own entry point, not a branch here.
     */
    animate?: Animate;
    /**
     * Which pose the blobatar holds. Import one from `blobatar/expression`.
     *
     * ```ts
     * import { happy } from "blobatar/expression";
     * blobatar(name, { expression: happy });
     * ```
     *
     * Passed as a value rather than named as a string so that the expressions you
     * do not import cost nothing — and so that the core carries no pose code at
     * all. Omitting it is `idle`; `idle` is also exported, for when writing it
     * reads better than `undefined`.
     *
     * Set by you and held until you change it — nothing here returns to idle on
     * its own, and there are no timers. A burst is `setExpression(happy)` followed
     * by your own `setTimeout`, which is four lines in your code and zero bytes in
     * this bundle.
     *
     * Independent of `animate` in both directions. Without `animate` the blobatar
     * renders the pose statically, which is what makes this work in the string API
     * and under `prefers-reduced-motion`; **the morph between poses requires
     * `animate`**, because that is what puts the blobatar in inline SVG where CSS
     * can reach it. Setting `expression` never turns `animate` on for you: that
     * would silently flip a 400-blobatar grid from 400 `<img>`s to 400 SVG trees.
     *
     * `idle` emits byte-identical markup to omitting the option.
     */
    expression?: Expression;
}
export interface Style<L> {
    layout(t: Traits): L;
    /**
     * `mo` is set when animating, absent otherwise. It is a flag rather than the
     * root class it used to be: the root `<g>` is the caller's now, because a
     * class inside this string is a class inside `dangerouslySetInnerHTML`. See
     * `makeParts`.
     */
    render(l: L, p: Palette, mo?: boolean): string;
    background: boolean | "square" | "circle" | "squircle";
}
export declare function resolve(seed: string, opts: BlobatarOptions): {
    t: Traits;
    palette: Palette;
};
/** The plate behind the figure, as geometry rather than as markup. */
export interface Backdrop {
    d: string;
    fill: string;
}
/**
 * What a motion factory hands back: the root class, and the seeded timing to
 * put on the outer element.
 *
 * Passed *in* rather than imported, so `src/animate.ts` never enters a bundle
 * that does not animate. That indirection is the entire reason the static path
 * still costs what it did before the motion layer existed — a plain
 * `if (opts.animate)` here would pull the motion module into every consumer,
 * animating or not.
 */
export interface Motion {
    cls: string;
    vars: Record<string, string>;
}
/**
 * The palette is handed to the factory because a tinting expression needs it:
 * the hot pair it mixes toward is derived from the colors the blobatar is actually
 * wearing, overrides included. It arrives as an argument rather than being
 * looked up so that `src/color.ts`'s `tinted()` stays reachable only from an
 * expression value — the same indirection that keeps `animate.ts` out of static
 * bundles.
 */
export type MotionFactory = (t: Traits, p: Palette) => Motion;
/** Binds one package major's frozen style into `blobatar(name, opts)`. */
export declare function makeBlobatar<L>(style: Style<L>): (name: string, opts?: BlobatarOptions) => string;
/**
 * The blobatar in the pieces a renderer that owns the outer element needs.
 *
 * Split out from `makeBlobatar` because the React adapter has to own the `<svg>`
 * when animating — it needs real JSX props on it — and recovering the inner
 * markup by regex-stripping a serialized `<svg>` is the kind of thing that
 * works until someone passes a `title` containing a `>`.
 *
 * The split runs one level deeper than that, and this is the load-bearing part:
 * **nothing that varies with `expression` may appear in `inner`.** `inner` is
 * handed to `dangerouslySetInnerHTML`, so a single byte of drift makes React
 * replace the whole subtree — and a brand-new element has no previous computed
 * value, which is precisely the rule that stops transitions running on first
 * style resolution. The morph would not be slow or wrong; it would not exist,
 * and every idle animation underneath would restart from phase zero on top of
 * it. So the root class lives in `cls` and the pose lives in `vars`, both of
 * which land on real attributes that React can diff in place, and the backdrop
 * comes back as geometry because it belongs outside the root `<g>` — a plate
 * that hover-lifts with the creature stops being a plate.
 *
 * `test/expression.test.ts` pins the invariant directly.
 */
export declare function makeParts<L>(style: Style<L>): (name: string, opts?: BlobatarOptions, motion?: MotionFactory) => {
    /** Goes on the root `<g>`, which the caller renders. */
    cls: string | undefined;
    bg: Backdrop | undefined;
    /** Everything below the root `<g>`. Free of both of the above. */
    inner: string;
    vars: Record<string, string> | undefined;
};
//# sourceMappingURL=render.d.ts.map