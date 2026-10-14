/**
 * Palette construction.
 *
 * Hue is the only value the seed controls. Lightness and chroma are authored
 * constants, which is what makes every blobatar look like it came from the same
 * designer rather than from a random number generator.
 *
 * Colors are resolved to hex rather than emitted as `oklch()`. Browsers handle
 * `oklch()` in SVG fine, but server-side rasterizers (resvg, librsvg, sharp)
 * largely do not — and blobatars get rasterized server-side constantly. Doing the
 * conversion here also means the contrast guarantee is enforced against real
 * sRGB luminance instead of assumed from OKLab lightness, which drifts by up to
 * ~1.4:1 between hues at equal L.
 */
export interface Oklch {
    l: number;
    c: number;
    h: number;
}
/** Every color slot a blobatar has. */
export type ColorKey = "bg" | "head" | "eye";
export type Palette = Partial<Record<ColorKey, string>>;
export declare function contrast(a: Oklch, b: Oklch): number;
/**
 * Pushes `fg`'s lightness away from `bg` until the pair clears `min`.
 *
 * Walks in the direction it is already leaning first, so a dark ink on a light
 * head gets darker rather than flipping to light. If that direction runs out of
 * range, it tries the other way before giving up at pure black or white.
 */
export declare function ensureContrast(fg: Oklch, bg: Oklch, min: number): Oklch;
export declare function toHex(color: Oklch): string;
/**
 * sRGB hex → OKLCh. The inverse of `toLinear` plus `resolve`'s decode, and the
 * only way back into the color space from a palette that has already been
 * serialized.
 *
 * It exists because the tint (§ `hot` below) has to start from the colors that
 * are actually on screen, not from the ramp that produced them: `BlobatarOptions.palette`
 * lets a consumer override `head` or `eye` outright, and a hot pair derived from
 * the ramp instead would tint toward a color the blobatar never wore.
 */
export declare function fromHex(hex: string): Oklch;
/**
 * Blend two colors in OKLab — `color-mix(in oklab, a, b t)`, done here.
 *
 * Here rather than in CSS because a hot pose has to resolve to a *finished*
 * color before it reaches the stylesheet: the hot endpoint would otherwise be a
 * custom property that vanishes the instant an expression is cleared, snapping
 * the fill back to base while the rest of the pose eases out over 360ms. See
 * `heatTint` in `src/expression.ts`.
 *
 * Interpolating in OKLab means lerping cartesian `a`/`b`, not the polar `c`/`h`
 * this module otherwise speaks — a hue lerp would swing a desaturated color
 * around the wheel and pick up chroma that is in neither endpoint.
 */
export declare function mix(a: Oklch, b: Oklch, t: number): Oklch;
/** `mix` between two serialized colors, serialized. */
export declare const mixHex: (a: string, b: string, t: number) => string;
/**
 * Where a tinting pose is heading.
 *
 * Four numbers rather than an authored colour, because the endpoint has to be
 * derived per seed — see `tinted` below. A `Tint` says *which way*, and the
 * blobatar's own palette says where that lands.
 *
 * This was three constants and a red until the roster wanted more than anger.
 * Generalising it is what makes a second tinting pose cost its numbers rather
 * than a second copy of the contrast walk — and the walk is the part that is
 * easy to get subtly wrong, so having exactly one of it is the point.
 */
export interface Tint {
    /** Hue the body arrives at, in degrees. Reached outright, not approached. */
    h: number;
    /** Lightness it heads toward. */
    l: number;
    /** How far of the way to `l` the body actually travels, 0–1. */
    pull: number;
    /** Chroma floor. The body never desaturates on the way. */
    c: number;
}
/**
 * Red, because every reference for anger is — and only 60% of the way there in
 * lightness, so the tone set survives the trip.
 */
export declare const HOT: Tint;
/**
 * The rest of the targets, kept here beside `HOT` rather than in `expression.ts`
 * so that the module that owns the guarantee owns the full set of endpoints it
 * has to hold for. `test/color.test.ts` iterates this list; a target added over
 * there and not here would be a tint nothing verifies.
 *
 * `pull` is the dial that keeps them apart as much as `h` is. `BLUSH` travels
 * only 0.4 of the way and lands pale — a shy blobatar that goes as red as an
 * angry one is an angry one.
 */
export declare const ROSE: Tint;
export declare const BLUSH: Tint;
export declare const BILE: Tint;
/** Every target the suite has to hold the contrast guarantee across. */
export declare const TINTS: [string, Tint][];
/**
 * The palette a tinting pose heads toward, given the one it is tinting from.
 *
 * Derived per seed rather than being a single authored colour, and the reason is
 * polarity: `blob` flips its eye between near-black and near-white depending on
 * the body's lightness, and no fixed red clears 4.5:1 against both.
 *
 * So the tinted body meets its target **partway** rather than landing on it.
 * Holding the body's own lightness was the first attempt and it is too quiet — a
 * pastel goes pink rather than angry, because at L 0.86 there is no red to be
 * had. Travelling the whole way is the opposite failure: every blobatar in the
 * roster converges on one red and the tone set, which is most of what makes a
 * grid look like a crowd, disappears at the exact moment the grid is loudest.
 * `pull` keeps a pale blobatar recognisably pale and an ink one recognisably
 * dark while giving both somewhere to go.
 *
 * The eye endpoint is then pushed until **every point along the mix** clears the
 * floor, not merely both ends. A straight line in OKLab between two passing
 * pairs is not itself a passing pair: the body travels further than the eye, so
 * the two lightnesses can close on each other in the middle of a transition that
 * is legible at both stops. This walks the mix and fixes the worst point.
 *
 * Tree-shaken out of any bundle that imports no tinting expression — it is
 * reached only through `tint` on the expression value, the same indirection that
 * keeps `expression.ts` itself out of the core. One walk serves every target, so
 * a bundle with three tinting poses in it carries this once.
 */
export declare function tinted(head: string, eye: string, t: Tint): [string, string];
/**
 * Minimum contrast ratios as [foreground, background, ratio], applied in order.
 * Later pairs resolve against already-final earlier colors, so the chain
 * converges. `4.5` on the eyes is the WCAG text floor: they are small marks
 * that have to read at 24px.
 *
 * The body/backdrop floor is deliberately weak. The backdrop is off by default,
 * and the pale swatches are meant to sit quietly on a light surface — forcing
 * 1.6:1 there would darken exactly the tones the style exists for.
 */
declare const FLOORS: [string, string, number][];
export { FLOORS };
/** The palette in OKLCh, before hex encoding. The test suite asserts against this. */
export declare function ramp(hue: number, enforce?: boolean, tone?: number): Record<string, Oklch>;
export declare function palette(hue: number, enforce?: boolean, tone?: number): Palette;
//# sourceMappingURL=color.d.ts.map