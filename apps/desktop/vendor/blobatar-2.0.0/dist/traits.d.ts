/**
 * A trait reader. Every value is addressed by a string key rather than drawn
 * from a sequential stream, so trait keys are an append-only namespace:
 * introducing `t.num("freckles.size", ...)` in a later version leaves every
 * other trait — and therefore every existing blobatar — untouched.
 *
 * The one thing that is NOT free to change is the contents of a `pick` array,
 * since option index is part of the mapping. Those are frozen per major.
 */
export interface Traits {
    /** Uniform float in [0, 1). */
    (key: string): number;
    /** Uniform float in [min, max). */
    num(key: string, min: number, max: number): number;
    /** Uniform integer in [min, max]. */
    int(key: string, min: number, max: number): number;
    /** Uniform choice. Appending to `options` remaps existing seeds — frozen per major. */
    pick<T>(key: string, options: readonly T[]): T;
    /** True with probability `p`. */
    bool(key: string, p?: number): boolean;
    /** Symmetric jitter in [-amount, amount). */
    jitter(key: string, amount: number): number;
}
/**
 * Fixed values for individual traits, keyed exactly as the layout reads them —
 * `{ "eye.gap": 0.82 }`.
 *
 * Every value is the position in [0, 1) that the hash would otherwise have
 * produced, which is what makes this a complete configuration surface rather
 * than a set of escape hatches: the ranges, the derived clamps and the
 * categorical thresholds all live in the layout, so an override is read in the
 * same units, in the same place, under the same guarantees as a hashed value.
 * There is nothing a seed can express that an override cannot, and nothing an
 * override can express that a seed could not have.
 *
 * The corollary is that this is the *only* configuration seam. The layout
 * function itself stays private: its return shape, its containment arithmetic
 * and its per-shape decoration would all become public API, and a caller who
 * replaced it would be one arithmetic slip away from eyes outside the body.
 * Overriding the input keeps every invariant in `styles/blob.ts` running.
 *
 * Sparse by design. Keys you omit still come from the seed, so
 * `{ shape: 0.95 }` means "always a sun, everything else per seed".
 *
 * An **array** is the third position between those two: `{ shape: [0.11, 0.965] }`
 * means "round or sun, whichever this name comes out as". A fixed value narrows
 * a key to one outcome and an omitted key leaves it at all of them; a list
 * narrows it to the ones you name and lets the seed choose among those. That is
 * the case a single number cannot state — "any of these" is not a position —
 * and it is the shape of most real requests for one, since a house style is
 * usually a handful of silhouettes rather than exactly one.
 *
 * The choice rides on the key's own hash, so it is per seed, stable, uniform
 * over what is listed, and independent of every other trait — the same
 * guarantees an unconfigured value has, because it *is* the unconfigured value,
 * read against a shorter list. An empty array selects nothing and is therefore
 * the same as omitting the key.
 *
 * Trait keys are an append-only namespace (see `Traits` above), so an override
 * map keeps meaning what it meant. The numeric *ranges* those keys are read
 * into are what a stated position is relative to, which makes them part of the
 * same frozen-per-major contract as a `pick` array's contents — retuning
 * `t.num("eye.gap", 0.1, 0.24)` moves every blobatar, seeded or configured.
 */
export type TraitOverrides = Record<string, number | number[]>;
/**
 * Overrides are clamped rather than trusted.
 *
 * `t.pick` and `t.int` index and floor, so a value of exactly 1 selects one
 * past the end of a `pick` array and one past `max` — `undefined` options and
 * out-of-range counts, from an input that looks entirely reasonable to whoever
 * typed it. NaN falls to 0 through the same comparison, so a bad parse renders
 * a blobatar instead of a stack of `NaN`s in the path data.
 */
export declare function traits(seed: string, normalize?: boolean, overrides?: TraitOverrides): Traits;
//# sourceMappingURL=traits.d.ts.map