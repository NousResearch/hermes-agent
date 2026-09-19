import type { Traits } from "./traits";
/**
 * Idle animation for the `blob` variant. See docs/motion-spec.md.
 *
 * `"hover"` animates one blobatar at a time, which is both the aesthetic answer
 * (ambient motion seen constantly is motion worth removing) and the performance
 * one. `"always"` is the escape hatch for the single-blobatar case — a profile
 * header, an onboarding screen — where that frequency argument does not apply.
 */
export type Animate = "hover" | "always";
/**
 * Root class. Amplitude, and therefore everything else, hangs off this.
 *
 * `mo-expr` marks "wearing a non-idle expression" and exists for exactly one
 * reason: a transition takes its duration from the state it is heading *to*, so
 * the class is what lets adopting an expression and returning to idle run on
 * different clocks. It selects no pose of its own — the pose is eight custom
 * properties, and this file never learns which expression is on.
 */
export declare const rootClass: (mode: Animate, expressive?: boolean) => string;
/**
 * Per-blobatar timing, as custom properties for the stylesheet to read.
 *
 * A grid where every blobatar breathes in unison does not read as a crowd of
 * creatures; it reads as a heartbeat. Seeded offsets are what make it a crowd,
 * and they are the single most load-bearing 40 bytes in the motion layer.
 *
 * Delays are negated **here**, at the source. A positive `animation-delay`
 * postpones the start rather than offsetting the phase, so the whole grid would
 * still open in unison — after an awkward pause. Same keystroke, opposite
 * behavior, and it only shows on first paint.
 *
 * Breathe and bob get independent offsets. Sharing one preserves the drift
 * between their two periods but locks every blobatar into the *same* drift, which
 * is the unison problem again, one level up.
 *
 * These keys cost nothing in compatibility: traits are string-addressed, so
 * adding `motion.*` cannot perturb any existing blobatar.
 */
export declare function motionVars(t: Traits): Record<string, string>;
/** `--a:1;--b:2` — for the string API, which has no style object to hand. */
export declare const serializeVars: (vars: Record<string, string>) => string;
//# sourceMappingURL=animate.d.ts.map