export {
  blobatar,
  _layout,
  type Animate,
  type BlobatarOptions,
  type Expression,
} from "./blobatar";
export {
  palette,
  ramp,
  contrast,
  FLOORS,
  type Palette,
  type Oklch,
  type ColorKey,
} from "./color";
export { traits, type Traits, type TraitOverrides } from "./traits";
export { normalizeSeed } from "./hash";

/**
 * The version this build came from. Asserted against package.json in
 * `test/blobatar.test.ts`, so it cannot drift.
 *
 * It is also load-bearing, which is the part worth knowing before deleting it.
 * On Bun 1.3.14, bundling an entry whose body is *nothing but* named re-exports
 * against a package declaring `sideEffects` produces a module that re-exports
 * names it never imported — `export { a as palette }` with no `a` anywhere —
 * and Node throws `SyntaxError: Export 'a' is not defined in module` the moment
 * it links the file. One real binding in the module body is enough to stop the
 * whole graph being dropped. `scripts/smoke.mjs` links the built barrel under
 * Node on every build and is what will tell you if this stops being true.
 */
export const VERSION = "2.0.0";
