/** Keep `prev`'s reference when it's element-equal to `next`, so a nanostores
 *  `computed` (notifies on `!==`) skips the emit when its projected list didn't
 *  actually change — e.g. status-id sets recomputed on every stream delta.
 *  `next` is frozen: the ref is shared across ticks, so an in-place mutation
 *  would corrupt the cache — fail loud instead. */
export const stableArray = <T>(prev: readonly T[], next: T[]): readonly T[] =>
  prev.length === next.length && prev.every((v, i) => v === next[i]) ? prev : Object.freeze(next)

/** {@link stableArray} for a keyed projection. */
export const stableRecord = <T>(
  prev: Readonly<Record<string, T>>,
  next: Record<string, T>
): Readonly<Record<string, T>> => {
  const keys = Object.keys(next)

  return keys.length === Object.keys(prev).length && keys.every(k => prev[k] === next[k]) ? prev : Object.freeze(next)
}
