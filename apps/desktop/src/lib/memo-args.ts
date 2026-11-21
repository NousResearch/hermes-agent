/**
 * Memoize a pure derivation by its arguments' identities, across component mounts.
 *
 * `useMemo` forgets with its component, so a page that derives from a large cached query result
 * (the ~100k-row skills feed) redid the whole derivation on every visit although the inputs were
 * the very same objects React Query handed out last time. Keyed by identity (`Object.is`), a hit
 * costs a few comparisons; the last `slots` distinct argument lists are kept (most recent first),
 * so alternating between a couple of inputs (two catalog tabs) does not thrash.
 */
export function memoArgs<A extends readonly unknown[], R>(fn: (...args: A) => R, slots = 4): (...args: A) => R {
  const recent: { args: A; result: R }[] = []

  return (...args: A) => {
    const index = recent.findIndex(
      entry => entry.args.length === args.length && entry.args.every((value, i) => Object.is(value, args[i]))
    )

    if (index >= 0) {
      const [hit] = recent.splice(index, 1)
      recent.unshift(hit)

      return hit.result
    }

    const result = fn(...args)
    recent.unshift({ args, result })
    recent.length = Math.min(recent.length, slots)

    return result
  }
}
