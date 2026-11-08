// A store that reads its value once (a plain atom such as the favorite models)
// keeps what it read when a newer machine value lands later (a first snapshot
// past the boot bound, another window's change), and its next edit carries
// that stale view whole. Written as is, it would drop every entry the user
// never touched (#145536 re-review). Rebasing keeps the user's edit and the
// store's other entries: for a list, the entries the user removed go and the
// ones they added join; for a record, the properties they changed win.
type Json = unknown[] | Record<string, unknown>

function parse(raw: null | string): Json | null | undefined {
  if (raw === null) {
    return null
  }

  try {
    const value = JSON.parse(raw) as unknown

    return value && typeof value === 'object' ? (value as Json) : undefined
  } catch {
    return undefined
  }
}

const same = (left: unknown, right: unknown) => JSON.stringify(left) === JSON.stringify(right)

// A three-way merge: each side's removals and additions both apply, so an
// entry another client removed stays removed. The order is this window's,
// with the store's additions appended, unless only the store reordered.
function rebaseList(base: unknown[], theirs: unknown[], mine: unknown[]): unknown[] {
  const has = (list: unknown[], item: unknown) => list.some(entry => same(entry, item))

  const reordered = (list: unknown[]) =>
    !same(
      list.filter(item => has(base, item)),
      base.filter(item => has(list, item))
    )

  const [first, second] = reordered(theirs) && !reordered(mine) ? [theirs, mine] : [mine, theirs]
  const removed = base.filter(item => !has(theirs, item) || !has(mine, item))

  const merged = first.filter(item => !has(removed, item))

  return [...merged, ...second.filter(item => !has(merged, item) && !has(base, item))]
}

function rebaseRecord(
  base: Record<string, unknown>,
  theirs: Record<string, unknown>,
  mine: Record<string, unknown>
): Record<string, unknown> {
  const result = { ...theirs }

  for (const name of new Set([...Object.keys(base), ...Object.keys(mine)])) {
    if (!same(base[name], mine[name])) {
      if (Object.hasOwn(mine, name)) {
        result[name] = mine[name]
      } else {
        delete result[name]
      }
    }
  }

  return result
}

/**
 * `mine` (this window's write) rebased from `base` (the value this window's
 * store last saw) onto `theirs` (the store's current value). A scalar, or any
 * value that is not a list or record on all sides, is the user's choice: mine
 * wins. An absent key reads as an empty list or record.
 */
export function rebaseWrite(base: null | string, theirs: null | string, mine: null | string): null | string {
  if (theirs === base) {
    return mine
  }

  const sides = [parse(base), parse(theirs), parse(mine)]
  const shape = sides.find(side => side !== null)

  if (
    !shape ||
    sides.some(side => side === undefined || (side !== null && Array.isArray(side) !== Array.isArray(shape)))
  ) {
    return mine
  }

  const [b, t, m] = sides.map(side => side ?? (Array.isArray(shape) ? [] : {})) as Json[]

  const rebased = Array.isArray(shape)
    ? rebaseList(b as unknown[], t as unknown[], m as unknown[])
    : rebaseRecord(b as Record<string, unknown>, t as Record<string, unknown>, m as Record<string, unknown>)

  const empty = Array.isArray(rebased) ? rebased.length === 0 : Object.keys(rebased).length === 0

  return mine === null && empty ? null : JSON.stringify(rebased)
}
