/**
 * Shared A→Z ordering for the "Sort sidebar alphabetically" preference.
 *
 * Case-insensitive, numbers in natural order ("v2" before "v10"); punctuation
 * is not ignored, so dot-prefixed labels stay on top. Callers add an id
 * tiebreak so equal labels never swap between renders.
 */
export const compareAlphabetically = (a: string, b: string): number =>
  a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })

/** A→Z by label, id as the tiebreak. Returns a new array. */
export function sortByLabel<T extends { id: string; label: string }>(nodes: readonly T[]): T[] {
  return [...nodes].sort((a, b) => compareAlphabetically(a.label, b.label) || a.id.localeCompare(b.id))
}
