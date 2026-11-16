// User-defined rail order for named profile squares. Names absent from the
// order alphabetise at the tail. Pure so both the active strip and the at-rest
// fleet groups sort identically without pulling stores into fleet-rail.
export function sortByProfileOrder<T>(items: readonly T[], order: readonly string[], name: (item: T) => string): T[] {
  const rank = new Map(order.map((entry, index) => [entry, index]))

  return [...items].sort((a, b) => {
    const ra = rank.get(name(a))
    const rb = rank.get(name(b))

    if (ra != null && rb != null) {
      return ra - rb
    }

    return ra != null ? -1 : rb != null ? 1 : name(a).localeCompare(name(b))
  })
}
