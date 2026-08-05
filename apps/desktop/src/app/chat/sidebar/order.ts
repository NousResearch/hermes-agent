/** New ids first, then ids still present in the persisted order. */
export function reconcileFreshFirst(currentIds: string[], orderIds: string[]): string[] {
  const current = new Set(currentIds)
  const retained = orderIds.filter(id => current.has(id))
  const retainedSet = new Set(retained)

  return [...currentIds.filter(id => !retainedSet.has(id)), ...retained]
}

export function resolveManualSessionOrderIds(currentIds: string[], orderIds: string[], manual: boolean): string[] {
  if (!manual || !currentIds.length || !orderIds.length) {
    return []
  }

  const current = new Set(currentIds)
  const retained = orderIds.filter(id => current.has(id))

  if (!retained.length) {
    return []
  }

  return reconcileFreshFirst(currentIds, orderIds)
}

/** Reorder `items` by `orderIds`; items missing from the order surface first. */
export function orderByIds<T>(items: T[], getId: (item: T) => string, orderIds: string[]): T[] {
  if (!orderIds.length) {
    return items
  }

  const byId = new Map(items.map(item => [getId(item), item]))
  const seen = new Set<string>()
  const ordered: T[] = []

  for (const id of orderIds) {
    const item = byId.get(id)

    if (item) {
      ordered.push(item)
      seen.add(id)
    }
  }

  // Items missing from the persisted order are new since it was last
  // reconciled. Callers pass recency-sorted lists (newest first), so surface
  // these at the TOP instead of burying them beneath the saved order —
  // otherwise a brand-new session sinks to the bottom of the sidebar and reads
  // as "my latest session never showed up".
  const fresh = items.filter(item => !seen.has(getId(item)))

  return fresh.length ? [...fresh, ...ordered] : ordered
}

/** Reconcile a persisted order against the live id set (fresh-first). */
export function reconcileOrderIds(currentIds: string[], orderIds: string[]): string[] {
  if (!currentIds.length) {
    return []
  }

  if (!orderIds.length) {
    return currentIds
  }

  return reconcileFreshFirst(currentIds, orderIds)
}

/** True when two id lists are element-for-element identical. */
export function sameIds(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((item, index) => item === right[index])
}

/**
 * Cluster items whose titles share a `[Topic]` prefix so they sit adjacent,
 * while preserving the caller's recency baseline: scan the (recency-sorted)
 * list and, on first sight of a topic, pull every other item with the same
 * prefix up to follow it. Untitled / unprefixed items stay in their original
 * positions. Sibling order inside a cluster follows the baseline order.
 */
export function clusterByTopic<T>(
  items: T[],
  getId: (item: T) => string,
  getTitle: (item: T) => string | null | undefined
): T[] {
  const topicOf = (title?: string | null) => title?.match(/^\[([^\]]+)\]/)?.[1] ?? null

  const byTopic = new Map<string, T[]>()

  for (const item of items) {
    const topic = topicOf(getTitle(item))

    if (topic) {
      const siblings = byTopic.get(topic)

      if (siblings) {
        siblings.push(item)
      } else {
        byTopic.set(topic, [item])
      }
    }
  }

  const placed = new Set<string>()
  const out: T[] = []

  for (const item of items) {
    const id = getId(item)

    if (placed.has(id)) {
      continue
    }

    out.push(item)
    placed.add(id)

    const topic = topicOf(getTitle(item))

    if (topic) {
      for (const sibling of byTopic.get(topic) ?? []) {
        const siblingId = getId(sibling)

        if (!placed.has(siblingId)) {
          out.push(sibling)
          placed.add(siblingId)
        }
      }
    }
  }

  return out
}
