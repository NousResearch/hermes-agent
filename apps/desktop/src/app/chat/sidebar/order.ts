export function reconcileOrderIds(currentIds: string[], orderIds: string[]): string[] {
  if (!currentIds.length || !orderIds.length) {
    return []
  }

  const current = new Set(currentIds)
  const next = orderIds.filter(id => current.has(id))

  if (!next.length) {
    return []
  }

  const ordered = new Set(next)
  const naturalKnownOrder = currentIds.filter(id => ordered.has(id))

  if (naturalKnownOrder.length === next.length && naturalKnownOrder.every((id, index) => id === next[index])) {
    return []
  }

  const known = new Set(next)

  for (const id of currentIds) {
    if (!known.has(id)) {
      next.push(id)
      known.add(id)
    }
  }

  return next
}
