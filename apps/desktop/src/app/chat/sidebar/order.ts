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
