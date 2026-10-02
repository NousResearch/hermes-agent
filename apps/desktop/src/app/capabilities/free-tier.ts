/** Free Capabilities until super admin unlocks the brand on Brand billing. */
export const FREE_CAPABILITY_LIMIT = 3

/** The three brand connectors a free login can use. Mail credit is separate. */
export const FREE_CONNECTOR_IDS = ['chatwoot', 'notifuse', 'twenty'] as const

export function freeListAllowed(ids: readonly string[], id: string, unlocked: boolean): boolean {
  if (unlocked) {
    return true
  }

  const rank = [...ids].sort((a, b) => a.localeCompare(b)).indexOf(id)

  return rank >= 0 && rank < FREE_CAPABILITY_LIMIT
}

export const FREE_LOCK_MESSAGE =
  'This stays locked until a super admin unlocks the desktop app for this brand.'

export function freeConnectorAllowed(connectorId: string, unlocked: boolean): boolean {
  if (unlocked) {
    return true
  }

  const key = connectorId.trim().toLowerCase()

  return (FREE_CONNECTOR_IDS as readonly string[]).some(id => key === id || key.endsWith(`-${id}`))
}
