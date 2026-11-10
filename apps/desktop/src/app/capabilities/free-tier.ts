/** Free Capabilities until super admin unlocks the brand on Brand billing. */
export const FREE_CAPABILITY_LIMIT = 3

/** Extra tools drawn under the lock. The rest of the catalog stays off the page. */
export const BRANDING_LOCKED_VISIBLE = 6

export function brandingLockedSlice<T>(locked: readonly T[]): T[] {
  return locked.slice(0, BRANDING_LOCKED_VISIBLE)
}

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
  'Pay to unlock all skills, tools, connectors, and plugins for this brand. After payment, a super admin turns desktop access on.'

export function freeConnectorAllowed(connectorId: string, unlocked: boolean): boolean {
  if (unlocked) {
    return true
  }

  const key = connectorId.trim().toLowerCase()

  return (FREE_CONNECTOR_IDS as readonly string[]).some(id => key === id || key.endsWith(`-${id}`))
}
