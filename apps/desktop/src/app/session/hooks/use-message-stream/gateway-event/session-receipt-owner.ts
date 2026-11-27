import { normalizeProfileKey } from '@/store/profile'
import { secondaryProfileOwnerForEvent } from '@/store/session-event-provenance'
import { knownOwnerForSession } from '@/store/session-states'

import type { GatewayEventContext } from './types'

/** Compare tagged receipts to the session's bound owner. Untagged frames and
 *  sessions without a known owner retain the caller's legacy source policy. */
export function matchesReceiptOwner({ explicitSid, event }: GatewayEventContext): boolean | undefined {
  const secondary = secondaryProfileOwnerForEvent(event)

  if (!explicitSid || (!event.connectionId && !secondary)) {
    return undefined
  }

  const owner = knownOwnerForSession(explicitSid)

  if (!owner) {
    return undefined
  }

  if (event.connectionId) {
    return (
      typeof owner !== 'string' &&
      owner.connectionId === event.connectionId &&
      normalizeProfileKey(owner.profile) === normalizeProfileKey(event.profile)
    )
  }

  return typeof owner === 'string' && normalizeProfileKey(owner) === normalizeProfileKey(secondary)
}
