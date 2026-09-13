import type { GatewayEvent } from '@hermes/shared'

import type { SessionOwnerScope } from '@/store/session-request-router'

/** Normalized source tags of a gateway event: the registry connection whose
 *  socket delivered it ('' for the untagged local primary) and its profile. */
export interface GatewayEventSource {
  connectionId: string
  profile: string
}

export function gatewayEventSource(event: Pick<GatewayEvent, 'connectionId' | 'profile'>): GatewayEventSource {
  return {
    connectionId: event.connectionId?.trim() || '',
    profile: event.profile?.trim() || 'default'
  }
}

/**
 * Whether a session-scoped event from `source` may act on a session owned by
 * `owner`. Runtime ids are only unique per backend, so a `session.replay_gap`
 * naming a colliding id from another gateway must not re-resume this one.
 *
 * A profile-only owner is the legacy/local pool route: it matches only an
 * UNTAGGED primary event for that profile. An explicitly tagged source needs
 * an exact route so same-named remote profiles stay isolated.
 */
export function eventSourceMatchesOwner(source: GatewayEventSource, owner: SessionOwnerScope): boolean {
  if (!owner) {
    return false
  }

  if (typeof owner === 'string') {
    return !source.connectionId && source.profile === (owner.trim() || 'default')
  }

  return source.connectionId === owner.connectionId.trim() && source.profile === (owner.profile.trim() || 'default')
}
