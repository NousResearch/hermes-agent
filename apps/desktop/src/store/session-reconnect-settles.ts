import type { SessionOwnerRoute } from './session-request-router'

/** A primary reconnect retains the route, not whatever Bot Chat is focused later. */
export interface ReconnectStatusSource extends Omit<SessionOwnerRoute, 'connectionId'> {
  connectionId: null | string
}

// Runtime-keyed: identical stored ids can belong to different backend scopes.
export const unconfirmedReconnectSettles = new Map<string, {
  storedId: string
  scope: string
  source: ReconnectStatusSource
}>()

/** Only known reconnect claims need an extra read; this is not discovery. */
export function pendingReconnectStatusSources(): ReconnectStatusSource[] {
  return [...new Map([...unconfirmedReconnectSettles.values()].map(p => [p.scope, p.source])).values()]
}
