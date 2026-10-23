import { registryBackendScopeKey } from '@hermes/shared'

import type { SessionOwnerRoute } from '@/store/session-request-router'

/** A legacy profile-only socket has no registry connection id. */
export interface LiveStatusScope extends Omit<SessionOwnerRoute, 'connectionId'> {
  connectionId: null | string
}
export interface CapturedLiveStatusRequest {
  scope: LiveStatusScope
  request: <T>() => Promise<T>
}
export type CaptureLiveStatusRequest = (source?: LiveStatusScope) => Promise<CapturedLiveStatusRequest>

export function normalizeLiveStatusScope(scope: LiveStatusScope | string): LiveStatusScope {
  return typeof scope === 'string' ? { connectionId: null, profile: scope } : scope
}

export function liveStatusScopeKey(scope: LiveStatusScope): string {
  const source = registryBackendScopeKey(scope.connectionId, scope.profile)

  return scope.targetProfile ? JSON.stringify([source, scope.targetProfile]) : source
}
