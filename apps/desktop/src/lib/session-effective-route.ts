import type { SessionInfo } from '@/types/hermes'

import { displayModelName } from './model-status-label'

/** Route halves stored by the backend (`agent/session_route_record.py`). */
export interface SessionRouteEnd {
  model: string
  provider: string
}

/** Requested vs effective model of a session, from `sessions.model_config.effective_route`.
 *  `active` while a provider fallback is answering in place of the requested model; a later turn
 *  back on the requested model clears it (the switch stays as `last_fallback`). */
export interface SessionEffectiveRoute {
  active: boolean
  at?: number
  effective: SessionRouteEnd
  reason?: string
  requested: SessionRouteEnd
  route_source: 'fallback' | 'primary'
}

/** `{ effective, requested }` display names while a fallback answers this session, else null. */
export function sessionFallbackRoute(
  session: Pick<SessionInfo, 'effective_route'>
): null | { effective: string; requested: string } {
  const route = session.effective_route

  if (!route?.active || route.route_source !== 'fallback' || !route.effective?.model) {
    return null
  }

  return {
    effective: displayModelName(route.effective.model),
    requested: displayModelName(route.requested?.model || '') || route.requested?.provider || '?'
  }
}
