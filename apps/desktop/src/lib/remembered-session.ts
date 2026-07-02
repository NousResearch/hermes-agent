import type { SessionInfo } from '@/hermes'

export function rememberedSessionIdForRoute(routedSessionId: string | null, sessions: SessionInfo[]): string | null {
  if (!routedSessionId) {
    return null
  }

  const routed = sessions.find(session => session.id === routedSessionId || session._lineage_root_id === routedSessionId)
  if (!routed) {
    return routedSessionId
  }

  if (routed.source === 'subagent') {
    return routed._delegate_from || routed.parent_session_id || null
  }

  return routedSessionId
}
