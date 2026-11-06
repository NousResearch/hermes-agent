import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import type { SessionInfo } from '@/hermes'
import { newSessionInAgent, newSessionInProfile } from '@/store/profile'

import type { SimplifiedAgent } from './use-simplified-agents'

export function startSimplifiedAgentSession(
  agent: Pick<SimplifiedAgent, 'connectionId' | 'profile' | 'targetProfile'>
): void {
  setWorkspaceScope('sessions')

  if (agent.connectionId) {
    newSessionInAgent({
      connectionId: agent.connectionId,
      profile: agent.profile,
      ...(agent.targetProfile ? { targetProfile: agent.targetProfile } : {})
    })
  } else {
    newSessionInProfile(agent.profile)
  }
}

/** Rank the current agent first, then frequency weighted by recency in available chat history. */
export function rankNewSessionAgents(
  agents: SimplifiedAgent[],
  sessions: SessionInfo[],
  connectionId: string | null,
  profile: string,
  now = Date.now() / 1000
): SimplifiedAgent[] {
  const scores = new Map<string, number>()

  for (const session of sessions) {
    const id = JSON.stringify([session.connection_id || connectionId, session.profile || 'default'])
    const ageWeeks = Math.max(0, now - (session.last_active ?? session.started_at)) / (7 * 86400)
    scores.set(id, (scores.get(id) ?? 0) + 1 / (1 + ageWeeks))
  }

  const active = JSON.stringify([connectionId, profile])

  return [...agents].sort(
    (a, b) =>
      Number(b.id === active) - Number(a.id === active) ||
      (scores.get(b.id) ?? 0) - (scores.get(a.id) ?? 0) ||
      a.label.localeCompare(b.label) ||
      a.id.localeCompare(b.id)
  )
}
