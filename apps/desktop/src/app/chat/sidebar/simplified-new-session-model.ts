import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import type { SessionInfo } from '@/hermes'
import { newSessionInAgent, newSessionInProfile } from '@/store/profile'

import { agentKey, sessionAgentKey, type SimplifiedAgent } from './use-simplified-agents'

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

/** Rank the current agent (`active`, an `agentKey`) first, then frequency weighted by recency in
 *  available chat history. A chat with no machine is `owner`'s, as the sidebar files it. */
export function rankNewSessionAgents(
  agents: SimplifiedAgent[],
  sessions: SessionInfo[],
  owner: string | null,
  active: string,
  now = Date.now() / 1000
): SimplifiedAgent[] {
  const scores = new Map<string, number>()

  for (const session of sessions) {
    const key = sessionAgentKey(session, owner)
    const ageWeeks = Math.max(0, now - (session.last_active ?? session.started_at)) / (7 * 86400)
    scores.set(key, (scores.get(key) ?? 0) + 1 / (1 + ageWeeks))
  }

  const score = (agent: SimplifiedAgent) => scores.get(agentKey(agent)) ?? 0

  return [...agents].sort(
    (a, b) =>
      Number(agentKey(b) === active) - Number(agentKey(a) === active) ||
      score(b) - score(a) ||
      a.label.localeCompare(b.label) ||
      a.id.localeCompare(b.id)
  )
}
