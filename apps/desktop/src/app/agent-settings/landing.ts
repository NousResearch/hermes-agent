import { atom } from 'nanostores'

import { SETTINGS_ROUTE } from '../routes'

/** The agent a page last showed in this window: in memory only, so every window keeps its own and a
 *  fresh window starts from the default agent. */
export interface AgentRef {
  connectionId: null | string
  profile: string
}

// The app's own machine is spelled both `null` and 'local' depending on who resolved it.
const machine = (agent: AgentRef) => agent.connectionId ?? 'local'

const sameAgent = (a: AgentRef, b: AgentRef) => a.profile === b.profile && machine(a) === machine(b)

export const $lastViewedAgent = atom<AgentRef | null>(null)

export function rememberViewedAgent(agent: AgentRef) {
  const current = $lastViewedAgent.get()

  if (!current || !sameAgent(current, agent)) {
    $lastViewedAgent.set(agent)
  }
}

/** Which agent the Agents destination opens: the one last viewed here while it still exists, else
 *  the default agent, else the first listed. `null` only when there is no agent at all. */
export function pickLandingAgent<T extends AgentRef>(
  agents: readonly T[],
  lastViewed: AgentRef | null,
  isDefault: (agent: T) => boolean
): T | null {
  return (
    (lastViewed && agents.find(agent => sameAgent(agent, lastViewed))) ?? agents.find(isDefault) ?? agents[0] ?? null
  )
}

/** The agent page's URL (Overview tab): the same route the chat header's "Agent settings" opens. */
export function agentPageHref(agent: AgentRef) {
  const params = new URLSearchParams({ agentProfile: agent.profile })

  if (agent.connectionId) {
    params.set('connection', agent.connectionId)
  }

  return `${SETTINGS_ROUTE}?${params}`
}
