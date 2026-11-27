import { $apiRequestScope, resolveOwnerNow } from '@/api/client'
import { getHermesConfigRecord, getProfileSoul, getSkills, getToolsets } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { $activeConnectionId } from '@/store/connections'
import { $fleetRoster } from '@/store/fleet-roster'
import { $profiles } from '@/store/profile'
import { $portalShell } from '@/store/shell-variant'

import { skillsQueryKey } from '../capabilities/skills/skills-data'
import { toolsetsQueryKey } from '../capabilities/toolsets/toolsets-data'
import { hermesConfigKey } from '../hooks/use-config-record'

import {
  type AgentRouteOwner,
  agentSettingsKey,
  type AgentSettingsOwner,
  getAgentMessaging,
  getAgentRecentSessions,
  getAgentRoutines
} from './data'
import { $lastViewedAgent, agentMachine } from './landing'

/** The agent page's owner-scoped reads, shared by the page and the idle prefetch so both hit one
 *  cache row: a rail click then paints the Overview from cache in its first frame. */
export const agentOverviewQueries = {
  routines: (owner: AgentSettingsOwner) => ({
    queryKey: [...agentSettingsKey(owner), 'routines'],
    queryFn: () => getAgentRoutines(owner)
  }),
  messaging: (owner: AgentSettingsOwner) => ({
    queryKey: [...agentSettingsKey(owner), 'messaging'],
    queryFn: () => getAgentMessaging(owner)
  }),
  recent: (owner: AgentSettingsOwner) => ({
    queryKey: [...agentSettingsKey(owner), 'recent', 5],
    queryFn: () => getAgentRecentSessions(owner)
  }),
  soul: (owner: AgentSettingsOwner) => ({
    queryKey: [...agentSettingsKey(owner), 'soul'],
    queryFn: () => getProfileSoul(owner.profile, owner)
  })
}

/** The owner an agent route pins on entry (entry.tsx): the route's machine, else the ambient one,
 *  always named, so every read, write and cache key of the page agrees on one identity. */
export function agentRouteOwner(connectionId: null | string | undefined, profile: string): AgentRouteOwner {
  return {
    connectionId: agentMachine(
      resolveOwnerNow({ connectionId: connectionId ?? undefined, profile }).connectionId ?? $activeConnectionId.get()
    ),
    profile
  }
}

/** The agent the Agents destination opens (landing.ts), resolved from stores outside React. */
function landingOwner(): AgentSettingsOwner | null {
  const last = $lastViewedAgent.get()

  if (last) {
    return agentRouteOwner(last.connectionId, last.profile)
  }

  const profile = $profiles.get().find(item => item.is_default)?.name

  if (!profile) {
    return null
  }

  const machine = $apiRequestScope.get().connectionId ?? $fleetRoster.get()?.primaryConnectionId ?? null

  return agentRouteOwner(machine, profile)
}

/** Warm the landing agent's Overview reads (the classic shell has no Overview). Rows already
 *  cached and fresh are not refetched. */
const PREFETCH_STALE_MS = 30_000

export async function prefetchLandingAgentOverview(): Promise<void> {
  const owner = $portalShell.get() ? landingOwner() : null

  if (!owner) {
    return
  }

  await Promise.all([
    ...Object.values(agentOverviewQueries).map(query =>
      queryClient.prefetchQuery<unknown>({ ...query(owner), staleTime: PREFETCH_STALE_MS })
    ),
    queryClient.prefetchQuery({
      queryFn: () => getHermesConfigRecord(owner),
      queryKey: hermesConfigKey(owner),
      staleTime: PREFETCH_STALE_MS
    }),
    queryClient.prefetchQuery({
      queryFn: () => getSkills(owner),
      queryKey: skillsQueryKey(owner),
      staleTime: PREFETCH_STALE_MS
    }),
    queryClient.prefetchQuery({
      queryFn: () => getToolsets(owner),
      queryKey: toolsetsQueryKey(owner),
      staleTime: PREFETCH_STALE_MS
    })
  ])
}
