import { useStore } from '@nanostores/react'

import type { CronJob } from '@/hermes'
import { agentSidebarKey } from '@/store/agent-sidebar-preferences'
import { $cronJobs, getCronJobsSnapshotScope } from '@/store/cron'

import { compareRoutines } from './routine-model'

/** Each agent's routines from the shared snapshot, keyed like the chat sidebar's agents and in the
 *  Routines page's order. A snapshot with no owning scope (wiped by a switch) says nothing about
 *  anyone's routines. */
export function routinesByAgent(jobs: readonly CronJob[], scope: string): Map<string, CronJob[]> {
  const byAgent = new Map<string, CronJob[]>()

  if (!scope) {
    return byAgent
  }

  const [connection, listed] = scope.split('\u0000')

  for (const job of [...jobs].sort(compareRoutines)) {
    const profile = listed === 'all' ? job.profile : listed

    if (!profile) {
      continue
    }

    const key = agentSidebarKey({ connectionId: connection || null, profile })
    byAgent.set(key, [...(byAgent.get(key) ?? []), job])
  }

  return byAgent
}

/** One agent's routines; none while its owner is unknown. */
export function useAgentRoutines(owner: { connectionId?: null | string; profile?: string }): CronJob[] {
  const jobs = useStore($cronJobs)

  if (!owner.profile) {
    return []
  }

  const scope = getCronJobsSnapshotScope()
  // An owner on the ambient source is on the connection the snapshot was read from.
  const connectionId = owner.connectionId ?? (scope.split('\u0000')[0] || null)

  return routinesByAgent(jobs, scope).get(agentSidebarKey({ connectionId, profile: owner.profile })) ?? []
}
