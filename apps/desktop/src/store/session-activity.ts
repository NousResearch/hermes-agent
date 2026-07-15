import { computed } from 'nanostores'

import { $sessions, $workingSessionIds } from './session'
import { $subagentsBySession, type SubagentProgress } from './subagents'

const RUNNING = (subagent: SubagentProgress) => subagent.status === 'running' || subagent.status === 'queued'

interface SessionActivityRow {
  _lineage_root_id?: null | string
  id: string
}

/**
 * Sessions with visible work in flight.
 *
 * Foreground turns live in `$workingSessionIds`. A top-level delegate finishes
 * that parent turn before its independent reviewers finish, so the sidebar and
 * profile rail must additionally project active subagents onto both the parent
 * conversation and each child's own persisted row. This intentionally does not
 * make the composer busy again — its calm background-resume treatment remains
 * authoritative for the foreground chat surface.
 */
export function deriveSessionActivityIds(
  foregroundSessionIds: readonly string[],
  subagentsBySession: Record<string, SubagentProgress[]>,
  sessions: readonly SessionActivityRow[] = []
): string[] {
  const active = new Set(foregroundSessionIds)

  for (const [runtimeOwnerId, subagents] of Object.entries(subagentsBySession)) {
    const running = subagents.filter(RUNNING)

    if (running.length === 0) {
      continue
    }

    // New events carry the durable stored owner. Fall back to the map key for
    // older backends/events where runtime and stored ids were commonly equal.
    active.add(running.find(subagent => subagent.ownerSessionId)?.ownerSessionId ?? runtimeOwnerId)

    for (const subagent of running) {
      if (subagent.sessionId) {
        active.add(subagent.sessionId)
      }
    }
  }

  // A reviewer may span parent-session compression. Project an active lineage
  // root onto its current continuation tip so exact-id sidebar/gateway lookups
  // remain live as well as profile lookups (which already understand lineage).
  for (const session of sessions) {
    if (session._lineage_root_id && active.has(session._lineage_root_id)) {
      active.add(session.id)
    }
  }

  return [...active]
}

export const $sessionActivityIds = computed(
  [$workingSessionIds, $subagentsBySession, $sessions],
  (foregroundSessionIds, subagentsBySession, sessions) =>
    deriveSessionActivityIds(foregroundSessionIds, subagentsBySession, sessions)
)
