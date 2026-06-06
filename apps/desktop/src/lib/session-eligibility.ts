import type { SessionInfo } from '@/types/hermes'

const ACTIVE_GRACE_SECONDS = 300

/**
 * Determine which sessions are eligible for archival and which are protected.
 * Matches the backend eligibility logic in `archive_surfaced_sessions()`.
 *
 * Protected sessions:
 * - Pinned sessions (by id or lineage root)
 * - Running/working sessions
 * - Recently active (within ACTIVE_GRACE_SECONDS) sessions that haven't ended
 */
export interface SessionEligibilitySummary {
  /** Total sessions considered */
  total: number
  /** Sessions that will be archived */
  eligible: number
  /** Sessions that are protected from archival */
  protected: number
}

/**
 * Compute which sessions are eligible for bulk archival given a set of
 * preserve IDs. Matches backend `archive_surfaced_sessions()` logic.
 */
export function computeSessionEligibility(
  sessions: SessionInfo[],
  preserveIds: Set<string>
): SessionEligibilitySummary {
  if (!sessions.length) {
    return { total: 0, eligible: 0, protected: 0 }
  }

  const now = Date.now() / 1000
  let eligible = 0
  let protectedCount = 0
  const seenTargets = new Set<string>()

  for (const session of sessions) {
    const sid = String(session.id).trim()
    if (!sid) {
      continue
    }

    const rootId = String(session._lineage_root_id ?? sid).trim()
    const targetId = rootId || sid

    if (seenTargets.has(targetId)) {
      continue
    }
    seenTargets.add(targetId)

    const isPreserved = preserveIds.has(sid) || preserveIds.has(targetId)

    const startedAt = Number(session.started_at) || 0
    const lastActive = Number(session.last_active) || startedAt
    const endedAt = session.ended_at
    const isRecentlyActive =
      endedAt === null &&
      ACTIVE_GRACE_SECONDS > 0 &&
      now - lastActive < ACTIVE_GRACE_SECONDS

    if (isPreserved || isRecentlyActive) {
      protectedCount++
    } else {
      eligible++
    }
  }

  return {
    total: eligible + protectedCount,
    eligible,
    protected: protectedCount
  }
}
