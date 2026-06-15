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

interface SessionArchivePreserveOptions {
  activeSessionId?: null | string
  pinnedSessionIds?: Iterable<string>
  selectedSessionId?: null | string
  workingSessionIds?: Iterable<string>
}

function addPreserveId(ids: Set<string>, value: null | string | undefined) {
  const id = String(value ?? '').trim()

  if (id) {
    ids.add(id)
  }
}

/**
 * Build the id set shared by the archive-all action and preview dialog. The
 * backend accepts both live session ids and lineage roots, so include both for
 * the current selected/active row when that row is present in the surfaced list.
 */
export function sessionArchivePreserveIds(
  sessions: SessionInfo[],
  {
    activeSessionId,
    pinnedSessionIds = [],
    selectedSessionId,
    workingSessionIds = []
  }: SessionArchivePreserveOptions
): Set<string> {
  const preserveIds = new Set<string>()

  for (const id of pinnedSessionIds) {
    addPreserveId(preserveIds, id)
  }

  for (const id of workingSessionIds) {
    addPreserveId(preserveIds, id)
  }

  addPreserveId(preserveIds, selectedSessionId)
  addPreserveId(preserveIds, activeSessionId)

  for (const session of sessions) {
    const sessionId = String(session.id ?? '').trim()
    const rootId = String(session._lineage_root_id ?? sessionId).trim()

    const isCurrent =
      preserveIds.has(sessionId) ||
      (rootId !== '' && preserveIds.has(rootId))

    if (isCurrent) {
      addPreserveId(preserveIds, sessionId)
      addPreserveId(preserveIds, rootId)
    }
  }

  return preserveIds
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
