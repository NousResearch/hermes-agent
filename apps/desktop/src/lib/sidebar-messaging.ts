import { sessionRecency } from '@/app/chat/sidebar/projects/workspace-groups'
import type { SessionInfo } from '@/hermes'
import { isMessagingSource, normalizeSessionSource, sessionSourceLabel } from '@/lib/session-source'

/** One self-managed messaging platform section in the sidebar. */
export interface MessagingSection {
  sourceId: string
  label: string
  sessions: SessionInfo[]
  total: number
  hasMore: boolean
}

/** What the backend still holds past the loaded messaging rows. */
export interface MessagingPaging {
  /** Exact per-platform total, once a "load more" has resolved it. */
  totalFor: (sourceId: string) => number | undefined
  /** The seed fetch was capped, so older threads may remain on disk. */
  truncated: boolean
}

/**
 * Split the archive the way the live sidebar queries split live rows: messaging
 * platform threads go to their platform sections, everything else to the
 * sessions list. Each archived row lands in exactly one of the two.
 */
export function splitArchivedSessions(archived: readonly SessionInfo[]): {
  messaging: SessionInfo[]
  sessions: SessionInfo[]
} {
  const messaging: SessionInfo[] = []
  const sessions: SessionInfo[] = []

  for (const session of archived) {
    if (isMessagingSource(session.source)) {
      messaging.push(session)
    } else {
      sessions.push(session)
    }
  }

  return { messaging, sessions }
}

/**
 * Split the messaging rows by source, newest platform first, rows within a
 * platform by recency. `paging` is null when the rows are already the complete
 * set (the archive), so no section offers "load more" or a backend total.
 */
export function buildMessagingSections(
  messaging: readonly SessionInfo[],
  isPinned: (session: SessionInfo) => boolean,
  paging: MessagingPaging | null
): MessagingSection[] {
  const bySource = new Map<string, SessionInfo[]>()
  // Rows this platform owns that the Pinned section is showing instead. The
  // backend's per-platform total counts them, so discount it or "load more"
  // promises rows that will never appear.
  const pinnedBySource = new Map<string, number>()

  for (const session of messaging) {
    const sourceId = normalizeSessionSource(session.source)

    if (!sourceId) {
      continue
    }

    if (isPinned(session)) {
      pinnedBySource.set(sourceId, (pinnedBySource.get(sourceId) ?? 0) + 1)

      continue
    }

    const list = bySource.get(sourceId) ?? []
    list.push(session)
    bySource.set(sourceId, list)
  }

  return [...bySource.entries()]
    .map(([sourceId, list]) => {
      const ordered = [...list].sort((a, b) => sessionRecency(b) - sessionRecency(a))
      const known = paging?.totalFor(sourceId)
      const unpinnedKnown = known == null ? null : Math.max(0, known - (pinnedBySource.get(sourceId) ?? 0))

      return {
        // Known exact total → more exist iff total exceeds loaded; otherwise
        // the seed fetch was capped, so assume more until a per-platform load
        // resolves the count.
        hasMore: unpinnedKnown != null ? unpinnedKnown > ordered.length : Boolean(paging?.truncated),
        label: sessionSourceLabel(sourceId) ?? sourceId,
        sessions: ordered,
        sourceId,
        total: Math.max(ordered.length, unpinnedKnown ?? 0)
      }
    })
    .sort((a, b) => sessionRecency(b.sessions[0]) - sessionRecency(a.sessions[0]))
}
