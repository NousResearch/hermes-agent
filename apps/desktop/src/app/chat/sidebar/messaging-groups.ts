import { hasSessionSourcePartition } from '@/contrib/session-sources'
import type { Contribution } from '@/contrib/types'
import { sessionSourcePresentation } from '@/lib/session-source'
import type { SessionInfo } from '@/types/hermes'

import { sessionRecency } from './projects/workspace-groups'

export interface MessagingSection {
  groupId: string
  sourceId: string
  /** Platform id for the section icon (differs from sourceId when a plugin rule sets `icon`). */
  iconId: string
  label: string
  sessions: SessionInfo[]
  total: number
  hasMore: boolean
}

interface BuildMessagingGroupsOptions {
  sessions: readonly SessionInfo[]
  getSourceTotal: (sourceId: string) => number | undefined
  truncated: boolean
  isPinned: (session: SessionInfo) => boolean
  profile: string
  contributions?: readonly Contribution[]
}

export function buildMessagingGroups({
  sessions,
  getSourceTotal,
  truncated,
  isPinned,
  profile,
  contributions
}: BuildMessagingGroupsOptions): MessagingSection[] {
  const byGroup = new Map<string, MessagingSection>()
  const pinnedBySource = new Map<string, number>()
  const loadedBySource = new Map<string, number>()

  for (const session of sessions) {
    const presentation = sessionSourcePresentation(session, contributions)

    if (!presentation) {
      continue
    }

    const { groupId, iconId, label, sourceId } = presentation

    if (isPinned(session)) {
      pinnedBySource.set(sourceId, (pinnedBySource.get(sourceId) ?? 0) + 1)

      continue
    }

    loadedBySource.set(sourceId, (loadedBySource.get(sourceId) ?? 0) + 1)

    const group = byGroup.get(groupId) ?? { groupId, hasMore: false, iconId, label, sessions: [], sourceId, total: 0 }

    group.sessions.push(session)
    byGroup.set(groupId, group)
  }

  return [...byGroup.values()]
    .map(group => {
      const ordered = [...group.sessions].sort((a, b) => sessionRecency(b) - sessionRecency(a))
      const known = getSourceTotal(group.sourceId)
      // Backend totals count the transport, including every presentation group
      // and pinned row. Pagination must keep using that same transport identity.
      const unpinnedKnown = known == null ? null : Math.max(0, known - (pinnedBySource.get(group.sourceId) ?? 0))
      const loaded = loadedBySource.get(group.sourceId) ?? 0
      const partitioned = hasSessionSourcePartition(group.sourceId, profile, contributions)

      return {
        ...group,
        hasMore: unpinnedKnown != null ? unpinnedKnown > loaded : truncated,
        sessions: ordered,
        // Neither a filtered group nor its remainder owns the source's exact
        // total, even when the current page has not loaded a matching row yet.
        total: partitioned ? ordered.length : Math.max(ordered.length, unpinnedKnown ?? 0)
      }
    })
    .sort((a, b) => sessionRecency(b.sessions[0]) - sessionRecency(a.sessions[0]))
}
