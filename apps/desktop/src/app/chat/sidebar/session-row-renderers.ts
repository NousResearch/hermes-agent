import type * as React from 'react'
import { useCallback } from 'react'

import type { SessionInfo } from '@/hermes'
import { flattenSessionsWithBranches } from '@/lib/session-branch-tree'
import {
  groupEntriesByRecency,
  hideCollapsedGroupRows,
  type SidebarListRow,
  toSessionRows
} from '@/lib/session-date-groups'

import { orderRowsWithinGroups } from './order'

export interface SessionRowRenderers {
  /** A project row's preview rows: complete recency groups, so a burst and its branches stay together
   *  and a divider reads as the project's own ("project:<id>" keys). */
  renderPreviewRows: (items: SessionInfo[], projectId: string) => React.ReactNode
  /** Sessions inside repos/worktrees: date-ordered and static. */
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  /** `renderRows` with date dividers folded in — entered-project lanes spanning days read
   *  chronologically, matching the flat recents list. */
  renderRowsDated: (sessions: SessionInfo[]) => React.ReactNode
}

/**
 * The three ways the section renders a list of sessions, all built on the section's own row builders.
 *
 * `renderPreviewRows` limits complete groups (not sessions), so a burst and its branches stay
 * together; the boundaries come from the whole pool, exactly like the Updated list. The hand-picked
 * order is applied INSIDE each group, so dragging a row ranks it among its own day's chats.
 */
export function useSessionRowRenderers({
  grouping,
  isListGroupOpen,
  manualOrderIds,
  renderListRow,
  renderRow,
  showAllSessions
}: {
  grouping: 'date' | 'none' | 'status'
  isListGroupOpen: (key: string) => boolean
  manualOrderIds?: string[]
  renderListRow: (row: SidebarListRow, draggable: boolean, action?: React.ReactNode) => React.ReactNode
  renderRow: (session: SessionInfo, draggable: boolean, branchStem?: string) => React.ReactNode
  showAllSessions: boolean
}): SessionRowRenderers {
  const renderRows = useCallback(
    (items: SessionInfo[]) =>
      flattenSessionsWithBranches(items).map(({ branchStem, session }) => renderRow(session, false, branchStem)),
    [renderRow]
  )

  const renderPreviewRows = useCallback(
    (items: SessionInfo[], projectId: string) => {
      const rows = groupEntriesByRecency(
        flattenSessionsWithBranches(items),
        undefined,
        undefined,
        showAllSessions ? Infinity : 2
      ).map(row => (row.kind === 'divider' ? { ...row, key: `project:${projectId}:${row.key}` } : row))

      const ordered = manualOrderIds?.length ? orderRowsWithinGroups(rows, manualOrderIds) : rows

      return hideCollapsedGroupRows(ordered, isListGroupOpen).map(row => renderListRow(row, false))
    },
    [isListGroupOpen, manualOrderIds, renderListRow, showAllSessions]
  )

  const renderRowsDated = useCallback(
    (items: SessionInfo[]) => {
      const entries = flattenSessionsWithBranches(items)

      const rows = grouping === 'date' ? groupEntriesByRecency(entries) : toSessionRows(entries)

      return hideCollapsedGroupRows(rows, isListGroupOpen).map(row => renderListRow(row, false))
    },
    [grouping, isListGroupOpen, renderListRow]
  )

  return { renderPreviewRows, renderRows, renderRowsDated }
}
