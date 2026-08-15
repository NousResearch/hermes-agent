import { useSortable } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useVirtualizer } from '@tanstack/react-virtual'
import type * as React from 'react'
import { type FC, useRef } from 'react'

import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { type SidebarListRow } from '@/lib/session-date-groups'
import { sessionBucketLabel } from '@/lib/time'
import { cn } from '@/lib/utils'
import { sessionPinId } from '@/store/session'

import { SidebarDateDivider } from './chrome'
import { SidebarSessionRow } from './session-row'

interface SessionRowCommonProps {
  branchStem?: string
  card?: boolean
  isPinned: boolean
  isSelected: boolean
  onArchive: () => void
  onBranch?: () => void
  onDelete: () => void
  onPin: () => void
  onResume: () => void
  reorderable?: boolean
  showProfile?: boolean
}

export interface VirtualSessionListProps {
  activeSessionId: null | string
  /** Render every session row as the three-line inbox card. */
  card?: boolean
  className?: string
  /** Hover-revealed control for date dividers (the group-level "+"). */
  dividerAction?: React.ReactNode
  rows: SidebarListRow[]
  onArchiveSession: (sessionId: string) => void
  onBranchSession?: (sessionId: string, profile?: string) => void
  onDeleteSession: (sessionId: string) => void
  onResumeSession: (sessionId: string) => void
  onTogglePin: (sessionId: string) => void
  pinned: boolean
  showProfileTags?: boolean
  sortable: boolean
}

const ROW_ESTIMATE_PX = 28
// Matches the card's typical rendered height (four lines when a preview
// exists) so long card lists don't jump under the scroll thumb before
// self-measurement catches up.
const CARD_ROW_ESTIMATE_PX = 66
const OVERSCAN_ROWS = 12

export const VirtualSessionList: FC<VirtualSessionListProps> = ({
  activeSessionId,
  card = false,
  className,
  dividerAction,
  rows: listRows,
  onArchiveSession,
  onBranchSession,
  onDeleteSession,
  onResumeSession,
  onTogglePin,
  pinned,
  showProfileTags = false,
  sortable
}) => {
  const { t } = useI18n()
  const dividerLabels = t.sidebar.dateDivider
  const scrollerRef = useRef<HTMLDivElement | null>(null)

  const virtualizer = useVirtualizer({
    count: listRows.length,
    estimateSize: () => (card ? CARD_ROW_ESTIMATE_PX : ROW_ESTIMATE_PX),
    getItemKey: index => {
      const row = listRows[index]

      return row ? (row.kind === 'divider' ? row.key : row.entry.session.id) : index
    },
    getScrollElement: () => scrollerRef.current,
    // jsdom-friendly default; the real rect takes over on first observe.
    initialRect: { height: 600, width: 240 },
    overscan: OVERSCAN_ROWS
  })

  const virtualItems = virtualizer.getVirtualItems()
  const totalSize = virtualizer.getTotalSize()

  const rows = virtualItems.map(virtualItem => {
    const row = listRows[virtualItem.index]

    if (!row) {
      return null
    }

    const itemStyle: React.CSSProperties = {
      left: 0,
      position: 'absolute',
      top: 0,
      transform: `translateY(${virtualItem.start}px)`,
      width: '100%'
    }

    // Dividers are non-sortable, self-measured rows interleaved with sessions.
    if (row.kind === 'divider') {
      return (
        <div data-index={virtualItem.index} key={row.key} ref={virtualizer.measureElement} style={itemStyle}>
          <SidebarDateDivider
            action={dividerAction}
            label={'label' in row ? row.label : sessionBucketLabel(row.bucket, dividerLabels)}
          />
        </div>
      )
    }

    const { branchStem, session } = row.entry
    const reorderable = sortable && !branchStem

    const commonProps: SessionRowCommonProps = {
      branchStem,
      card,
      isPinned: pinned,
      isSelected: session.id === activeSessionId,
      onArchive: () => onArchiveSession(session.id),
      onBranch: onBranchSession ? () => onBranchSession(session.id, session.profile) : undefined,
      onDelete: () => onDeleteSession(session.id),
      onPin: () => onTogglePin(sessionPinId(session)),
      onResume: () => onResumeSession(session.id),
      reorderable,
      showProfile: showProfileTags
    }

    return reorderable ? (
      <div data-index={virtualItem.index} key={session.id} ref={virtualizer.measureElement} style={itemStyle}>
        <VirtualSortableRow rowProps={commonProps} session={session} />
      </div>
    ) : (
      <div data-index={virtualItem.index} key={session.id} ref={virtualizer.measureElement} style={itemStyle}>
        <SidebarSessionRow {...commonProps} session={session} />
      </div>
    )
  })

  // When sortable, the caller wraps this in a ReorderableList that owns the
  // DndContext + SortableContext (keyed on the same ids); the virtualized rows
  // just consume that context via useSortable.
  return (
    <div
      // scrollbar-fade, NOT scrollbar-overlay: overlay opts out of the themed
      // thin scrollbar entirely, and on Windows (no native overlay scrollbars)
      // Chromium then paints the classic always-visible gutter. The themed
      // fade bar reserves its 4px on every platform but stays invisible until
      // hover — and the wrapper no longer stacks a second scroller, so the
      // double-gutter this class change was reaching for is already gone.
      className={cn(
        'scrollbar-fade relative min-h-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain',
        className
      )}
      ref={scrollerRef}
    >
      <div className="relative" style={{ height: `${totalSize}px` }}>
        {rows}
      </div>
    </div>
  )
}

interface VirtualSortableRowProps {
  rowProps: SessionRowCommonProps
  session: SessionInfo
}

function VirtualSortableRow({ rowProps, session }: VirtualSortableRowProps) {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({ id: session.id })

  return (
    <SidebarSessionRow
      {...rowProps}
      dragging={isDragging}
      dragHandleProps={{ ...attributes, ...listeners }}
      ref={setNodeRef}
      reorderable
      session={session}
      style={{ transform: CSS.Transform.toString(transform), transition }}
    />
  )
}
