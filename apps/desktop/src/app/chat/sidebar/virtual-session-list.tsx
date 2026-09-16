import { useSortable } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useStore } from '@nanostores/react'
import { useVirtualizer } from '@tanstack/react-virtual'
import type * as React from 'react'
import { type FC, useEffect, useRef } from 'react'

import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { type SidebarListRow } from '@/lib/session-date-groups'
import { sessionBucketLabel } from '@/lib/time'
import { cn } from '@/lib/utils'
import { sessionPinId } from '@/store/session'
import { $sessionListDensity } from '@/store/session-list-density'

import { SidebarDateDivider } from './chrome'
import { SidebarSessionRow } from './session-row'
import { sessionRowEstimate } from './session-row-details'

interface SessionRowCommonProps {
  branchStem?: string
  card?: boolean
  isPinned: boolean
  isSelected: boolean
  unread: boolean
  onArchive: () => void
  onBranch?: () => void
  onDelete: () => void
  onPin: () => void
  onToggleUnread: () => void
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
  /** Collapse/expand the sessions under a date or status divider. */
  dividerToggle?: {
    ariaLabel: (label: string, open: boolean) => string
    onToggle: (key: string) => void
    open: (key: string) => boolean
  }
  rows: SidebarListRow[]
  onArchiveSession: (sessionId: string) => void
  onBranchSession?: (sessionId: string, profile?: string) => void
  onDeleteSession: (sessionId: string) => void
  onResumeSession: (sessionId: string, session?: SessionInfo) => void
  onTogglePin: (sessionId: string) => void
  onToggleUnread: (sessionId: string) => void
  pinned: boolean
  showProfileTags?: boolean
  sortable: boolean
}

// Cover the tallest allowed card before self-measurement: 12px body padding,
// the 20px header action, two 6.4px card gaps, two 13px title lines, the 2.4px
// title/preview gap, and 13.5px preview + footer lines = 100.2px. Rounding up
// keeps an initially unmeasured list from correcting downward under the thumb.
const CARD_ROW_ESTIMATE_PX = 101
const DIVIDER_ESTIMATE_PX = 28
const OVERSCAN_ROWS = 12

export const VirtualSessionList: FC<VirtualSessionListProps> = ({
  activeSessionId,
  card = false,
  className,
  dividerAction,
  dividerToggle,
  rows: listRows,
  onArchiveSession,
  onBranchSession,
  onDeleteSession,
  onResumeSession,
  onTogglePin,
  onToggleUnread,
  pinned,
  showProfileTags = false,
  sortable
}) => {
  const { t } = useI18n()
  const dividerLabels = t.sidebar.dateDivider
  const scrollerRef = useRef<HTMLDivElement | null>(null)
  const density = useStore($sessionListDensity)

  const virtualizer = useVirtualizer({
    count: listRows.length,
    estimateSize: (index: number) => {
      const row = listRows[index]

      if (row?.kind === 'divider') {
        return DIVIDER_ESTIMATE_PX
      }

      return card ? CARD_ROW_ESTIMATE_PX : sessionRowEstimate(density)
    },
    getItemKey: index => {
      const row = listRows[index]

      return row ? (row.kind === 'divider' ? row.key : row.entry.session.id) : index
    },
    getScrollElement: () => scrollerRef.current,
    // jsdom-friendly default; the real rect takes over on first observe.
    initialRect: { height: 600, width: 240 },
    overscan: OVERSCAN_ROWS
  })

  // Rows are measured after paint, so changing density must invalidate cached
  // measurements from the previous mode before off-screen rows re-enter.
  useEffect(() => virtualizer.measure(), [density, virtualizer])

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
      const label = 'label' in row ? row.label : sessionBucketLabel(row.bucket, dividerLabels)
      const open = dividerToggle?.open(row.key) ?? true

      return (
        <div data-index={virtualItem.index} key={row.key} ref={virtualizer.measureElement} style={itemStyle}>
          <SidebarDateDivider
            action={dividerAction}
            label={label}
            toggle={
              dividerToggle
                ? {
                    ariaLabel: dividerToggle.ariaLabel(label, open),
                    onToggle: () => dividerToggle.onToggle(row.key),
                    open
                  }
                : undefined
            }
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
      onToggleUnread: () => onToggleUnread(session.id),
      onResume: () => onResumeSession(session.id, session),
      reorderable,
      showProfile: showProfileTags,
      unread: session.unread === true
    }

    // Key by (profile, id): twins with the same stored id in two profiles are
    // distinct rows (#92454) — a bare-id key misattributes rendered state.
    const rowKey = `${session.profile ?? ''}::${session.id}`

    return reorderable ? (
      <div data-index={virtualItem.index} key={rowKey} ref={virtualizer.measureElement} style={itemStyle}>
        <VirtualSortableRow rowProps={commonProps} session={session} />
      </div>
    ) : (
      <div data-index={virtualItem.index} key={rowKey} ref={virtualizer.measureElement} style={itemStyle}>
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
      //
      // No `overscroll-contain` here: this scroller is NESTED inside the
      // sidebar's own scroll container (index.tsx SCROLL_Y). Containing
      // overscroll on the inner scroller swallowed wheel events at its scroll
      // boundaries instead of chaining them to the outer sidebar scroller,
      // which read as a wheel dead-zone mid-list once 25+ sessions
      // virtualized (#84964) — the scrollbar still dragged, only the wheel
      // died. The outer sidebar scroller keeps its own overscroll-contain, so
      // the gesture still never escapes the sidebar.
      className={cn('scrollbar-fade relative min-h-0 flex-1 overflow-x-hidden overflow-y-auto', className)}
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
