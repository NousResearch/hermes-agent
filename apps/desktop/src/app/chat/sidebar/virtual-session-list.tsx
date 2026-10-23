import { useSortable } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useStore } from '@nanostores/react'
import { useVirtualizer } from '@tanstack/react-virtual'
import type * as React from 'react'
import { type FC, memo, useCallback, useEffect, useRef } from 'react'

import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { type SidebarListRow } from '@/lib/session-date-groups'
import { sessionBucketLabel } from '@/lib/time'
import { cn } from '@/lib/utils'
import { sessionPinId } from '@/store/session'
import { $sessionListDensity } from '@/store/session-list-density'

import { SidebarDateDivider } from './chrome'
import { SidebarSessionRow } from './session-row'
import { SESSION_CARD_ROW_ESTIMATE_PX, sessionRowEstimate } from './session-row-details'

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

// Matches the card's typical rendered height (four lines when a preview
// exists) so long card lists don't jump under the scroll thumb before
// self-measurement catches up. Kept at/above the wrapped-title worst case —
// see SESSION_CARD_ROW_ESTIMATE_PX (#88473).
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

  // Option closures must be IDENTITY-STABLE across renders. The react-virtual
  // adapter calls setOptions every render, and virtual-core's measurement
  // memos key on these closures: a fresh estimateSize/getItemKey each render
  // invalidated the memo and rebuilt the ENTIRE measurement cache (a fresh
  // position/size array over every row) on every store tick — O(N) allocation
  // churn that GC'd the renderer into a launch OOM at ~11.5k sessions
  // (#62964). useCallback keeps them referentially equal while still reading
  // the current rows/card through the refs.
  const rowsRef = useRef(listRows)
  rowsRef.current = listRows
  const cardRef = useRef(card)
  cardRef.current = card

  const estimateSize = useCallback(
    (index: number) => {
      const row = rowsRef.current[index]

      if (row?.kind === 'divider') {
        return DIVIDER_ESTIMATE_PX
      }

      return cardRef.current ? SESSION_CARD_ROW_ESTIMATE_PX : sessionRowEstimate(density)
    },
    [density]
  )

  // Key by (profile, id), matching the React row key: twins with one stored
  // id across profiles are distinct rows (#92454) and must not share a
  // measurement-cache slot — a bare id made each twin's measured size thrash
  // the other's on every observe.
  const getItemKey = useCallback((index: number) => {
    const row = rowsRef.current[index]

    if (!row) {
      return index
    }

    return row.kind === 'divider' ? row.key : `${row.entry.session.profile ?? ''}::${row.entry.session.id}`
  }, [])

  const getScrollElement = useCallback(() => scrollerRef.current, [])

  const virtualizer = useVirtualizer({
    count: listRows.length,
    estimateSize,
    getItemKey,
    getScrollElement,
    // jsdom-friendly default; the real rect takes over on first observe.
    initialRect: { height: 600, width: 240 },
    overscan: OVERSCAN_ROWS
  })

  // Rows are measured after paint, so changing density OR toggling Inbox
  // cards must invalidate cached measurements from the previous mode before
  // off-screen rows re-enter (#88473).
  useEffect(() => virtualizer.measure(), [card, density, virtualizer])

  // Latest-handler bag: the row shells below are memoized on identity-stable
  // inputs, so the caller's callbacks reach them through this ref (one stable
  // object for the list's lifetime) instead of through per-row closures that
  // would defeat the shell's memo. A store tick that touched no session then
  // re-renders NOTHING in the list — the piece #62964 was missing. Handler
  // swaps still take effect: the ref always reads the latest.
  const handlersRef = useRef({ onArchiveSession, onBranchSession, onDeleteSession, onResumeSession, onTogglePin, onToggleUnread })
  handlersRef.current = { onArchiveSession, onBranchSession, onDeleteSession, onResumeSession, onTogglePin, onToggleUnread }

  // Stable measureElement prop for the shells (same identity as the
  // virtualizer's own method; wrapping keeps the shell's memo honest).
  const measure = useCallback((node: HTMLElement | null) => virtualizer.measureElement(node), [virtualizer])

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

    // Key by (profile, id): twins with the same stored id in two profiles are
    // distinct rows (#92454) — a bare-id key misattributes rendered state.
    const rowKey = `${session.profile ?? ''}::${session.id}`

    // The row SHELL is memoized on identity-stable props only (session ref,
    // geometry, primitive flags). A parent re-render that moved nothing
    // re-renders no row: every mounted row's re-render would re-run dnd-kit's
    // O(items) useSortable work (items.indexOf + items.slice for the
    // resize-observer id list), which multiplied by the sidebar's per-tick
    // re-renders was the launch OOM (#62964). Per-session callbacks bind
    // inside the shell through the handler ref, so callback identity never
    // re-renders a row and a handler swap still takes effect on next call.
    return (
      <VirtualRowShell
        branchStem={branchStem}
        card={card}
        handlersRef={handlersRef}
        index={virtualItem.index}
        key={rowKey}
        measure={measure}
        pinned={pinned}
        selected={session.id === activeSessionId}
        session={session}
        showProfile={showProfileTags}
        sortable={sortable && !branchStem}
        start={virtualItem.start}
        unread={session.unread === true}
      />
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

/** The list-level callbacks a row shell reads through a latest-handler ref. */
export interface VirtualSessionListHandlers {
  onArchiveSession: (sessionId: string) => void
  onBranchSession?: (sessionId: string, profile?: string) => void
  onDeleteSession: (sessionId: string) => void
  onResumeSession: (sessionId: string, session?: SessionInfo) => void
  onTogglePin: (sessionId: string) => void
  onToggleUnread: (sessionId: string) => void
}

interface VirtualRowShellProps {
  branchStem: string | undefined
  card: boolean
  /** Latest-handler bag, identity-stable for the list's lifetime. */
  handlersRef: React.RefObject<VirtualSessionListHandlers>
  index: number
  measure: (node: HTMLElement | null) => void
  pinned: boolean
  selected: boolean
  session: SessionInfo
  showProfile: boolean
  sortable: boolean
  start: number
  unread: boolean
}

function virtualRowShellPropsEqual(a: VirtualRowShellProps, b: VirtualRowShellProps): boolean {
  return (
    a.session === b.session &&
    a.branchStem === b.branchStem &&
    a.card === b.card &&
    a.handlersRef === b.handlersRef &&
    a.index === b.index &&
    a.measure === b.measure &&
    a.pinned === b.pinned &&
    a.selected === b.selected &&
    a.showProfile === b.showProfile &&
    a.sortable === b.sortable &&
    a.start === b.start &&
    a.unread === b.unread
  )
}

/** The per-row prop bundle both row variants need, with per-session callbacks
 * bound through the latest-handler ref (never through fresh closures — those
 * would defeat the shell's memo). */
function shellRowProps(
  {
    branchStem,
    card,
    handlersRef,
    pinned,
    selected,
    showProfile,
    sortable,
    unread
  }: Omit<VirtualRowShellProps, 'index' | 'measure' | 'session' | 'start'>,
  session: SessionInfo
): SessionRowCommonProps {
  const handlers = handlersRef.current

  return {
    branchStem,
    card,
    isPinned: pinned,
    isSelected: selected,
    onArchive: () => handlers.onArchiveSession(session.id),
    onBranch: handlers.onBranchSession ? () => handlers.onBranchSession?.(session.id, session.profile) : undefined,
    onDelete: () => handlers.onDeleteSession(session.id),
    onPin: () => handlers.onTogglePin(sessionPinId(session)),
    onToggleUnread: () => handlers.onToggleUnread(session.id),
    onResume: () => handlers.onResumeSession(session.id, session),
    reorderable: sortable,
    showProfile,
    unread
  }
}

/**
 * One virtualized session row: the measured absolutely-positioned cell plus
 * its sortable (dnd-kit) or plain row. Memoized on identity-stable inputs so
 * a section re-render that changed nothing about THIS row renders nothing —
 * the budget the render-budget test pins (#62964).
 */
const VirtualRowShell = memo(
  function VirtualRowShell({ index, measure, session, sortable, start, ...flags }: VirtualRowShellProps) {
    return (
      <div
        data-index={index}
        ref={measure}
        style={{ left: 0, position: 'absolute', top: 0, transform: `translateY(${start}px)`, width: '100%' }}
      >
        {sortable ? (
          <VirtualSortableRow flags={flags} session={session} />
        ) : (
          <SidebarSessionRow {...shellRowProps({ ...flags, sortable: false }, session)} session={session} />
        )}
      </div>
    )
  },
  virtualRowShellPropsEqual
)

interface VirtualSortableRowProps {
  flags: Omit<VirtualRowShellProps, 'index' | 'measure' | 'session' | 'start' | 'sortable'>
  session: SessionInfo
}

function VirtualSortableRow({ flags, session }: VirtualSortableRowProps) {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({ id: session.id })
  // The sortable row IS reorderable by construction.
  const rowProps = shellRowProps({ ...flags, sortable: true }, session)

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
