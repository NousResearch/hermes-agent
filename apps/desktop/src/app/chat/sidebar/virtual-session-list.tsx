import { useSortable } from '@dnd-kit/sortable'
import { useVirtualizer } from '@tanstack/react-virtual'
import { type FC, useCallback, useMemo, useRef } from 'react'

import type { SessionDragPayload } from '@/app/chat/composer/inline-refs'
import type { SessionInfo } from '@/hermes'
import { sessionTitle } from '@/lib/chat-runtime'
import { cn } from '@/lib/utils'
import { sessionPinId } from '@/store/session'
import type { SessionPresenceRecord } from '@/types/hermes'

import { SidebarSessionRow } from './session-row'

interface SessionRowCommonProps {
  isPinned: boolean
  isSelected: boolean
  isWorking: boolean
  onArchive: () => void
  onDelete: () => void
  onPin: () => void
  onResume: () => void
  archived?: boolean
  onRestore?: () => void
  selectable?: boolean
  selectionActive?: boolean
  checked?: boolean
  dragging?: boolean
  onSessionDragEnd?: () => void
  onSessionDragStart?: (payload: SessionDragPayload) => void
  onToggleSelect?: (mode: 'range' | 'single') => void
  bulkSelectedSessionIds?: readonly string[]
  onArchiveSelectedSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onDeleteSelectedSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onHaltSelectedSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onPromptSelectedSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  onRestoreSelectedSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onSteerSelectedSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
}

interface VirtualSessionListProps {
  activeSessionId: null | string
  className?: string
  dropActive?: boolean
  onArchiveSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onResumeSession: (sessionId: string) => void
  onTogglePin: (sessionId: string) => void
  pinned: boolean
  reorderable?: boolean
  sortable?: boolean
  sessions: SessionInfo[]
  workingSessionIdSet: Set<string>
  draggingSessionId?: string
  onSessionDragEnd?: () => void
  onSessionDragStart?: (payload: SessionDragPayload) => void
  /** Presence lookup map — flags rows live on another device/client. */
  presenceBySession?: Map<string, SessionPresenceRecord>
  /** Rows belong to the Archived section (restore instead of archive). */
  archived?: boolean
  onRestoreSession?: (sessionId: string) => void
  /** Multi-select wiring, provided by the owning section. */
  selectable?: boolean
  selectionActive?: boolean
  selectedSessionIds?: readonly string[]
  selectedIds?: ReadonlySet<string>
  sectionKey?: string
  sessionDragEnabled?: boolean
  sourceSectionKey?: null | string
  onToggleSelect?: (sessionId: string, mode: 'range' | 'single') => void
  onArchiveSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onDeleteSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onHaltSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onPromptSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  onRestoreSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onSteerSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
}

const ROW_ESTIMATE_PX = 28
const OVERSCAN_ROWS = 12
const SIDEBAR_SESSION_DND_ID_PREFIX = 'session:'
const SIDEBAR_SESSION_SORTABLE_TRANSITION = { duration: 150, easing: 'cubic-bezier(0.2, 0, 0, 1)' }

const sidebarSessionDndId = (id: string) => `${SIDEBAR_SESSION_DND_ID_PREFIX}${id}`

function sessionDragPayloadFor(
  session: SessionInfo,
  options: { archived: boolean; pinned: boolean }
): SessionDragPayload {
  return {
    archived: options.archived,
    id: session.id,
    pinId: sessionPinId(session),
    pinned: options.pinned,
    profile: session.profile || 'default',
    title: sessionTitle(session)
  }
}

export const VirtualSessionList: FC<VirtualSessionListProps> = ({
  activeSessionId,
  className,
  dropActive = false,
  onArchiveSession,
  onDeleteSession,
  onResumeSession,
  onTogglePin,
  pinned,
  reorderable = false,
  sortable = false,
  sessions,
  workingSessionIdSet,
  draggingSessionId,
  onSessionDragEnd,
  onSessionDragStart,
  presenceBySession,
  archived = false,
  onRestoreSession,
  selectable = false,
  selectionActive = false,
  selectedSessionIds,
  selectedIds,
  sectionKey,
  sessionDragEnabled = false,
  sourceSectionKey,
  onToggleSelect,
  onArchiveSessions,
  onDeleteSessions,
  onHaltSessions,
  onPromptSessions,
  onRestoreSessions,
  onSteerSessions
}) => {
  const scrollerRef = useRef<HTMLDivElement | null>(null)

  const virtualizer = useVirtualizer({
    count: sessions.length,
    estimateSize: () => ROW_ESTIMATE_PX,
    getItemKey: index => sessions[index]?.id ?? index,
    getScrollElement: () => scrollerRef.current,
    // jsdom-friendly default; the real rect takes over on first observe.
    initialRect: { height: 600, width: 240 },
    overscan: OVERSCAN_ROWS
  })

  const virtualItems = virtualizer.getVirtualItems()
  const totalSize = virtualizer.getTotalSize()
  const paddingTop = virtualItems[0]?.start ?? 0
  const paddingBottom = Math.max(0, totalSize - (virtualItems[virtualItems.length - 1]?.end ?? 0))

  const rows = virtualItems.map(virtualItem => {
    const session = sessions[virtualItem.index]

    if (!session) {
      return null
    }

    const rowIsChecked = selectedIds?.has(session.id) ?? false

    const commonProps: SessionRowCommonProps = {
      archived,
      bulkSelectedSessionIds:
        rowIsChecked && selectedSessionIds && selectedSessionIds.length > 1 ? selectedSessionIds : undefined,
      checked: rowIsChecked,
      dragging: draggingSessionId === session.id,
      isPinned: pinned,
      isSelected: session.id === activeSessionId,
      isWorking: workingSessionIdSet.has(session.id),
      onArchive: () => onArchiveSession(session.id),
      onArchiveSelectedSessions: onArchiveSessions,
      onDelete: () => onDeleteSession(session.id),
      onDeleteSelectedSessions: onDeleteSessions,
      onHaltSelectedSessions: archived ? undefined : onHaltSessions,
      onPin: () => onTogglePin(sessionPinId(session)),
      onPromptSelectedSessions: archived ? undefined : onPromptSessions,
      onRestore: onRestoreSession ? () => onRestoreSession(session.id) : undefined,
      onRestoreSelectedSessions: onRestoreSessions,
      onResume: () => onResumeSession(session.id),
      onSteerSelectedSessions: archived ? undefined : onSteerSessions,
      onSessionDragEnd,
      onSessionDragStart,
      onToggleSelect: onToggleSelect ? mode => onToggleSelect(session.id, mode) : undefined,
      selectable,
      selectionActive
    }

    const presence = presenceBySession?.get(session.id)

    const crossSectionPreview =
      sessionDragEnabled && dropActive && draggingSessionId === session.id && sourceSectionKey !== sectionKey

    return (sessionDragEnabled || sortable) && !crossSectionPreview ? (
      <VirtualSortableSessionRow
        archived={archived}
        commonProps={commonProps}
        index={virtualItem.index}
        key={session.id}
        measureRef={virtualizer.measureElement}
        pinned={pinned}
        presence={presence}
        sectionKey={sectionKey}
        session={session}
      />
    ) : (
      <SidebarSessionRow
        {...commonProps}
        data-index={virtualItem.index}
        key={session.id}
        nativeDraggable={!crossSectionPreview}
        presence={presence}
        ref={virtualizer.measureElement}
        reorderable={reorderable || crossSectionPreview}
        session={session}
      />
    )
  })

  const list = (
    <div className={cn('relative min-h-0 flex-1 overflow-y-auto overscroll-contain', className)} ref={scrollerRef}>
      <div className="grid gap-px" style={{ paddingBottom: `${paddingBottom}px`, paddingTop: `${paddingTop}px` }}>
        {rows}
      </div>
    </div>
  )

  return list
}

interface VirtualSortableSessionRowProps {
  archived: boolean
  commonProps: SessionRowCommonProps
  index: number
  measureRef: (node: Element | null) => void
  pinned: boolean
  presence?: SessionPresenceRecord
  sectionKey?: string
  session: SessionInfo
}

function VirtualSortableSessionRow({
  archived,
  commonProps,
  index,
  measureRef,
  pinned,
  presence,
  sectionKey,
  session
}: VirtualSortableSessionRowProps) {
  const payload = useMemo(() => sessionDragPayloadFor(session, { archived, pinned }), [archived, pinned, session])

  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({
    data: {
      sessionDragPayload: payload,
      sessionId: session.id,
      sourceSectionKey: sectionKey
    },
    id: sidebarSessionDndId(session.id),
    transition: SIDEBAR_SESSION_SORTABLE_TRANSITION
  })

  const refMerged = useCallback(
    (node: HTMLDivElement | null) => {
      setNodeRef(node)
      measureRef(node)
    },
    [measureRef, setNodeRef]
  )

  return (
    <SidebarSessionRow
      {...commonProps}
      {...attributes}
      {...listeners}
      data-index={index}
      dragging={isDragging}
      key={session.id}
      nativeDraggable
      presence={presence}
      ref={refMerged}
      reorderable
      session={session}
      style={{
        transform: transform ? `translate3d(0px, ${transform.y}px, 0)` : undefined,
        transition: isDragging ? undefined : transition,
        willChange: isDragging ? 'transform' : undefined
      }}
    />
  )
}
