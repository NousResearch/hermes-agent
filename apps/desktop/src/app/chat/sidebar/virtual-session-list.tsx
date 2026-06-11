import { useVirtualizer } from '@tanstack/react-virtual'
import { type FC, useRef } from 'react'

import type { SessionDragPayload } from '@/app/chat/composer/inline-refs'
import type { SessionInfo } from '@/hermes'
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
  onRestoreSelectedSessions?: (sessionIds: string[]) => Promise<unknown> | void
}

interface VirtualSessionListProps {
  activeSessionId: null | string
  className?: string
  onArchiveSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onResumeSession: (sessionId: string) => void
  onTogglePin: (sessionId: string) => void
  pinned: boolean
  reorderable?: boolean
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
  onToggleSelect?: (sessionId: string, mode: 'range' | 'single') => void
  onArchiveSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onDeleteSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onRestoreSessions?: (sessionIds: string[]) => Promise<unknown> | void
}

const ROW_ESTIMATE_PX = 28
const OVERSCAN_ROWS = 12

export const VirtualSessionList: FC<VirtualSessionListProps> = ({
  activeSessionId,
  className,
  onArchiveSession,
  onDeleteSession,
  onResumeSession,
  onTogglePin,
  pinned,
  reorderable = false,
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
  onToggleSelect,
  onArchiveSessions,
  onDeleteSessions,
  onRestoreSessions
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
      bulkSelectedSessionIds: rowIsChecked && selectedSessionIds && selectedSessionIds.length > 1 ? selectedSessionIds : undefined,
      checked: rowIsChecked,
      dragging: draggingSessionId === session.id,
      isPinned: pinned,
      isSelected: session.id === activeSessionId,
      isWorking: workingSessionIdSet.has(session.id),
      onArchive: () => onArchiveSession(session.id),
      onArchiveSelectedSessions: onArchiveSessions,
      onDelete: () => onDeleteSession(session.id),
      onDeleteSelectedSessions: onDeleteSessions,
      onPin: () => onTogglePin(sessionPinId(session)),
      onRestore: onRestoreSession ? () => onRestoreSession(session.id) : undefined,
      onRestoreSelectedSessions: onRestoreSessions,
      onResume: () => onResumeSession(session.id),
      onSessionDragEnd,
      onSessionDragStart,
      onToggleSelect: onToggleSelect ? mode => onToggleSelect(session.id, mode) : undefined,
      selectable,
      selectionActive
    }

    const presence = presenceBySession?.get(session.id)

    return (
      <SidebarSessionRow
        {...commonProps}
        data-index={virtualItem.index}
        key={session.id}
        presence={presence}
        ref={virtualizer.measureElement}
        reorderable={reorderable}
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
