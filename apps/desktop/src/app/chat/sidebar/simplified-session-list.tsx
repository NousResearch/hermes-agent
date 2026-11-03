import { useStore } from '@nanostores/react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { useRef } from 'react'
import { useNavigate } from 'react-router'

import { $apiRequestScope } from '@/api/client'
import { openSession } from '@/app/open-session'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { RowButton } from '@/components/ui/row-button'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { cn } from '@/lib/utils'
import { $activeGatewayProfile } from '@/store/profile'
import { $sessionListDensity } from '@/store/session-list-density'
import { sessionOwnerRouteFromRow } from '@/store/session-request-router'

import { SessionStatusDot } from '../session-status-dot'

import { SessionActionsMenu } from './session-actions-menu'
import { sessionRowEstimate } from './session-row-details'
import { sessionRowKey } from './simplified-session-identity'

interface SimplifiedSessionListProps {
  sessions: SessionInfo[]
  activeSessionId: string | null
  pinned?: boolean
  ambiguousIds: ReadonlySet<string>
  onResumeSession: (id: string, row?: SessionInfo) => void
  onDeleteSession: (id: string) => void
  onArchiveSession: (id: string) => void
  onBranchSession: (id: string) => void
  onUpdate: (row: SessionInfo, field: 'pinned' | 'unread', value: boolean) => void
}

/** A flat v2 list: owner-qualified keys, with the desktop's existing dots and row menu. */
export function SimplifiedSessionList(props: SimplifiedSessionListProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const activeProfile = useStore($activeGatewayProfile)
  const activeScope = useStore($apiRequestScope)
  const scrollRef = useRef<HTMLDivElement>(null)
  const density = useStore($sessionListDensity)
  const virtual = props.sessions.length >= 60

  const list = useVirtualizer({
    count: virtual ? props.sessions.length : 0,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => sessionRowEstimate(density),
    getItemKey: index => sessionRowKey(props.sessions[index]),
    overscan: 8
  })

  const renderRow = (row: SessionInfo) => {
    const title = sessionTitle(row)

    const selected =
      row.id === props.activeSessionId &&
      (row.profile || 'default') === activeProfile &&
      (!row.connection_id || row.connection_id === activeScope.connectionId)

    return (
      <div className="group flex min-w-0 items-center gap-1" key={sessionRowKey(row)}>
        <RowButton
          aria-current={selected ? 'page' : undefined}
          className={cn(
            'flex min-w-0 flex-1 items-center gap-2 rounded-sm px-2 py-1.5 text-left text-xs hover:bg-(--chrome-action-hover)',
            selected && 'bg-(--chrome-action-hover)'
          )}
          onClick={event => {
            if (event.metaKey || event.ctrlKey)
              {openSession(row.id, navigate, event.shiftKey ? 'window' : 'tab', {
                workspaceMode: 'sessions',
                ownerRoute: sessionOwnerRouteFromRow(row)
              })}
            else {props.onResumeSession(row.id, row)}
          }}
        >
          <SessionStatusDot session={row} storedSessionId={row.id} />
          <span className="min-w-0 flex-1 truncate">{title}</span>
          <span
            aria-label={`${t.simplifiedAgents.agents}: ${row.profile || 'default'}`}
            className="max-w-24 truncate text-xs text-(--ui-text-tertiary)"
          >
            {row.profile || 'default'}
          </span>
        </RowButton>
        {/* shortcut: legacy actions accept bare IDs; restore menus for twins when those actions accept owners. */}
        {!props.ambiguousIds.has(row.id) && (
          <SessionActionsMenu
            onArchive={() => props.onArchiveSession(row.id)}
            onBranch={() => props.onBranchSession(row.id)}
            onDelete={() => props.onDeleteSession(row.id)}
            onPin={() => props.onUpdate(row, 'pinned', !props.pinned)}
            onToggleUnread={() => props.onUpdate(row, 'unread', row.unread !== true)}
            pinned={props.pinned}
            profile={row.profile}
            sessionId={row.id}
            title={title}
            unread={row.unread === true}
          >
            <Button aria-label={t.sidebar.row.sessionActions} size="icon" variant="ghost">
              <Codicon name="kebab-vertical" />
            </Button>
          </SessionActionsMenu>
        )}
      </div>
    )
  }

  return (
    <div className={virtual ? 'h-64 overflow-y-auto' : 'flex flex-col gap-px'} ref={scrollRef}>
      {virtual ? (
        <div style={{ height: list.getTotalSize(), position: 'relative' }}>
          {list.getVirtualItems().map(item => (
            <div
              data-index={item.index}
              key={item.key}
              ref={list.measureElement}
              style={{ position: 'absolute', top: 0, left: 0, width: '100%', transform: `translateY(${item.start}px)` }}
            >
              {renderRow(props.sessions[item.index])}
            </div>
          ))}
        </div>
      ) : (
        props.sessions.map(renderRow)
      )}
    </div>
  )
}
