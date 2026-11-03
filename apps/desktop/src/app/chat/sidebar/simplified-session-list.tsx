import { useStore } from '@nanostores/react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { $apiRequestScope } from '@/api/client'
import { openSession } from '@/app/open-session'
import { formatMessageTimestamp } from '@/components/assistant-ui/thread/timestamp'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { RowButton } from '@/components/ui/row-button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { coarseElapsed } from '@/lib/time'
import { cn } from '@/lib/utils'
import { $activeGatewayProfile } from '@/store/profile'
import { $sessionListDensity } from '@/store/session-list-density'
import { sessionOwnerRouteFromRow } from '@/store/session-request-router'

import { SessionStatusDot } from '../session-status-dot'

import { SessionActionsMenu } from './session-actions-menu'
import { sessionRowEstimate } from './session-row-details'
import { sessionRowKey } from './simplified-session-identity'

interface SimplifiedSessionListProps {
  showAgent?: boolean
  sessions: SessionInfo[]
  activeSessionId: string | null
  pinnedIds: ReadonlySet<string>
  pinPending?: (row: SessionInfo) => boolean
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
  const [now, setNow] = useState(Date.now)
  useEffect(() => {
    if (props.showAgent) {
      return
    }

    const timer = window.setInterval(() => setNow(Date.now()), 60_000)

    return () => window.clearInterval(timer)
  }, [props.showAgent])

  const list = useVirtualizer({
    count: virtual ? props.sessions.length : 0,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => sessionRowEstimate(density),
    getItemKey: index => sessionRowKey(props.sessions[index]),
    overscan: 8
  })

  const renderRow = (row: SessionInfo) => {
    const title = sessionTitle(row)
    const pinned = props.pinnedIds.has(sessionRowKey(row))
    const timestamp = new Date((row.last_active || row.started_at) * 1000)
    const { unit, value } = coarseElapsed(now - timestamp.getTime())
    const units = { day: t.sidebar.row.ageDay, hour: t.sidebar.row.ageHour, minute: t.sidebar.row.ageMin }
    const age = unit === 'second' ? t.sidebar.row.ageNow : `${value}${units[unit]}`
    const absoluteAge = formatMessageTimestamp(timestamp, t.assistant.thread)

    const selected =
      row.id === props.activeSessionId &&
      (row.profile || 'default') === activeProfile &&
      (!row.connection_id || row.connection_id === activeScope.connectionId)

    return (
      <div className="group/session relative flex min-w-0 items-center" key={sessionRowKey(row)}>
        <RowButton
          aria-current={selected ? 'page' : undefined}
          className={cn(
            'flex min-h-8.5 min-w-0 flex-1 items-center gap-2 rounded-sm px-2 text-left text-xs hover:bg-(--chrome-action-hover) focus-visible:outline-2 focus-visible:outline-ring [@media(hover:none)]:pr-8',
            selected && 'bg-(--chrome-action-hover)'
          )}
          onClick={event => {
            if (event.metaKey || event.ctrlKey) {
              openSession(row.id, navigate, event.shiftKey ? 'window' : 'tab', {
                workspaceMode: 'sessions',
                ownerRoute: sessionOwnerRouteFromRow(row)
              })
            } else {
              props.onResumeSession(row.id, row)
            }
          }}
        >
          <span className="grid size-4 shrink-0 place-items-center">
            <SessionStatusDot session={row} storedSessionId={row.id} />
          </span>
          <span className="min-w-0 flex-1 truncate" title={title}>
            {title}
          </span>
          <span
            aria-label={props.showAgent ? `${t.simplifiedAgents.agents}: ${row.profile || 'default'}` : undefined}
            className={cn(
              'min-w-6 max-w-[30%] truncate text-right text-[0.6875rem] text-(--ui-text-tertiary)',
              !props.ambiguousIds.has(row.id) && 'group-hover/session:opacity-0 group-focus-within/session:opacity-0'
            )}
          >
            {props.showAgent ? (
              row.profile || 'default'
            ) : (
              <time aria-label={`${age}, ${absoluteAge}`} dateTime={timestamp.toISOString()} title={absoluteAge}>
                {age}
              </time>
            )}
          </span>
        </RowButton>
        {/* shortcut: legacy actions accept bare IDs; restore menus for twins when those actions accept owners. */}
        {!props.ambiguousIds.has(row.id) && (
          <Tooltip>
            <SessionActionsMenu
              onArchive={() => props.onArchiveSession(row.id)}
              onBranch={() => props.onBranchSession(row.id)}
              onDelete={() => props.onDeleteSession(row.id)}
              onPin={props.pinPending?.(row) ? undefined : () => props.onUpdate(row, 'pinned', !pinned)}
              onToggleUnread={() => props.onUpdate(row, 'unread', row.unread !== true)}
              pinned={pinned}
              profile={row.profile}
              sessionId={row.id}
              title={title}
              unread={row.unread === true}
            >
              <TooltipTrigger asChild>
                <Button
                  aria-label={`${t.sidebar.row.sessionActions}: ${title}`}
                  className="absolute right-0 top-1/2 -translate-y-1/2 opacity-0 group-hover/session:opacity-100 group-focus-within/session:opacity-100 focus-visible:opacity-100 [@media(hover:none)]:opacity-100"
                  size="icon-xs"
                  variant="ghost"
                >
                  <Codicon name="kebab-vertical" />
                </Button>
              </TooltipTrigger>
            </SessionActionsMenu>
            <TooltipContent>{t.sidebar.row.sessionActions}</TooltipContent>
          </Tooltip>
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
