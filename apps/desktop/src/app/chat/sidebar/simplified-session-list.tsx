import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useState } from 'react'

import { $apiRequestScope } from '@/api/client'
import { openSession } from '@/app/open-session'
import { useStableNavigate } from '@/app/stable-navigate'
import { formatMessageTimestamp } from '@/components/assistant-ui/thread/timestamp'
import { Button } from '@/components/ui/button'
import { RowButton } from '@/components/ui/row-button'
import { OverflowTip, Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { MoreHorizontal } from '@/lib/icons'
import { SESSION_ROW_AREAS } from '@/lib/session-row-slots'
import { coarseElapsed } from '@/lib/time'
import { useStoreSelector } from '@/lib/use-session-slice'
import { recordIaAction } from '@/store/ia-metrics'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, knownSessionOwner, ownerLookupSessionRows, sessionPinId } from '@/store/session'
import { $sessionDotStateById } from '@/store/session-dot-state'
import { $focusedSessionIsTile } from '@/store/session-focus'
import { sessionOwnerRouteFromRow } from '@/store/session-request-router'
import { $sessionTiles, canOpenSessionOwner, runtimeSessionOwner } from '@/store/session-states'

import { SessionStatusDot } from '../session-status-dot'

import { capAgentChats } from './portal/agent-chat-cap'
import { SessionActionsMenu, SessionContextMenu } from './session-actions-menu'
import { SessionRowSlot } from './session-row-slots'
import { sessionRowKey } from './simplified-session-identity'

/** An agent card's cap: its most recent chats, then a row that shows the rest or folds them back. */
interface SessionListCap {
  limit: number
  showAll: boolean
  onToggle: () => void
}

interface SimplifiedSessionListProps {
  cap?: SessionListCap
  searching?: boolean
  showAgent?: boolean
  sessions: SessionInfo[]
  activeSessionId: string | null
  pinnedIds: ReadonlySet<string>
  pinPending?: (row: SessionInfo) => boolean
  ambiguousIds: ReadonlySet<string>
  onResumeSession: (id: string, row?: SessionInfo) => void
  onDeleteSession: (id: string, session?: SessionInfo) => void
  onArchiveSession: (id: string, session?: SessionInfo) => void
  onBranchSession: (id: string, profile?: string, session?: SessionInfo) => void
  onUpdate: (row: SessionInfo, field: 'pinned' | 'unread', value: boolean) => void
}

type ListSurface = 'search' | 'pinned' | 'agent'

const OPEN_METRIC = {
  search: 'ia.session.opened_search',
  pinned: 'ia.session.opened_pinned',
  agent: 'ia.session.opened_agent'
} as const

const PIN_METRIC = {
  search: 'ia.pin.search_request',
  pinned: 'ia.pin.pinned_request',
  agent: 'ia.pin.agent_request'
} as const

/** Live status marks the row. Every row leads with a dot, quiet when settled (notif 16292:15533,
 *  16292:15604), so the titles of a list share one inset. */
function SessionRowMark({ row }: { row: SessionInfo }) {
  const state = useStoreSelector($sessionDotStateById, states => states[row.id] ?? 'idle')

  if (state === 'idle') {
    return (
      <span aria-hidden className="portal-row-mark" data-slot="session-status" data-state="idle">
        <span>
          <span />
        </span>
      </span>
    )
  }

  return (
    <span className="portal-row-mark" data-slot="session-status" data-state={state}>
      <SessionStatusDot session={row} storedSessionId={row.id} />
    </span>
  )
}

function SessionAge({ row, now }: { row: SessionInfo; now: number }) {
  const { t } = useI18n()
  const timestamp = new Date((row.last_active || row.started_at) * 1000)
  const { unit, value } = coarseElapsed(now - timestamp.getTime())
  const units = { day: t.sidebar.row.ageDay, hour: t.sidebar.row.ageHour, minute: t.sidebar.row.ageMin }
  const age = unit === 'second' ? t.sidebar.row.ageNow : `${value}${units[unit]}`
  const absoluteAge = formatMessageTimestamp(timestamp, t.assistant.thread)

  return (
    <time aria-label={`${age}, ${absoluteAge}`} dateTime={timestamp.toISOString()} title={absoluteAge}>
      {age}
    </time>
  )
}

interface SessionRowProps extends Omit<SimplifiedSessionListProps, 'sessions' | 'activeSessionId' | 'cap'> {
  row: SessionInfo
  selected: boolean
  surface: ListSurface
  meta: ReactNode
}

function SessionRow({ row, selected, surface, meta, ...props }: SessionRowProps) {
  const { t } = useI18n()
  const navigate = useStableNavigate()
  const title = sessionTitle(row)
  const pinned = props.pinnedIds.has(sessionRowKey(row))
  const ambiguous = props.ambiguousIds.has(row.id)

  const actions = {
    onArchive: () => props.onArchiveSession(row.id, row),
    onBranch: () => props.onBranchSession(row.id, row.profile, row),
    onDelete: () => props.onDeleteSession(row.id, row),
    onPin: props.pinPending?.(row)
      ? undefined
      : () => {
          recordIaAction(PIN_METRIC[surface], 'menu')
          props.onUpdate(row, 'pinned', !pinned)
        },
    onToggleUnread: () => props.onUpdate(row, 'unread', row.unread !== true),
    ownerRoute: sessionOwnerRouteFromRow(row),
    pinned,
    profile: row.profile,
    sessionId: row.id,
    title,
    unread: row.unread === true
  }

  // The row is the click target (a modified click opens a tab or window), its title the one
  // focusable button, as in the classic sidebar: plugin decorations (`sessionRow.*`) sit in the row
  // beside that button, never inside it, so an interactive one stays a control of its own.
  const button = (
    <div
      aria-current={selected ? 'page' : undefined}
      className="portal-session-row"
      onClick={event => {
        recordIaAction(OPEN_METRIC[surface])

        if (surface === 'search') {
          recordIaAction('ia.search.selected')
        }

        if (event.metaKey || event.ctrlKey) {
          openSession(row.id, navigate, event.shiftKey ? 'window' : 'tab', {
            workspaceMode: 'sessions',
            ownerRoute: sessionOwnerRouteFromRow(row)
          })
        } else if (canOpenSessionOwner(row.id, sessionOwnerRouteFromRow(row))) {
          props.onResumeSession(row.id, row)
        }
      }}
    >
      <SessionRowMark row={row} />
      {/* Wrappers with no box: a row no plugin decorates (or one it declines) keeps its geometry. */}
      <span className="contents" data-slot="session-row-leading">
        <SessionRowSlot area={SESSION_ROW_AREAS.leading} sessionId={sessionPinId(row)} />
      </span>
      <RowButton className="portal-session-open">
        <OverflowTip label={title}>
          <span className="portal-session-title" data-slot="session-title">
            {title}
          </span>
        </OverflowTip>
      </RowButton>
      <span className="contents" data-slot="session-row-trailing">
        <SessionRowSlot area={SESSION_ROW_AREAS.trailing} sessionId={sessionPinId(row)} />
      </span>
      <span className="portal-session-meta" data-slot="session-meta">
        {meta}
      </span>
    </div>
  )

  const item = (
    <div
      className="portal-session-item"
      data-actions={ambiguous ? undefined : ''}
      data-session-key={sessionRowKey(row)}
    >
      {button}
      {!ambiguous && (
        <Tooltip>
          <SessionActionsMenu {...actions}>
            <TooltipTrigger asChild>
              <Button
                aria-label={`${t.sidebar.row.sessionActions}: ${title}`}
                className="portal-row-action"
                size="icon-xs"
                variant="ghost"
              >
                <MoreHorizontal aria-hidden />
              </Button>
            </TooltipTrigger>
          </SessionActionsMenu>
          <TooltipContent>{t.sidebar.row.sessionActions}</TooltipContent>
        </Tooltip>
      )}
    </div>
  )

  // shortcut: legacy actions accept bare IDs; restore menus for twins when those actions accept owners.
  return ambiguous ? item : <SessionContextMenu {...actions}>{item}</SessionContextMenu>
}

function useFocusedOwner(activeSessionId: string | null) {
  const activeScope = useStore($apiRequestScope)
  const activeProfile = useStore($activeGatewayProfile)
  const focusedIsTile = useStore($focusedSessionIsTile)
  const tiles = useStore($sessionTiles)
  const activeRuntime = useStore($activeSessionId)
  const focusedTile = focusedIsTile ? tiles.find(tile => tile.storedSessionId === activeSessionId) : undefined

  const owner =
    focusedTile?.ownerRoute ??
    runtimeSessionOwner(focusedIsTile ? focusedTile?.runtimeId : activeRuntime) ??
    knownSessionOwner(ownerLookupSessionRows(), activeSessionId)

  const profile = (typeof owner === 'string' ? owner : owner?.profile) || activeProfile
  const connection = owner && typeof owner === 'object' ? owner.connectionId : activeScope.connectionId

  return (row: SessionInfo) =>
    row.id === activeSessionId &&
    (row.profile || 'default') === profile &&
    (row.connection_id || activeScope.connectionId) === connection
}

/** Live age labels refresh each minute while the list shows them. */
function useMinuteClock(enabled: boolean) {
  const [now, setNow] = useState(Date.now)

  useEffect(() => {
    if (!enabled) {
      return
    }

    const timer = window.setInterval(() => setNow(Date.now()), 60_000)

    return () => window.clearInterval(timer)
  }, [enabled])

  return now
}

/** A flat v2 list: owner-qualified keys, live status marks and the desktop's row menu. */
export function SimplifiedSessionList({ cap, sessions, activeSessionId, ...props }: SimplifiedSessionListProps) {
  const { t } = useI18n()
  const isSelected = useFocusedOwner(activeSessionId)
  const dots = useStore($sessionDotStateById)
  const now = useMinuteClock(!props.showAgent)
  const surface: ListSurface = props.searching ? 'search' : props.showAgent ? 'pinned' : 'agent'

  const ordered = props.showAgent
    ? sessions
    : [...sessions].sort(
        (a, b) => Number(props.pinnedIds.has(sessionRowKey(b))) - Number(props.pinnedIds.has(sessionRowKey(a)))
      )

  const metaFor = (row: SessionInfo) => {
    if (props.showAgent) {
      const profile = row.profile || 'default'

      return <span aria-label={`${t.simplifiedAgents.agents}: ${profile}`}>{profile}</span>
    }

    // Figma 147:7470 draws no pin glyph in an agent's list; the pinned-first order is still named
    // for assistive tech.
    return (
      <>
        {props.pinnedIds.has(sessionRowKey(row)) && (
          <span aria-label={t.sidebar.pinned} className="sr-only" role="img" />
        )}
        <SessionAge now={now} row={row} />
      </>
    )
  }

  // A search lists every match; otherwise the cap keeps whatever needs attention in reach.
  const capped = cap && !cap.showAll && !props.searching
  const keep = (row: SessionInfo) => isSelected(row) || row.unread === true || (dots[row.id] ?? 'idle') !== 'idle'
  const shown = capped ? capAgentChats(ordered, cap.limit, keep) : ordered
  const foldable = cap && !props.searching && ordered.length > cap.limit

  return (
    <div className="portal-session-list">
      {shown.map(row => (
        <SessionRow
          {...props}
          key={sessionRowKey(row)}
          meta={metaFor(row)}
          row={row}
          selected={isSelected(row)}
          surface={surface}
        />
      ))}
      {foldable && (
        <RowButton aria-expanded={cap.showAll} className="portal-session-more" onClick={cap.onToggle}>
          {cap.showAll ? t.simplifiedAgents.showFewerChats : t.simplifiedAgents.showAllChats(ordered.length)}
        </RowButton>
      )}
    </div>
  )
}
