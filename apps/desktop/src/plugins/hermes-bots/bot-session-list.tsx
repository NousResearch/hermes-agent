/** Source-scoped child conversations shown beneath an expanded bot row. */

import * as sdk from '@hermes/plugin-sdk'
import { atom, Button, coarseElapsed, Codicon, haptic, host, RowButton, useI18n } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { saveSelectedRosterBot } from './bot-state'
import { CANONICAL_CHAT_TITLE, PROFILE_SESSION_LIST_LIMIT } from './canonical-chat'
import { botOwner, botRosterKey, newBotChat } from './data'
import { useBots } from './i18n'
import { backendTargetProfile, botWorkspaceOwnerKey, setBotsWorkspaceOwner } from './routing'
import type { RosterRow } from './types'

export const $expandedBotSessions = atom<ReadonlySet<string>>(new Set())

export function toggleBotSessions(rosterKey: string): void {
  const next = new Set($expandedBotSessions.get())
  next.has(rosterKey) ? next.delete(rosterKey) : next.add(rosterKey)
  $expandedBotSessions.set(next)
}

export interface BotNamedSession {
  id: string
  lastActive: number
  messageCount?: number
  title: string
}

const RECENT_SESSION_HEAD = 5

function sessionAge(
  seconds: number,
  now: number,
  labels: { ageDay: string; ageHour: string; ageMin: string; ageNow: string }
): string | null {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return null
  }

  const { unit, value } = coarseElapsed(Math.max(0, now - seconds * 1000))

  return unit === 'second'
    ? labels.ageNow
    : `${value}${unit === 'day' ? labels.ageDay : unit === 'hour' ? labels.ageHour : labels.ageMin}`
}

interface BotNamedSessionsResult {
  hasMore: boolean
  sessions: BotNamedSession[]
}

function botSessionOwner(bot: RosterRow) {
  const { name, route } = botOwner(bot)

  return { profile: backendTargetProfile(route, name), route }
}

export async function listBotNamedSessions(bot: RosterRow): Promise<BotNamedSessionsResult> {
  if (typeof host.listPersistedSessions !== 'function') {
    throw new Error('This Hermes Desktop version cannot list a bot’s conversations')
  }

  const { profile, route } = botSessionOwner(bot)
  const res = await host.listPersistedSessions(route, { profile, limit: PROFILE_SESSION_LIST_LIMIT })
  const profileError = res?.errors?.find(error => error.profile === profile)

  if (profileError) {
    throw new Error(`Could not read ${profile}'s conversations: ${profileError.error}`)
  }

  const rows = Array.isArray(res?.sessions) ? res.sessions : []

  const sessions = rows
    .filter(row => Boolean(row?.id))
    .map(row => ({
      id: String(row.id),
      lastActive: Number(row.last_active) || 0,
      messageCount:
        typeof row.message_count === 'number' && Number.isFinite(row.message_count) ? row.message_count : undefined,
      title: String(row.title || '').trim()
    }))
    .filter(session => session.title !== CANONICAL_CHAT_TITLE)
    .sort((a, b) => b.lastActive - a.lastActive)

  return { hasMore: Number(res?.total) > rows.length, sessions }
}

export async function openBotNamedSession(bot: RosterRow, session: BotNamedSession): Promise<void> {
  const { id: storedSessionId, messageCount } = session

  if (!storedSessionId || typeof host.openSession !== 'function') {
    throw new Error('This Hermes Desktop version cannot open stored sessions')
  }

  const { bot: owner, name, route } = botOwner(bot)
  const ownerKey = botWorkspaceOwnerKey(owner)

  haptic('tap')
  saveSelectedRosterBot(owner)
  setBotsWorkspaceOwner(ownerKey, owner)
  await host.openSession(storedSessionId, {
    ...(route ? { route } : {}),
    profile: name,
    intent: 'tab',
    awaitHydration: true,
    expectHistory: messageCount === undefined || messageCount > 0,
    forceResume: true,
    hydrationTimeoutMs: Number.isFinite(sdk.BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS)
      ? sdk.BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS
      : 60_000,
    keepAllProfilesScope: true,
    workspaceMode: 'bots',
    workspaceOwnerKey: ownerKey,
    retryHydrationTimeoutOnce: true
  })
}

type BotSessionsState =
  | { hasMore: boolean; sessions: BotNamedSession[]; status: 'loading' }
  | { hasMore: boolean; sessions: BotNamedSession[]; status: 'ready' }
  | { hasMore: boolean; message: string; sessions: BotNamedSession[]; status: 'error' }

const NOTE_ROW = 'mb-0.5 ml-[3.25rem] truncate px-2 py-1 text-[0.6875rem] text-(--ui-text-quaternary)'

const errorMessage = (error: unknown) =>
  String((error as { message?: unknown })?.message || error || '').trim() || 'unknown error'

function sameSessionList(current: BotSessionsState, result: BotNamedSessionsResult): boolean {
  return (
    current.status === 'ready' &&
    current.hasMore === result.hasMore &&
    current.sessions.length === result.sessions.length &&
    current.sessions.every((session, index) => {
      const next = result.sessions[index]

      return session.id === next.id && session.title === next.title &&
        session.lastActive === next.lastActive && session.messageCount === next.messageCount
    })
  )
}

export function BotSessionList({ bot }: { bot: RosterRow }) {
  const b = useBots()
  const { t } = useI18n()
  const rosterKey = botRosterKey(bot)
  const [refreshTick, setRefreshTick] = useState(0)
  const [showEarlier, setShowEarlier] = useState(false)
  const [state, setState] = useState<BotSessionsState>({ hasMore: false, sessions: [], status: 'loading' })

  useEffect(() => {
    let live = true
    let refreshId = 0

    const refresh = () => {
      const requestId = ++refreshId
      // A background session event must not collapse an already painted list.
      // Keep the last good rows until the replacement is ready (or fails).
      listBotNamedSessions(bot)
        .then(result => {
          if (live && requestId === refreshId) {
            setState(current => sameSessionList(current, result) ? current : { ...result, status: 'ready' })
          }
        })
        .catch((error: unknown) => {
          if (live && requestId === refreshId) {
            setState(previous => ({
              hasMore: previous.hasMore,
              message: errorMessage(error),
              sessions: previous.sessions,
              status: 'error'
            }))
          }
        })
    }

    refresh()

    const disposers =
      typeof host.onEvent === 'function'
        ? [host.onEvent('sessions.changed', refresh), host.onEvent('session.title', refresh)]
        : []

    return () => {
      live = false
      disposers.forEach(dispose => dispose())
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rosterKey, refreshTick])

  if (state.status === 'loading') {
    return <p className={NOTE_ROW}>Loading conversations…</p>
  }

  if (state.status === 'error' && !state.sessions.length) {
    return (
      <div className={NOTE_ROW}>
        <p>{`Could not load conversations — ${state.message}`}</p>
        <Button onClick={() => setRefreshTick(tick => tick + 1)} size="inline" variant="text">
          Retry
        </Button>
      </div>
    )
  }

  return (
    <div
      data-bot-sessions={rosterKey}
      onContextMenu={event => {
        // The enclosing bot's context menu includes Delete *profile*. Child
        // conversations have no context menu until session actions are designed.
        event.preventDefault()
        event.stopPropagation()
      }}
    >
      {state.status === 'error' ? (
        <p className={NOTE_ROW}>
          {`Could not refresh conversations — ${state.message} `}
          <Button onClick={() => setRefreshTick(tick => tick + 1)} size="inline" variant="text">
            Retry
          </Button>
        </p>
      ) : null}
      <Button
        aria-label={b.bot.newChatWith}
        className="mb-0.5 ml-[3.25rem] text-[0.6875rem]"
        onClick={() => newBotChat(bot)}
        size="inline"
        variant="text"
      >
        {b.bot.newChatWith}
      </Button>
      {state.sessions.length ? (
        <ul className="mb-0.5 ml-[3.25rem] grid min-w-0 gap-0.5 pr-1">
          {state.sessions.slice(0, showEarlier ? undefined : RECENT_SESSION_HEAD).map(session => (
            <li className="min-w-0" key={session.id}>
              <RowButton
                className="flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1 text-left text-[0.75rem] text-(--ui-text-secondary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
                data-bot-session-id={session.id}
                onClick={() =>
                  void openBotNamedSession(bot, session).catch(error =>
                    host.notifyError?.(error, 'Could not open that conversation')
                  )
                }
              >
                <Codicon className="shrink-0 text-(--ui-text-quaternary)" name="comment" size="0.75rem" />
                <span className="min-w-0 flex-1 truncate">{session.title || '(untitled)'}</span>
                {sessionAge(session.lastActive, Date.now(), t.sidebar.row) ? (
                  <span className="shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)">
                    {sessionAge(session.lastActive, Date.now(), t.sidebar.row)}
                  </span>
                ) : null}
              </RowButton>
            </li>
          ))}
        </ul>
      ) : (
        <p className={NOTE_ROW}>No other conversations</p>
      )}
      {state.sessions.length > RECENT_SESSION_HEAD ? (
        <Button
          aria-expanded={showEarlier}
          aria-label="Earlier conversations"
          className={NOTE_ROW}
          onClick={() => setShowEarlier(value => !value)}
          size="inline"
          variant="text"
        >
          {showEarlier ? 'Hide earlier conversations' : 'Earlier conversations'}
        </Button>
      ) : null}
      {state.hasMore ? <p className={NOTE_ROW}>Showing up to 200 recent conversations</p> : null}
    </div>
  )
}
