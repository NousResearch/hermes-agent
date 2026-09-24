/**
 * A bot's other conversations: the named sessions its own profile/source
 * already holds, listed under its row so they are one click away instead of a
 * trip through Sessions (hermes-agent#112184).
 *
 * DISCOVERY AND NAVIGATION ONLY. The list is a read of the session records
 * that already exist (`host.listPersistedSessions`) and the open that already
 * exists (`host.openSession`): no second session store, no stored id pointer,
 * no routing between conversations, and no opinion about the canonical Bot
 * Chat — that stays the row's own click target with its own identity. The list
 * is collapsed until it is asked for, so a rail nobody expands behaves exactly
 * as it did before.
 */

import { atom, Codicon, haptic, host, RowButton } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { saveSelectedRosterBot } from './bot-state'
import { CANONICAL_CHAT_TITLE, PROFILE_SESSION_LIST_LIMIT } from './canonical-chat'
import { botOwner, botRosterKey } from './data'
import { backendTargetProfile, botWorkspaceOwnerKey, setBotsWorkspaceOwner } from './routing'
import type { RosterRow } from './types'

/** Roster keys whose conversation list is open. Session-only, like the hidden
 *  -bots reveal: every launch starts collapsed, which is the experience the
 *  rail already had. */
export const $expandedBotSessions = atom<ReadonlySet<string>>(new Set())

export function toggleBotSessions(rosterKey: string): void {
  const next = new Set($expandedBotSessions.get())

  if (next.has(rosterKey)) {
    next.delete(rosterKey)
  } else {
    next.add(rosterKey)
  }

  $expandedBotSessions.set(next)
}

/** One conversation under a bot, as the row renders it. */
export interface BotNamedSession {
  id: string
  /** `last_active`, epoch SECONDS — the unit every session listing reports. */
  lastActive: number
  /** The title the user gave it; '' when the session has none yet. */
  title: string
}

/** The exact owner whose sessions belong to this bot: its own source, and the
 *  backend profile that source actually files them under (an alias row's
 *  logical name is not the profile the sessions live on). */
function botSessionOwner(bot: RosterRow) {
  const { name, route } = botOwner(bot)

  return { profile: backendTargetProfile(route, name), route }
}

/** Every named Desktop session the bot's own profile holds, newest first.
 *
 *  Reads the persisted listing (visible rows only, which is what makes the
 *  hidden Bot Mode plumbing — the forever-chat, room sessions — exclude
 *  itself). A failed read rejects: the caller owns the message, because only
 *  it knows whether the list is still on screen. */
export async function listBotNamedSessions(bot: RosterRow): Promise<BotNamedSession[]> {
  if (typeof host.listPersistedSessions !== 'function') {
    throw new Error('This Hermes Desktop version cannot list a bot’s conversations')
  }

  const { profile, route } = botSessionOwner(bot)
  const res = await host.listPersistedSessions(route, { profile, limit: PROFILE_SESSION_LIST_LIMIT })
  const rows = Array.isArray(res?.sessions) ? res.sessions : []

  return (
    rows
      .filter(row => Boolean(row?.id))
      .map(row => ({
        id: String(row.id),
        lastActive: Number(row.last_active) || 0,
        title: String(row.title || '').trim()
      }))
      // The forever-chat is the row, never a conversation under it. The
      // visibility sweep already hides it; dropping the title here too keeps the
      // two from disagreeing during the window before that write lands.
      .filter(session => session.title !== CANONICAL_CHAT_TITLE)
      .sort((a, b) => b.lastActive - a.lastActive)
  )
}

/** Open one of the bot's conversations. Same lane the canonical open uses —
 *  front an already-open tile, else load into main, chrome left on its own
 *  profile — so a click here reads like every other Bot Mode open. */
export async function openBotNamedSession(bot: RosterRow, storedSessionId: string): Promise<void> {
  if (!storedSessionId || typeof host.openSession !== 'function') {
    throw new Error('This Hermes Desktop version cannot open stored sessions')
  }

  const { bot: owner, name, route } = botOwner(bot)
  const ownerKey = botWorkspaceOwnerKey(owner)

  haptic('tap')
  saveSelectedRosterBot(owner)
  setBotsWorkspaceOwner(ownerKey, owner)
  await host.openSession(storedSessionId, {
    ...(route
      ? {
          route
        }
      : {}),
    profile: name,
    intent: 'in-place',
    keepAllProfilesScope: true,
    workspaceMode: 'bots',
    workspaceOwnerKey: ownerKey
  })
}

type BotSessionsState =
  | { sessions: BotNamedSession[]; status: 'loading' }
  | { sessions: BotNamedSession[]; status: 'ready' }
  | { message: string; sessions: BotNamedSession[]; status: 'error' }

const NOTE_ROW = 'mb-0.5 ml-[3.25rem] truncate px-2 py-1 text-[0.6875rem] text-(--ui-text-quaternary)'

function errorMessage(error: unknown) {
  const message = String((error as { message?: unknown })?.message || error || '').trim()

  return message || 'unknown error'
}

/** The expanded list. Mounted only while its row is open, so the read happens
 *  on the user's ask and a collapsed rail issues nothing. */
export function BotSessionList({ bot }: { bot: RosterRow }) {
  const rosterKey = botRosterKey(bot)
  const [state, setState] = useState<BotSessionsState>({ sessions: [], status: 'loading' })

  useEffect(() => {
    let live = true

    setState({ sessions: [], status: 'loading' })
    listBotNamedSessions(bot)
      .then(sessions => {
        if (live) {
          setState({ sessions, status: 'ready' })
        }
      })
      .catch((error: unknown) => {
        if (live) {
          setState({ message: errorMessage(error), sessions: [], status: 'error' })
        }
      })

    return () => {
      live = false
    }
    // Keyed on the row's IDENTITY, not its object: the roster re-polls every
    // few seconds with a fresh row for the same bot, and re-reading then would
    // re-issue the listing for a list the user is already looking at.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rosterKey])

  if (state.status === 'loading') {
    return <p className={NOTE_ROW}>Loading conversations…</p>
  }

  if (state.status === 'error') {
    return <p className={NOTE_ROW}>{`Could not load conversations — ${state.message}`}</p>
  }

  if (!state.sessions.length) {
    return <p className={NOTE_ROW}>No other conversations</p>
  }

  return (
    <ul className="mb-0.5 ml-[3.25rem] grid min-w-0 gap-0.5 pr-1" data-bot-sessions={rosterKey}>
      {state.sessions.map(session => (
        <li className="min-w-0" key={session.id}>
          <RowButton
            className="flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1 text-left text-[0.75rem] text-(--ui-text-secondary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
            data-bot-session-id={session.id}
            onClick={() => {
              void openBotNamedSession(bot, session.id).catch(error =>
                host.notifyError?.(error, 'Could not open that conversation')
              )
            }}
          >
            <Codicon className="shrink-0 text-(--ui-text-quaternary)" name="comment" size="0.75rem" />
            <span className="min-w-0 flex-1 truncate">{session.title || '(untitled)'}</span>
          </RowButton>
        </li>
      ))}
    </ul>
  )
}
