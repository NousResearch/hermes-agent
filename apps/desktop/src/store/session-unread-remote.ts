/**
 * Persisted unread flag sync (backend read-state watermark via
 * PATCH /api/sessions/{id} → SessionDB.set_session_read).
 *
 * The sidebar's dot is fed by TWO sources (see session-dot-state.ts): the
 * runtime "turn finished in background" marker ($unreadFinishedSessionIds,
 * transient) and the backend's derived `unread` key (last_read_at watermark
 * vs last_active — survives restarts and is visible to every surface). This
 * module owns the WRITE side of the persisted flag: the row-level
 * "Mark as unread"/"Mark as read" toggle and the automatic clear when a
 * session is opened. The read side lives in session-dot-state.ts.
 *
 * Optimistic, then honest (AGENTS.md): paint the row immediately, PATCH the
 * backend, roll back visibly on failure. A list page already in flight when
 * we PATCH can land after the ack carrying the OLD value — the write guard
 * lets our value outrank that stale page briefly (#74570 pattern, same as
 * session-pin-sync.ts).
 *
 * NOTE: import cycle with ./session is inert — both modules only touch each
 * other's exports inside function bodies, never at module evaluation time.
 */
import { atom } from 'nanostores'

import { setSessionUnreadRemote } from '@/hermes'
import type { SessionInfo } from '@/types/hermes'

import { $sessions, setSessions } from './session'
import { sessionMutationScopeFromRow, sessionOwnerRowKey } from './session-request-router'

export const UNREAD_WRITE_GUARD_MS = 10_000

/** Immutable owner + id -> the value we wrote and when. Guarded rows outrank list pages. */
export const $unreadWriteGuard = atom<Map<string, { at: number; value: boolean }>>(new Map())
interface UnreadWriteState {
  active: null | symbol
  confirmed: boolean
  latest: symbol
  tail: Promise<void>
}

const unreadWrites = new Map<string, UnreadWriteState>()

export function unreadWriteKey(row: Pick<SessionInfo, '_desktop_route_binding' | 'connection_id' | 'id' | 'profile'>) {
  return sessionOwnerRowKey(row)
}

function rowFor(session: SessionInfo | string) {
  if (typeof session === 'string') {
    const matches = $sessions.get().filter(row => row.id === session)

    return matches.length === 1 ? matches[0] : undefined
  }

  const key = unreadWriteKey(session)

  return $sessions.get().find(row => unreadWriteKey(row) === key)
}

/** Toggle the persisted unread flag: optimistic row update, then PATCH, then
 *  roll back visibly if the write fails. No-op for runtime-only sessions (a
 *  brand-new chat with no persisted row yet — there is nothing to flag). */
export async function markSessionUnread(session: SessionInfo | string, unread: boolean): Promise<void> {
  const row = rowFor(session)

  if (!row) {
    return
  }

  const storedId = row.id
  const revision = Symbol(storedId)
  const scope = sessionMutationScopeFromRow(row) ?? row.profile
  const writeKey = unreadWriteKey(row)

  const state =
    unreadWrites.get(writeKey) ??
    ({ active: null, confirmed: row.unread === true, latest: revision, tail: Promise.resolve() } satisfies UnreadWriteState)

  state.latest = revision
  unreadWrites.set(writeKey, state)

  // Paint before publishing the guard so the guard watcher cannot mistake
  // this local optimistic row for a confirming backend page.
  setSessions(rows => rows.map(r => (unreadWriteKey(r) === writeKey ? { ...r, unread } : r)))

  const guard = new Map($unreadWriteGuard.get())
  guard.set(writeKey, { at: Date.now(), value: unread })
  $unreadWriteGuard.set(guard)

  const write = state.tail.then(async () => {
    state.active = revision

    try {
      await setSessionUnreadRemote(storedId, unread, scope)
    } finally {
      if (state.active === revision) {
        state.active = null
      }
    }
  })

  // Serialize transport per session so late server completions cannot restore
  // an older value after a newer click. A failed predecessor does not block
  // the next requested value.
  state.tail = write.catch(() => undefined)

  try {
    await write

    state.confirmed = unread
  } catch (err) {
    // An older rejection must not undo a newer click. The latest rejection
    // restores the last backend-confirmed value, never another pending edit.
    if (state.latest !== revision) {
      throw err
    }

    // Roll back visibly: the backend kept the old value.
    const guard2 = new Map($unreadWriteGuard.get())
    guard2.delete(writeKey)
    $unreadWriteGuard.set(guard2)
    setSessions(rows =>
      rows.map(r => (unreadWriteKey(r) === writeKey && r.unread === unread ? { ...r, unread: state.confirmed } : r))
    )
    throw err
  } finally {
    if (state.latest === revision) {
      unreadWrites.delete(writeKey)
    }
  }
}

/** Opening a session clears its persisted unread flag (auto-mark-read).
 *  Best-effort: a failed PATCH is healed by the next honest refresh. */
export async function clearUnreadOnOpen(session: SessionInfo | string): Promise<void> {
  const row = rowFor(session)

  if (!row || row.unread !== true) {
    return
  }

  try {
    await markSessionUnread(row, false)
  } catch {
    // Ignore: the dot simply returns until a refresh reconciles.
  }
}

/** Release guard entries once a list page confirms the value we wrote. Call
 *  once at boot, next to watchSessionPins(). */
export function watchUnreadWriteGuard(): void {
  $sessions.listen(rows => {
    const guard = $unreadWriteGuard.get()
    let changed = false

    for (const [key, entry] of guard) {
      const row = rows.find(r => unreadWriteKey(r) === key)
      const state = unreadWrites.get(key)

      // A page can confirm only the write currently on the wire. A value that
      // merely matches a later queued intent may predate every pending PATCH.
      if (row && row.unread === entry.value && (!state || state.active === state.latest)) {
        guard.delete(key)

        if (state) {
          state.confirmed = entry.value
          state.latest = Symbol(key)
          unreadWrites.delete(key)
        }

        changed = true
      }
    }

    if (changed) {
      $unreadWriteGuard.set(new Map(guard))
    }
  })
}
