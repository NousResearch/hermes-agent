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

import { $cronSessions, $messagingSessions, $sessions, ownerLookupSessionRows, setCronSessions, setMessagingSessions, setSessions } from './session'

export const UNREAD_WRITE_GUARD_MS = 10_000

/** id -> the value we wrote and when. Guarded rows outrank list pages. */
export const $unreadWriteGuard = atom<Map<string, { at: number; value: boolean }>>(new Map())

function rowFor(storedId: string) {
  // Messaging rows (telegram, …) live outside recents — resolve across every
  // slice or their unread toggle silently no-ops (#126732).
  return ownerLookupSessionRows().find(row => row.id === storedId)
}

function patchUnread(storedId: string, unread: boolean): void {
  // Optimistic paint must land in the slice that owns the row: patching
  // recents alone leaves a messaging row's dot stale until the next refresh.
  setSessions(rows => rows.map(r => (r.id === storedId ? { ...r, unread } : r)))
  setMessagingSessions(rows => rows.map(r => (r.id === storedId ? { ...r, unread } : r)))
  setCronSessions(rows => rows.map(r => (r.id === storedId ? { ...r, unread } : r)))
}

/** Toggle the persisted unread flag: optimistic row update, then PATCH, then
 *  roll back visibly if the write fails. No-op for runtime-only sessions (a
 *  brand-new chat with no persisted row yet — there is nothing to flag). */
export async function markSessionUnread(storedId: string, unread: boolean): Promise<void> {
  const row = rowFor(storedId)

  if (!row) {
    return
  }

  const guard = new Map($unreadWriteGuard.get())
  guard.set(storedId, { at: Date.now(), value: unread })
  $unreadWriteGuard.set(guard)

  patchUnread(storedId, unread)

  try {
    await setSessionUnreadRemote(storedId, unread, row.profile)
  } catch (err) {
    // Roll back visibly: the backend kept the old value.
    const guard2 = new Map($unreadWriteGuard.get())
    guard2.delete(storedId)
    $unreadWriteGuard.set(guard2)
    patchUnread(storedId, !unread)
    throw err
  }
}

/** Opening a session clears its persisted unread flag (auto-mark-read).
 *  Best-effort: a failed PATCH is healed by the next honest refresh. */
export async function clearUnreadOnOpen(storedId: string): Promise<void> {
  const row = rowFor(storedId)

  if (!row || row.unread !== true) {
    return
  }

  try {
    await markSessionUnread(storedId, false)
  } catch {
    // Ignore: the dot simply returns until a refresh reconciles.
  }
}

/** Release guard entries once a list page confirms the value we wrote. Call
 *  once at boot, next to watchSessionPins(). Listens to every slice — a
 *  messaging row's confirm arrives on $messagingSessions, never recents. */
export function watchUnreadWriteGuard(): void {
  const check = () => {
    const guard = $unreadWriteGuard.get()
    let changed = false

    for (const [id, entry] of guard) {
      const row = ownerLookupSessionRows().find(r => r.id === id)

      if (row && row.unread === entry.value) {
        guard.delete(id)
        changed = true
      }
    }

    if (changed) {
      $unreadWriteGuard.set(new Map(guard))
    }
  }

  $sessions.listen(check)
  $messagingSessions.listen(check)
  $cronSessions.listen(check)
}
