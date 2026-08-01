/**
 * Reconcile the sidebar's pins with the backend "keep" flag, both directions.
 *
 * Pins drive the sidebar UI out of `$pinnedSessionIds` (localStorage), but the
 * durable record is `sessions.pinned` in each profile's state.db. Two things
 * depend on the backend copy: the `sessions.auto_archive` sweep runs
 * server-side and would otherwise hide a pinned chat, and a second Desktop app
 * pointed at the same gateway has its own, separate localStorage.
 *
 * Push: PATCH `pinned` whenever the local set changes, and re-assert the whole
 * set at boot — which transparently migrates pre-existing pins with no user
 * action.
 *
 * Pull: session rows now carry `pinned`, and the list endpoints back-fill
 * pinned conversations past their LIMIT, so a row's absence from a page no
 * longer says anything about its pin state. That makes the server row
 * authoritative: adopt pins this app hasn't seen, and drop local pins the
 * server says are gone. Only rows actually present in the payload are
 * consulted, so a backend predating the flag (`pinned === undefined`) leaves
 * the local set untouched.
 */

import { setSessionPinnedRemote } from '@/hermes'
import { $pinnedSessionIds, pinSession, unpinSession } from '@/store/layout'
import { $messagingSessions, $sessions, sessionMatchesStoredId, sessionPinId } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

// pin ids we've successfully PATCHed pinned=true this session.
const mirrored = new Set<string>()
// pin ids awaiting their row so we can resolve the owning profile before PATCH.
const pending = new Set<string>()
// Writes we've issued but not yet had acked, id -> value written. A list page
// already in flight when we PATCH still carries the old value, so it must not
// be read as the server disagreeing with us. Cleared when the write settles —
// the request's own lifetime is the guard, so nothing can leave one open.
const unconfirmed = new Map<string, boolean>()

interface PinSyncConversation {
  pinId: string
  representative: SessionInfo
  rows: SessionInfo[]
}

/** One identity group per durable conversation across both sidebar slices. */
function pinSyncConversations(): PinSyncConversation[] {
  const byPinId = new Map<string, PinSyncConversation>()

  for (const row of [...$sessions.get(), ...$messagingSessions.get()]) {
    const pinId = sessionPinId(row)
    const existing = byPinId.get(pinId)

    if (!existing) {
      byPinId.set(pinId, { pinId, representative: row, rows: [row] })

      continue
    }

    existing.rows.push(row)

    const existingHasPin = typeof existing.representative.pinned === 'boolean'
    const rowHasPin = typeof row.pinned === 'boolean'

    if (
      (!existingHasPin && rowHasPin) ||
      (existingHasPin === rowHasPin && row.id === pinId && existing.representative.id !== pinId)
    ) {
      existing.representative = row
    }
  }

  return [...byPinId.values()]
}

function rowFor(conversations: PinSyncConversation[], storedId: string): SessionInfo | undefined {
  for (const conversation of conversations) {
    const row = conversation.rows.find(entry => sessionMatchesStoredId(entry, storedId))

    if (row) {
      return row
    }
  }

  return undefined
}

function profileFor(conversations: PinSyncConversation[], pinId: string): null | string | undefined {
  return rowFor(conversations, pinId)?.profile
}

/** PATCH the flag, guarding reads against pages that predate the write. */
function writePin(id: string, pinned: boolean, profile?: null | string): Promise<void> {
  unconfirmed.set(id, pinned)

  return setSessionPinnedRemote(id, pinned, profile).then(
    () => {
      unconfirmed.delete(id)
    },
    (err: unknown) => {
      unconfirmed.delete(id)
      throw err
    }
  )
}

/**
 * Adopt the server's pin state for every row in the current page.
 *
 * Runs before the push pass so a remote pin is already in the local set by the
 * time we reconcile — it gets marked as mirrored rather than echoed straight
 * back as a redundant PATCH.
 */
function pullRemotePins(conversations: PinSyncConversation[]): void {
  const local = new Set($pinnedSessionIds.get())

  for (const conversation of conversations) {
    const { pinId, representative: row, rows } = conversation

    const pinnedValues = new Set(
      rows.map(entry => entry.pinned).filter((pinned): pinned is boolean => typeof pinned === 'boolean')
    )

    // Both slices are filtered views of the same backend data; neither is more
    // authoritative. If overlapping rows disagree, preserve local state until
    // a later refresh converges instead of choosing a synthetic winner.
    if (pinnedValues.size > 1) {
      continue
    }

    // A backend without the flag has no opinion; never act on `undefined`.
    if (typeof row.pinned !== 'boolean') {
      continue
    }

    // Pins are keyed on the durable lineage root so they survive compression
    // tip rotation; the row may surface under either identity.
    const heldId = local.has(pinId) ? pinId : rows.find(entry => local.has(entry.id))?.id
    const heldLocally = heldId !== undefined

    // A write of ours the page hasn't caught up to yet is newer than the page.
    const awaitedId = [pinId, ...rows.map(entry => entry.id)].find(id => unconfirmed.has(id))
    const awaited = awaitedId === undefined ? undefined : unconfirmed.get(awaitedId)

    if (awaited !== undefined && awaited !== row.pinned) {
      continue
    }

    if (row.pinned && !heldLocally) {
      pinSession(pinId)
      // Already true server-side; record it so the push pass doesn't re-PATCH.
      mirrored.add(pinId)
    } else if (!row.pinned && heldId !== undefined) {
      unpinSession(heldId)
      mirrored.delete(pinId)

      for (const entry of rows) {
        mirrored.delete(entry.id)
      }
    }
  }
}

function reconcile(): void {
  // Config/session REST is only reachable through the Electron bridge.
  if (!window.hermesDesktop) {
    return
  }

  const conversations = pinSyncConversations()

  pullRemotePins(conversations)

  const current = new Set($pinnedSessionIds.get())

  // Unpinned: anything we were tracking that's no longer in the set.
  for (const id of [...mirrored, ...pending]) {
    if (!current.has(id)) {
      mirrored.delete(id)
      pending.delete(id)
      void writePin(id, false, profileFor(conversations, id)).catch(() => {})
    }
  }

  // Newly pinned: hold until we can resolve the row (for its profile).
  for (const id of current) {
    if (!mirrored.has(id)) {
      pending.add(id)
    }
  }

  // Flush whatever we can resolve now; unresolved ids (row not loaded yet)
  // retry on the next session-list change.
  for (const id of [...pending]) {
    const row = rowFor(conversations, id)

    if (!row) {
      continue
    }

    pending.delete(id)
    mirrored.add(id)
    void writePin(id, true, row.profile).catch(() => {
      // Let a later reconcile retry the mirror.
      mirrored.delete(id)
      pending.add(id)
    })
  }
}

// Sync once, then re-sync on pin-set and session-list changes. Call once per app.
export function watchSessionPins(): void {
  reconcile()
  $pinnedSessionIds.listen(reconcile)
  $sessions.listen(reconcile)
  $messagingSessions.listen(reconcile)
}
