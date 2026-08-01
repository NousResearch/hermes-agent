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
 *
 * Order matters for BOTH directions. Local intent (pin and unpin) must be
 * written and guarded BEFORE pull runs. Otherwise a lagging list page wins the
 * same tick:
 *   - still-true page re-adopts a just-unpinned chat
 *   - still-false page strips a just-pinned chat
 */

import { setSessionPinnedRemote } from '@/hermes'
import { $pinnedSessionIds, pinSession, unpinSession } from '@/store/layout'
import { $sessions, sessionMatchesStoredId, sessionPinId } from '@/store/session'

// pin ids we've successfully PATCHed pinned=true this session.
const mirrored = new Set<string>()
// pin ids awaiting their row so we can resolve the owning profile before PATCH.
const pending = new Set<string>()
// Writes we've issued but not yet had acked, id -> value written. A list page
// already in flight when we PATCH still carries the old value, so it must not
// be read as the server disagreeing with us. Cleared when the write settles.
const unconfirmed = new Map<string, boolean>()
// Local unpin intent. Survives lagging pinned=true pages until the server
// reports false (or the user pins again).
const unpinSticky = new Set<string>()
const unpinAcked = new Set<string>()
// Local pin intent. Survives lagging pinned=false pages until the server
// reports true (or the user unpins).
const pinSticky = new Set<string>()
const pinAcked = new Set<string>()

function profileFor(pinId: string): null | string | undefined {
  return $sessions.get().find(row => sessionMatchesStoredId(row, pinId))?.profile
}

/** Every id that should travel with a pin write for this conversation. */
function aliasIdsFor(pinId: string): string[] {
  const ids = new Set<string>([pinId])
  for (const row of $sessions.get()) {
    if (sessionMatchesStoredId(row, pinId)) {
      ids.add(row.id)
      ids.add(sessionPinId(row))
    }
  }
  return [...ids]
}

function markUnconfirmed(id: string, pinned: boolean): void {
  for (const alias of aliasIdsFor(id)) {
    unconfirmed.set(alias, pinned)
  }
}

function clearUnconfirmed(id: string): void {
  for (const alias of aliasIdsFor(id)) {
    unconfirmed.delete(alias)
  }
}

function markUnpinSticky(id: string): void {
  for (const alias of aliasIdsFor(id)) {
    unpinSticky.add(alias)
    unpinAcked.delete(alias)
    pinSticky.delete(alias)
    pinAcked.delete(alias)
  }
}

function markUnpinAcked(id: string): void {
  for (const alias of aliasIdsFor(id)) {
    unpinAcked.add(alias)
  }
}

function clearUnpinSticky(id: string): void {
  for (const alias of aliasIdsFor(id)) {
    unpinSticky.delete(alias)
    unpinAcked.delete(alias)
  }
}

function markPinSticky(id: string): void {
  for (const alias of aliasIdsFor(id)) {
    pinSticky.add(alias)
    pinAcked.delete(alias)
    unpinSticky.delete(alias)
    unpinAcked.delete(alias)
  }
}

function markPinAcked(id: string): void {
  for (const alias of aliasIdsFor(id)) {
    pinAcked.add(alias)
  }
}

function clearPinSticky(id: string): void {
  for (const alias of aliasIdsFor(id)) {
    pinSticky.delete(alias)
    pinAcked.delete(alias)
  }
}

function isUnpinSticky(pinId: string, liveId: string): boolean {
  return unpinSticky.has(pinId) || unpinSticky.has(liveId)
}

function isPinSticky(pinId: string, liveId: string): boolean {
  return pinSticky.has(pinId) || pinSticky.has(liveId)
}

function guardedValue(pinId: string, liveId: string): boolean | undefined {
  if (unconfirmed.has(pinId)) {
    return unconfirmed.get(pinId)
  }
  if (unconfirmed.has(liveId)) {
    return unconfirmed.get(liveId)
  }
  return undefined
}

/** PATCH the flag, guarding reads against pages that predate the write. */
function writePin(id: string, pinned: boolean, profile?: null | string): Promise<void> {
  markUnconfirmed(id, pinned)
  if (pinned) {
    markPinSticky(id)
  } else {
    markUnpinSticky(id)
  }

  return setSessionPinnedRemote(id, pinned, profile).then(
    () => {
      clearUnconfirmed(id)
      if (pinned) {
        markPinAcked(id)
      } else {
        markUnpinAcked(id)
      }
    },
    (err: unknown) => {
      clearUnconfirmed(id)
      throw err
    }
  )
}

/**
 * Push local removals to the backend.
 *
 * Must run before pull: a session list still carrying `pinned: true` would
 * otherwise re-adopt the id in the same reconcile tick and swallow the unpin.
 */
function pushLocalUnpins(current: Set<string>): void {
  const toUnpin = new Set<string>()

  for (const id of [...mirrored, ...pending]) {
    if (!current.has(id)) {
      mirrored.delete(id)
      pending.delete(id)
      toUnpin.add(id)
    }
  }

  // Retry sticky unpins that never got an ack (failed PATCH).
  for (const id of [...unpinSticky]) {
    if (!current.has(id) && !unconfirmed.has(id) && !unpinAcked.has(id)) {
      toUnpin.add(id)
    }
  }

  for (const id of toUnpin) {
    if (unconfirmed.has(id)) {
      continue
    }

    void writePin(id, false, profileFor(id)).catch(() => {
      // unpinSticky remains; next reconcile retries.
    })
  }
}

/**
 * Push local additions to the backend.
 *
 * Must run before pull: a session list still carrying `pinned: false` would
 * otherwise strip a just-pinned id in the same tick.
 */
function pushLocalPins(current: Set<string>): void {
  for (const id of current) {
    if (!mirrored.has(id)) {
      pending.add(id)
      // Guard immediately so pull in this same reconcile cannot drop it.
      markPinSticky(id)
    }
  }

  for (const id of [...pending]) {
    // Sticky unpin that the user has not re-added must not be flipped true.
    if (unpinSticky.has(id) && !current.has(id)) {
      pending.delete(id)
      continue
    }

    const row = $sessions.get().find(entry => sessionMatchesStoredId(entry, id))

    if (!row) {
      continue
    }

    // Already mirrored and acked: nothing to do unless a prior write failed.
    if (mirrored.has(id) && pinAcked.has(id) && !unconfirmed.has(id)) {
      pending.delete(id)
      continue
    }

    pending.delete(id)
    mirrored.add(id)
    void writePin(id, true, row.profile).catch(() => {
      mirrored.delete(id)
      pending.add(id)
    })
  }

  // Retry sticky pins that never got an ack (failed PATCH).
  for (const id of [...pinSticky]) {
    if (!current.has(id) || unconfirmed.has(id) || pinAcked.has(id) || pending.has(id)) {
      continue
    }

    const row = $sessions.get().find(entry => sessionMatchesStoredId(entry, id))
    if (!row) {
      continue
    }

    mirrored.add(id)
    void writePin(id, true, row.profile).catch(() => {
      mirrored.delete(id)
      pending.add(id)
    })
  }
}

/**
 * Adopt the server's pin state for every row in the current page.
 *
 * Runs after local push so in-flight / sticky intent always beats a lagging
 * page. Remote-only changes still flow in once intent is cleared.
 */
function pullRemotePins(): void {
  const local = new Set($pinnedSessionIds.get())

  for (const row of $sessions.get()) {
    // A backend without the flag has no opinion; never act on `undefined`.
    if (typeof row.pinned !== 'boolean') {
      continue
    }

    // Pins are keyed on the durable lineage root so they survive compression
    // tip rotation; the row may surface under either identity.
    const pinId = sessionPinId(row)
    const heldLocally = local.has(pinId) || local.has(row.id)

    // A write of ours the page hasn't caught up to yet is newer than the page.
    const awaited = guardedValue(pinId, row.id)

    if (awaited !== undefined && awaited !== row.pinned) {
      continue
    }

    // Local unpin intent: ignore lagging pinned=true until the server agrees.
    if (isUnpinSticky(pinId, row.id)) {
      if (row.pinned) {
        continue
      }
      clearUnpinSticky(pinId)
      clearUnpinSticky(row.id)
    }

    // Local pin intent: ignore lagging pinned=false until the server agrees.
    if (isPinSticky(pinId, row.id)) {
      if (!row.pinned) {
        continue
      }
      clearPinSticky(pinId)
      clearPinSticky(row.id)
    }

    if (row.pinned && !heldLocally) {
      pinSession(pinId)
      // Already true server-side; record it so the push pass doesn't re-PATCH.
      mirrored.add(pinId)
      local.add(pinId)
    } else if (!row.pinned && heldLocally) {
      // Drop every alias that names this conversation (legacy live-id pins and
      // durable lineage-root pins both count).
      unpinSession(pinId, row.id)
      mirrored.delete(pinId)
      mirrored.delete(row.id)
      local.delete(pinId)
      local.delete(row.id)
      clearPinSticky(pinId)
      clearPinSticky(row.id)
    }
  }
}

function reconcile(): void {
  // Config/session REST is only reachable through the Electron bridge.
  if (!window.hermesDesktop) {
    return
  }

  const current = new Set($pinnedSessionIds.get())

  // 1) Push local intent first (sets unconfirmed + sticky guards).
  pushLocalUnpins(current)
  pushLocalPins(new Set($pinnedSessionIds.get()))
  // 2) Then adopt remote truth for ids we have no opposing local intent about.
  pullRemotePins()
}

// Sync once, then re-sync on pin-set and session-list changes. Call once per app.
export function watchSessionPins(): void {
  reconcile()
  $pinnedSessionIds.listen(reconcile)
  $sessions.listen(reconcile)
}

/** Test-only: wipe module mirrors so cases don't bleed across tests. */
export function __resetSessionPinSyncForTests(): void {
  mirrored.clear()
  pending.clear()
  unconfirmed.clear()
  unpinSticky.clear()
  unpinAcked.clear()
  pinSticky.clear()
  pinAcked.clear()
}
