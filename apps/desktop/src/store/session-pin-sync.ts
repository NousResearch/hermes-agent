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
import { $sessions, sessionMatchesStoredId, sessionPinId } from '@/store/session'

// pin ids we've successfully PATCHed pinned=true this session.
const mirrored = new Set<string>()
// pin ids awaiting their row so we can resolve the owning profile before PATCH.
const pending = new Set<string>()
// Writes we've issued but not yet had acked, id -> value written. A list page
// already in flight when we PATCH still carries the old value, so it must not
// be read as the server disagreeing with us. Cleared when the write settles —
// the request's own lifetime is the guard, so nothing can leave one open.
const unconfirmed = new Map<string, boolean>()
// The pinned set as of the last reconcile. Ids that appear in the set now but
// not here are FRESH local pins whose PATCH has not been sent yet; they must
// be protected from the pull pass (which would otherwise see the page's stale
// pinned=false row and undo the pin before the write even leaves). Null until
// the first reconcile so a pre-existing pin set at boot counts as historical,
// not fresh — the server stays authoritative for those.
let lastSeen: ReadonlySet<string> | null = null

function profileFor(pinId: string): null | string | undefined {
  return $sessions.get().find(row => sessionMatchesStoredId(row, pinId))?.profile
}

/** PATCH the flag, guarding reads against pages that predate the write. */
function writePin(id: string, pinned: boolean, profile?: null | string): Promise<void> {
  unconfirmed.set(id, pinned)

  return setSessionPinnedRemote(id, pinned, profile).then(
    () => {
      unconfirmed.delete(id)
      // The sidebar row cache still carries the pre-PATCH value until the WS
      // row update lands. Refresh it on the ack so a later pull pass never
      // reads our own confirmed write as the server disagreeing (the ack can
      // outlive the row update by seconds, and that window used to undo the
      // pin on the very next reconcile).
      const rows = $sessions.get()

      if (rows.some(row => sessionMatchesStoredId(row, id))) {
        $sessions.set(rows.map(row => (sessionMatchesStoredId(row, id) ? { ...row, pinned } : row)))
      }
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
function pullRemotePins(freshlyUnpinned: ReadonlySet<string> = new Set()): void {
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
    const awaited = unconfirmed.has(pinId) ? unconfirmed.get(pinId) : unconfirmed.get(row.id)

    if (awaited !== undefined && awaited !== row.pinned) {
      continue
    }

    if (row.pinned && !heldLocally && !freshlyUnpinned.has(pinId) && !freshlyUnpinned.has(row.id)) {
      pinSession(pinId)
      // Already true server-side; record it so the push pass doesn't re-PATCH.
      mirrored.add(pinId)
    } else if (!row.pinned && heldLocally && !pending.has(pinId) && !pending.has(row.id)) {
      unpinSession(local.has(pinId) ? pinId : row.id)
      mirrored.delete(pinId)
      mirrored.delete(row.id)
    }
  }
}

function reconcile(): void {
  // Config/session REST is only reachable through the Electron bridge.
  if (!window.hermesDesktop) {
    return
  }

  // Snapshot taken BEFORE the pull pass: ids added since the last reconcile
  // are FRESH local pins whose PATCH has not been sent yet. Register them as
  // pending before the pull so a page row still carrying the pre-PATCH
  // pinned=false value can't undo the pin (the unconfirmed map can't protect
  // this window — the write hasn't started). Symmetrically, ids REMOVED since
  // the last reconcile are fresh unpins whose unpin PATCH has not been sent;
  // the pull pass must not re-adopt them from a stale pinned=true row. The
  // first reconcile (boot) seeds `lastSeen` with the current set instead, so
  // historical pins stay subject to the server's authority both ways.
  const before = new Set($pinnedSessionIds.get())
  const freshlyUnpinned = new Set<string>()

  if (lastSeen === null) {
    lastSeen = before
  } else {
    for (const id of before) {
      if (!lastSeen.has(id) && !mirrored.has(id)) {
        pending.add(id)
      }
    }

    for (const id of lastSeen) {
      if (!before.has(id)) {
        freshlyUnpinned.add(id)
      }
    }

    lastSeen = before
  }

  pullRemotePins(freshlyUnpinned)

  // Snapshot taken AFTER the pull pass: pins the pull just adopted are part
  // of the current set and must not be treated as stale in the passes below.
  const current = new Set($pinnedSessionIds.get())

  // Unpinned: anything we were tracking that's no longer in the set.
  for (const id of [...mirrored, ...pending]) {
    if (!current.has(id)) {
      mirrored.delete(id)
      pending.delete(id)
      void writePin(id, false, profileFor(id)).catch(() => {})
    }
  }

  // Newly pinned: hold until we can resolve the row (for its profile).
  for (const id of current) {
    if (!mirrored.has(id)) {
      pending.add(id)
    }
  }

  // Flush whatever we can resolve now; unresolved ids (row not loaded yet)
  // retry on the next $sessions change.
  for (const id of [...pending]) {
    const row = $sessions.get().find(entry => sessionMatchesStoredId(entry, id))

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
}
