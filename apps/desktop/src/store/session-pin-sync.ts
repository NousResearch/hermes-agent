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

function profileFor(pinId: string): null | string | undefined {
  return $sessions.get().find(row => sessionMatchesStoredId(row, pinId))?.profile
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
 *
 * ``justChanged`` carries ids whose local state was just changed by the user
 * (the symmetric difference between the pinned set and what we've successfully
 * mirrored).  Those ids must be skipped: the ``$sessions`` page is stale until
 * the push pass writes the new value and the next refresh carries it, so
 * adopting the server's old state would immediately undo the user's click.
 */
function pullRemotePins(justChanged: Set<string>): void {
  const local = new Set($pinnedSessionIds.get())

  for (const row of $sessions.get()) {
    // A backend without the flag has no opinion; never act on `undefined`.
    if (typeof row.pinned !== 'boolean') {
      continue
    }

    // Pins are keyed on the durable lineage root so they survive compression
    // tip rotation; the row may surface under either identity.
    const pinId = sessionPinId(row)

    // Skip ids whose local state was just changed by the user — the page is
    // stale and the push pass will write the correct value.
    if (justChanged.has(pinId) || justChanged.has(row.id)) {
      continue
    }

    const heldLocally = local.has(pinId) || local.has(row.id)

    // A write of ours the page hasn't caught up to yet is newer than the page.
    const awaited = unconfirmed.has(pinId) ? unconfirmed.get(pinId) : unconfirmed.get(row.id)

    if (awaited !== undefined && awaited !== row.pinned) {
      continue
    }

    if (row.pinned && !heldLocally) {
      pinSession(pinId)
      // Already true server-side; record it so the push pass doesn't re-PATCH.
      mirrored.add(pinId)
    } else if (!row.pinned && heldLocally) {
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

  // Compute the set of ids whose local pin state was just changed by the user
  // but hasn't been mirrored to the server yet.  Pull must skip these: the
  // $sessions page still carries the old server value and would immediately
  // undo the user's click.  This is the symmetric difference between what the
  // local set says and what we've successfully mirrored.
  const prePull = new Set($pinnedSessionIds.get())
  const justChanged = new Set<string>()

  // Newly pinned (in local set, not yet mirrored) — the push pass will write
  // pinned=true; pull must not see pinned=false and undo it.
  for (const id of prePull) {
    if (!mirrored.has(id) && !pending.has(id)) {
      justChanged.add(id)
    }
  }

  // Newly unpinned (was mirrored/pending, no longer in local set) — the push
  // pass will write pinned=false; pull must not see pinned=true and re-pin it.
  for (const id of [...mirrored, ...pending]) {
    if (!prePull.has(id)) {
      justChanged.add(id)
    }
  }

  pullRemotePins(justChanged)

  // Re-read after pull — pullRemotePins may have adopted/dropped pins via
  // pinSession/unpinSession, and the push pass must see the updated set.
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
