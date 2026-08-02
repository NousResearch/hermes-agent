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
 * Runs after the push pass so local intent is already fenced (`pending` /
 * `unconfirmed`) by the time the page is read — a fresh local toggle whose
 * PATCH hasn't landed yet must win over the stale row, not be reverted by it
 * (#74570). Remote pins adopted here are marked mirrored before the local set
 * changes, so the re-entrant reconcile doesn't echo them back as a PATCH.
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
    const awaited = unconfirmed.has(pinId) ? unconfirmed.get(pinId) : unconfirmed.get(row.id)

    if (awaited !== undefined && awaited !== row.pinned) {
      continue
    }

    // Local intent still waiting on its PATCH (row unresolved when the push
    // pass ran) is also newer than the page — never revert it.
    if (pending.has(pinId) || pending.has(row.id)) {
      continue
    }

    if (row.pinned && !heldLocally) {
      // Mark mirrored first: pinSession fires the pin listener synchronously,
      // and the nested reconcile must not see this as a new pin to PATCH.
      mirrored.add(pinId)
      pinSession(pinId)
    } else if (!row.pinned && heldLocally) {
      // Same discipline on the way down: forget the mirror before the nested
      // reconcile runs, or it re-PATCHes pinned=false the server already has.
      mirrored.delete(pinId)
      mirrored.delete(row.id)
      unpinSession(local.has(pinId) ? pinId : row.id)
    }
  }
}

function reconcile(): void {
  // Config/session REST is only reachable through the Electron bridge.
  if (!window.hermesDesktop) {
    return
  }

  // Push before pull. The pin listener fires synchronously on a local toggle,
  // so this reconcile runs before the PATCH for that toggle exists anywhere.
  // The push pass below records the intent (`pending`, then `unconfirmed` via
  // writePin) — only then may the pull read the page, where those fences stop
  // the still-stale row from silently reverting the user's action (#74570).
  const current = new Set($pinnedSessionIds.get())

  // Unpinned: anything we were tracking that's no longer in the set.
  for (const id of [...mirrored, ...pending]) {
    if (!current.has(id)) {
      mirrored.delete(id)
      pending.delete(id)
      void writePin(id, false, profileFor(id)).catch((err) => {
        console.warn('[pin-sync] unpin failed:', id, err)
      })
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

  pullRemotePins()
}

// Sync once, then re-sync on pin-set and session-list changes. Call once per app.
export function watchSessionPins(): void {
  reconcile()
  $pinnedSessionIds.listen(reconcile)
  $sessions.listen(reconcile)
}
