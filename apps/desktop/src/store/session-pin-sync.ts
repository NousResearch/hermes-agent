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
// Unpins local intent has issued but the server hasn't confirmed yet (failed
// PATCH or a page that still reports pinned=true), keyed by the id with the
// owning profile captured at unpin time. The pull must not adopt a stale
// pinned=true row while one of these is outstanding, or a transient unpin
// failure would silently re-pin the chat; settled when a page confirms
// pinned=false, the user re-pins, or the unpin write lands on a row that is
// no longer in the list.
const unpinPending = new Map<string, null | string | undefined>()
// One chain per id so a quick pin->unpin (or pin->unpin->pin) can't land out
// of order: the server applies PATCHes in arrival order, and two parallel
// writes for the same session can arrive swapped, leaving a pin the user just
// unpinned. The chain lives until every queued write for the id has settled,
// and its presence fences the pull — a page read while any write is queued or
// in flight predates the newest local intent and must not be adopted.
const writeChains = new Map<string, Promise<void>>()

function profileFor(pinId: string): null | string | undefined {
  return $sessions.get().find(row => sessionMatchesStoredId(row, pinId))?.profile
}

/**
 * Resolves once every queued or in-flight write for the id has settled.
 *
 * Used by flows that mutate the session outside the pin-sync write path (the
 * archive request) so their own PATCH cannot be overtaken by an earlier
 * in-flight pin write for the same session.
 */
export function flushSessionPinWrites(id: string): Promise<void> {
  const chain = writeChains.get(id)

  return chain ? chain.catch(() => {}) : Promise.resolve()
}

/** PATCH the flag. Writes for the same id are serialized per session. */
function writePin(id: string, pinned: boolean, profile?: null | string): Promise<void> {
  const prev = writeChains.get(id) ?? Promise.resolve()

  const next: Promise<void> = prev
    .then(() => setSessionPinnedRemote(id, pinned, profile))
    .then(() => undefined)

  // The chain swallows rejections so a failed write doesn't break the queue;
  // callers observe the rejection on `next` and decide on retries.
  const settled = next.catch(() => {})
  writeChains.set(id, settled)
  void settled.finally(() => {
    // Only the newest chain owns the entry; an earlier settling chain must not
    // delete a newer write's fence (the ABA hazard for boolean fences).
    if (writeChains.get(id) === settled) {
      writeChains.delete(id)
    }
  })

  return next
}

/**
 * Adopt the server's pin state for every row in the current page.
 *
 * Runs after the push pass so local intent is already fenced (`pending` /
 * the per-id write chain) by the time the page is read — a fresh local toggle
 * whose PATCH hasn't landed yet must win over the stale row, not be reverted
 * by it (#74570). Remote pins adopted here are marked mirrored before the
 * local set changes, so the re-entrant reconcile doesn't echo them back as a
 * PATCH.
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

    // Any write of ours that is still queued or in flight is newer than this
    // page — the page predates the newest local intent, so never adopt it.
    if (writeChains.has(pinId) || writeChains.has(row.id)) {
      continue
    }

    // Local intent still waiting on its PATCH (row unresolved when the push
    // pass ran) is also newer than the page — never revert it.
    if (pending.has(pinId) || pending.has(row.id)) {
      continue
    }

    // An unpin the server hasn't confirmed is newer than any pinned=true page;
    // adopting it would silently undo the user's action.
    if (unpinPending.has(pinId) || unpinPending.has(row.id)) {
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
  // The push pass below records the intent (`pending`, then the per-id write
  // chain) — only then may the pull read the page, where those fences stop
  // the still-stale row from silently reverting the user's action (#74570).
  const current = new Set($pinnedSessionIds.get())

  // Unpinned: anything we were tracking that's no longer in the set.
  for (const id of [...mirrored, ...pending]) {
    if (!current.has(id)) {
      mirrored.delete(id)
      pending.delete(id)
      // Record the intent (with the owning profile captured now, while the row
      // is still resolvable) so a stale page can't re-pin it before the write
      // is confirmed; settleUnpins() re-asserts and retires the entry.
      unpinPending.set(id, profileFor(id))
      void writePin(id, false, profileFor(id)).catch(() => {})
    }
  }

  // Newly pinned: hold until we can resolve the row (for its profile).
  for (const id of current) {
    // A re-pin supersedes any outstanding unpin intent for the same chat.
    unpinPending.delete(id)

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
  settleUnpins()
}

/**
 * Re-assert unpins the server hasn't confirmed yet, and retire the ones it has.
 *
 * A failed (or still-queued) unpin write leaves the backend believing the chat
 * is pinned. Retrying here keeps local intent honest without swallowing the
 * failure, and a page that finally reports pinned=false retires the entry —
 * the pull only adopts server rows again once they match what the user asked
 * for. A re-pin clears the entry in the push pass above.
 */
function settleUnpins(): void {
  const current = new Set($pinnedSessionIds.get())

  for (const [id, profile] of [...unpinPending]) {
    if (current.has(id)) {
      // User re-pinned; the push pass owns the id from here.
      unpinPending.delete(id)

      continue
    }

    const row = $sessions.get().find(entry => sessionMatchesStoredId(entry, id))

    if (row && row.pinned === false) {
      // Server truth caught up with local intent.
      unpinPending.delete(id)

      continue
    }

    if (writeChains.has(id)) {
      continue // a write for this id is already queued or in flight
    }

    if (!row) {
      // The row is not in the current list (profile switch, list-scope
      // refresh, or the session was archived/deleted). That is not
      // confirmation: re-assert with the profile captured when the unpin was
      // issued. Retire only once the write lands (durable flag is then 0 and
      // any later pinned=true is new truth to adopt) or the session is gone.
      void writePin(id, false, profile).then(
        () => unpinPending.delete(id),
        (err: unknown) => {
          if (isSessionGoneError(err)) {
            unpinPending.delete(id)
          }
        }
      )

      continue
    }

    // Row still present and still reports pinned — keep re-asserting.
    void writePin(id, false, profile).catch(() => {})
  }
}

/** A 404 from the session PATCH means the row itself is gone. */
function isSessionGoneError(err: unknown): boolean {
  return String(err instanceof Error ? err.message : err).includes('404')
}

// Sync once, then re-sync on pin-set and session-list changes. Call once per app.
export function watchSessionPins(): void {
  reconcile()
  $pinnedSessionIds.listen(reconcile)
  $sessions.listen(reconcile)
}
