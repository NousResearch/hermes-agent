/**
 * Reconcile the sidebar's pins with the backend "keep" flag, both directions.
 *
 * Pins drive the sidebar UI out of `$pinnedSessionIds` (localStorage), but the
 * durable record is `sessions.pinned` in each profile's state.db. Two things
 * depend on the backend copy: the `sessions.auto_archive` sweep runs
 * server-side and would otherwise hide a pinned chat, and a second Desktop app
 * pointed at the same gateway has its own, separate localStorage.
 *
 * Push: PATCH `pinned` whenever the local set changes. Pins restored from
 * machine-local storage wait for their first server row: a boolean flag seeds
 * from backend truth, while a row with no flag identifies a legacy backend and
 * keeps the old transparent migration path.
 *
 * Pull: session rows now carry `pinned`, and the list endpoints back-fill
 * pinned conversations past their LIMIT, so a row's absence from a page no
 * longer says anything about its pin state. That makes the server row
 * authoritative: adopt pins this app hasn't seen, and drop local pins the
 * server says are gone. Only rows actually present in the payload are
 * consulted, so a backend predating the flag (`pinned === undefined`) leaves
 * the local set untouched — and a page that predates one of our own writes is
 * fenced out until a later page confirms the value we wrote.
 */

import { atom } from 'nanostores'

import { setSessionPinnedRemote } from '@/hermes'
import { onConnectionScopeChange } from '@/lib/connection-scoped'
import { $pinnedSessionIds, pinSession, unpinSession } from '@/store/layout'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $cronSessions, $messagingSessions, $sessions, sessionMatchesStoredId, sessionPinId } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

// pin ids we've successfully PATCHed pinned=true this session.
const mirrored = new Set<string>()
// pin ids awaiting their row so we can resolve the owning profile before PATCH.
const pending = new Set<string>()
// Fresh unpins of restored boot pins whose row has not loaded yet. Unlike a
// restored membership, this absence is user intent and must survive until the
// row identifies the owning profile.
const pendingUnpins = new Set<string>()
// Pins restored from machine-local storage have no fresh user intent behind
// them. Do not push them until their first row establishes whether this backend
// has authoritative pin state; otherwise a second device can resurrect a pin
// already removed elsewhere.
const unseededLocalPins = new Set<string>()
let observedLocalPins = new Set<string>()
let captureScopedPinSnapshot = false
// Writes we've issued, id -> the value we wrote and when. A list page already
// in flight when we PATCH still carries the OLD value, and it can land after
// our ack — so the ack is not proof the page we're reading is newer than the
// write. Hold the guard until a page actually CONFIRMS the written value,
// with a cooldown so a row that never comes back can't fence itself forever.
const unconfirmed = new Map<string, { at: number; value: boolean }>()

/**
 * The ids `unconfirmed` currently fences, for readers outside this module.
 *
 * The sidebar's Pinned section falls back to the server `pinned` flag for rows
 * the local set doesn't know about, and that fallback needs the same fence the
 * pull pass uses: a row whose flag our own in-flight write contradicts is not
 * news, it's the past. Without it an unpin re-lists the session under Pinned
 * until the next page lands.
 *
 * Re-published only when the key set actually changes, so a sidebar memo keyed
 * on it survives an ordinary session refresh.
 */
export const $unconfirmedPinWrites = atom<ReadonlySet<string>>(new Set())

// How long an unconfirmed write outranks a page that contradicts it. Long
// enough to cover a list request issued just before the PATCH (those are the
// slow ones), short enough that a genuine server-side change still wins.
const WRITE_GUARD_MS = 10_000

function captureScopedPins(): void {
  const current = new Set($pinnedSessionIds.get())

  unseededLocalPins.clear()

  for (const id of current) {
    unseededLocalPins.add(id)
  }

  observedLocalPins = current
  captureScopedPinSnapshot = false
}

function publishUnconfirmed(): void {
  const published = $unconfirmedPinWrites.get()

  if (published.size === unconfirmed.size && [...unconfirmed.keys()].every(id => published.has(id))) {
    return
  }

  $unconfirmedPinWrites.set(new Set(unconfirmed.keys()))
}

function profileFor(pinId: string): null | string | undefined {
  return loadedRowFor(pinId)?.profile
}

function loadedSessionRows(): SessionInfo[] {
  return [...$sessions.get(), ...$cronSessions.get(), ...$messagingSessions.get()]
}

/**
 * The row a stored pin id resolves to, across every slice. Same tie-break as
 * `rowsByPinId`: when two profiles share the id, the write must target the
 * row the pull adopted — the active gateway's — or an unpin PATCHes the other
 * profile and the next page re-adopts the pin.
 */
function loadedRowFor(pinId: string): SessionInfo | undefined {
  const rows = loadedSessionRows().filter(row => sessionMatchesStoredId(row, pinId))
  const gateway = normalizeProfileKey($activeGatewayProfile.get())

  return rows.find(row => normalizeProfileKey(row.profile) === gateway) ?? rows[0]
}

/**
 * One authoritative row per durable pin id. Session ids are only unique inside
 * a profile, so the cross-profile list can legitimately hold two rows with the
 * same `sessionPinId` but different `pinned` flags (copied/imported profile
 * databases). Iterating both would pin then unpin the same id in one pass and
 * re-fire `reconcile` forever — the runaway that overflows nanostores'
 * listenerQueue. Collapse to a single row per id, preferring the active
 * gateway's profile (the same tie-break `resolveLoadedRow` uses), so the pull
 * is deterministic and never oscillates.
 */
function rowsByPinId(rows: readonly SessionInfo[]): Map<string, SessionInfo> {
  const byId = new Map<string, SessionInfo>()
  const gateway = normalizeProfileKey($activeGatewayProfile.get())

  for (const row of rows) {
    const pinId = sessionPinId(row)
    const existing = byId.get(pinId)

    if (!existing) {
      byId.set(pinId, row)

      continue
    }

    // Prefer the active gateway's profile; otherwise keep the first seen.
    if (normalizeProfileKey(row.profile) === gateway && normalizeProfileKey(existing.profile) !== gateway) {
      byId.set(pinId, row)
    }
  }

  return byId
}

function forgetUnseededAliases(row: SessionInfo): void {
  unseededLocalPins.delete(sessionPinId(row))
  unseededLocalPins.delete(row.id)
}

/** PATCH the flag, guarding reads against pages that predate the write. */
function writePin(id: string, pinned: boolean, profile?: null | string): Promise<void> {
  unconfirmed.set(id, { at: Date.now(), value: pinned })

  return setSessionPinnedRemote(id, pinned, profile).then(
    () => {
      // Deliberately NOT cleared here: a list request issued before this PATCH
      // can still land after the ack carrying the pre-write value. The guard
      // is released by pullRemotePins when a page confirms the written value,
      // or by the cooldown if none ever does.
    },
    (err: unknown) => {
      // A failed write leaves the server on the old value, so the guard would
      // be fencing out the truth. Drop it and let the page win.
      unconfirmed.delete(id)
      publishUnconfirmed()
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

  for (const row of rowsByPinId(loadedSessionRows()).values()) {
    // A backend without the flag has no opinion; never act on `undefined`.
    if (typeof row.pinned !== 'boolean') {
      continue
    }

    // Pins are keyed on the durable lineage root so they survive compression
    // tip rotation; the row may surface under either identity.
    const pinId = sessionPinId(row)
    const heldLocally = local.has(pinId) || local.has(row.id)

    // This is the first authoritative answer for a pin restored at boot. Drop
    // the local-only hold before adopting it, so `pinned=false` can remove the
    // stale cache without first being turned into our own guarded write.
    forgetUnseededAliases(row)

    // A write of ours this page may predate. Confirmed (page agrees) → release
    // the guard, the server has caught up. Contradicted but still inside the
    // cooldown → the page was almost certainly issued before our PATCH, so our
    // write is newer: skip the row. Contradicted past the cooldown → no page
    // ever confirmed us, so stop fencing and let the server win.
    const guardKey = unconfirmed.has(pinId) ? pinId : unconfirmed.has(row.id) ? row.id : null
    const guard = guardKey ? unconfirmed.get(guardKey) : undefined

    if (guard && guardKey) {
      if (guard.value === row.pinned) {
        unconfirmed.delete(guardKey)
      } else if (Date.now() - guard.at < WRITE_GUARD_MS) {
        continue
      } else {
        unconfirmed.delete(guardKey)
      }
    }

    // Local intent still waiting on its PATCH (row unresolved when the push
    // pass ran) is also newer than the page — never revert it.
    if (pending.has(pinId) || pending.has(row.id)) {
      continue
    }

    if (row.pinned && heldLocally) {
      // The backend confirms a restored local pin. It is mirrored already;
      // recording that prevents a redundant PATCH on the next refresh.
      mirrored.add(pinId)
    } else if (row.pinned) {
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

// Re-entrancy guard: reconcile() is subscribed to every loaded-session slice
// and $pinnedSessionIds, and pullRemotePins() mutates $pinnedSessionIds (via
// pinSession/unpinSession), which fires reconcile() again synchronously.
// Without this guard, a session whose pin state oscillates — two rows with the
// same durable id but conflicting `pinned` flags, possible when profile
// databases share session ids — drives an unbounded re-entrant loop that
// overflows nanostores' shared listenerQueue and crashes the renderer with
// `RangeError: Invalid array length`.
let reconciling = false

function reconcile(): void {
  if (reconciling) {
    return
  }

  reconciling = true

  try {
    reconcileInner()
  } finally {
    reconciling = false
    // One publish per top-level pass: writePin adds guards and pullRemotePins
    // retires them, and re-entrant calls above returned without touching either.
    publishUnconfirmed()
  }
}

function reconcileInner(): void {
  // Config/session REST is only reachable through the Electron bridge.
  if (!window.hermesDesktop) {
    return
  }

  // A row without `pinned` comes from a backend predating authoritative pin
  // state. Preserve the old boot migration there: once the row identifies that
  // compatibility case, its restored local pin may enter the push pass below.
  for (const row of rowsByPinId(loadedSessionRows()).values()) {
    if (typeof row.pinned !== 'boolean') {
      forgetUnseededAliases(row)
    }
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
      void writePin(id, false, profileFor(id)).catch(() => {})
    }
  }

  // Newly pinned: hold until we can resolve the row (for its profile).
  for (const id of current) {
    if (!mirrored.has(id) && !unseededLocalPins.has(id)) {
      pending.add(id)
    }
  }

  // A restored pin can be removed before its row loads. Keep that negative
  // intent explicitly: absence from `current` alone cannot distinguish it from
  // a pin that was never present in this scope.
  for (const id of [...pendingUnpins]) {
    const row = loadedRowFor(id)

    if (!row) {
      continue
    }

    pendingUnpins.delete(id)
    void writePin(id, false, row.profile).catch(() => pendingUnpins.add(id))
  }

  // Flush whatever we can resolve now; unresolved ids (row not loaded yet)
  // retry on the next loaded-session slice change.
  for (const id of [...pending]) {
    const row = loadedRowFor(id)

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
  // A connection rescope repaints $pinnedSessionIds from the new backend's
  // storage scope; the mirrored/pending/unconfirmed bookkeeping describes
  // the PREVIOUS backend and must reset before that reload reconciles.
  resetSessionPinMirror()
  onConnectionScopeChange(() => {
    resetSessionPinMirror()
    // Scope listeners run before connection-scoped atoms reload. Treat the
    // next pin-set emission as the new backend's boot cache, not a user toggle.
    captureScopedPinSnapshot = true

    // A missing key loads the atom's fallback. Two empty scopes therefore
    // reuse the same array reference and nanostores emits nothing; consume the
    // snapshot explicitly once every scoped atom has finished reloading.
    return () => {
      if (captureScopedPinSnapshot) {
        captureScopedPins()
      }
    }
  })
  reconcile()
  $pinnedSessionIds.listen(() => {
    const current = new Set($pinnedSessionIds.get())

    if (captureScopedPinSnapshot) {
      captureScopedPins()
    } else {
      // Any membership change after the boot snapshot is fresh local intent.
      // This includes toggling a restored pin off and back on before rows load.
      for (const id of new Set([...observedLocalPins, ...current])) {
        if (observedLocalPins.has(id) !== current.has(id)) {
          if (unseededLocalPins.has(id) && !current.has(id)) {
            pendingUnpins.add(id)
          } else if (current.has(id)) {
            pendingUnpins.delete(id)
          }

          unseededLocalPins.delete(id)
        }
      }
    }

    observedLocalPins = new Set($pinnedSessionIds.get())
    reconcile()
  })
  $sessions.listen(reconcile)
  $cronSessions.listen(reconcile)
  $messagingSessions.listen(reconcile)
}

/**
 * Forget what we've mirrored, because the backend we mirrored it TO is gone.
 *
 * `mirrored` / `pending` / `unconfirmed` all mean "relative to the gateway we
 * are talking to". After a soft switch the next backend has its own state.db
 * and has never seen these pins, but `mirrored` would report them as already
 * pushed and suppress the PATCHes — so the user's pins silently fail to reach
 * the new gateway (and its auto-archive sweep is free to hide them). Dropping
 * the bookkeeping makes the next reconcile re-assert the whole set, which is
 * the same path that migrates pre-existing pins at boot.
 */
export function resetSessionPinMirror(): void {
  mirrored.clear()
  pending.clear()
  pendingUnpins.clear()
  unconfirmed.clear()
  unseededLocalPins.clear()

  for (const id of $pinnedSessionIds.get()) {
    unseededLocalPins.add(id)
  }

  observedLocalPins = new Set($pinnedSessionIds.get())
  captureScopedPinSnapshot = false
  publishUnconfirmed()
}
