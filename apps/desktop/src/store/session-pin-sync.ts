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
 * the local set untouched — and a page that predates one of our own writes is
 * fenced out until a later page confirms the value we wrote.
 */

import { atom } from 'nanostores'

import { ambientOwnerConnectionId, getApiRequestConnection } from '@/api/client'
import { setSessionPinnedRemote } from '@/hermes'
import { translateNow } from '@/i18n'
import {
  type ConnectionScopeDescriptor,
  connectionScopeSuffix,
  onConnectionScopeChange,
  updateConnectionScopedValue
} from '@/lib/connection-scoped'
import { sessionPinWriteKey } from '@/lib/session-pin-key'
import { insertUniqueId } from '@/lib/storage'
import { $primaryGatewayRoute } from '@/store/gateway-primary-route'
import { $pinnedSessionIds, pinSession as pinSessionLocally, unpinSession as unpinSessionLocally } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import {
  $connection,
  $cronSessions,
  $messagingSessions,
  $sessions,
  sessionMatchesStoredId,
  sessionPinId,
  setCronSessions,
  setMessagingSessions,
  setSessions
} from '@/store/session'
import { isSessionRemovalPending } from '@/store/session-removal'
import type { SessionInfo } from '@/types/hermes'

interface ForeignPinWrite {
  row: SessionInfo
  connectionId: string
  at: number
  pending: boolean
  value: boolean
  index?: number
  write?: Promise<void>
  acknowledged?: boolean
  rollbackValue: boolean
  connection?: ConnectionScopeDescriptor
  cachedPins?: { id: string; index: number }[]
}

// Unlike the ambient mirror, these writes name an owner and must survive a
// gateway switch until a real page confirms them (or the cooldown expires).
const foreignPinWrites = new Map<string, ForeignPinWrite>()

// pin ids we've successfully PATCHed pinned=true this session.
let mirrored = new Set<string>()
// pin ids awaiting their row so we can resolve the owning profile before PATCH.
let pending = new Set<string>()
// Writes we've issued, id -> the value we wrote and when. A list page already
// in flight when we PATCH still carries the OLD value, and it can land after
// our ack — so the ack is not proof the page we're reading is newer than the
// write. Hold the guard until a page actually CONFIRMS the written value,
// with a cooldown so a row that never comes back can't fence itself forever.
let unconfirmed = new Map<string, { at: number; value: boolean }>()

const connectionMirrors = new Map<
  string,
  {
    mirrored: Set<string>
    pending: Set<string>
    unconfirmed: Map<string, { at: number; value: boolean }>
  }
>()

let mirrorScope = ''
// Marker stamped on rows this module repainted to match a local toggle. They
// carry OUR intent, not the server's, so they may never confirm (release) a
// write guard — only a row a real list page delivered can do that. It is an
// enumerable symbol so a value copy (`{ ...row }`) carries it too: rows are
// re-spread on ordinary user paths (touchSessionActivity on send,
// applySessionTitle on rename), and identity-keyed tracking would drop
// exactly those copies, letting our own painted value read as a remote
// confirmation and release the guard mid-flight.
const optimisticPin = Symbol('optimisticPin')

type OptimisticSessionRow = SessionInfo & { [optimisticPin]?: true }

function isOptimisticRow(row: SessionInfo): boolean {
  return (row as OptimisticSessionRow)[optimisticPin] === true
}

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

function publishUnconfirmed(): void {
  const published = $unconfirmedPinWrites.get()

  const keys = new Set([
    ...unconfirmed.keys(),
    ...[...foreignPinWrites].filter(([, intent]) => !intent.value).map(([key]) => key)
  ])

  if (published.size === keys.size && [...keys].every(id => published.has(id))) {
    return
  }

  $unconfirmedPinWrites.set(keys)
}

function profileFor(pinId: string): null | string | undefined {
  return loadedRowFor(pinId)?.profile
}

/**
 * Whether `row` lives on the gateway the pin set is scoped to.
 *
 * `$pinnedSessionIds` follows `$connection` (setConnection rescopes it), but
 * the loaded slices can hold other gateways' rows: a gateway pick publishes
 * the new connection while the previous one's rows are still painted, and
 * kept gateways pool their rows into the same lists. Another gateway's
 * `pinned` flag says nothing about this gateway's state.db, and a PATCH for it
 * would ride the ambient connection to a backend that never had the row. A
 * row with no `connection_id` belongs to the primary pool (the row stamp never
 * tags `local` or the primary's own rows): `local` for a local primary, else
 * the primary's registry id; a primary with no registry id keeps counting as
 * the active backend's, as before. A local descriptor owns the local pool;
 * a remote descriptor can inherit an explicit API registry pin, but never
 * the API's potentially stale local-mode default during a rescope.
 */
function ownedByPinScope(row: SessionInfo): boolean {
  const primary = $primaryGatewayRoute.get()
  const owner = row.connection_id?.trim() || (primary.mode === 'local' ? 'local' : primary.connectionId?.trim())

  if (!owner) {
    return true
  }

  const connection = $connection.get()

  // setConnection publishes the descriptor before it updates the API's local
  // mode. During rescope that old mode must not make a remote primary local.
  const active = connection
    ? connection.connectionId?.trim() ||
      (connection.mode === 'local' ? 'local' : (getApiRequestConnection() ?? undefined))
    : ambientOwnerConnectionId()

  return owner === active
}

function allSessionRows(): SessionInfo[] {
  return [...$sessions.get(), ...$cronSessions.get(), ...$messagingSessions.get()]
}

function loadedSessionRows(): SessionInfo[] {
  return allSessionRows().filter(ownedByPinScope)
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

/**
 * Paint the local intent onto every loaded row for `pinId`, in every slice.
 *
 * Without this the cached row keeps the pre-toggle flag until a page replaces
 * it — and a row kept alive as a survivor (open tile, selection, recently
 * settled, or simply no refresh yet) never is. Once the write guard's cooldown
 * lapsed, the next unrelated slice change read that frozen `pinned: true` as a
 * remote pin and re-listed the chat the user had just unpinned (and the mirror
 * image for a fresh pin). Rows without the flag (backend predating it) are left
 * alone so the pull still has no opinion on them.
 */
function applyRowPinned(
  pinId: string,
  pinned: boolean,
  profile?: null | string,
  owns: (row: SessionInfo) => boolean = ownedByPinScope
): void {
  // Session ids are only unique per profile; when the write names its owner,
  // leave a same-id twin in another profile carrying its own server flag.
  const owner = profile == null ? null : normalizeProfileKey(profile)

  const patchRows = (rows: SessionInfo[]): SessionInfo[] => {
    let changed = false

    const next = rows.map(row => {
      if (
        typeof row.pinned !== 'boolean' ||
        row.pinned === pinned ||
        !sessionMatchesStoredId(row, pinId) ||
        !owns(row) ||
        (owner != null && normalizeProfileKey(row.profile) !== owner)
      ) {
        return row
      }

      changed = true
      const updated: OptimisticSessionRow = { ...row, pinned, [optimisticPin]: true }

      return updated
    })

    return changed ? next : rows
  }

  setSessions(patchRows)
  setCronSessions(patchRows)
  setMessagingSessions(patchRows)
}

function updateForeignPinCache(intent: ForeignPinWrite, connection = intent.connection): void {
  if (!connection) {
    return
  }

  const { row } = intent
  intent.connection ??= connection
  const scope = connectionScopeSuffix(connection, false)
  const mirror = scope === mirrorScope ? { mirrored, pending, unconfirmed } : connectionMirrors.get(scope)

  for (const id of [row.id, sessionPinId(row)]) {
    mirror?.mirrored.delete(id)
    mirror?.pending.delete(id)
    mirror?.unconfirmed.delete(id)
  }

  if (intent.value) {
    mirror?.mirrored.add(sessionPinId(row))
  }

  updateConnectionScopedValue($pinnedSessionIds, connection, ids => {
    intent.cachedPins ??= ids.flatMap((id, index) => (sessionMatchesStoredId(row, id) ? [{ id, index }] : []))
    const next = ids.filter(id => !sessionMatchesStoredId(row, id))

    return intent.value
      ? insertUniqueId(next, sessionPinId(row), intent.index ?? next.length)
      : next.length === ids.length
        ? ids
        : next
  })
}

function restoreForeignPinCache(intent: ForeignPinWrite): void {
  if (!intent.connection || !intent.cachedPins) {
    return
  }

  const scope = connectionScopeSuffix(intent.connection, false)
  const mirror = scope === mirrorScope ? { mirrored, pending, unconfirmed } : connectionMirrors.get(scope)

  for (const id of [intent.row.id, sessionPinId(intent.row)]) {
    mirror?.mirrored.delete(id)
    mirror?.pending.delete(id)
    mirror?.unconfirmed.delete(id)
  }

  updateConnectionScopedValue($pinnedSessionIds, intent.connection, ids => {
    const restored = ids.filter(id => !sessionMatchesStoredId(intent.row, id))

    for (const { id, index } of intent.cachedPins!) {
      mirror?.mirrored.add(id)
      restored.splice(index, 0, id)
    }

    return restored
  })
}

/** User actions retain the selected row's owner. A foreign row must never
 *  mutate the active connection's local pin set to trigger an ambient PATCH. */
export function pinSession(id: string, session?: SessionInfo, index?: number): void {
  setSessionPin(id, true, session, index)
}

export function unpinSession(id: string, session?: SessionInfo): void {
  setSessionPin(id, false, session)
}

function setSessionPin(id: string, value: boolean, session?: SessionInfo, index?: number): void {
  const row = session ?? loadedRowFor(id) ?? allSessionRows().find(candidate => sessionMatchesStoredId(candidate, id))
  const key = row ? sessionPinWriteKey(row) : ''
  const previous = foreignPinWrites.get(key)

  if (!row || (ownedByPinScope(row) && !previous)) {
    if (value) {
      pinSessionLocally(id, index)
    } else {
      unpinSessionLocally(id)
    }

    return
  }

  const primary = $primaryGatewayRoute.get()
  const connectionId = row.connection_id?.trim() || (primary.mode === 'local' ? 'local' : primary.connectionId)

  if (!connectionId || previous?.value === value) {
    return
  }

  const intent: ForeignPinWrite = {
    row,
    connectionId,
    at: Date.now(),
    pending: true,
    value,
    index,
    rollbackValue: previous?.value ?? row.pinned ?? !value
  }

  foreignPinWrites.set(key, intent)
  publishUnconfirmed()
  applyRowPinned(sessionPinId(row), value, row.profile, candidate => sessionPinWriteKey(candidate) === key)

  // Opposite clicks share an owner but have different mutation bodies. Wait
  // for the previous write to settle so a slow pin cannot land after Unpin.
  intent.write = writeForeignPin(key, intent, previous).catch(async error => {
    await settlePreviousPin(intent, previous)

    if (foreignPinWrites.get(key) !== intent) {
      return
    }

    foreignPinWrites.delete(key)
    applyRowPinned(
      sessionPinId(row),
      intent.rollbackValue,
      row.profile,
      candidate => sessionPinWriteKey(candidate) === key
    )
    restoreForeignPinCache(intent)
    publishUnconfirmed()
    notifyError(error, translateNow('errors.genericFailure'))
  })
}

async function settlePreviousPin(intent: ForeignPinWrite, previous?: ForeignPinWrite): Promise<void> {
  if (!previous) {
    return
  }

  await previous.write

  if (!previous.acknowledged) {
    intent.rollbackValue = previous.rollbackValue
    intent.cachedPins = previous.cachedPins ?? intent.cachedPins
  }
}

async function writeForeignPin(key: string, intent: ForeignPinWrite, previous?: ForeignPinWrite): Promise<void> {
  const profile = normalizeProfileKey(intent.row.profile)
  const scope = { connectionId: intent.connectionId, profile }
  // Resolve the real descriptor: registry ids alone cannot name a remote
  // localStorage key, which is based on the gateway URL.
  intent.connection = await window.hermesDesktop.getConnectionFor?.(scope)

  if (foreignPinWrites.get(key) !== intent) {
    return
  }

  if (!intent.connection) {
    throw new Error('Session owner connection is unavailable')
  }

  updateForeignPinCache(intent)

  if (previous) {
    await settlePreviousPin(intent, previous)
  }

  if (foreignPinWrites.get(key) !== intent) {
    return
  }

  await setSessionPinnedRemote(sessionPinId(intent.row), intent.value, scope)
  intent.acknowledged = true

  if (foreignPinWrites.get(key) === intent) {
    intent.pending = false
    intent.at = Date.now()
    reconcile()
  }
}

function reconcileForeignPinWrites(): void {
  const rows = allSessionRows()

  for (const [key, intent] of foreignPinWrites) {
    // Explicit local atom edits still supersede a completed foreign action.
    // The public actions above serialize edits while a write is pending.
    if (ownedByPinScope(intent.row) && intent.connection) {
      const pinned = $pinnedSessionIds.get().some(id => sessionMatchesStoredId(intent.row, id))

      if (pinned !== intent.value) {
        foreignPinWrites.delete(key)

        continue
      }
    }

    const confirmed = rows.some(
      row => sessionPinWriteKey(row) === key && row.pinned === intent.value && !isOptimisticRow(row)
    )

    if (!intent.pending && (confirmed || Date.now() - intent.at >= WRITE_GUARD_MS)) {
      foreignPinWrites.delete(key)
    } else {
      applyRowPinned(sessionPinId(intent.row), intent.value, intent.row.profile, row => sessionPinWriteKey(row) === key)
    }
  }
}

/** PATCH the flag, guarding reads against pages that predate the write. */
function writePin(id: string, pinned: boolean, profile?: null | string): Promise<void> {
  const writes = unconfirmed
  const write = { at: Date.now(), value: pinned }
  writes.set(id, write)
  applyRowPinned(id, pinned, profile)

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
      // The painted rows are left as-is (pin or unpin alike): a failed pin is
      // retried by the caller (repainting here would fire a reconcile before
      // that retry is booked and drop the pin), a failed unpin keeps the local
      // value until the next real page, and the next page replaces them anyway.
      if (writes.get(id) === write) {
        writes.delete(id)
      }

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
  const rows = loadedSessionRows().filter(row => !foreignPinWrites.has(sessionPinWriteKey(row)))

  for (const row of rowsByPinId(rows).values()) {
    // A backend without the flag has no opinion; never act on `undefined`.
    if (typeof row.pinned !== 'boolean') {
      continue
    }

    // Pins are keyed on the durable lineage root so they survive compression
    // tip rotation; the row may surface under either identity.
    const pinId = sessionPinId(row)
    const heldLocally = local.has(pinId) || local.has(row.id)

    // A write of ours this page may predate. Confirmed (page agrees) → release
    // the guard, the server has caught up. Contradicted but still inside the
    // cooldown → the page was almost certainly issued before our PATCH, so our
    // write is newer: skip the row. Contradicted past the cooldown → no page
    // ever confirmed us, so stop fencing and let the server win.
    const guardKey = unconfirmed.has(pinId) ? pinId : unconfirmed.has(row.id) ? row.id : null
    const guard = guardKey ? unconfirmed.get(guardKey) : undefined

    if (guard && guardKey) {
      if (isOptimisticRow(row)) {
        // Our own painted row agrees by construction (as does any value copy
        // of it — the marker is enumerable and spreads with the row), so it is
        // not a confirmation. Let the cooldown retire the guard as usual; the
        // row then matches the local set and the checks below are a no-op.
        if (Date.now() - guard.at >= WRITE_GUARD_MS) {
          unconfirmed.delete(guardKey)
        }
      } else if (guard.value === row.pinned) {
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

    if (row.pinned && !heldLocally) {
      // Mark mirrored first: pinSession fires the pin listener synchronously,
      // and the nested reconcile must not see this as a new pin to PATCH.
      mirrored.add(pinId)
      persistServerPin(row, true)
    } else if (!row.pinned && heldLocally) {
      // Same discipline on the way down: forget the mirror before the nested
      // reconcile runs, or it re-PATCHes pinned=false the server already has.
      mirrored.delete(pinId)
      mirrored.delete(row.id)
      persistServerPin(row, false)
    }
  }
}

/** Server corrections made during a scope reload must bypass its ordinary
 *  persistence-echo suppression, or the next visit reads the same stale id. */
function persistServerPin(row: SessionInfo, pinned: boolean): void {
  const update = (ids: string[]): string[] =>
    pinned ? insertUniqueId(ids, sessionPinId(row), ids.length) : ids.filter(id => !sessionMatchesStoredId(row, id))

  const connection = $connection.get()

  if (connection) {
    updateConnectionScopedValue($pinnedSessionIds, connection, update)
  } else {
    $pinnedSessionIds.set(update($pinnedSessionIds.get()))
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

  reconcileForeignPinWrites()

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

      // Removal owns the durable write; a separate unpin would race its
      // revision snapshot. Archive clears the pin in the same PATCH.
      if (!isSessionRemovalPending(id)) {
        void writePin(id, false, profileFor(id)).catch(() => {})
      }
    }
  }

  // Newly pinned: hold until we can resolve the row (for its profile).
  for (const id of current) {
    if (!mirrored.has(id)) {
      pending.add(id)
    }
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
    const sentMirror = mirrored
    const sentPending = pending
    void writePin(id, true, row.profile).catch(() => {
      // Let a later reconcile retry the mirror.
      sentMirror.delete(id)
      sentPending.add(id)
    })
  }

  pullRemotePins()
}

// Sync once, then re-sync on pin-set and session-list changes. Call once per app.
export function watchSessionPins(): void {
  mirrorScope = connectionScopeSuffix($connection.get(), false)
  // Returning to a gateway reloads a cache, not a new user pin. Keep its
  // acknowledged mirror so a confirming false row can retire a stale cached
  // id instead of the push pass reasserting true before the pull sees it.
  onConnectionScopeChange(() => {
    connectionMirrors.set(mirrorScope, { mirrored, pending, unconfirmed })
    mirrorScope = connectionScopeSuffix($connection.get(), false)
    const saved = connectionMirrors.get(mirrorScope)
    mirrored = saved?.mirrored ?? new Set()
    pending = saved?.pending ?? new Set()
    unconfirmed = saved?.unconfirmed ?? new Map()
    applyRescopedForeignPins()
    publishUnconfirmed()
  })
  reconcile()
  $pinnedSessionIds.listen(reconcile)
  $sessions.listen(reconcile)
  $cronSessions.listen(reconcile)
  $messagingSessions.listen(reconcile)
}

/**
 * Forget the mirrors after a backend reset. Ordinary gateway picks retain
 * each scope's mirror in watchSessionPins; an explicit reset invalidates
 * them all so the next reconcile can migrate the cache as it does at boot.
 */
export function resetSessionPinMirror(): void {
  connectionMirrors.clear()
  mirrored = new Set()
  pending = new Set()
  unconfirmed = new Map()
  mirrorScope = connectionScopeSuffix($connection.get(), false)
  applyRescopedForeignPins()
  publishUnconfirmed()
}

function applyRescopedForeignPins(): void {
  // Scope listeners run before the pin atom reloads. Apply the pending intent
  // to the persisted owner cache, even while descriptor lookup is pending.
  for (const intent of foreignPinWrites.values()) {
    if (ownedByPinScope(intent.row)) {
      updateForeignPinCache(intent, $connection.get() ?? undefined)
    }
  }
}

/**
 * Forget one id's sync bookkeeping. The dead-session prune calls this before
 * unpinning a pin whose session is gone, so the reconcile listener (fired
 * synchronously by `unpinSession`) doesn't re-PATCH the dead id — which would
 * produce exactly the 404 the prune exists to remove.
 */
export function forgetPinSyncState(id: string): void {
  mirrored.delete(id)
  pending.delete(id)
  unconfirmed.delete(id)
}
