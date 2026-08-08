/**
 * Reconcile local sidebar pins with the backend pinned flag.
 *
 * The local pin cache is scoped to the active Desktop connection. Backend rows
 * are authoritative when they expose `pinned`, while local writes are guarded
 * against stale list responses that were already in flight.
 */

import type { HermesConnection } from '@/global'
import { setSessionPinnedRemote } from '@/hermes'
import { desktopConnectionScope } from '@/lib/connection-scope'
import { $pinnedSessionIds, pinSession, unpinSession } from '@/store/layout'
import { $connection, $sessions, sessionMatchesStoredId, sessionPinId } from '@/store/session'
import {
  activatePinnedSessionConnection,
  initializePinnedSessionScope,
  legacyPinnedSessionIds,
  pinnedSessionScopeInitialized
} from '@/store/session-pins'

interface PinSyncState {
  pending: Set<string>
  mirrored: Set<string>
  // A write remains guarded until a page confirms it or the cooldown expires.
  unconfirmed: Map<string, { at: number; value: boolean }>
}

const WRITE_GUARD_MS = 10_000

let activeSyncScope: null | string = null
let activeSyncState: null | PinSyncState = null
let hydratingPinScope = false

function activatePinSyncConnection(connection: HermesConnection | null): void {
  const scope = desktopConnectionScope(connection)

  activeSyncScope = scope
  activeSyncState = scope
    ? { mirrored: new Set<string>(), pending: new Set<string>(), unconfirmed: new Map() }
    : null
}

function hydratePinScope(connection: HermesConnection | null): void {
  activatePinSyncConnection(connection)
  hydratingPinScope = true

  try {
    activatePinnedSessionConnection(connection)
  } finally {
    hydratingPinScope = false
  }

  // A scoped localStorage value is cache state, not new user intent.
  if (activeSyncState) {
    activeSyncState.mirrored = new Set($pinnedSessionIds.get())
  }
}

function initializePinScopeFromSessions(): void {
  if (pinnedSessionScopeInitialized()) {
    return
  }

  const sessions = $sessions.get()

  if (sessions.length === 0) {
    return
  }

  // Modern backends own pin truth. For legacy backends, migrate only pins that
  // resolve to a row in this connection; never copy the old global list whole.
  const ids = sessions.some(row => typeof row.pinned === 'boolean')
    ? []
    : legacyPinnedSessionIds().filter(id => sessions.some(row => sessionMatchesStoredId(row, id)))

  initializePinnedSessionScope(ids)
}

function profileFor(pinId: string): null | string | undefined {
  return $sessions.get().find(row => sessionMatchesStoredId(row, pinId))?.profile
}

/** PATCH the flag and keep a per-connection stale-read guard. */
function writePin(state: PinSyncState, id: string, pinned: boolean, profile?: null | string): Promise<void> {
  const confirmation = { at: Date.now(), value: pinned }
  state.unconfirmed.set(id, confirmation)

  return setSessionPinnedRemote(id, pinned, profile).then(
    () => {
      // Do not clear here: a pre-write list request can resolve after this ack.
    },
    (err: unknown) => {
      // A failed write leaves the server on the old value, so let the page win.
      if (state.unconfirmed.get(id) === confirmation) {
        state.unconfirmed.delete(id)
      }

      throw err
    }
  )
}

function pullRemotePins(state: PinSyncState): void {
  const local = new Set($pinnedSessionIds.get())

  for (const row of $sessions.get()) {
    if (typeof row.pinned !== 'boolean') {
      continue
    }

    const pinId = sessionPinId(row)
    const heldLocally = local.has(pinId) || local.has(row.id)

    // Confirmed pages release the guard. Contradictory pages are ignored while
    // the cooldown is active, then the server wins if no confirmation arrives.
    const guardKey = state.unconfirmed.has(pinId) ? pinId : state.unconfirmed.has(row.id) ? row.id : null
    const guard = guardKey ? state.unconfirmed.get(guardKey) : undefined

    if (guard && guardKey) {
      if (guard.value === row.pinned) {
        state.unconfirmed.delete(guardKey)
      } else if (Date.now() - guard.at < WRITE_GUARD_MS) {
        continue
      } else {
        state.unconfirmed.delete(guardKey)
      }
    }

    // A local toggle whose row was not loaded when the push pass ran is newer.
    if (state.pending.has(pinId) || state.pending.has(row.id)) {
      continue
    }

    if (row.pinned && !heldLocally) {
      state.mirrored.add(pinId)
      pinSession(pinId)
    } else if (!row.pinned && heldLocally) {
      state.mirrored.delete(pinId)
      state.mirrored.delete(row.id)
      unpinSession(local.has(pinId) ? pinId : row.id)
    }
  }
}

function reconcile(): void {
  if (!window.hermesDesktop) {
    return
  }

  const connection = $connection.get()

  if (!connection || desktopConnectionScope(connection) !== activeSyncScope) {
    return
  }

  const state = activeSyncState

  if (!state) {
    return
  }

  initializePinScopeFromSessions()
  const current = new Set($pinnedSessionIds.get())

  // Push before pull so local intent is fenced before reading session rows.
  for (const id of [...state.mirrored, ...state.pending]) {
    if (!current.has(id)) {
      state.mirrored.delete(id)
      state.pending.delete(id)
      void writePin(state, id, false, profileFor(id)).catch(() => {})
    }
  }

  for (const id of current) {
    if (!state.mirrored.has(id)) {
      state.pending.add(id)
    }
  }

  for (const id of [...state.pending]) {
    const row = $sessions.get().find(entry => sessionMatchesStoredId(entry, id))

    if (!row) {
      continue
    }

    state.pending.delete(id)
    state.mirrored.add(id)
    void writePin(state, id, true, row.profile).catch(() => {
      state.mirrored.delete(id)
      state.pending.add(id)
    })
  }

  pullRemotePins(state)
}

// Sync on connection, pin-set, and session-list changes.
export function watchSessionPins(): () => void {
  const offPins = $pinnedSessionIds.listen(() => {
    if (!hydratingPinScope) {
      reconcile()
    }
  })

  const offSessions = $sessions.listen(reconcile)

  const offConnection = $connection.listen(connection => {
    hydratePinScope(connection)
    reconcile()
  })

  hydratePinScope($connection.get())
  reconcile()

  return () => {
    offConnection()
    offSessions()
    offPins()
  }
}

/** Forget mirror bookkeeping when the active backend changes. */
export function resetSessionPinMirror(): void {
  if (activeSyncState) {
    activeSyncState.mirrored.clear()
    activeSyncState.pending.clear()
    activeSyncState.unconfirmed.clear()
  }
}
