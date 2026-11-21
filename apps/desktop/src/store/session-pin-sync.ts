/** Server-authoritative session pins. localStorage is a read-through UI cache.
 * Only a local pin-set change may PATCH; boot and list updates pull only. */
import { atom } from 'nanostores'

import { ambientOwnerConnectionId } from '@/api/client'
import { getSession, setSessionPinnedRemote } from '@/hermes'
import { onConnectionScopeChange } from '@/lib/connection-scoped'
import { $primaryGatewayRoute } from '@/store/gateway-primary-route'
import { $pinnedSessionIds, pinSession, unpinSession } from '@/store/layout'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import {
  $connection, $cronSessions, $messagingSessions, $sessions,
  sessionMatchesStoredId, sessionPinId, setCronSessions, setMessagingSessions, setSessions
} from '@/store/session'
import { isSessionRemovalPending } from '@/store/session-removal'
import type { SessionInfo } from '@/types/hermes'

export interface SessionPinnedChangedPayload {
  profile: string
  pinned: boolean
  revision: number
  session_id: string
}

const mirrored = new Set<string>()
const pending = new Set<string>()
const unconfirmed = new Map<string, { at: number; value: boolean }>()
const optimisticPin = Symbol('optimisticPin')
type OptimisticSessionRow = SessionInfo & { [optimisticPin]?: true }
const isOptimisticRow = (row: SessionInfo): boolean => (row as OptimisticSessionRow)[optimisticPin] === true

export const $unconfirmedPinWrites = atom<ReadonlySet<string>>(new Set())
const WRITE_GUARD_MS = 10_000

function publishUnconfirmed(): void {
  const published = $unconfirmedPinWrites.get()

  if (published.size === unconfirmed.size && [...unconfirmed.keys()].every(id => published.has(id))) {return}
  $unconfirmedPinWrites.set(new Set(unconfirmed.keys()))
}

function ownedByPinScope(row: SessionInfo): boolean {
  const primary = $primaryGatewayRoute.get()
  const owner = row.connection_id?.trim() || (primary.mode === 'local' ? 'local' : primary.connectionId?.trim())

  return !owner || owner === ($connection.get()?.connectionId?.trim() || ambientOwnerConnectionId())
}

function loadedSessionRows(): SessionInfo[] {
  return [...$sessions.get(), ...$cronSessions.get(), ...$messagingSessions.get()].filter(ownedByPinScope)
}

function loadedRowFor(pinId: string): SessionInfo | undefined {
  const rows = loadedSessionRows().filter(row => sessionMatchesStoredId(row, pinId))
  const gateway = normalizeProfileKey($activeGatewayProfile.get())

  return rows.find(row => normalizeProfileKey(row.profile) === gateway) ?? rows[0]
}

function rowsByPinId(rows: readonly SessionInfo[]): Map<string, SessionInfo> {
  const byId = new Map<string, SessionInfo>()
  const gateway = normalizeProfileKey($activeGatewayProfile.get())

  for (const row of rows) {
    const pinId = sessionPinId(row)
    const existing = byId.get(pinId)

    if (!existing || (normalizeProfileKey(row.profile) === gateway && normalizeProfileKey(existing.profile) !== gateway)) {
      byId.set(pinId, row)
    }
  }

  return byId
}

function applyRowPinned(pinId: string, pinned: boolean, profile?: null | string): void {
  const owner = profile == null ? null : normalizeProfileKey(profile)

  const patchRows = (rows: SessionInfo[]): SessionInfo[] => {
    let changed = false

    const next = rows.map(row => {
      if (typeof row.pinned !== 'boolean' || row.pinned === pinned || !sessionMatchesStoredId(row, pinId)
        || !ownedByPinScope(row) || (owner != null && normalizeProfileKey(row.profile) !== owner)) {return row}

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

let reconciling = false

function adoptAuthoritativePin(row: SessionInfo): void {
  const id = sessionPinId(row)
  forgetPinSyncState(id)
  unconfirmed.delete(row.id)
  applyRowPinned(id, Boolean(row.pinned), row.profile)
  reconciling = true

  try {
    if (row.pinned) {
      mirrored.add(id)
      pinSession(id)
    } else {
      const local = $pinnedSessionIds.get()
      unpinSession(local.includes(id) ? id : row.id)
      mirrored.delete(id)
      mirrored.delete(row.id)
    }
  } finally {
    reconciling = false
    publishUnconfirmed()
  }
}

function isRevisionConflict(error: unknown): boolean {
  if (typeof error !== 'object' || error === null) {return String(error).includes('revision_conflict')}
  const status = Number((error as { statusCode?: unknown }).statusCode)

  return status === 409 || String(error).includes('revision_conflict')
}

async function repullAfterConflict(id: string, profile?: null | string): Promise<void> {
  adoptAuthoritativePin(await getSession(id, profile))
}

/** Apply a committed WS event immediately, scoped to the active profile. */
export function applySessionPinnedChanged(change: SessionPinnedChangedPayload): void {
  if (!change || typeof change.session_id !== 'string' || typeof change.pinned !== 'boolean'
    || normalizeProfileKey(change.profile) !== normalizeProfileKey($activeGatewayProfile.get())) {return}

  adoptAuthoritativePin({
    ...loadedRowFor(change.session_id), id: change.session_id, pinned: change.pinned,
    profile: change.profile, runtime_revision: change.revision
  } as SessionInfo)
}

function writePin(id: string, pinned: boolean, profile?: null | string): Promise<void> {
  unconfirmed.set(id, { at: Date.now(), value: pinned })
  applyRowPinned(id, pinned, profile)

  return setSessionPinnedRemote(id, pinned, profile, loadedRowFor(id)?.runtime_revision).then(
    () => {},
    (error: unknown) => {
      unconfirmed.delete(id)
      publishUnconfirmed()

      if (isRevisionConflict(error)) {return repullAfterConflict(id, profile)}
      throw error
    }
  )
}

function pullRemotePins(): void {
  const local = new Set($pinnedSessionIds.get())

  for (const row of rowsByPinId(loadedSessionRows()).values()) {
    if (typeof row.pinned !== 'boolean') {continue}
    const pinId = sessionPinId(row)
    const heldLocally = local.has(pinId) || local.has(row.id)
    const guardKey = unconfirmed.has(pinId) ? pinId : unconfirmed.has(row.id) ? row.id : null
    const guard = guardKey ? unconfirmed.get(guardKey) : undefined

    if (guard && guardKey) {
      if (isOptimisticRow(row)) {
        if (Date.now() - guard.at >= WRITE_GUARD_MS) {unconfirmed.delete(guardKey)}
      } else if (guard.value === row.pinned) {
        unconfirmed.delete(guardKey)
      } else if (Date.now() - guard.at < WRITE_GUARD_MS) {
        continue
      } else {
        unconfirmed.delete(guardKey)
      }
    }

    if (pending.has(pinId) || pending.has(row.id)) {continue}

    if (row.pinned && !heldLocally) {
      mirrored.add(pinId)
      pinSession(pinId)
    } else if (!row.pinned && heldLocally) {
      mirrored.delete(pinId)
      mirrored.delete(row.id)
      unpinSession(local.has(pinId) ? pinId : row.id)
    }
  }
}

function reconcile(userToggle = false): void {
  if (reconciling) {return}
  reconciling = true

  try {
    if (!window.hermesDesktop) {return}

    // Only an explicit local change creates pending intent. A deferred pin
    // may flush when its row arrives; boot/list pulls never create a write.
    if (userToggle || pending.size > 0) {
      const current = new Set($pinnedSessionIds.get())

      for (const id of [...mirrored, ...pending]) {
        if (!current.has(id)) {
          mirrored.delete(id)
          pending.delete(id)

          if (!isSessionRemovalPending(id)) {void writePin(id, false, loadedRowFor(id)?.profile).catch(() => {})}
        }
      }

      for (const id of current) {if (!mirrored.has(id)) {pending.add(id)}}

      for (const id of [...pending]) {
        const row = loadedRowFor(id)

        if (!row) {continue}
        pending.delete(id)
        mirrored.add(id)
        void writePin(id, true, row.profile).catch(() => {
          mirrored.delete(id)
          pending.add(id)
        })
      }
    }

    pullRemotePins()
  } finally {
    reconciling = false
    publishUnconfirmed()
  }
}

export function watchSessionPins(): () => void {
  const offScope = onConnectionScopeChange(resetSessionPinMirror)
  reconcile()
  const offPins = $pinnedSessionIds.listen(() => reconcile(true))
  const offSessions = $sessions.listen(() => reconcile(false))
  const offCron = $cronSessions.listen(() => reconcile(false))
  const offMessaging = $messagingSessions.listen(() => reconcile(false))

  return () => {
    offScope()
    offPins()
    offSessions()
    offCron()
    offMessaging()
  }
}

export function resetSessionPinMirror(): void {
  mirrored.clear()
  pending.clear()
  unconfirmed.clear()
  publishUnconfirmed()
}

export function forgetPinSyncState(id: string): void {
  mirrored.delete(id)
  pending.delete(id)
  unconfirmed.delete(id)
}
