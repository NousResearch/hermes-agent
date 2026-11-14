/**
 * Prune renderer-stored session references whose sessions no longer exist on
 * the backend.
 *
 * Connection-scoped pins (`hermes.desktop.pinnedSessions`) can go stale when a
 * session is removed from state.db and repeatedly trigger 404s at boot.
 *
 * This module sweeps for dead ids after the sidebar list has spoken. A session
 * is declared dead ONLY when a by-id probe 404s on every known profile of the
 * selected connection. A 404 proves nothing about another gateway. Global
 * drafts and queued prompts are NEVER auto-pruned: their shared localStorage
 * maps cannot atomically delete one entry without losing another window's write.
 * Interrupted sends (`hermes:composer-outbox:v1:*`) are one key per entry, so
 * a drop would be atomic — but those keys carry no connection scope either, and
 * a 404 from the selected connection cannot prove one dead; they are left to
 * the outbox's own expiry. Any non-404 probe failure (gateway mid-restart,
 * network) is inconclusive: the id is deferred to a later pass instead of
 * being dropped.
 */

import { getSession } from '@/hermes'
import { mapPool } from '@/lib/pool'
import { $pinnedSessionIds, unpinSession } from '@/store/layout'
import {
  $activeGatewayProfile,
  $freshProfileRosterRevision,
  freshProfileRoster,
  normalizeProfileKey
} from '@/store/profile'
import { $connection, $sessions } from '@/store/session'

import { forgetPinSyncState } from './session-pin-sync'

const SWEEP_CONCURRENCY = 4
// After the first list payload, let boot settle (pin reconcile, resume probes)
// before probing; later list changes re-sweep debounced.
const FIRST_PASS_DELAY_MS = 2_000
const RERUN_DELAY_MS = 5_000
// How long an 'alive' verdict is trusted. Without expiry, a session deleted
// mid-session would stay cached alive until the next app restart — re-creating
// exactly the boot 404s this prune exists to remove. With expiry, a later
// sweep re-probes and heals it within one TTL.
const ALIVE_TTL_MS = 10 * 60_000

// Ids verified ALIVE this app session, id -> verdict expiry timestamp — later
// sweeps skip re-probing ids whose verdict is still fresh.
const alive = new Map<string, number>()
// Ids currently mid-probe, so overlapping sweeps don't double-probe.
const inFlight = new Map<string, number>()

let listLoaded = false
// Bumped by resetDeadSessionPrune: in-flight probes from a pre-reset sweep
// must not write verdicts (alive cache, drops) that belong to the OLD backend.
let epoch = 0
// Debounce deadline, clamped to the EARLIEST change since the last pass: a
// busy list that keeps changing must not starve the sweep indefinitely.
let sweepDueAt = 0
let sweepTimer: ReturnType<typeof setTimeout> | undefined

type Verdict = 'alive' | 'dead' | 'unknown'

function isNotFoundError(error: unknown): boolean {
  // The HTTP transport prefixes status and raw response body. A route/proxy
  // 404 is not proof that the session detail endpoint looked up this id.
  const message = error instanceof Error ? error.message : String(error)
  const match = /^404:\s+(.+)$/s.exec(message)

  if (!match) {
    return false
  }

  try {
    const body: unknown = JSON.parse(match[1])

    return typeof body === 'object' && body !== null && 'detail' in body && body.detail === 'Session not found'
  } catch {
    return false
  }
}

/** The selected connection's registry id; a local connection without one is `local`, a remote one has none. */
function currentConnectionId(): string | undefined {
  const connection = $connection.get()

  return connection?.connectionId?.trim() || (connection?.mode === 'local' ? 'local' : undefined)
}

/**
 * A stored id is dead only if a by-id lookup 404s on the active profile AND on
 * every named profile of this connection. Pin the connection explicitly so an
 * unrelated same-id hint cannot redirect the probe to another backend.
 */
async function sessionVerdict(id: string, profiles: readonly string[], connectionId: string): Promise<Verdict> {
  const activeKey = normalizeProfileKey($activeGatewayProfile.get())

  for (const profile of [activeKey, ...profiles.filter(key => key !== activeKey)]) {
    try {
      // Pin BOTH axes. A session owner hint must not reroute a pin's probe
      // to another gateway, and an ambient dial may change during the await.
      await getSession(id, { connectionId, profile })

      return 'alive'
    } catch (error) {
      if (!isNotFoundError(error)) {
        return 'unknown'
      }
    }
  }

  return 'dead'
}

/** Stored ids no loaded sidebar row covers — the only ids that can be dead. */
function collectCandidates(): string[] {
  const rows = $sessions.get()
  const covered = new Set<string>()

  for (const row of rows) {
    if (row.id) {
      covered.add(row.id)
    }

    if (row._lineage_root_id) {
      covered.add(row._lineage_root_id)
    }
  }

  return $pinnedSessionIds.get().filter(id => {
    const expiresAt = alive.get(id)

    return !covered.has(id) && !(expiresAt !== undefined && Date.now() < expiresAt) && !inFlight.has(id)
  })
}

function dropDeadId(id: string, connectionId: string, probedProfiles: number): void {
  // Cheap re-verification before destructive drops: if a row for this id
  // appeared while the probe was in flight (a session resurrected or a slow
  // page landed), the id is no longer dead — keep every entry.
  const covered = $sessions.get().some(row => row.id === id || row._lineage_root_id === id)

  if (covered) {
    return
  }

  // Forget the pin's sync bookkeeping BEFORE unpinning: the reconcile listener
  // fires synchronously on unpin, and with the id still tracked it would
  // re-PATCH the dead id (a fresh 404 on every boot — the noise we're removing).
  if ($pinnedSessionIds.get().includes(id)) {
    forgetPinSyncState(id)
    unpinSession(id)
    // The one destructive step this module takes leaves a trace in desktop.log
    // (id and verdict basis only, never content), so a vanished pin can be tied
    // to this sweep instead of being inferred from timestamps (#144702).
    window.hermesDesktop?.logLine?.(
      `[dead-session-prune] unpinned ${id}: 404 on ${String(probedProfiles)} profile(s) of connection "${connectionId}"`
    )
  }
}

async function sweepOnce(requireLoaded = true): Promise<void> {
  if (requireLoaded && !listLoaded) {
    return
  }

  if (typeof window === 'undefined' || !window.hermesDesktop) {
    return
  }

  const sweepEpoch = epoch
  const connection = $connection.get()
  const connectionId = currentConnectionId()

  if (!connectionId) {
    return
  }

  // A fresh successful roster from this connection and routing epoch is required;
  // the picker list can be restored from cache after a switch.
  const freshRoster = freshProfileRoster(connectionId)

  if (!freshRoster) {
    return
  }

  const profiles = [...new Set(freshRoster.map(profile => normalizeProfileKey(profile.name)))]
  const activeProfile = normalizeProfileKey($activeGatewayProfile.get())

  if (profiles.length === 0 || !profiles.includes(activeProfile)) {
    return
  }

  const candidates = collectCandidates()

  if (candidates.length === 0) {
    return
  }

  for (const id of candidates) {
    inFlight.set(id, sweepEpoch)
  }

  try {
    await mapPool(candidates, SWEEP_CONCURRENCY, async id => {
      if (epoch !== sweepEpoch) {
        return
      }

      const verdict = await sessionVerdict(id, profiles, connectionId)

      // A gateway switch (reset) while this probe was in flight invalidates
      // the verdict — it is about the backend we measured, not the live one.
      if (
        epoch !== sweepEpoch ||
        $connection.get() !== connection ||
        currentConnectionId() !== connectionId ||
        normalizeProfileKey($activeGatewayProfile.get()) !== activeProfile ||
        freshProfileRoster(connectionId) !== freshRoster
      ) {
        return
      }

      if (verdict === 'alive') {
        alive.set(id, Date.now() + ALIVE_TTL_MS)
      } else if (verdict === 'dead') {
        dropDeadId(id, connectionId, profiles.length)
      }
      // 'unknown' — neither cached nor dropped; a later sweep retries it.
    })
  } finally {
    for (const id of candidates) {
      if (inFlight.get(id) === sweepEpoch) {
        inFlight.delete(id)
      }
    }
  }
}

/** Run one prune pass now. Test-only hook — production never calls it. */
export function __runDeadSessionPrunePass(): Promise<void> {
  // Bypass the list-loaded gate WITHOUT mutating `listLoaded` (a mutation
  // here would leak into the watcher's first-payload timing).
  return sweepOnce(false)
}

/** Schedule a bounded pass after a sidebar change or an accepted fresh roster. */
function scheduleSweep(delay: number): void {
  // Clamp to the EARLIEST change since the last pass: constant list churn
  // must delay the sweep, not starve it.
  const dueAt = Date.now() + delay
  const nextDueAt = sweepTimer ? Math.min(sweepDueAt, dueAt) : dueAt

  if (sweepTimer && nextDueAt !== sweepDueAt) {
    clearTimeout(sweepTimer)
    sweepTimer = undefined
  }

  sweepDueAt = nextDueAt

  // Exactly one timer at a time: a churn that doesn't move the deadline
  // keeps the existing timer instead of stacking a second one.
  if (!sweepTimer) {
    sweepTimer = setTimeout(
      () => {
        sweepDueAt = 0
        sweepTimer = undefined
        void sweepOnce()
      },
      Math.max(0, nextDueAt - Date.now())
    )
  }
}

/**
 * Sweep after the first sidebar payload, then on later list changes and
 * accepted fresh rosters. A roster arriving after the first pass must wake
 * the sweep even when the sidebar list keeps its reference identity.
 */
export function watchDeadSessionPrune(): () => void {
  const unlistenSessions = $sessions.listen(() => {
    const firstPayload = !listLoaded
    listLoaded = true
    scheduleSweep(firstPayload ? FIRST_PASS_DELAY_MS : RERUN_DELAY_MS)
  })

  const unlistenRoster = $freshProfileRosterRevision.listen(() => {
    const connectionId = currentConnectionId()

    if (listLoaded && connectionId && freshProfileRoster(connectionId)) {
      scheduleSweep(RERUN_DELAY_MS)
    }
  })

  let disposed = false

  return () => {
    if (disposed) {
      return
    }

    disposed = true
    unlistenSessions()
    unlistenRoster()
    // Cancel the pending pass and invalidate any probe already awaiting the backend.
    resetDeadSessionPrune()
  }
}

/**
 * Forget probe results, because the backend they were measured against is
 * gone. `alive` means "alive on the gateway we probed"; a switched gateway has
 * its own state.db and must be re-probed from scratch.
 */
export function resetDeadSessionPrune(): void {
  epoch += 1
  alive.clear()
  inFlight.clear()
  listLoaded = false

  if (sweepTimer) {
    clearTimeout(sweepTimer)
    sweepTimer = undefined
  }

  sweepDueAt = 0
}
