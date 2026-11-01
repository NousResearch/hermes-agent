/**
 * Prune renderer-stored session references whose sessions no longer exist on
 * the backend.
 *
 * Pins (`hermes.desktop.pinnedSessions`), composer drafts
 * (`hermes:composer-drafts:v4`, mirrored to v3), queued prompts
 * (`hermes.desktop.composerQueue.v1`) and interrupted sends
 * (`hermes:composer-outbox:v1:*`) are keyed by session id and re-asserted
 * or probed at boot. When a session is removed from state.db (retention purge,
 * manual delete, another surface pruning it), those keys go stale: nothing
 * ever drops them, so every boot re-requests the dead ids and the gateway
 * answers `404 {"detail":"Session not found"}` once per stale reference,
 * forever.
 *
 * This module sweeps for dead ids after the sidebar list has spoken. A session
 * is declared dead ONLY when a by-id probe 404s on every known profile — a 404
 * is profile-scoped ("not on this profile's state.db"), so a single miss never
 * prunes anything (a session can legitimately live on a non-active profile).
 * Any non-404 probe failure (gateway mid-restart, network) is inconclusive:
 * the id is deferred to a later pass instead of being dropped.
 *
 * Not every known profile can be asked, though. One gateway process serves only
 * its own set of profiles (`gateway_shared_with`); a parked or standalone
 * profile is installed here but the gateway refuses every by-id read for it,
 * with an error whose wording changes between versions. Such a profile is
 * never declared dead on: it is skipped when the served set says so, and when
 * no set is known its refusals are remembered with a growing wait, so a sweep
 * every ~10 s does not log an error against it forever. An id only those
 * profiles could own is `'unreachable'` — not alive, not dead — and is left
 * untouched until one of them becomes askable again.
 */

import { getSession } from '@/hermes'
import { mapPool } from '@/lib/pool'
import { clearSessionDraft, isFreshDraftScope, stashedDraftScopes } from '@/store/composer'
import { dropInterruptedSends, interruptedSendScopes } from '@/store/composer-outbox'
import { $queuedPromptsBySession, clearQueuedPrompts } from '@/store/composer-queue'
import { $pinnedSessionIds, unpinSession } from '@/store/layout'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import { $sessions } from '@/store/session'

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
// First wait before re-asking a profile that refused a by-id read, then the
// growth per further refusal. Capped: a parked profile that never comes back
// is re-checked a few times an hour at most, which is enough to notice it
// being served again without ever becoming per-sweep noise.
const REFUSAL_WAIT_FIRST_MS = 30_000
const REFUSAL_WAIT_MAX_MS = 10 * 60_000

// Ids verified ALIVE this app session, id -> verdict expiry timestamp — later
// sweeps skip re-probing ids whose verdict is still fresh.
const alive = new Map<string, number>()
// Ids currently mid-probe, so overlapping sweeps don't double-probe.
const inFlight = new Set<string>()
// Profiles the gateway refused a by-id read for, key -> when to ask again and
// how long to wait after the next refusal. Keyed by normalized profile name.
const refusals = new Map<string, { retryAt: number; waitMs: number }>()
// Ids no askable profile could rule out ('unreachable'), id -> the profiles it
// is waiting on. The id is not a candidate again until one of them opens up,
// which is what keeps an unserved profile free of per-sweep probe traffic.
const waitingFor = new Map<string, Set<string>>()
// Profiles the gateway serves, or null while unknown (standalone gateway,
// older backend, status not read yet) — then every profile counts as served.
let servedProfiles: null | Set<string> = null

let listLoaded = false
// Bumped by resetDeadSessionPrune: in-flight probes from a pre-reset sweep
// must not write verdicts (alive cache, drops) that belong to the OLD backend.
let epoch = 0
// Debounce deadline, clamped to the EARLIEST change since the last pass: a
// busy list that keeps changing must not starve the sweep indefinitely.
let sweepDueAt = 0
let sweepTimer: ReturnType<typeof setTimeout> | undefined

type Verdict = 'alive' | 'dead' | 'unknown' | 'unreachable'

// What one sweep learned about one id, plus the per-profile bookkeeping the
// caller applies only if the backend it probed is still the live one.
type ProbeOutcome = {
  verdict: Verdict
  /** Profiles that could still own the id but were not asked. */
  waitingOn: string[]
  /** Profiles that refused the read this pass (their wait grows). */
  refused: string[]
  /** Profiles that answered for real, clearing any earlier refusal. */
  answered: string[]
}

function isNotFoundError(error: unknown): boolean {
  return /session not found/i.test(error instanceof Error ? error.message : String(error))
}

/** A profile this sweep may send a by-id read to. The active profile always is:
 * the probe for it is the unscoped one this whole module starts from. */
function profileAskable(key: string, activeKey: string): boolean {
  if (key === activeKey) {
    return true
  }

  const refusal = refusals.get(key)

  if (refusal && Date.now() < refusal.retryAt) {
    return false
  }

  return servedProfiles === null || servedProfiles.has(key)
}

/** Next wait after another refusal from this profile: 30 s, 1 min, 2 min, … 10 min. */
function refusalWaitMs(current: number | undefined): number {
  return current === undefined ? REFUSAL_WAIT_FIRST_MS : Math.min(current * 2, REFUSAL_WAIT_MAX_MS)
}

/** Same contents, not the same object: the status poll hands us a fresh array
 *  every 60 s, and re-learning an unchanged set would clear refusals for
 *  nothing. */
function sameServedSet(left: null | Set<string>, right: null | Set<string>): boolean {
  if (left === right) {
    return true
  }

  if (left === null || right === null) {
    return false
  }

  return left.size === right.size && [...right].every(name => left.has(name))
}

/**
 * The set of profiles the gateway serves (`gateway_shared_with`), or
 * null/undefined while that is not known — then every profile counts as
 * served, which is how the sweep treated them before the set existed.
 *
 * A profile newly on the list gets a clean slate: its earlier refusals were
 * earned while the gateway could not reach it, so it is asked on the next
 * sweep instead of waiting out a backoff from a set that no longer exists.
 */
export function noteServedProfiles(names: readonly string[] | null | undefined): void {
  const next = Array.isArray(names) ? new Set(names.map(name => normalizeProfileKey(name))) : null

  if (sameServedSet(servedProfiles, next)) {
    return
  }

  servedProfiles = next

  // An unknown list serves everything, so nothing stays refused.
  if (next === null) {
    refusals.clear()

    return
  }

  for (const key of [...refusals.keys()]) {
    if (next.has(key)) {
      refusals.delete(key)
    }
  }
}

/**
 * A stored id is dead only if a by-id lookup 404s on the active profile AND on
 * every named profile that can be asked. The active-scope probe covers
 * single-profile installs; named-profile probes rule out sessions living on
 * another profile.
 *
 * Completeness assumption: `$profiles` is the same profile list the UI uses to
 * route and resolve sessions (resolveStoredSession, the picker). A session on
 * a profile the app does not know about is unreachable in the UI — it cannot
 * be opened, pinned, or drafted — so ids stored locally always reference a
 * listed profile. When the list is EMPTY (not yet loaded) nothing is declared
 * dead: another profile we can't rule out might own the id.
 *
 * A profile that cannot be asked (not served, or still inside its refusal
 * wait) leaves the id `'unreachable'` rather than dead: its 404 is missing,
 * not obtained.
 */
async function sessionVerdict(id: string): Promise<ProbeOutcome> {
  const activeKey = normalizeProfileKey($activeGatewayProfile.get())

  // No known profiles → another profile we can't rule out might own the id.
  // Skip the probe entirely; there is nothing to gain from a lone unscoped hit.
  if ($profiles.get().length === 0) {
    return { verdict: 'unknown', waitingOn: [], refused: [], answered: [] }
  }

  try {
    await getSession(id)

    return { verdict: 'alive', waitingOn: [], refused: [], answered: [] }
  } catch (error) {
    // The gateway itself is not answering: nothing is remembered, and the
    // next sweep retries. This is the one failure that must stay unnamed —
    // blaming a single profile for it would park it for no reason.
    if (!isNotFoundError(error)) {
      return { verdict: 'unknown', waitingOn: [], refused: [], answered: [] }
    }
  }

  const knownProfiles = [...new Set($profiles.get().map(profile => normalizeProfileKey(profile.name)))]

  // Without the profile list we can't rule out another profile — defer rather
  // than risk pruning a session that lives elsewhere.
  if (knownProfiles.length === 0) {
    return { verdict: 'unknown', waitingOn: [], refused: [], answered: [] }
  }

  const waitingOn: string[] = []
  const refused: string[] = []
  const answered: string[] = []

  for (const profile of knownProfiles.filter(key => key !== activeKey)) {
    if (!profileAskable(profile, activeKey)) {
      waitingOn.push(profile)

      continue
    }

    try {
      await getSession(id, profile)
      answered.push(profile)

      return { verdict: 'alive', waitingOn: [], refused, answered }
    } catch (error) {
      if (!isNotFoundError(error)) {
        // The active profile answered above, so the gateway is up and this one
        // profile declined the read. Any non-404 wording counts as a refusal:
        // the text differs between versions and must not be matched here.
        refused.push(profile)
        waitingOn.push(profile)

        continue
      }

      answered.push(profile)
    }
  }

  if (waitingOn.length > 0) {
    return { verdict: 'unreachable', waitingOn, refused, answered }
  }

  return { verdict: 'dead', waitingOn: [], refused, answered }
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

  const candidates = new Set<string>()

  for (const id of $pinnedSessionIds.get()) {
    if (!covered.has(id)) {
      candidates.add(id)
    }
  }

  for (const scope of stashedDraftScopes()) {
    if (!covered.has(scope)) {
      candidates.add(scope)
    }
  }

  for (const key of Object.keys($queuedPromptsBySession.get())) {
    if (!covered.has(key)) {
      candidates.add(key)
    }
  }

  // Fresh-chat keys are never session references (they expire with the outbox entry instead).
  for (const scope of interruptedSendScopes()) {
    if (!covered.has(scope) && !isFreshDraftScope(scope)) {
      candidates.add(scope)
    }
  }

  // An 'unreachable' id waits for one of the profiles it could not ask to
  // open up (now served, or past its refusal wait). Probing it earlier would
  // only repeat a question that was already skipped or refused — the exact
  // per-sweep error traffic this module exists to remove.
  const activeKey = normalizeProfileKey($activeGatewayProfile.get())

  return [...candidates].filter(id => {
    const expiresAt = alive.get(id)

    if (expiresAt !== undefined && Date.now() < expiresAt) {
      return false
    }

    if (inFlight.has(id)) {
      return false
    }

    const waitingOn = waitingFor.get(id)

    if (waitingOn && ![...waitingOn].some(profile => profileAskable(profile, activeKey))) {
      return false
    }

    return true
  })
}

function dropDeadId(id: string): void {
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
  }

  // No-ops for surfaces that don't hold the id — the guards also skip the
  // localStorage write a no-op clear would still perform.
  if (stashedDraftScopes().includes(id)) {
    clearSessionDraft(id)
  }

  if ($queuedPromptsBySession.get()[id]?.length) {
    clearQueuedPrompts(id)
  }

  dropInterruptedSends(id)
}

async function sweepOnce(requireLoaded = true): Promise<void> {
  if (requireLoaded && !listLoaded) {
    return
  }

  if (typeof window === 'undefined' || !window.hermesDesktop) {
    return
  }

  const sweepEpoch = epoch
  const candidates = collectCandidates()

  if (candidates.length === 0) {
    return
  }

  for (const id of candidates) {
    inFlight.add(id)
  }

  try {
    await mapPool(candidates, SWEEP_CONCURRENCY, async id => {
      if (epoch !== sweepEpoch) {
        return
      }

      const { verdict, waitingOn, refused, answered } = await sessionVerdict(id)

      // A gateway switch (reset) while this probe was in flight invalidates
      // the verdict — it is about the backend we measured, not the live one.
      if (epoch !== sweepEpoch) {
        return
      }

      for (const profile of answered) {
        refusals.delete(profile)
      }

      for (const profile of refused) {
        const previous = refusals.get(profile)

        // One sweep can refuse the same profile once per stored id (up to
        // SWEEP_CONCURRENCY probes run at once): those are one refusal of the
        // profile, not several. Only a refusal after the previous wait has
        // ended grows it, so an existing record still in the future is kept.
        if (previous && Date.now() < previous.retryAt) {
          continue
        }

        const waitMs = refusalWaitMs(previous?.waitMs)

        refusals.set(profile, { retryAt: Date.now() + waitMs, waitMs })
      }

      if (verdict === 'alive') {
        alive.set(id, Date.now() + ALIVE_TTL_MS)
        waitingFor.delete(id)
      } else if (verdict === 'dead') {
        waitingFor.delete(id)
        dropDeadId(id)
      } else if (verdict === 'unreachable') {
        waitingFor.set(id, new Set(waitingOn))
      } else {
        // 'unknown' — neither cached nor dropped, and not gated on any profile:
        // the next sweep retries the id as a whole.
        waitingFor.delete(id)
      }
    })
  } finally {
    for (const id of candidates) {
      inFlight.delete(id)
    }
  }
}

/** Run one prune pass now. Test-only hook — production never calls it. */
export function __runDeadSessionPrunePass(): Promise<void> {
  // Bypass the list-loaded gate WITHOUT mutating `listLoaded` (a mutation
  // here would leak into the watcher's first-payload timing).
  return sweepOnce(false)
}

/**
 * Sweep after the first sidebar payload, then re-sweep (debounced) whenever
 * the list changes — a purge while the app is running heals within one pass.
 * Call once per app, next to `watchSessionPins`. Returns the store unsubscribe
 * (used by tests).
 */
export function watchDeadSessionPrune(): () => void {
  return $sessions.listen(() => {
    const firstPayload = !listLoaded
    listLoaded = true

    const delay = firstPayload ? FIRST_PASS_DELAY_MS : RERUN_DELAY_MS

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
  })
}

/**
 * Forget probe results, because the backend they were measured against is
 * gone. `alive` means "alive on the gateway we probed"; a switched gateway has
 * its own state.db, its own served set and its own refusals, so every one of
 * them is re-read from scratch.
 */
export function resetDeadSessionPrune(): void {
  epoch += 1
  alive.clear()
  inFlight.clear()
  refusals.clear()
  waitingFor.clear()
  servedProfiles = null
  listLoaded = false

  if (sweepTimer) {
    clearTimeout(sweepTimer)
    sweepTimer = undefined
  }

  sweepDueAt = 0
}
