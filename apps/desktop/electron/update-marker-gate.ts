/**
 * The update gate's marker probe (R6, SPEC section 6 "Electron gate").
 *
 * A marker whose owner and delegate are dead is not necessarily a finished
 * update: an inheriting completion process can still hold the checkout's
 * kernel lock after the script's immediate child died. The Desktop cannot see
 * that lock portably and must never delete the marker (A7 rule 3), so a
 * dead/malformed marker is routed through the checkout script's `reclaim`
 * helper, which decides under the `<marker>.lock` sidecar:
 *
 * - `held` / `busy` / `live <pid>` => an update still owns the checkout: keep waiting;
 * - `reclaimed` / `absent` => nothing runs: proceed;
 * - no marker at all => the helper is asked too (it checks the checkout lock
 *   under the same rule); only its `held` keeps the gate closed, any other
 *   answer proceeds as before;
 * - `unsupported` (older checkout, no helper) => proceed without deleting
 *   (dead = not running, as before minus the deletion) — unless this process
 *   already saw the same body block: then the script is only missing or
 *   unreadable for a moment (git rewriting it mid-update), so the answer counts
 *   as `error` and is re-asked (review R8 M5).
 *
 * The helper is asked once per distinct dead marker body per wait; a `held` /
 * `busy` / `error` / `live` answer is re-asked every `reprobeMs` (5 s), or on
 * the next poll after a Retry, because only the script can see the lock being
 * released.
 *
 * No positive answer has a ceiling (review R8 D3). `held` means no owner
 * identity is alive but some process still holds the checkout lock; it may be
 * a leaked long-lived one, and it may still be writing the install. `busy` /
 * `error` mean the helper could not establish ownership at all. The one
 * bounded case is a helper that has never answered for this body: after
 * HELPER_ERROR_ATTEMPTS consecutive `error`s (bash / PowerShell blocked, a
 * broken script) the gate proceeds as main did for a dead marker, minus the
 * deletion — otherwise such a machine would park every boot after every
 * crashed update. Once the helper has answered for a body, a later `error` is
 * a transient failure of a working helper (R8 M5) and keeps waiting. Nothing
 * else ever opens the gate by itself: past a short grace the caller shows a blocked
 * boot screen (what holds the install, Retry, Quit) whose only way through
 * while the hold lasts is an explicit, confirmed, logged "Start anyway"
 * (`allowStartOverHold`), scoped to the exact marker body the user saw. The
 * marker is never touched. A helper `live <pid>` names a live identity and is
 * waited out like any live marker (C1 rule 3).
 */

import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { type CreateTimeProbe, inspectUpdateMarker, markerPath } from './update-marker'
import type { UpdateMarker } from './update-marker-judge'
import type { MarkerHelperVerdict } from './updater/marker-helper'

export const HELD_REPROBE_MS = 5_000

/**
 * Consecutive `error` answers about a body the helper never answered for
 * before the gate proceeds anyway (~10 s at HELD_REPROBE_MS): a helper that
 * cannot run at all must not park boot behind every dead marker.
 */
export const HELPER_ERROR_ATTEMPTS = 3

/** How long a blocking verdict must last in one wait before the boot shows the blocked screen. */
export const HOLD_SCREEN_GRACE_MS = 5_000

/** Why a dead marker still keeps the gate closed, as the script helper put it. */
export interface HeldState {
  verdict: 'held' | 'busy' | 'live' | 'error'
  /** Line 1 of the marker: the update process that started it (exited), when it parses. */
  ownerPid: number | null
  /** The live process the helper named (`live <pid>`), else null. */
  livePid: number | null
  /** Stable id of the marker body this answer is about; "Start anyway" is scoped to it. */
  holdId: string
  /** Epoch ms this process first saw this marker body keep the gate closed. */
  since: number
  /** Epoch ms of the helper answer this state reflects. */
  checkedAt: number
  /** No live identity owns the install (`held`), or ownership could not be established (`busy`/`error`). */
  blocking: boolean
}

export interface LiveMarkerProbeOptions {
  hermesHome: string
  /** The script helper `reclaim`, or null when the checkout's script predates protocol 2. */
  reclaim: (() => Promise<MarkerHelperVerdict>) | null
  createTime?: CreateTimeProbe
  onLiveMarker?: (marker: { startedAt: number | null; runId: string | null }) => void
  /** Every answer that comes from a running helper verdict (boot progress, the blocked screen). */
  onHeld?: (state: HeldState) => void
  /** The gate let this wait through over a blocking hold because the user chose Start anyway. */
  onOverride?: (holdId: string) => void
  log?: (line: string) => void
  now?: () => number
  reprobeMs?: number
}

const STILL_RUNNING = new Set(['held', 'busy', 'live', 'error'])

/**
 * The hold id of "no marker, but the helper says the checkout lock is held"
 * (an update that holds the lock before publishing, or after retiring, its
 * marker). There is no body to scope it to, so one id stands for that state.
 */
const ABSENT_HOLD_ID = markerHoldId(Buffer.from('\0no update marker\0'))

// First sighting of each held body, process-wide: a later gate wait (a pool
// backend, a reconnect) reports the same "since".
const firstHeldAt = new Map<string, number>()

function heldSince(key: string, at: number): number {
  if (!firstHeldAt.has(key)) {
    if (firstHeldAt.size >= 16) {
      firstHeldAt.clear()
    }

    firstHeldAt.set(key, at)
  }

  return firstHeldAt.get(key)!
}

// Marker bodies the user explicitly chose to start over (R8 D3), process-wide
// so a pool backend wait honours the same decision. A different body (a new
// update, a new owner) blocks again.
const startAnywayHolds = new Set<string>()

// Bodies whose helper never answered within HELPER_ERROR_ATTEMPTS, process-wide
// so a later wait (a pool backend) proceeds without re-running the retries.
const unanswerableHolds = new Set<string>()

// Bumped by an explicit Retry: every probe re-asks the helper on its next poll.
let recheckGeneration = 0

/**
 * The stable id of one marker file's body (what `HeldState.holdId` carries).
 * The file's inode and mtime are mixed in so two byte-identical bodies — every
 * empty marker, say — written by different updates are different holds
 * (review R8 m7).
 */
export function markerHoldId(raw: Buffer, file?: { ino: number; mtimeMs: number } | null): string {
  const hash = createHash('sha256').update(raw)

  if (file) {
    hash.update(`\0${file.ino}:${file.mtimeMs}`)
  }

  return hash.digest('hex').slice(0, 16)
}

/**
 * The blocked screen's holds, one per blocked wait: the primary boot wait
 * (`PRIMARY_HOLD_OWNER`) and each pool/profile backend wait. The primary's
 * own hold wins; otherwise the first blocked pool wait is shown, so a hold
 * only a profile backend meets (a remote primary, or a hold that appeared
 * after boot) still gets a screen with its ways out (review R8 M6).
 */
export const PRIMARY_HOLD_OWNER = 'primary'

export class UpdateHoldBoard {
  private readonly holds = new Map<string, HeldState>()

  set(owner: string, state: HeldState): void {
    this.holds.set(owner, state)
  }

  clear(owner: string): void {
    this.holds.delete(owner)
  }

  shown(): HeldState | null {
    return this.holds.get(PRIMARY_HOLD_OWNER) ?? this.holds.values().next().value ?? null
  }
}

/**
 * One wait's hold state machine, shared by the boot and pool/profile waits:
 * `onHeld` keeps the probe's latest held state, and `tick` (once per wait
 * poll) consumes it, shows it once a blocking hold has lasted
 * HOLD_SCREEN_GRACE_MS without a break, and clears this wait's hold
 * otherwise. Never a timeout: past the grace the wait stays parked behind
 * the blocked screen (R8 D3).
 */
export function holdTicker(screen: { show: (state: HeldState) => void; clear: () => void; now?: () => number }) {
  const now = screen.now ?? Date.now
  let held: HeldState | null = null
  let blockedSince: number | null = null

  return {
    onHeld(state: HeldState) {
      held = state
    },
    tick(reason: string | null): { held: HeldState | null; shown: boolean } {
      const seen = held
      held = null
      blockedSince = reason === 'marker' && seen?.blocking ? (blockedSince ?? now()) : null
      const shown = seen !== null && blockedSince !== null && now() - blockedSince >= HOLD_SCREEN_GRACE_MS

      if (shown) {
        screen.show(seen)
      } else {
        screen.clear()
      }

      return { held: seen, shown }
    }
  }
}

/**
 * The user confirmed "Start anyway" over this exact held marker body. The
 * caller logs the override; the marker is left in place.
 */
export function allowStartOverHold(holdId: string): void {
  startAnywayHolds.add(holdId)
}

/** Retry: ask the script helper again on the next poll instead of waiting out the re-probe interval. */
export function requestHoldRecheck(): void {
  recheckGeneration += 1
}

/** Test seam: forget process-wide hold state. */
export function resetHoldStateForTests(): void {
  firstHeldAt.clear()
  startAnywayHolds.clear()
  unanswerableHolds.clear()
  recheckGeneration = 0
}

interface AskedEntry {
  verdict: MarkerHelperVerdict
  at: number
  generation: number
  overrideLogged?: boolean
  /** Consecutive `error` answers. */
  errors?: number
  /** The helper gave a real verdict (anything but `error`) for this body at least once. */
  answered?: boolean
}

function statMarkerFile(hermesHome: string): fs.Stats | null {
  try {
    return fs.statSync(markerPath(hermesHome))
  } catch {
    // Gone since the read: the body alone names it until the next poll.
    return null
  }
}

function verdictDue(previous: AskedEntry | undefined, now: () => number, reprobeMs: number): boolean {
  return (
    !previous ||
    (STILL_RUNNING.has(previous.verdict.kind) &&
      (now() - previous.at >= reprobeMs || previous.generation !== recheckGeneration))
  )
}

/** Ask the script helper again about one dead marker body and record the answer. */
async function refreshVerdict(
  previous: AskedEntry | undefined,
  holdId: string,
  reclaim: () => Promise<MarkerHelperVerdict>,
  now: () => number,
  log: ((line: string) => void) | undefined,
  subject = 'dead update marker'
): Promise<AskedEntry> {
  const generation = recheckGeneration
  let verdict = await reclaim()

  // A body that blocked never clears on `unsupported`: the checkout's
  // script went missing or unreadable after answering for it (R8 M5).
  if (verdict.kind === 'unsupported' && (STILL_RUNNING.has(previous?.verdict.kind ?? '') || firstHeldAt.has(holdId))) {
    verdict = { kind: 'error' }
  }

  if (!previous || STILL_RUNNING.has(previous.verdict.kind) !== STILL_RUNNING.has(verdict.kind)) {
    log?.(
      `[updates] ${subject}: script helper says ${verdict.kind}${'pid' in verdict ? ` ${verdict.pid}` : ''}`
    )
  }

  const errors = verdict.kind === 'error' ? (previous?.errors ?? 0) + 1 : 0

  return { ...previous, verdict, at: now(), generation, errors, answered: previous?.answered || verdict.kind !== 'error' }
}

/** The helper never answered for this body and has used its retries: proceed as main did for a dead marker. */
function helperUnanswerable(entry: AskedEntry, holdId: string, log: ((line: string) => void) | undefined): boolean {
  if (entry.verdict.kind !== 'error' || entry.answered || (entry.errors ?? 0) < HELPER_ERROR_ATTEMPTS) {
    return false
  }

  unanswerableHolds.add(holdId)
  log?.(
    `[updates] dead update marker (hold ${holdId}): the script helper failed ${entry.errors} times and never ` +
      'answered; starting as for any dead marker. The marker is left in place.'
  )

  return true
}

function logOverrideOnce(entry: AskedEntry, state: HeldState, log: ((line: string) => void) | undefined): void {
  if (entry.overrideLogged) {
    return
  }

  entry.overrideLogged = true
  log?.(
    `[updates] update marker still ${state.verdict} (hold ${state.holdId}); not blocking start-up because the user ` +
      'chose Start anyway. The marker is left in place.'
  )
}

/**
 * A v1 marker (no `ct:`, no delegate) reads live on a bare alive pid for up to
 * V1_MAX_AGE_S, so a recycled pid (Windows, #122206) parks boot ~20 min. A
 * failed receipt that finished after the marker started says that update is
 * over. Only v1: a v2 owner's creation time already tells a reused pid apart,
 * and a retry's marker starts after the old receipt, so a retry still waits
 * (V2: `latest.json` keeps reading `failed` until the retry finalizes).
 */
async function failedAfterV1Start(hermesHome: string, marker: UpdateMarker | null): Promise<boolean> {
  if (!marker || marker.ct !== null || marker.delegate !== null) {
    return false
  }

  try {
    const text = await fs.promises.readFile(path.join(hermesHome, 'logs', 'update_receipts', 'latest.json'), 'utf8')
    const receipt = JSON.parse(text.replace(/^\uFEFF/, ''))
    const finishedS = Date.parse(receipt?.finished_at) / 1000

    // Line 2 is whole seconds: a receipt from the marker's own start second is ambiguous, so it waits.
    return receipt?.outcome === 'failed' && finishedS >= marker.startedAt + 1
  } catch {
    return false
  }
}

/** `hasLiveMarker` for one gate wait (create it per wait, never module-wide). */
export function liveMarkerProbe({
  hermesHome,
  reclaim,
  createTime,
  onLiveMarker,
  onHeld,
  onOverride,
  log,
  now = Date.now,
  reprobeMs = HELD_REPROBE_MS
}: LiveMarkerProbeOptions): () => Promise<boolean> {
  const asked = new Map<string, AskedEntry>()
  let finishedLogged = false

  return async () => {
    const inspection = await inspectUpdateMarker(hermesHome, { createTime, now })

    if (inspection.state === 'live') {
      if (await failedAfterV1Start(hermesHome, inspection.marker)) {
        if (!finishedLogged) {
          finishedLogged = true
          log?.('[updates] latest update receipt records a failure after this v1 marker started; not parking the boot on it')
        }

        return false
      }

      onLiveMarker?.({ startedAt: inspection.marker?.startedAt ?? null, runId: inspection.marker?.run ?? null })

      return true
    }

    const absent = inspection.state === 'absent'

    if ((inspection.state !== 'dead' && !absent) || !reclaim) {
      return false
    }

    const holdId = absent ? ABSENT_HOLD_ID : markerHoldId(inspection.raw, statMarkerFile(hermesHome))

    if (unanswerableHolds.has(holdId)) {
      return false
    }

    const previous = asked.get(holdId)
    let entry = previous!

    if (verdictDue(previous, now, reprobeMs)) {
      entry = await refreshVerdict(previous, holdId, reclaim, now, log, absent ? 'no update marker' : undefined)
      asked.set(holdId, entry)
    }

    const { verdict } = entry

    // No marker: only the helper's positive `held` (a process holds the
    // checkout lock) closes the gate; a helper that cannot answer keeps the
    // old "no marker, nothing runs" so a broken script never blocks boot.
    if (!STILL_RUNNING.has(verdict.kind) || (absent && verdict.kind !== 'held') || helperUnanswerable(entry, holdId, log)) {
      return false
    }

    const state: HeldState = {
      verdict: verdict.kind as HeldState['verdict'],
      ownerPid: absent ? null : (inspection.marker?.pid ?? null),
      livePid: 'pid' in verdict ? verdict.pid : null,
      holdId,
      since: heldSince(holdId, now()),
      checkedAt: entry.at,
      // An `error` still inside its retries is not a hold yet: no blocked screen.
      blocking: verdict.kind !== 'live' && (verdict.kind !== 'error' || Boolean(entry.answered))
    }

    if (state.blocking && startAnywayHolds.has(holdId)) {
      logOverrideOnce(entry, state, log)
      onOverride?.(holdId)

      return false
    }

    onHeld?.(state)
    onLiveMarker?.(absent ? { startedAt: null, runId: null } : { startedAt: inspection.marker?.startedAt ?? null, runId: inspection.marker?.run ?? null })

    return true
  }
}

/** Boot-progress text while a dead marker's checkout is still held. */
export function heldWaitMessage(state: HeldState): string {
  if (state.verdict === 'busy' || state.verdict === 'error') {
    return 'Hermes could not verify update ownership yet — startup is paused while it retries. Details are in logs/update.log.'
  }

  if (state.livePid !== null) {
    return `An update is still finishing (process ${state.livePid}) — Hermes will start automatically when it completes…`
  }

  const who = state.ownerPid ? `the update (process ${state.ownerPid}) exited, but a process` : 'a process'

  return `An update is still finishing: ${who} it started still holds the Hermes install. Hermes will start when it lets go.`
}

/** The log line for a confirmed "Start anyway" (R8 D3). */
export function startAnywayLogLine(state: HeldState): string {
  return (
    `[updates] USER OVERRIDE: Start anyway over an update marker the helper reports ${state.verdict}` +
    `${state.ownerPid ? ` (update pid ${state.ownerPid}, exited)` : ''}, hold ${state.holdId}, ` +
    `blocking since ${new Date(state.since).toISOString()}; starting the local backend without waiting. ` +
    'The marker is left in place.'
  )
}
