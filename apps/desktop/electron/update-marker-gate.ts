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
 * - `unsupported` (older checkout, no helper) => proceed without deleting
 *   (dead = not running, as before minus the deletion).
 *
 * The helper is asked once per distinct dead marker body per wait; a `held` /
 * `busy` / `live` answer is re-asked every `reprobeMs` (5 s) because only the
 * script can see the lock being released.
 *
 * Only verified `held` has a ceiling (review R6 m7): no owner identity is alive,
 * only some process still holding the checkout lock, and that can be a
 * leaked long-lived one. The scripts stop waiting on it after RELEASE_WAIT_S
 * (7200 s); the gate stops blocking after the same span, counted from the
 * marker's line 2 (the last time an owner wrote it; the scripts keep it young
 * while they live) or, when line 2 is unreadable, from the first time this
 * process saw the body held. Past the ceiling the probe answers "not
 * running", logs it once and reports it through `onHeld` so the caller can
 * tell the user; the marker is left exactly as it is. A helper `live <pid>`
 * names a live identity and is waited out like any live marker (C1 rule 3).
 * `busy`/`error` are indeterminate, retried without granting clearance.
 */

import { type CreateTimeProbe, inspectUpdateMarker } from './update-marker'
import type { MarkerHelperVerdict } from './updater/marker-helper'

export const HELD_REPROBE_MS = 5_000

/** The scripts' own wait on the checkout lock (posix.sh RELEASE_WAIT_S, marker.ps1 MarkerReleaseWaitSeconds). */
export const HELD_CEILING_MS = 7_200_000

/** Why a dead marker still keeps the gate closed, as the script helper put it. */
export interface HeldState {
  verdict: 'held' | 'busy' | 'live' | 'error'
  /** Line 1 of the marker: the update process that started it (exited), when it parses. */
  ownerPid: number | null
  /** The live process the helper named (`live <pid>`), else null. */
  livePid: number | null
  /** Epoch ms the ceiling counts from. */
  since: number
  /** Ms until the gate stops blocking; null when there is no ceiling (`live`). */
  remainingMs: number | null
  /** The ceiling passed: the gate is open and the marker untouched. */
  expired: boolean
}

export interface LiveMarkerProbeOptions {
  hermesHome: string
  /** The script helper `reclaim`, or null when the checkout's script predates protocol 2. */
  reclaim: (() => Promise<MarkerHelperVerdict>) | null
  createTime?: CreateTimeProbe
  onLiveMarker?: (marker: { startedAt: number | null }) => void
  /** Every answer that comes from a running helper verdict (boot progress, the ceiling notice). */
  onHeld?: (state: HeldState) => void
  log?: (line: string) => void
  now?: () => number
  reprobeMs?: number
  /** Test seam; production is HELD_CEILING_MS. */
  heldCeilingMs?: number
}

const STILL_RUNNING = new Set(['held', 'busy', 'live', 'error'])

// First sighting of each held body, process-wide: a later gate wait (a pool
// backend, a reconnect) continues the same ceiling instead of restarting it.
const firstHeldAt = new Map<string, number>()

function heldSince(key: string, startedAt: number | null, at: number): number {
  if (!firstHeldAt.has(key)) {
    if (firstHeldAt.size >= 16) {
      firstHeldAt.clear()
    }

    firstHeldAt.set(key, at)
  }

  const written = startedAt !== null && Number.isFinite(startedAt) && startedAt > 0 ? startedAt * 1000 : Infinity

  return Math.min(firstHeldAt.get(key)!, written)
}

/** `hasLiveMarker` for one gate wait (create it per wait, never module-wide). */
export function liveMarkerProbe({
  hermesHome,
  reclaim,
  createTime,
  onLiveMarker,
  onHeld,
  log,
  now = Date.now,
  reprobeMs = HELD_REPROBE_MS,
  heldCeilingMs = HELD_CEILING_MS
}: LiveMarkerProbeOptions): () => Promise<boolean> {
  const asked = new Map<string, { verdict: MarkerHelperVerdict; at: number; expiryLogged?: boolean }>()

  return async () => {
    const inspection = await inspectUpdateMarker(hermesHome, { createTime, now })

    if (inspection.state === 'live') {
      onLiveMarker?.({ startedAt: inspection.marker?.startedAt ?? null })

      return true
    }

    if (inspection.state !== 'dead' || !reclaim) {
      return false
    }

    const key = inspection.raw.toString('hex')
    const previous = asked.get(key)
    let entry = previous

    if (!previous || (STILL_RUNNING.has(previous.verdict.kind) && now() - previous.at >= reprobeMs)) {
      const verdict = await reclaim()

      if (!previous || STILL_RUNNING.has(previous.verdict.kind) !== STILL_RUNNING.has(verdict.kind)) {
        log?.(`[updates] dead update marker: script helper says ${verdict.kind}${'pid' in verdict ? ` ${verdict.pid}` : ''}`)
      }

      entry = { ...previous, verdict, at: now() }
      asked.set(key, entry)
    }

    const { verdict } = entry!

    if (!STILL_RUNNING.has(verdict.kind)) {
      return false
    }

    const startedAt = inspection.marker?.startedAt ?? null
    const since = heldSince(key, startedAt, now())
    // `busy`/`error` did not establish ownership: stale bytes cannot grant clearance.
    const remainingMs = verdict.kind === 'held' ? Math.max(0, since + heldCeilingMs - now()) : null

    const state: HeldState = {
      verdict: verdict.kind as HeldState['verdict'],
      ownerPid: inspection.marker?.pid ?? null,
      livePid: 'pid' in verdict ? verdict.pid : null,
      since,
      remainingMs,
      expired: remainingMs === 0
    }

    onHeld?.(state)

    if (state.expired) {
      if (!entry!.expiryLogged) {
        entry!.expiryLogged = true
        log?.(
          `[updates] update marker still ${state.verdict} ${Math.round((now() - since) / 60_000)} min after its owner` +
            `${state.ownerPid ? ` (pid ${state.ownerPid})` : ''} last wrote it; no longer blocking start-up` +
            ' (the scripts stop waiting at the same ceiling). The marker is left in place.'
        )
      }

      return false
    }

    onLiveMarker?.({ startedAt })

    return true
  }
}

function duration(ms: number): string {
  const m = Math.max(1, Math.ceil(ms / 60_000))

  return m >= 120 ? `${Math.round(m / 60)} hours` : m === 1 ? '1 minute' : `${m} minutes`
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

  return (
    `An update is still finishing: ${who} it started still holds the Hermes install. ` +
    `Hermes will start when that process exits, or in ${duration(state.remainingMs ?? 0)} at the latest.`
  )
}

/** Dialog detail once the ceiling let Hermes start over a still-held checkout. */
export function heldCeilingNotice(state: HeldState): string {
  const owner = state.ownerPid ? ` (process ${state.ownerPid}, now exited)` : ''

  return (
    `An earlier update${owner} left a process that still held the Hermes install ${duration(HELD_CEILING_MS)} ` +
    'later, so Hermes started without waiting for it. The update marker was left in place.\n\n' +
    'If something looks wrong: quit Hermes, end leftover git or hermes processes (or restart the computer), ' +
    'then run the update again from Settings or with `hermes update`. Details are in logs/update.log.'
  )
}
