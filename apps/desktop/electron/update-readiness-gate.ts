// Main-process wiring for the update readiness gate (see update-readiness.ts):
// reads every local backend's idle proof and, right before a hand-off, closes
// each backend's work admission so nothing can start between the last idle
// read and the teardown. Kept out of main.ts so the gate is testable with
// fake backends.

import type { ActiveWork } from './quit-guard'
import {
  type BackendIdleAnswer,
  describeUpdateBusy,
  type IdleProofWire,
  mergeUpdateReadiness,
  type UpdateBusyReason,
  type UpdateReadiness,
  type UpdateWhenIdle
} from './update-readiness'

const IDLE_PROBE_TIMEOUT_MS = 5000
// The backend's admission permit lives 30 s; a hand-off still downloading/verifying renews it well inside that.
export const UPDATE_FENCE_RENEW_MS = 10_000
export const UPDATE_WHEN_IDLE_POLL_MS = 5000
// Long enough for a delegated research run to finish, short enough that a
// forgotten wait does not fire hours later into a session the user just began.
export const UPDATE_WHEN_IDLE_MAX_MS = 60 * 60 * 1000

/** The renderer's apply request: no mode = refuse when busy. */
export type UpdateApplyMode = 'check' | 'now' | 'when-idle'

export function updateApplyMode(payload: unknown): UpdateApplyMode {
  const mode = (payload as { mode?: unknown } | null)?.mode

  return mode === 'now' || mode === 'when-idle' ? mode : 'check'
}

// A type alias (not an interface) so it stays assignable to the indexed apply-result wire.
export type BusyApplyResult = {
  ok: false
  error: 'active-work'
  message: string
  waitingFor: UpdateBusyReason[]
}

export function busyApplyResult(reasons: UpdateBusyReason[]): BusyApplyResult {
  return {
    ok: false,
    error: 'active-work',
    message: `Hermes is still working (${describeUpdateBusy(reasons)}).`,
    waitingFor: reasons
  }
}

export interface LocalBackend {
  baseUrl: string
}

export interface UpdateReadinessHost<B extends LocalBackend> {
  activeWork: () => ActiveWork
  /** Local backends this app is connected to right now (never starts one). */
  localBackends: () => Promise<B[]>
  request: (backend: B, path: string, opts: { method?: string; body?: unknown; timeoutMs?: number }) => Promise<unknown>
  log: (line: string) => void
  /** Run `tick` every `ms` until the returned stop is called (tests pass a manual clock). */
  every?: (ms: number, tick: () => void) => () => void
}

function everyInterval(ms: number, tick: () => void): () => void {
  const timer = setInterval(tick, ms)

  return () => clearInterval(timer)
}

/**
 * `ok`: admission is frozen (and its permits renewed) until `release`; call it whenever the hand-off
 * does not happen (cancel, failure). Otherwise `reasons` name the work.
 */
export interface UpdateFreeze {
  ok: boolean
  reasons: UpdateBusyReason[]
  release: () => Promise<void>
}

const noRelease = async (): Promise<void> => undefined

function busyFreeze(reasons: UpdateBusyReason[]): UpdateFreeze {
  return { ok: false, reasons, release: noRelease }
}

interface FencePrepare {
  ok?: boolean
  token?: string
}

// A backend older than the admission fence answers 404/405 there; its idle
// read stays the authority.
function isRouteMissing(error: unknown): boolean {
  const status = (error as { statusCode?: unknown } | null)?.statusCode

  return status === 404 || status === 405
}

export function createUpdateReadinessGate<B extends LocalBackend>(host: UpdateReadinessHost<B>) {
  async function probe(backend: B): Promise<BackendIdleAnswer> {
    try {
      return (await host.request(backend, '/api/health/idle', { timeoutMs: IDLE_PROBE_TIMEOUT_MS })) as IdleProofWire
    } catch (error) {
      host.log(`[updates] idle probe failed for ${backend.baseUrl}; counting it as busy: ${String(error)}`)

      return null
    }
  }

  async function read(): Promise<UpdateReadiness> {
    const backends = await host.localBackends()
    const answers = await Promise.all(backends.map(probe))

    return mergeUpdateReadiness(host.activeWork(), answers)
  }

  async function renewFence(backend: B, token: string): Promise<void> {
    const answer = (await host
      .request(backend, '/api/health/retirement', {
        method: 'POST',
        body: { action: 'renew', token },
        timeoutMs: IDLE_PROBE_TIMEOUT_MS
      })
      .catch((error: unknown) => ({ ok: false, error: String(error) }))) as { ok?: boolean; error?: string }

    if (!answer?.ok) {
      // A backend older than renew, or one that restarted: its 30 s permit is all the hand-off has.
      host.log(`[updates] admission permit for ${backend.baseUrl} could not be renewed: ${answer?.error ?? 'refused'}`)
    }
  }

  async function cancelFence(backend: B, token: string): Promise<void> {
    await host
      .request(backend, '/api/health/retirement', { method: 'POST', body: { action: 'cancel', token } })
      .catch(error => host.log(`[updates] admission fence cancel failed for ${backend.baseUrl}: ${String(error)}`))
  }

  /**
   * The last check before a hand-off. Re-reads readiness, then asks every
   * local backend to freeze admission and prove idle in one step (the
   * backend's retirement fence holds new turns, messaging turns included, cron
   * ticks and delegations; the 30 s permit is renewed every
   * UPDATE_FENCE_RENEW_MS until `release`). Busy anywhere releases the fences already taken and
   * answers the reasons. `release` reopens admission when the hand-off does
   * not happen. A backend that predates the fence route falls back to the
   * idle read alone.
   */
  async function freeze(): Promise<UpdateFreeze> {
    const readiness = await read()

    if (!readiness.ready) {
      return busyFreeze(readiness.reasons)
    }

    const held: Array<[B, string]> = []
    let stopRenewing = () => undefined as void

    const release = async () => {
      stopRenewing()
      await Promise.all(held.splice(0).map(([backend, token]) => cancelFence(backend, token)))
    }

    for (const backend of await host.localBackends()) {
      let answer: FencePrepare

      try {
        answer = (await host.request(backend, '/api/health/retirement', {
          method: 'POST',
          body: { action: 'prepare' },
          timeoutMs: IDLE_PROBE_TIMEOUT_MS
        })) as FencePrepare
      } catch (error) {
        if (isRouteMissing(error)) {
          continue
        }

        host.log(`[updates] admission fence failed for ${backend.baseUrl}; not handing off: ${String(error)}`)
        await release()

        return busyFreeze([{ kind: 'unknown' }])
      }

      if (answer?.ok && typeof answer.token === 'string') {
        held.push([backend, answer.token])

        continue
      }

      await release()
      const again = await read()

      return busyFreeze(again.reasons.length ? again.reasons : [{ kind: 'unknown' }])
    }

    // The renderer may have started a turn while the fences were being taken.
    const work = host.activeWork()

    if (work.count > 0) {
      await release()

      return busyFreeze(mergeUpdateReadiness(work, []).reasons)
    }

    // Held until the hand-off stops these backends (or `release`): a long download must not outlive the permit.
    stopRenewing = (host.every ?? everyInterval)(UPDATE_FENCE_RENEW_MS, () => {
      for (const [backend, token] of held) {
        void renewFence(backend, token)
      }
    })

    return { ok: true, reasons: [], release }
  }

  return { read, freeze }
}

export interface WhenIdleApplyDeps<R> {
  scheduler: UpdateWhenIdle<R>
  read: () => Promise<UpdateReadiness>
  /**
   * The gated apply (no consent): answers a BusyApplyResult when work started in the gap.
   * `stillWanted` is false once the wait was cancelled; check it before tearing anything down.
   */
  applyGated: (stillWanted: () => boolean) => Promise<R | BusyApplyResult>
  emitWaiting: (reasons: UpdateBusyReason[], deadlineAt: number) => void
  /** Every window shows the wait, so every window hears how it ended (not only the one that started it). */
  emitEnded: (ending: UpdateWaitEnding) => void
  log: (line: string) => void
  pollMs?: number
  maxWaitMs?: number
  sleep?: (ms: number) => Promise<void>
  now?: () => number
}

/** Broadcast to every window when a wait ends without a hand-off; `idle` returns the overlay to its idle view. */
export type UpdateWaitEnding =
  { stage: 'idle' } | { stage: 'error'; error: 'idle-timeout'; message: string; waitingFor: UpdateBusyReason[] }

/** The cancel IPC: stop the wait and tell every window at once (the loop itself notices at its next wake). */
export function cancelUpdateWhenIdle(
  scheduler: Pick<UpdateWhenIdle<unknown>, 'cancel'>,
  emitEnded: (ending: UpdateWaitEnding) => void
): { cancelled: boolean } {
  const cancelled = scheduler.cancel()

  if (cancelled) {
    emitEnded({ stage: 'idle' })
  }

  return { cancelled }
}

function isBusyResult(result: unknown): result is BusyApplyResult {
  return (result as { error?: unknown } | null)?.error === 'active-work'
}

/** "Update when idle": wait on the named reasons, then run the gated apply. */
export async function applyUpdatesWhenIdle<R>(
  deps: WhenIdleApplyDeps<R>
): Promise<R | { ok: false; error: 'cancelled' | 'idle-timeout'; message?: string; waitingFor?: UpdateBusyReason[] }> {
  let lastPublished = ''

  deps.log('[updates] update armed: it runs once Hermes is idle')

  const outcome = await deps.scheduler.run({
    read: deps.read,
    fire: async stillWanted => {
      const result = await deps.applyGated(stillWanted)

      return isBusyResult(result) ? { busy: result.waitingFor } : { result }
    },
    publish: (reasons, deadlineAt) => {
      const key = JSON.stringify(reasons)

      if (key !== lastPublished) {
        lastPublished = key
        deps.emitWaiting(reasons, deadlineAt)
      }
    },
    sleep: deps.sleep ?? (ms => new Promise(resolve => setTimeout(resolve, ms))),
    now: deps.now ?? Date.now,
    pollMs: deps.pollMs ?? UPDATE_WHEN_IDLE_POLL_MS,
    maxWaitMs: deps.maxWaitMs ?? UPDATE_WHEN_IDLE_MAX_MS
  })

  if (outcome.kind === 'fired') {
    return outcome.result
  }

  if (outcome.kind === 'cancelled') {
    deps.log('[updates] update-when-idle cancelled')

    return { ok: false, error: 'cancelled' }
  }

  deps.log(`[updates] update-when-idle gave up after its bound: ${describeUpdateBusy(outcome.reasons)}`)

  const timedOut = {
    ok: false as const,
    error: 'idle-timeout' as const,
    message: `Hermes stayed busy (${describeUpdateBusy(outcome.reasons)}). Nothing was updated.`,
    waitingFor: outcome.reasons
  }

  deps.emitEnded({ stage: 'error', error: timedOut.error, message: timedOut.message, waitingFor: timedOut.waitingFor })

  return timedOut
}
