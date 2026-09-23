import type { HealthEvidence, ScopeCapture, ScopeEvidence } from '../src/lib/managed-rollout-contract'
import { validateHealthEvidence } from '../src/lib/managed-rollout-contract'
import { MAX_EFFECTIVE_WAVE_SIZE, MAX_SWEEP_PROBES } from '../src/lib/managed-rollout-waves'

import { probeRemoteHermesHome, readRemoteInstallId } from './remote-lifecycle'

export const MAX_PROBE_CONCURRENCY = 8
export const PROBE_DEADLINE_MS = 10_000
export const SWEEP_DEADLINE_MS = 5 * 60 * 1000
export const SWEEP_FRESHNESS_MS = 10 * 1000

export interface HealthEvidenceInput extends Omit<HealthEvidence, 'scopeCapture' | 'scopes'> {
  scopes: readonly ScopeEvidence[] | null
  scopeCapture?: ScopeCapture
}

/**
 * Build the one normalized health shape shared by preflight and progression.
 * `null` is deliberately retained as a missing capture marker; it is never
 * treated as a known-empty scope list by policy.
 */
export function buildHealthEvidence(input: HealthEvidenceInput): HealthEvidence {
  const scopeCapture: ScopeCapture = input.scopes === null ? 'missing' : (input.scopeCapture ?? 'complete')
  const scopes = input.scopes === null ? [] : input.scopes.slice()

  if (scopeCapture === 'missing' && scopes.length !== 0) {
    throw new Error('missing-scope-capture-has-scopes')
  }

  return validateHealthEvidence({ ...input, scopeCapture, scopes })
}

export interface RemoteLifecycleIdentity {
  hermesHome: string
  installId: string | null
}

/**
 * Compose the existing lifecycle readers instead of adding another remote
 * install/home probe. The injected executor is read-only SSH transport owned by
 * the caller; this helper never invokes a mutating command.
 */
export async function readRemoteLifecycleIdentity(input: {
  exec: (command: string) => Promise<string>
}): Promise<RemoteLifecycleIdentity> {
  const ssh = {
    exec: async (command: string): Promise<string> => String(await input.exec(command))
  }

  const hermesHome = await probeRemoteHermesHome(ssh)
  const installId = await readRemoteInstallId(ssh)

  return { hermesHome, installId: installId ?? null }
}

export interface SweepTarget {
  installId: string
  requiredScopeIds: readonly string[] | null
  wave: number
  excluded: boolean
}

export interface SweepObservation {
  installId: string
  wave: number
  health: HealthEvidence
}

export interface SweepProbeContext {
  epochId: string
  deadlineMono: number
  signal: AbortSignal
}

export type SweepProbe = (target: SweepTarget, context: SweepProbeContext) => Promise<{ health: HealthEvidence }>

export interface EvidenceSweepOptions {
  epochId: string
  nowMono?: () => number
  maxConcurrency?: number
  deadlineMs?: number
  freshnessMs?: number
}

export interface EvidenceSweepMetrics {
  requestedProbes: number
  completedProbes: number
  timedOutProbes: number
  retryCount: number
  queueDelayMs: number
}

export interface EvidenceSweepResult {
  ok: boolean
  epochId: string
  startedMono: number
  finishedMono: number
  observations: SweepObservation[]
  errors: Array<{ installId: string; reason: string }>
  completeScope: boolean
  fresh: boolean
  nextAdmissionInstallIds: string[]
  metrics: EvidenceSweepMetrics
}

function boundedReason(error: unknown): string {
  const reason = error instanceof Error ? error.message : String(error)

  // eslint-disable-next-line no-control-regex -- sanitize probe diagnostics before IPC
  return reason.replace(/[\x00\r\n]+/g, ' ').slice(0, 256) || 'probe-failed'
}

function unique(values: readonly string[]): boolean {
  return new Set(values).size === values.length
}

function hasRequiredScope(target: SweepTarget, health: HealthEvidence): boolean {
  if (target.requiredScopeIds === null || health.scopeCapture !== 'complete') {return false}
  const observed = health.scopes.map(scope => scope.scopeId)

  return (
    unique(target.requiredScopeIds) &&
    unique(observed) &&
    target.requiredScopeIds.length === observed.length &&
    target.requiredScopeIds.every(scopeId => observed.includes(scopeId))
  )
}

function targetIsComplete(target: SweepTarget, observation: SweepObservation | undefined): boolean {
  if (target.excluded) {return true}

  if (!observation) {return false}

  return (
    observation.health.observationId !== '' &&
    observation.health.installId === target.installId &&
    hasRequiredScope(target, observation.health)
  )
}

// The lease belongs to the actual probe promise, not the timeout race. A
// transport that ignores cancellation keeps its slot until it really settles;
// later sweeps then fail closed instead of creating a ninth live probe.
let liveProbeSlots = 0
const waitingProbeSlots: Array<{ grant: () => void; cancel: () => void }> = []

function acquireProbeSlot(signal: AbortSignal): Promise<() => void> {
  if (signal.aborted) {return Promise.reject(new Error('probe-slot-cancelled'))}

  return new Promise((resolve, reject) => {
    let released = false

    const release = () => {
      if (released) {return}
      released = true
      liveProbeSlots -= 1
      waitingProbeSlots.shift()?.grant()
    }

    const grant = () => {
      if (signal.aborted) {
        reject(new Error('probe-slot-cancelled'))
        waitingProbeSlots.shift()?.grant()

        return
      }
      signal.removeEventListener('abort', cancel)
      liveProbeSlots += 1
      resolve(release)
    }

    const cancel = () => {
      const index = waitingProbeSlots.findIndex(waiter => waiter.grant === grant)

      if (index >= 0) {waitingProbeSlots.splice(index, 1)}
      reject(new Error('probe-slot-cancelled'))
    }

    if (liveProbeSlots < MAX_PROBE_CONCURRENCY) {grant()}
    else {
      waitingProbeSlots.push({ grant, cancel })
      signal.addEventListener('abort', cancel, { once: true })
    }
  })
}

async function runBoundedProbe(
  probe: SweepProbe,
  target: SweepTarget,
  context: Omit<SweepProbeContext, 'signal'>,
  remainingMs: number,
  sweepSignal: AbortSignal
): Promise<{ health: HealthEvidence }> {
  if (remainingMs <= 0) {throw new Error('probe-timeout')}

  const controller = new AbortController()
  let timeout: ReturnType<typeof setTimeout> | undefined

  const timeoutPromise = new Promise<never>((_, reject) => {
    timeout = setTimeout(() => {
      controller.abort()
      reject(new Error('probe-timeout'))
    }, Math.max(1, remainingMs))
  })
  const onSweepAbort = () => controller.abort()
  sweepSignal.addEventListener('abort', onSweepAbort, { once: true })

  try {
    if (sweepSignal.aborted) {controller.abort()}
    const permit = acquireProbeSlot(controller.signal).then(release => {
      if (controller.signal.aborted) {
        release()
        throw new Error('probe-slot-cancelled')
      }

      return release
    })
    const release = await Promise.race([permit, timeoutPromise])
    const actualProbe = Promise.resolve().then(() => {
      if (controller.signal.aborted) {throw new Error('probe-timeout')}

      return probe(target, { ...context, signal: controller.signal })
    })
    void actualProbe.then(release, release)

    return await Promise.race([actualProbe, timeoutPromise])
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error(sweepSignal.aborted ? 'sweep-deadline-exceeded' : 'probe-timeout')
    }

    throw error
  } finally {
    if (timeout) {clearTimeout(timeout)}
    sweepSignal.removeEventListener('abort', onSweepAbort)
  }
}

function validateTargets(targets: readonly SweepTarget[]): void {
  const ids = targets.map(target => target.installId)

  if (!ids.length || ids.length > 500 || !unique(ids)) {throw new Error('invalid-sweep-targets')}

  if (targets.some(target => !target.installId || !Number.isSafeInteger(target.wave) || target.wave < 0)) {
    throw new Error('invalid-sweep-targets')
  }

  if (targets.some(target => target.requiredScopeIds !== null && !unique(target.requiredScopeIds))) {
    throw new Error('invalid-sweep-targets')
  }

  const included = targets.filter(target => !target.excluded)
  const waveCounts = new Map<number, number>()

  for (const target of included) {waveCounts.set(target.wave, (waveCounts.get(target.wave) ?? 0) + 1)}

  if (
    included.length > MAX_SWEEP_PROBES || waveCounts.size > 2 ||
    [...waveCounts.values()].some(count => count > MAX_EFFECTIVE_WAVE_SIZE)
  ) {throw new Error('sweep-budget-exceeded')}
}

/**
 * Run a bounded, complete-scope evidence epoch. Probe completion order never
 * changes the returned observation order, and a partial/old epoch cannot be
 * promoted by relabeling cached observations.
 */
export async function runEvidenceSweep(
  targets: readonly SweepTarget[],
  probe: SweepProbe,
  options: EvidenceSweepOptions
): Promise<EvidenceSweepResult> {
  validateTargets(targets)

  const nowMono = options.nowMono ?? (() => Number(process.hrtime.bigint() / 1_000_000n))
  const startedMono = nowMono()
  const deadlineMs = Math.min(Math.max(1, options.deadlineMs ?? SWEEP_DEADLINE_MS), SWEEP_DEADLINE_MS)
  const requestedConcurrency = options.maxConcurrency ?? MAX_PROBE_CONCURRENCY
  const concurrency = Math.min(Math.max(1, requestedConcurrency), MAX_PROBE_CONCURRENCY)
  const observationsByIndex = new Map<number, SweepObservation>()
  const errors: Array<{ installId: string; reason: string }> = []
  const requestedProbes = targets.filter(target => !target.excluded).length
  let firstProbeStartedMono: number | undefined
  let cursor = 0
  let closed = false
  const sweepController = new AbortController()

  const worker = async (): Promise<void> => {
    for (;;) {
      if (closed) {return}
      const index = cursor
      cursor += 1

      if (index >= targets.length) {return}

      const target = targets[index]

      if (target.excluded) {continue}

      if (nowMono() - startedMono >= deadlineMs) {
        errors.push({ installId: target.installId, reason: 'sweep-deadline-exceeded' })

        continue
      }

      try {
        const now = nowMono()
        firstProbeStartedMono ??= now
        const sweepDeadlineMono = startedMono + deadlineMs
        const probeDeadlineMono = Math.min(sweepDeadlineMono, now + PROBE_DEADLINE_MS)

        const result = await runBoundedProbe(
          probe,
          target,
          { epochId: options.epochId, deadlineMono: probeDeadlineMono },
          probeDeadlineMono - now,
          sweepController.signal
        )

        if (closed) {return}

        const health = validateHealthEvidence(result.health)

        if (health.observationId !== options.epochId) {
          errors.push({ installId: target.installId, reason: 'stale-observation-epoch' })

          continue
        }

        observationsByIndex.set(index, { installId: target.installId, wave: target.wave, health })
      } catch (error) {
        if (closed) {return}
        errors.push({ installId: target.installId, reason: boundedReason(error) })
      }
    }
  }

  const workers = Promise.all(Array.from({ length: concurrency }, () => worker()))
  let deadlineTimer: ReturnType<typeof setTimeout> | undefined
  const deadline = new Promise<void>(resolve => {
    deadlineTimer = setTimeout(() => {
      closed = true
      sweepController.abort()
      resolve()
    }, deadlineMs)
  })
  await Promise.race([workers, deadline])
  if (deadlineTimer) {clearTimeout(deadlineTimer)}
  const expired = closed
  closed = true

  if (expired) {
    const known = new Set([...errors.map(error => error.installId), ...[...observationsByIndex.values()].map(row => row.installId)])

    for (const target of targets) {
      if (!target.excluded && !known.has(target.installId)) {
        errors.push({ installId: target.installId, reason: 'sweep-deadline-exceeded' })
      }
    }
  }

  const observations = [...observationsByIndex.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, observation]) => observation)

  const observationById = new Map(observations.map(observation => [observation.installId, observation]))
  const completeScope = targets.every(target => targetIsComplete(target, observationById.get(target.installId)))
  const finishedMono = nowMono()

  const fresh =
    finishedMono >= startedMono &&
    finishedMono - startedMono <= deadlineMs &&
    !expired &&
    errors.length === 0 &&
    observations.every(observation => observation.health.observationId === options.epochId)

  const firstWave = Math.min(...targets.filter(target => !target.excluded).map(target => target.wave))
  const nextAdmissionInstallIds = targets
    .filter(target => !target.excluded && target.wave === firstWave + 1)
    .sort((left, right) => left.wave - right.wave || left.installId.localeCompare(right.installId))
    .map(target => target.installId)

  const timedOutProbes = errors.filter(
    error => error.reason === 'probe-timeout' || error.reason === 'sweep-deadline-exceeded'
  ).length

  return {
    ok: completeScope && fresh,
    epochId: options.epochId,
    startedMono,
    finishedMono,
    observations,
    errors: errors.sort((left, right) => left.installId.localeCompare(right.installId)),
    completeScope,
    fresh,
    nextAdmissionInstallIds,
    metrics: {
      requestedProbes,
      completedProbes: observations.length,
      timedOutProbes,
      retryCount: 0,
      queueDelayMs: firstProbeStartedMono === undefined ? 0 : Math.max(0, firstProbeStartedMono - startedMono)
    }
  }
}

export function isFreshSweep(
  result: Pick<EvidenceSweepResult, 'ok' | 'epochId' | 'finishedMono'>,
  nowMono: number,
  freshnessMs = SWEEP_FRESHNESS_MS
): boolean {
  const boundedFreshnessMs = Math.min(Math.max(0, freshnessMs), SWEEP_FRESHNESS_MS)

  return (
    result.ok &&
    result.epochId.length > 0 &&
    Number.isFinite(result.finishedMono) &&
    Number.isFinite(nowMono) &&
    nowMono >= result.finishedMono &&
    nowMono - result.finishedMono <= boundedFreshnessMs
  )
}
