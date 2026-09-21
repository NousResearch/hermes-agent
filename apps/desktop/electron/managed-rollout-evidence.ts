import type { HealthEvidence, ScopeEvidence, ScopeCapture } from '../src/lib/managed-rollout-contract'
import { validateHealthEvidence } from '../src/lib/managed-rollout-contract'
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
}

export type SweepProbe = (target: SweepTarget, context: SweepProbeContext) => Promise<{ health: HealthEvidence }>

export interface EvidenceSweepOptions {
  epochId: string
  nowMono?: () => number
  maxConcurrency?: number
  deadlineMs?: number
  freshnessMs?: number
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
}

function boundedReason(error: unknown): string {
  const reason = error instanceof Error ? error.message : String(error)
  return reason.replace(/[\x00\r\n]+/g, ' ').slice(0, 256) || 'probe-failed'
}

function unique(values: readonly string[]): boolean {
  return new Set(values).size === values.length
}

function hasRequiredScope(target: SweepTarget, health: HealthEvidence): boolean {
  if (target.requiredScopeIds === null || health.scopeCapture !== 'complete') return false
  const observed = health.scopes.map(scope => scope.scopeId)

  return (
    unique(target.requiredScopeIds) &&
    unique(observed) &&
    target.requiredScopeIds.length === observed.length &&
    target.requiredScopeIds.every(scopeId => observed.includes(scopeId))
  )
}

function targetIsComplete(target: SweepTarget, observation: SweepObservation | undefined): boolean {
  if (target.excluded) return true
  if (!observation) return false

  return (
    observation.health.observationId !== '' &&
    observation.health.installId === target.installId &&
    hasRequiredScope(target, observation.health)
  )
}

async function runBoundedProbe(
  probe: SweepProbe,
  target: SweepTarget,
  context: SweepProbeContext,
  remainingMs: number
): Promise<{ health: HealthEvidence }> {
  if (remainingMs <= 0) throw new Error('probe-timeout')

  let timeout: ReturnType<typeof setTimeout> | undefined
  const timeoutPromise = new Promise<never>((_, reject) => {
    timeout = setTimeout(() => reject(new Error('probe-timeout')), Math.max(1, remainingMs))
  })

  try {
    return await Promise.race([Promise.resolve().then(() => probe(target, context)), timeoutPromise])
  } finally {
    if (timeout) clearTimeout(timeout)
  }
}

function validateTargets(targets: readonly SweepTarget[]): void {
  const ids = targets.map(target => target.installId)

  if (!ids.length || ids.length > 500 || !unique(ids)) throw new Error('invalid-sweep-targets')
  if (targets.some(target => !target.installId || !Number.isSafeInteger(target.wave) || target.wave < 0)) {
    throw new Error('invalid-sweep-targets')
  }
  if (targets.some(target => target.requiredScopeIds !== null && !unique(target.requiredScopeIds))) {
    throw new Error('invalid-sweep-targets')
  }
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
  let cursor = 0

  const worker = async (): Promise<void> => {
    for (;;) {
      const index = cursor
      cursor += 1
      if (index >= targets.length) return

      const target = targets[index]
      if (target.excluded) continue

      if (nowMono() - startedMono > deadlineMs) {
        errors.push({ installId: target.installId, reason: 'sweep-deadline-exceeded' })
        continue
      }

      try {
        const now = nowMono()
        const sweepDeadlineMono = startedMono + deadlineMs
        const probeDeadlineMono = Math.min(sweepDeadlineMono, now + PROBE_DEADLINE_MS)
        const result = await runBoundedProbe(
          probe,
          target,
          { epochId: options.epochId, deadlineMono: probeDeadlineMono },
          probeDeadlineMono - now
        )
        const health = validateHealthEvidence(result.health)

        if (health.observationId !== options.epochId) {
          errors.push({ installId: target.installId, reason: 'stale-observation-epoch' })
          continue
        }

        observationsByIndex.set(index, { installId: target.installId, wave: target.wave, health })
      } catch (error) {
        errors.push({ installId: target.installId, reason: boundedReason(error) })
      }
    }
  }

  await Promise.all(Array.from({ length: concurrency }, () => worker()))

  const observations = [...observationsByIndex.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, observation]) => observation)
  const observationById = new Map(observations.map(observation => [observation.installId, observation]))
  const completeScope = targets.every(target => targetIsComplete(target, observationById.get(target.installId)))
  const finishedMono = nowMono()
  const fresh =
    finishedMono >= startedMono &&
    finishedMono - startedMono <= deadlineMs &&
    errors.length === 0 &&
    observations.every(observation => observation.health.observationId === options.epochId)
  const nextAdmissionInstallIds = targets
    .filter(target => !target.excluded)
    .sort((left, right) => left.wave - right.wave || left.installId.localeCompare(right.installId))
    .map(target => target.installId)

  return {
    ok: completeScope && fresh,
    epochId: options.epochId,
    startedMono,
    finishedMono,
    observations,
    errors: errors.sort((left, right) => left.installId.localeCompare(right.installId)),
    completeScope,
    fresh,
    nextAdmissionInstallIds
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
