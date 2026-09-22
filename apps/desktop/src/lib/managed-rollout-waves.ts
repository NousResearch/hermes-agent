import {
  MAX_ROLLOUT_CONCURRENCY,
  MAX_ROLLOUT_INSTALLATIONS,
  validateRolloutCapabilities,
  type PromotionPolicy,
  type RolloutCapabilities
} from './managed-rollout-contract'

export const MAX_EFFECTIVE_WAVE_SIZE = 120
export const MAX_SWEEP_PROBES = MAX_EFFECTIVE_WAVE_SIZE * 2
export const WAVE_SIZE_EXCEEDS_SWEEP_BUDGET = 'wave-size-exceeds-sweep-budget'

export interface SweepFeasibility {
  ok: boolean
  canaryCount: number
  successorCount: number
  probeBudget: number
  reason: string | null
}

export function promotionSweepFeasibility(canaryCount: number, successorCount: number): SweepFeasibility {
  const valid =
    Number.isSafeInteger(canaryCount) &&
    Number.isSafeInteger(successorCount) &&
    canaryCount >= 1 &&
    successorCount >= 1
  const ok = valid && canaryCount <= MAX_EFFECTIVE_WAVE_SIZE && successorCount <= MAX_EFFECTIVE_WAVE_SIZE

  return {
    ok,
    canaryCount,
    successorCount,
    probeBudget: MAX_SWEEP_PROBES,
    reason: ok ? null : WAVE_SIZE_EXCEEDS_SWEEP_BUDGET
  }
}

function invalid(message: string): never {
  throw new Error(message)
}

function uniqueNonEmpty(values: readonly string[], label: string): string[] {
  if (!Array.isArray(values) || values.length === 0) invalid(`invalid-${label}`)

  if (values.some(value => typeof value !== 'string')) invalid(`invalid-${label}`)
  const copy = values.slice()

  if (copy.some(value => value.length === 0 || /[\x00\r\n]/.test(value))) invalid(`invalid-${label}`)
  if (new Set(copy).size !== copy.length) invalid(`invalid-${label}`)

  return copy
}

function validateWaveSize(size: number): number {
  if (!Number.isInteger(size) || size < 1) invalid('invalid-wave-size')
  if (size > MAX_EFFECTIVE_WAVE_SIZE) invalid(WAVE_SIZE_EXCEEDS_SWEEP_BUDGET)

  return size
}

/**
 * Construct the one canonical ordered partition used by both renderer preview
 * and Electron admission. Canary order is explicit; all remaining order comes
 * from the operator's selected list. A wave never implies concurrent updates.
 */
export function makeWaves(selected: readonly string[], canaries: readonly string[], size: number): string[][] {
  const normalizedSelection = uniqueNonEmpty(selected, 'selection')
  if (!Array.isArray(canaries) || canaries.some(value => typeof value !== 'string')) invalid('invalid-canaries')
  const normalizedCanaries = canaries.slice()
  validateWaveSize(size)

  if (normalizedSelection.length > MAX_ROLLOUT_INSTALLATIONS) invalid('invalid-selection')
  if (normalizedCanaries.some(value => value.length === 0 || /[\x00\r\n]/.test(value))) invalid('invalid-canaries')
  if (new Set(normalizedCanaries).size !== normalizedCanaries.length) invalid('invalid-canaries')
  if (normalizedCanaries.some(value => !normalizedSelection.includes(value))) invalid('invalid-canaries')

  if (normalizedCanaries.length > MAX_EFFECTIVE_WAVE_SIZE) invalid(WAVE_SIZE_EXCEEDS_SWEEP_BUDGET)

  if (normalizedSelection.length === 1) {
    if (normalizedCanaries.length && normalizedCanaries[0] !== normalizedSelection[0]) invalid('invalid-canaries')

    return [[normalizedSelection[0]]]
  }

  if (normalizedCanaries.length === 0 || normalizedCanaries.length >= normalizedSelection.length)
    invalid('invalid-canaries')

  const canarySet = new Set(normalizedCanaries)
  const remaining = normalizedSelection.filter(value => !canarySet.has(value))
  if (
    remaining.length > 0 &&
    !promotionSweepFeasibility(normalizedCanaries.length, Math.min(size, remaining.length)).ok
  ) {
    invalid(WAVE_SIZE_EXCEEDS_SWEEP_BUDGET)
  }
  const waves = [normalizedCanaries.slice()]

  for (let index = 0; index < remaining.length; index += size) {
    waves.push(remaining.slice(index, index + size))
  }

  return waves
}

/** Number of human approvals required at wave boundaries. */
export function requiredApprovalCount(waves: readonly (readonly string[])[], policy: PromotionPolicy): number {
  if (waves.length <= 1) return 0
  return policy === 'manual' ? waves.length - 1 : 1
}

export const approvalCount = requiredApprovalCount

function validateEstimateInputs(waves: readonly (readonly string[])[], durations: Record<string, number>): void {
  if (!Array.isArray(waves) || waves.length === 0) invalid('invalid-waves')
  if (!durations || typeof durations !== 'object') invalid('invalid-duration-history')

  const ids = waves.flat()
  if (new Set(ids).size !== ids.length || ids.some(id => typeof id !== 'string' || id.length === 0)) {
    invalid('invalid-waves')
  }
}

/**
 * Estimate machine work only. Approval waiting is deliberately not included;
 * absent/invalid history returns null rather than invented timing.
 */
export function estimateOperationalMs(
  waves: readonly (readonly string[])[],
  durations: Record<string, number>,
  concurrency: number
): number | null {
  if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > MAX_ROLLOUT_CONCURRENCY) return null

  try {
    validateEstimateInputs(waves, durations)
  } catch {
    return null
  }

  let total = 0

  for (const wave of waves) {
    const lanes = Array<number>(concurrency).fill(0)

    for (const id of wave) {
      const duration = durations[id]

      if (!Number.isFinite(duration) || duration < 0) return null

      const lane = lanes.indexOf(Math.min(...lanes))
      lanes[lane] += duration
    }

    total += Math.max(...lanes, 0)
  }

  return total
}

/**
 * The caller may ask the pure helper about a future concurrency, but it must
 * never estimate or display work the currently advertised capability cannot
 * execute. This wrapper is the capability gate for that boundary.
 */
export function estimateCapabilityBoundOperationalMs(
  waves: readonly (readonly string[])[],
  durations: Record<string, number>,
  requestedConcurrency: number,
  capabilities: RolloutCapabilities
): number | null {
  let bounded: RolloutCapabilities

  try {
    bounded = validateRolloutCapabilities(capabilities)
  } catch {
    return null
  }

  if (!bounded.available || requestedConcurrency > bounded.maxConcurrency) return null
  if (bounded.maxInstallations < waves.flat().length) return null

  return estimateOperationalMs(waves, durations, requestedConcurrency)
}

export { MAX_ROLLOUT_CONCURRENCY as MAX_CONCURRENCY, MAX_ROLLOUT_INSTALLATIONS as MAX_INSTALLATIONS }
