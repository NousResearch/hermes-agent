import { canAdmitLocalBackend, type PoolEvictionEntry } from './pool-eviction'

export const SPECIALIST_POOL_PREFIX = 'specialist-run:'

export interface SpecialistDispatchAdmissionRequest {
  requestDigest: string
  runId: string
  profileIds: string[]
  /** Canonical runner input; never interpreted as a command by Electron. */
  runnerInput?: string
}

export interface SpecialistDispatchAdmissionResult {
  outcome: 'admitted' | 'rejected'
  reasonCode: 'ADMITTED' | 'IDEMPOTENCY_CONFLICT' | 'INVALID_REQUEST' | 'POOL_CAPACITY_EXCEEDED' | 'RUNNER_START_FAILED'
  maxCapacity?: number
  reservedCapacity?: number
  runId: string
}

export interface SpecialistDispatchAdmissionDeps {
  maxCapacity: number
  pool: Map<string, PoolEvictionEntry>
  spawnRunner: (request: SpecialistDispatchAdmissionRequest) => Promise<unknown>
}

const DIGEST = /^sha256:[0-9a-f]{64}$/
const PROFILE = /^[a-z0-9][a-z0-9_-]{0,63}$/
const RUN_ID = /^specialist-run-[a-z0-9][a-z0-9-]{7,63}$/

function valid(request: SpecialistDispatchAdmissionRequest): boolean {
  return Boolean(
    request &&
      typeof request.runId === 'string' && RUN_ID.test(request.runId) &&
      typeof request.requestDigest === 'string' && DIGEST.test(request.requestDigest) &&
      Array.isArray(request.profileIds) &&
      request.profileIds.length >= 1 &&
      request.profileIds.length <= 4 &&
      request.profileIds.every(profile => typeof profile === 'string' && PROFILE.test(profile)) &&
      request.profileIds.every((profile, index, profiles) => index === 0 || profiles[index - 1] < profile)
  )
}

/**
 * Electron-only weighted pool admission. The caller supplies no command,
 * capability assertion, token, connection URL, or capacity value; the fixed
 * Electron spawn path receives only an admitted request identity.
 */
export function createSpecialistDispatchAdmission(deps: SpecialistDispatchAdmissionDeps) {
  const starts = new Map<string, { digest: string; result: Promise<SpecialistDispatchAdmissionResult>; entry?: PoolEvictionEntry }>()
  const terminals = new Map<string, 'released' | 'runner_failed' | 'cleanup_failed'>()

  function poolKey(runId: string): string {
    return `${SPECIALIST_POOL_PREFIX}${runId}`
  }

  function release(runId: string, terminal: 'released' | 'runner_failed' | 'cleanup_failed'): void {
    const record = starts.get(runId)
    if (!record) {
      return
    }
    const key = poolKey(runId)
    if (record.entry && deps.pool.get(key) === record.entry) {
      deps.pool.delete(key)
    }
    terminals.set(runId, terminal)
  }

  return {
    async admit(request: SpecialistDispatchAdmissionRequest): Promise<SpecialistDispatchAdmissionResult> {
      if (!valid(request)) {
        return { outcome: 'rejected', reasonCode: 'INVALID_REQUEST', runId: String(request?.runId || '') }
      }
      const prior = starts.get(request.runId)
      if (prior) {
        if (prior.digest !== request.requestDigest) {
          return { outcome: 'rejected', reasonCode: 'IDEMPOTENCY_CONFLICT', runId: request.runId }
        }
        return prior.result
      }
      const reservedCapacity = request.profileIds.length
      const result = Promise.resolve().then(async (): Promise<SpecialistDispatchAdmissionResult> => {
        if (!canAdmitLocalBackend(deps.pool.entries(), deps.maxCapacity, reservedCapacity)) {
          return {
            outcome: 'rejected',
            reasonCode: 'POOL_CAPACITY_EXCEEDED',
            maxCapacity: deps.maxCapacity,
            runId: request.runId
          }
        }
        const entry: PoolEvictionEntry = {
          process: null,
          countsTowardPoolCap: true,
          capacityUnits: reservedCapacity,
          lastActiveAt: Date.now()
        }
        starts.get(request.runId)!.entry = entry
        deps.pool.set(poolKey(request.runId), entry)
        try {
          entry.process = await deps.spawnRunner(request)
          return { outcome: 'admitted', reasonCode: 'ADMITTED', reservedCapacity, runId: request.runId }
        } catch {
          if (deps.pool.get(poolKey(request.runId)) === entry) {
            deps.pool.delete(poolKey(request.runId))
          }
          starts.delete(request.runId)
          return { outcome: 'rejected', reasonCode: 'RUNNER_START_FAILED', runId: request.runId }
        }
      })
      starts.set(request.runId, { digest: request.requestDigest, result })
      return result
    },
    hasActive(): boolean {
      return [...starts.entries()].some(([runId, record]) => record.entry !== undefined && !terminals.has(runId))
    },
    poolKey,
    release,
    status(runId: string): 'running' | 'released' | 'runner_failed' | 'cleanup_failed' | 'unknown' {
      return terminals.get(runId) || (starts.has(runId) ? 'running' : 'unknown')
    }
  }
}
