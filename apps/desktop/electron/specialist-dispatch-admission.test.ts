import { expect, test, vi } from 'vitest'

import { createSpecialistDispatchAdmission } from './specialist-dispatch-admission'

type Entry = {
  capacityUnits?: number
  countsTowardPoolCap?: boolean
  lastActiveAt?: number
  process?: unknown
}

const request = (overrides: Record<string, unknown> = {}) => ({
  requestDigest: `sha256:${'a'.repeat(64)}`,
  runId: 'specialist-run-00000001',
  profileIds: ['explorer', 'longmemeval-bench', 'public', 'statistician'],
  ...overrides
})

test('Electron reserves all four requested slots before spawning and rejects the fifth without eviction', async () => {
  const pool = new Map<string, Entry>()
  const spawnRunner = vi.fn(async () => ({ pid: 17 }))
  const admission = createSpecialistDispatchAdmission({ maxCapacity: 4, pool, spawnRunner })

  const accepted = await admission.admit(request())
  const rejected = await admission.admit(request({
    requestDigest: `sha256:${'b'.repeat(64)}`,
    runId: 'specialist-run-00000002',
    profileIds: ['cognitive-scientist']
  }))

  expect(accepted).toMatchObject({ outcome: 'admitted', reservedCapacity: 4 })
  expect(rejected).toMatchObject({ outcome: 'rejected', reasonCode: 'POOL_CAPACITY_EXCEEDED', maxCapacity: 4 })
  expect(pool.get('specialist-run:specialist-run-00000001')).toMatchObject({ capacityUnits: 4, countsTowardPoolCap: true })
  expect(spawnRunner).toHaveBeenCalledTimes(1)
})

test('identical request coalesces, mismatched reuse is rejected, and exact release frees capacity', async () => {
  const pool = new Map<string, Entry>()
  const spawnRunner = vi.fn(async () => ({ pid: 18 }))
  const admission = createSpecialistDispatchAdmission({ maxCapacity: 4, pool, spawnRunner })

  const first = await admission.admit(request({ profileIds: ['explorer'] }))
  const duplicate = await admission.admit(request({ profileIds: ['explorer'] }))
  const conflict = await admission.admit(request({ requestDigest: `sha256:${'c'.repeat(64)}`, profileIds: ['explorer'] }))
  expect(admission.hasActive()).toBe(true)
  admission.release('specialist-run-00000001', 'released')
  expect(admission.hasActive()).toBe(false)
  const next = await admission.admit(request({ requestDigest: `sha256:${'d'.repeat(64)}`, runId: 'specialist-run-00000003', profileIds: ['explorer', 'public'] }))

  expect(duplicate).toEqual(first)
  expect(conflict).toMatchObject({ outcome: 'rejected', reasonCode: 'IDEMPOTENCY_CONFLICT' })
  expect(spawnRunner).toHaveBeenCalledTimes(2)
  expect(next).toMatchObject({ outcome: 'admitted', reservedCapacity: 2 })
})

test('invalid request is rejected before a runner can spawn', async () => {
  const spawnRunner = vi.fn(async () => ({ pid: 19 }))
  const admission = createSpecialistDispatchAdmission({ maxCapacity: 4, pool: new Map(), spawnRunner })

  const result = await admission.admit(request({ profileIds: [] }))

  expect(result).toMatchObject({ outcome: 'rejected', reasonCode: 'INVALID_REQUEST' })
  expect(spawnRunner).not.toHaveBeenCalled()
})

test('a capacity rejection is terminal admission evidence, not an active runner', async () => {
  const pool = new Map<string, Entry>([
    ['occupied', { capacityUnits: 4, countsTowardPoolCap: true, process: { pid: 20 } }]
  ])
  const admission = createSpecialistDispatchAdmission({ maxCapacity: 4, pool, spawnRunner: vi.fn() })

  const rejected = await admission.admit(request({ profileIds: ['explorer'], runId: 'specialist-run-00000004' }))

  expect(rejected).toMatchObject({ outcome: 'rejected', reasonCode: 'POOL_CAPACITY_EXCEEDED' })
  expect(admission.hasActive()).toBe(false)
})
