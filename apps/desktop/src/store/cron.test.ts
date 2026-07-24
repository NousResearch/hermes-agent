import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { getCachedCronRuns, invalidateCronJobRuns, loadCronJobRuns } from './cron'

const getCronJobRuns = vi.fn<(jobId: string, limit?: number) => Promise<SessionInfo[]>>()

vi.mock('@/hermes', () => ({
  getCronJobRuns: (jobId: string, limit?: number) => getCronJobRuns(jobId, limit)
}))

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void

  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })

  return { promise, reject, resolve }
}

function runs(prefix: string, count: number): SessionInfo[] {
  return Array.from({ length: count }, (_, index) => ({ id: `${prefix}-${index}` }) as SessionInfo)
}

describe('cron run cache', () => {
  beforeEach(() => {
    getCronJobRuns.mockReset()
  })

  it('keeps concurrent 5-row and 20-row requests separate when they resolve out of order', async () => {
    const fiveRows = deferred<SessionInfo[]>()
    const twentyRows = deferred<SessionInfo[]>()
    getCronJobRuns.mockImplementation((_jobId, limit) => (limit === 5 ? fiveRows.promise : twentyRows.promise))

    const fiveRequest = loadCronJobRuns('job-reordered', 5)
    const duplicateFiveRequest = loadCronJobRuns('job-reordered', 5)
    const twentyRequest = loadCronJobRuns('job-reordered')
    const duplicateTwentyRequest = loadCronJobRuns('job-reordered', 20)

    expect(duplicateFiveRequest).toBe(fiveRequest)
    expect(duplicateTwentyRequest).toBe(twentyRequest)
    expect(getCronJobRuns).toHaveBeenCalledTimes(2)
    expect(getCronJobRuns).toHaveBeenCalledWith('job-reordered', 5)
    expect(getCronJobRuns).toHaveBeenCalledWith('job-reordered', 20)

    const twenty = runs('twenty', 20)
    twentyRows.resolve(twenty)
    await expect(twentyRequest).resolves.toBe(twenty)

    expect(getCachedCronRuns('job-reordered')).toBe(twenty)
    expect(getCachedCronRuns('job-reordered', 5)).toBeNull()

    const five = runs('five', 5)
    fiveRows.resolve(five)
    await expect(fiveRequest).resolves.toBe(five)

    expect(getCachedCronRuns('job-reordered', 5)).toBe(five)
    expect(getCachedCronRuns('job-reordered', 20)).toBe(twenty)
  })

  it('deduplicates equivalent normalized limits and caches the normalized shape', async () => {
    const five = runs('normalized', 5)
    getCronJobRuns.mockResolvedValue(five)

    const decimalRequest = loadCronJobRuns('job-normalized', 5.9)
    const integerRequest = loadCronJobRuns('job-normalized', 5)

    expect(integerRequest).toBe(decimalRequest)
    await expect(decimalRequest).resolves.toBe(five)
    expect(getCronJobRuns).toHaveBeenCalledOnce()
    expect(getCronJobRuns).toHaveBeenCalledWith('job-normalized', 5)
    expect(getCachedCronRuns('job-normalized', 5.1)).toBe(five)
  })

  it('invalidates every cached and in-flight limit for one job only', async () => {
    const staleFiveRows = deferred<SessionInfo[]>()
    const twenty = runs('twenty', 20)
    const other = runs('other', 5)
    getCronJobRuns.mockImplementation((jobId, limit) => {
      if (jobId === 'job-invalidated' && limit === 5) {
        return staleFiveRows.promise
      }

      return Promise.resolve(jobId === 'job-invalidated' ? twenty : other)
    })

    const staleFiveRequest = loadCronJobRuns('job-invalidated', 5)
    await loadCronJobRuns('job-invalidated', 20)
    await loadCronJobRuns('job-preserved', 5)

    invalidateCronJobRuns('job-invalidated')

    expect(getCachedCronRuns('job-invalidated', 5)).toBeNull()
    expect(getCachedCronRuns('job-invalidated', 20)).toBeNull()
    expect(getCachedCronRuns('job-preserved', 5)).toBe(other)

    staleFiveRows.resolve(runs('stale', 5))
    await staleFiveRequest
    expect(getCachedCronRuns('job-invalidated', 5)).toBeNull()
  })
})
