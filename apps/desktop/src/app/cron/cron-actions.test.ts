import { beforeEach, describe, expect, it, vi } from 'vitest'

const getCronJobs = vi.fn()
const triggerCronJob = vi.fn()

vi.mock('@/hermes', () => ({
  getCronJobs: (...args: unknown[]) => getCronJobs(...args),
  triggerCronJob: (...args: unknown[]) => triggerCronJob(...args)
}))

import { beginCronJobsRequest } from '@/store/cron'

import { mutateAndRefreshCronJobs, refreshCronJobs, triggerAndRefreshCronJobs } from './cron-actions'

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(res => {
    resolve = res
  })

  return { promise, resolve }
}

describe('triggerAndRefreshCronJobs', () => {
  beforeEach(() => {
    getCronJobs.mockReset()
    triggerCronJob.mockReset()
  })

  it('replaces the local cache with the authoritative list after a trigger', async () => {
    const authoritative = [{ id: 'recurring-job', state: 'scheduled' }]
    triggerCronJob.mockResolvedValue({ id: 'deleted-one-shot', state: 'completed' })
    getCronJobs.mockResolvedValue(authoritative)

    const result = await triggerAndRefreshCronJobs('deleted-one-shot', 'work')

    expect(triggerCronJob).toHaveBeenCalledWith('deleted-one-shot')
    expect(getCronJobs).toHaveBeenCalledWith('work')
    expect(result).toEqual({ jobs: authoritative, refreshError: null, stale: false })
  })

  it('reports refresh failure separately after a successful trigger', async () => {
    const refreshError = new Error('refresh failed')
    triggerCronJob.mockResolvedValue({ id: 'job-1', state: 'scheduled' })
    getCronJobs.mockRejectedValue(refreshError)

    const result = await triggerAndRefreshCronJobs('job-1', 'all')

    expect(result).toEqual({ jobs: null, refreshError, stale: false })
  })

  it('still rejects when the trigger itself fails', async () => {
    const triggerError = new Error('trigger failed')
    triggerCronJob.mockRejectedValue(triggerError)

    await expect(triggerAndRefreshCronJobs('job-1', 'all')).rejects.toBe(triggerError)
    expect(getCronJobs).not.toHaveBeenCalled()
  })

  it('discards a trigger failure after the profile scope changes', async () => {
    const trigger = deferred<never>()
    triggerCronJob.mockReturnValue(trigger.promise)

    const resultPromise = triggerAndRefreshCronJobs('job-1', 'work')
    beginCronJobsRequest('personal')
    trigger.resolve(Promise.reject(new Error('old profile failed')) as never)

    await expect(resultPromise).resolves.toEqual({ jobs: null, refreshError: null, stale: true })
    expect(getCronJobs).not.toHaveBeenCalled()
  })

  it('discards a trigger refresh after the profile scope changes', async () => {
    const refresh = deferred<Array<{ id: string }>>()
    triggerCronJob.mockResolvedValue({ id: 'job-1', state: 'scheduled' })
    getCronJobs.mockReturnValue(refresh.promise)

    const resultPromise = triggerAndRefreshCronJobs('job-1', 'work')
    await triggerCronJob.mock.results[0]?.value

    beginCronJobsRequest('personal')
    refresh.resolve([{ id: 'work-job' }])

    await expect(resultPromise).resolves.toEqual({ jobs: null, refreshError: null, stale: true })
  })

  it('discards an older ordinary refresh that completes after a trigger refresh', async () => {
    const older = deferred<Array<{ id: string }>>()
    const newer = deferred<Array<{ id: string }>>()
    triggerCronJob.mockResolvedValue({ id: 'job-1', state: 'scheduled' })
    getCronJobs.mockReturnValueOnce(older.promise).mockReturnValueOnce(newer.promise)

    const olderPromise = refreshCronJobs('work')
    const newerPromise = triggerAndRefreshCronJobs('job-1', 'work')
    newer.resolve([{ id: 'newer' }])
    older.resolve([{ id: 'older' }])

    await expect(newerPromise).resolves.toEqual({
      jobs: [{ id: 'newer' }],
      refreshError: null,
      stale: false
    })
    await expect(olderPromise).resolves.toEqual({ jobs: null, refreshError: null, stale: true })
  })
})

describe('mutateAndRefreshCronJobs', () => {
  beforeEach(() => {
    getCronJobs.mockReset()
  })

  it('does not refresh the old profile after a successful mutation switches scope', async () => {
    const mutation = deferred<{ id: string }>()
    const resultPromise = mutateAndRefreshCronJobs('work', () => mutation.promise)

    beginCronJobsRequest('personal')
    mutation.resolve({ id: 'work-job' })

    await expect(resultPromise).resolves.toEqual({
      jobs: null,
      refreshError: null,
      stale: true,
      value: null
    })
    expect(getCronJobs).not.toHaveBeenCalled()
  })

  it('suppresses a mutation error after the profile scope changes', async () => {
    const mutation = deferred<never>()
    const resultPromise = mutateAndRefreshCronJobs('work', () => mutation.promise)

    beginCronJobsRequest('personal')
    mutation.resolve(Promise.reject(new Error('old profile failed')) as never)

    await expect(resultPromise).resolves.toEqual({
      jobs: null,
      refreshError: null,
      stale: true,
      value: null
    })
  })

  it('allows overlapping same-profile mutations to authoritatively refresh', async () => {
    const first = deferred<string>()
    const second = deferred<string>()
    getCronJobs
      .mockResolvedValueOnce([{ id: 'after-second' }])
      .mockResolvedValueOnce([{ id: 'after-both' }])

    const firstResult = mutateAndRefreshCronJobs('work', () => first.promise)
    const secondResult = mutateAndRefreshCronJobs('work', () => second.promise)

    second.resolve('second')
    await expect(secondResult).resolves.toMatchObject({ stale: false, value: 'second' })

    first.resolve('first')
    await expect(firstResult).resolves.toMatchObject({ stale: false, value: 'first' })
    expect(getCronJobs).toHaveBeenCalledTimes(2)
  })
})
