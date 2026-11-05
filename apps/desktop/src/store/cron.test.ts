import { beforeEach, describe, expect, it } from 'vitest'

import {
  $cronJobs,
  $cronJobsFailedScope,
  $cronJobsSnapshotScope,
  beginCronJobsRequest,
  commitCronJobsRequest,
  failCronJobsRequest,
  setCronJobs,
  updateCronJobs
} from './cron'

const oldJob = { id: 'old' } as never
const newJob = { id: 'new' } as never

describe('cron jobs request fencing', () => {
  beforeEach(() => {
    setCronJobs([])
  })

  it('rejects an older refresh after a newer refresh commits', () => {
    const older = beginCronJobsRequest('all')
    const newer = beginCronJobsRequest('all')

    expect(commitCronJobsRequest(newer, [newJob])).toBe(true)
    expect(commitCronJobsRequest(older, [oldJob])).toBe(false)
    expect($cronJobs.get()).toEqual([newJob])
  })

  it('rejects a refresh from the previous profile scope', () => {
    const work = beginCronJobsRequest('work')

    beginCronJobsRequest('personal')

    expect(commitCronJobsRequest(work, [oldJob])).toBe(false)
    expect($cronJobs.get()).toEqual([])
  })

  it('rejects an in-flight poll after a local mutation', () => {
    const poll = beginCronJobsRequest('all')

    updateCronJobs(() => [newJob])

    expect(commitCronJobsRequest(poll, [oldJob])).toBe(false)
    expect($cronJobs.get()).toEqual([newJob])
  })
})

describe('cron jobs first-load outcome', () => {
  beforeEach(() => {
    setCronJobs([])
  })

  it('ignores a superseded failure and records the newest one', () => {
    const older = beginCronJobsRequest('work')
    const newer = beginCronJobsRequest('work')

    expect(failCronJobsRequest(older)).toBe(false)
    expect($cronJobsFailedScope.get()).toBe('')

    expect(failCronJobsRequest(newer)).toBe(true)
    expect($cronJobsFailedScope.get()).toBe('work')
  })

  it('clears a recorded failure when a new request starts or the cache is reset', () => {
    failCronJobsRequest(beginCronJobsRequest('work'))
    beginCronJobsRequest('work')

    expect($cronJobsFailedScope.get()).toBe('')

    failCronJobsRequest(beginCronJobsRequest('work'))
    setCronJobs([])

    expect($cronJobsFailedScope.get()).toBe('')
    expect($cronJobsSnapshotScope.get()).toBe('')
  })
})
