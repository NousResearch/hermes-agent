import { beforeEach, describe, expect, it } from 'vitest'

import {
  $cronJobs,
  beginCronJobsRequest,
  commitCronJobsRequest,
  invalidateCronJobsRequests,
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

  it('restores the cached jobs when returning to a prior profile scope', () => {
    const work = beginCronJobsRequest('work')
    expect(commitCronJobsRequest(work, [oldJob])).toBe(true)

    const personal = beginCronJobsRequest('personal')
    expect(commitCronJobsRequest(personal, [newJob])).toBe(true)

    beginCronJobsRequest('work')

    expect($cronJobs.get()).toEqual([oldJob])
  })

  it('rejects a same-named profile request that completed after a gateway wipe', () => {
    const beforeSwitch = beginCronJobsRequest('local\u0000default')

    invalidateCronJobsRequests()
    const afterSwitch = beginCronJobsRequest('local\u0000default')

    expect(commitCronJobsRequest(afterSwitch, [newJob])).toBe(true)
    expect(commitCronJobsRequest(beforeSwitch, [oldJob])).toBe(false)
    expect($cronJobs.get()).toEqual([newJob])
  })
})
