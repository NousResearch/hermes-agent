import { beforeEach, describe, expect, it } from 'vitest'

import {
  $cronJobs,
  beginCronJobsRequest,
  commitCronJobsRequest,
  cronJobIdentity,
  removeCronJobForOwner,
  replaceCronJobForOwner,
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

describe('cron job owner identity', () => {
  it('keeps duplicate job ids in different profiles distinct', () => {
    expect(cronJobIdentity({ id: 'shared-job', profile: 'worker_alpha' })).not.toBe(
      cronJobIdentity({ id: 'shared-job', profile: 'worker_beta' })
    )
  })

  it('replaces only the matching profile when duplicate ids coexist', () => {
    const alpha = { id: 'shared-job', profile: 'worker_alpha', state: 'scheduled' }
    const beta = { id: 'shared-job', profile: 'worker_beta', state: 'scheduled' }
    const pausedAlpha = { ...alpha, state: 'paused' }

    expect(replaceCronJobForOwner([alpha, beta] as never, alpha as never, pausedAlpha as never)).toEqual([
      pausedAlpha,
      beta
    ])
  })

  it('deletes only the matching profile when duplicate ids coexist', () => {
    const alpha = { id: 'shared-job', profile: 'worker_alpha' }
    const beta = { id: 'shared-job', profile: 'worker_beta' }

    expect(removeCronJobForOwner([alpha, beta] as never, alpha as never)).toEqual([beta])
  })
})
