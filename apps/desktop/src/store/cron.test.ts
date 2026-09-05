import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { CronJob } from '@/types/hermes'

import {
  $cronJobs,
  beginCronJobsRequest,
  commitCronJobsRequest,
  sameCronJob,
  sameCronJobs,
  setCronJobs,
  updateCronJobs
} from './cron'

const oldJob = { id: 'old' } as never
const newJob = { id: 'new' } as never
const jobA: CronJob = { enabled: true, id: 'job-a', name: 'Job A' } as CronJob

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

  it('does not mutate $cronJobs or notify listeners when committing an identical job list', () => {
    const listener = vi.fn()
    const unsubscribe = $cronJobs.listen(listener)

    const req = beginCronJobsRequest('all')
    expect(commitCronJobsRequest(req, [])).toBe(true)
    expect(listener).not.toHaveBeenCalled()

    // Now set a non-empty job
    setCronJobs([jobA])
    listener.mockClear()

    const req2 = beginCronJobsRequest('all')
    expect(commitCronJobsRequest(req2, [{ ...jobA }])).toBe(true)
    expect(listener).not.toHaveBeenCalled()

    // And changing a job should notify
    const req3 = beginCronJobsRequest('all')
    expect(commitCronJobsRequest(req3, [{ ...jobA, name: 'Changed' }])).toBe(true)
    expect(listener).toHaveBeenCalledTimes(1)

    unsubscribe()
  })

  it('notifies listeners and publishes when a schedule-only change is committed', () => {
    const initialJob: CronJob = {
      ...jobA,
      schedule: {
        kind: 'cron',
        expr: '0 9 * * *',
        display: 'At 09:00 AM',
      },
    }

    setCronJobs([initialJob])

    const listener = vi.fn()
    const unsubscribe = $cronJobs.listen(listener)

    const updatedExprJob: CronJob = {
      ...initialJob,
      schedule: {
        kind: 'cron',
        expr: '0 10 * * *',
        display: 'At 09:00 AM',
      },
    }

    const req = beginCronJobsRequest('all')
    expect(commitCronJobsRequest(req, [updatedExprJob])).toBe(true)
    expect(listener).toHaveBeenCalledTimes(1)
    expect($cronJobs.get()).toEqual([updatedExprJob])

    listener.mockClear()

    const updatedDisplayJob: CronJob = {
      ...updatedExprJob,
      schedule: {
        kind: 'cron',
        expr: '0 10 * * *',
        display: 'At 10:00 AM',
      },
    }

    const req2 = beginCronJobsRequest('all')
    expect(commitCronJobsRequest(req2, [updatedDisplayJob])).toBe(true)
    expect(listener).toHaveBeenCalledTimes(1)
    expect($cronJobs.get()).toEqual([updatedDisplayJob])

    listener.mockClear()

    const updatedKindJob: CronJob = {
      ...updatedDisplayJob,
      schedule: {
        kind: 'interval',
        expr: '0 10 * * *',
        display: 'At 10:00 AM',
      },
    }

    const req3 = beginCronJobsRequest('all')
    expect(commitCronJobsRequest(req3, [updatedKindJob])).toBe(true)
    expect(listener).toHaveBeenCalledTimes(1)
    expect($cronJobs.get()).toEqual([updatedKindJob])

    unsubscribe()
  })

  it('does not mutate $cronJobs or notify listeners when committing identical schedule objects', () => {
    const scheduledJob: CronJob = {
      ...jobA,
      schedule: {
        kind: 'cron',
        expr: '0 9 * * *',
        display: 'At 09:00 AM'
      }
    }

    setCronJobs([scheduledJob])

    const listener = vi.fn()
    const unsubscribe = $cronJobs.listen(listener)

    const clonedJob: CronJob = {
      ...scheduledJob,
      schedule: {
        kind: 'cron',
        expr: '0 9 * * *',
        display: 'At 09:00 AM'
      }
    }

    const req = beginCronJobsRequest('all')
    expect(commitCronJobsRequest(req, [clonedJob])).toBe(true)
    expect(listener).not.toHaveBeenCalled()
    expect($cronJobs.get()).toEqual([scheduledJob])

    unsubscribe()
  })

  it('notifies listeners and publishes when schedule is added or removed', () => {
    setCronJobs([jobA])

    const listener = vi.fn()
    const unsubscribe = $cronJobs.listen(listener)

    const withSchedule: CronJob = {
      ...jobA,
      schedule: {
        kind: 'cron',
        expr: '0 9 * * *',
        display: 'At 09:00 AM'
      }
    }

    const req = beginCronJobsRequest('all')
    expect(commitCronJobsRequest(req, [withSchedule])).toBe(true)
    expect(listener).toHaveBeenCalledTimes(1)
    expect($cronJobs.get()).toEqual([withSchedule])

    listener.mockClear()

    const withoutSchedule: CronJob = {
      ...jobA,
      schedule: undefined
    }

    const req2 = beginCronJobsRequest('all')
    expect(commitCronJobsRequest(req2, [withoutSchedule])).toBe(true)
    expect(listener).toHaveBeenCalledTimes(1)
    expect($cronJobs.get()).toEqual([withoutSchedule])

    unsubscribe()
  })
})

describe('CronJob structural comparator', () => {
  const fullJob: CronJob = {
    deliver: 'webhook',
    enabled: true,
    id: 'job-1',
    last_error: null,
    last_run_at: '2026-09-01T00:00:00Z',
    model: 'gpt-4',
    name: 'Full Job',
    next_run_at: '2026-09-02T00:00:00Z',
    no_agent: false,
    prompt: 'Run task',
    provider: 'openai',
    schedule: {
      display: 'Every day',
      expr: '0 0 * * *',
      kind: 'cron'
    },
    schedule_display: 'Every day at midnight',
    script: 'run.sh',
    state: 'idle'
  }

  it('returns true when comparing identical job instances or structural clones', () => {
    expect(sameCronJob(fullJob, fullJob)).toBe(true)
    expect(sameCronJob(fullJob, { ...fullJob })).toBe(true)
    expect(sameCronJob(fullJob, { ...fullJob, schedule: { ...fullJob.schedule } })).toBe(true)
  })

  it('detects changes in any CronJob interface field', () => {
    expect(sameCronJob(fullJob, { ...fullJob, id: 'job-2' })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, name: 'Other' })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, enabled: false })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, state: 'running' })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, next_run_at: '2026-09-03T00:00:00Z' })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, last_run_at: '2026-09-02T00:00:00Z' })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, last_error: 'Failed' })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, schedule_display: 'Changed display' })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, prompt: 'New prompt' })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, model: 'claude-3' })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, provider: 'anthropic' })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, deliver: 'email' })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, no_agent: true })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, script: 'other.sh' })).toBe(false)
  })

  it('detects changes in nested schedule fields', () => {
    expect(sameCronJob(fullJob, { ...fullJob, schedule: { ...fullJob.schedule, kind: 'interval' } })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, schedule: { ...fullJob.schedule, expr: '0 1 * * *' } })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, schedule: { ...fullJob.schedule, display: 'Hourly' } })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, schedule: undefined })).toBe(false)
    expect(sameCronJob({ ...fullJob, schedule: undefined }, fullJob)).toBe(false)
    expect(sameCronJob({ ...fullJob, schedule: undefined }, { ...fullJob, schedule: undefined })).toBe(true)
  })

  it('compares job arrays correctly via sameCronJobs', () => {
    const listA = [fullJob]
    const listB = [{ ...fullJob, schedule: { ...fullJob.schedule } }]

    expect(sameCronJobs(listA, listA)).toBe(true)
    expect(sameCronJobs(listA, listB)).toBe(true)
    expect(sameCronJobs(listA, [])).toBe(false)
    expect(sameCronJobs(listA, [fullJob, fullJob])).toBe(false)
    expect(sameCronJobs(listA, [{ ...fullJob, name: 'Diff' }])).toBe(false)
  })
})
