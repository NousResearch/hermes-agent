import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { CronJob } from '@/types/hermes'

import {
  $cronJobs,
  beginCronJobsRequest,
  commitCronJobsRequest,
  sameCronJob,
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

  it('detects changes in every CronJob interface field', () => {
    const changedByField = {
      deliver: { ...fullJob, deliver: 'email' },
      enabled: { ...fullJob, enabled: false },
      id: { ...fullJob, id: 'job-2' },
      last_error: { ...fullJob, last_error: 'Failed' },
      last_run_at: { ...fullJob, last_run_at: '2026-09-02T00:00:00Z' },
      model: { ...fullJob, model: 'claude-3' },
      name: { ...fullJob, name: 'Other' },
      next_run_at: { ...fullJob, next_run_at: '2026-09-03T00:00:00Z' },
      no_agent: { ...fullJob, no_agent: true },
      prompt: { ...fullJob, prompt: 'New prompt' },
      provider: { ...fullJob, provider: 'anthropic' },
      schedule: { ...fullJob, schedule: undefined },
      schedule_display: { ...fullJob, schedule_display: 'Changed display' },
      script: { ...fullJob, script: 'other.sh' },
      state: { ...fullJob, state: 'running' }
    } satisfies Record<keyof CronJob, CronJob>

    for (const [field, changedJob] of Object.entries(changedByField)) {
      expect(sameCronJob(fullJob, changedJob), field).toBe(false)
    }
  })

  it('detects changes in nested schedule fields', () => {
    expect(sameCronJob(fullJob, { ...fullJob, schedule: { ...fullJob.schedule, kind: 'interval' } })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, schedule: { ...fullJob.schedule, expr: '0 1 * * *' } })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, schedule: { ...fullJob.schedule, display: 'Hourly' } })).toBe(false)
    expect(sameCronJob(fullJob, { ...fullJob, schedule: undefined })).toBe(false)
    expect(sameCronJob({ ...fullJob, schedule: undefined }, fullJob)).toBe(false)
    expect(sameCronJob({ ...fullJob, schedule: undefined }, { ...fullJob, schedule: undefined })).toBe(true)
  })
})
