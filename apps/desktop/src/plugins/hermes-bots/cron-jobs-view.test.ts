/**
 * Which cron jobs the pane shows, and why it can look empty.
 *
 * Jobs are namespaced `[bot:<name>]`. Newer gateways honour the `profile`
 * param and answer with the bot's OWN cron store (echoed back as `scoped`),
 * in which case every job in the reply belongs to the bot. Older gateways
 * ignore the param and return the launch-profile store, so the tag filter
 * stays the fallback — and a stale `scoped` marker must never be trusted to
 * leak another profile's jobs.
 */

import { describe, expect, it } from 'vitest'

import { routineFilterHint, selectRoutineJobs } from './cron'
import type { RoutineJob } from './types'

const jobs: RoutineJob[] = [
  { job_id: '1', name: '[bot:ops] Morning' },
  { job_id: '2', name: '[bot:research] Digest' }
]

describe('selecting the jobs for the active bot', () => {
  it('filters a live list down to the current bot', () => {
    const view = selectRoutineJobs({ jobs }, null, [], 'ops')

    expect(view.live).toHaveLength(2)
    expect(view.jobs.map(job => job.job_id)).toEqual(['1'])
  })

  it('shows every job in a profile-scoped list, tagged or not', () => {
    const profileJobs: RoutineJob[] = [
      { job_id: 'legacy', name: 'ordinary profile cronjob' },
      { job_id: 'routine', name: '[bot:ops] Bot Mode routine' }
    ]

    const view = selectRoutineJobs({ jobs: profileJobs, scoped: 'ops' }, null, [], 'ops')

    expect(view.jobs.map(job => job.job_id)).toEqual(['legacy', 'routine'])
  })

  it('keeps tag filtering for an unmarked list (older gateways)', () => {
    const profileJobs: RoutineJob[] = [
      { job_id: 'legacy', name: 'ordinary launch-profile cronjob' },
      { job_id: 'routine', name: '[bot:ops] Bot Mode routine' }
    ]

    const view = selectRoutineJobs({ jobs: profileJobs }, null, [], 'ops')

    expect(view.jobs.map(job => job.job_id)).toEqual(['routine'])
  })

  it('cannot leak another profile\u2019s jobs through a stale scope marker', () => {
    const profileJobs: RoutineJob[] = [
      { job_id: 'research', name: 'ordinary research cronjob' },
      { job_id: 'routine', name: '[bot:ops] Bot Mode routine' }
    ]

    const view = selectRoutineJobs({ jobs: profileJobs, scoped: 'research' }, null, [], 'ops')

    expect(view.jobs.map(job => job.job_id)).toEqual(['routine'])
  })

  it('shows untagged legacy cronjobs only on the default bot', () => {
    const legacy: RoutineJob = { job_id: 'legacy', name: 'Existing reminder' }

    expect(selectRoutineJobs({ jobs: [legacy] }, null, [], 'default').jobs).toEqual([legacy])
    expect(selectRoutineJobs({ jobs: [legacy] }, null, [], 'ops').jobs).toEqual([])
  })
})

describe('a failed refresh keeps the last good list', () => {
  it('falls back to the previous jobs, still filtered', () => {
    const view = selectRoutineJobs(undefined, new Error('down'), jobs, 'ops')

    expect(view.live).toBeNull()
    expect(view.all).toHaveLength(2)
    expect(view.jobs).toHaveLength(1)
  })

  it('does not resurrect a list a successful refresh emptied', () => {
    const view = selectRoutineJobs({ jobs: [] }, null, jobs, 'ops')

    expect(view.live).toBeTruthy()
    expect(view.jobs).toHaveLength(0)
  })

  it('has nothing to show when the FIRST load fails', () => {
    const view = selectRoutineJobs(undefined, new Error('down'), [], 'ops')

    expect(view.all).toHaveLength(0)
    expect(view.jobs).toHaveLength(0)
  })
})

// A bot's cron store can hold jobs while none are namespaced for the active
// bot. Without a hint the user stares at the generic empty state with no clue
// that cronjobs are present but hidden by the tag filter.
describe('explaining an empty pane over a non-empty store', () => {
  it('stays quiet when the active bot already has tagged jobs', () => {
    const all: RoutineJob[] = [{ job_id: '1', name: '[bot:ops] Morning' }]

    expect(routineFilterHint(all, all)).toBeNull()
  })

  it('stays quiet when the store is genuinely empty', () => {
    expect(routineFilterHint([], [])).toBeNull()
    expect(routineFilterHint(undefined as unknown as RoutineJob[], [])).toBeNull()
  })

  it('explains the hidden jobs when the store has jobs but none match', () => {
    const all: RoutineJob[] = [
      { job_id: '2', name: '[bot:research] Digest' },
      { job_id: '3', name: 'untagged job' }
    ]

    expect(routineFilterHint(all, [])).toMatch(/tagged for this bot/)
  })
})
