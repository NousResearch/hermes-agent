import { describe, expect, it } from 'vitest'

import type { CronJob, SessionInfo } from '@/hermes'

import {
  deliveryTargets,
  filterRoutines,
  groupRoutines,
  neighbourInList,
  remembersLastRun,
  routineGroup,
  routineIssue,
  routinesInPanelOrder,
  routineStatus,
  routineTitle,
  runChip,
  runLimit,
  toggleDelivery,
  withRememberLastRun
} from './routine-model'

const NOW = Date.parse('2026-11-12T12:00:00Z')

const job = (patch: Partial<CronJob>): CronJob =>
  ({
    enabled: true,
    id: patch.name ?? 'job',
    name: 'Routine',
    next_run_at: '2026-11-13T09:00:00Z',
    profile: 'default',
    state: 'scheduled',
    ...patch
  }) as CronJob

const failedRun = { id: 'x1', status: 'failed' }

describe('needs attention', () => {
  it('flags a failed last run, an undelivered one and an overdue slot, in that order', () => {
    expect(routineIssue(job({ latest_execution: failedRun }), NOW)).toBe('failed')
    expect(routineIssue(job({ last_status: 'error' }), NOW)).toBe('failed')
    expect(routineIssue(job({ latest_execution: { delivery_outcome: 'failed', id: 'x', status: 'completed' } }), NOW)).toBe(
      'delivery'
    )
    expect(routineIssue(job({ next_run_at: '2026-11-12T08:00:00Z' }), NOW)).toBe('overdue')
    expect(routineIssue(job({}), NOW)).toBeNull()
  })

  it('lets a run in flight supersede the previous failure, as the chat sidebar does', () => {
    const running = job({ latest_execution: { id: 'x2', status: 'running' }, last_error: 'boom' })

    expect(routineIssue(running, NOW)).toBeNull()
    expect(routineStatus(running, true, NOW)).toBe('running')
  })

  it('never calls a paused routine overdue', () => {
    expect(routineIssue(job({ next_run_at: '2026-11-01T00:00:00Z', state: 'paused' }), NOW)).toBeNull()
  })
})

describe('grouping by status', () => {
  const jobs = [
    job({ name: 'Weekly review' }),
    job({ latest_execution: failedRun, name: 'Flight price monitor' }),
    job({ enabled: false, name: 'Candidate follow-ups', state: 'paused' }),
    job({ enabled: false, name: 'Switched off', state: null }),
    job({ name: 'Done once', state: 'completed' }),
    job({ name: 'Daily brief' })
  ]

  it('puts every routine in exactly one group, problems first, each sorted by name', () => {
    const groups = groupRoutines(jobs, NOW)

    expect(groups.attention.map(routineTitle)).toEqual(['Flight price monitor'])
    expect(groups.scheduled.map(routineTitle)).toEqual(['Daily brief', 'Weekly review'])
    expect(groups.paused.map(routineTitle)).toEqual(['Candidate follow-ups', 'Switched off'])
    expect(groups.finished.map(routineTitle)).toEqual(['Done once'])
    expect(Object.values(groups).flat()).toHaveLength(jobs.length)
  })

  it('files a failing paused routine under Needs attention, not Paused', () => {
    expect(routineGroup(job({ latest_execution: failedRun, state: 'paused' }), NOW)).toBe('attention')
  })

  it('searches every word in name, instruction or agent', () => {
    const label = (profile: string) => (profile === 'ops' ? 'PED operations' : profile)
    const list = [...jobs, job({ name: 'Inbox sweep', profile: 'ops', prompt: 'Summarize urgent mail' })]

    expect(filterRoutines(list, 'urgent mail', label).map(routineTitle)).toEqual(['Inbox sweep'])
    expect(filterRoutines(list, 'operations', label).map(routineTitle)).toEqual(['Inbox sweep'])
  })

  it('lists routines in the panel’s order, and lands on the next one when one leaves', () => {
    const order = routinesInPanelOrder(jobs, NOW).map(routineTitle)

    expect(order).toEqual(['Flight price monitor', 'Daily brief', 'Weekly review', 'Candidate follow-ups', 'Switched off', 'Done once'])
    expect(neighbourInList(order, title => title === 'Daily brief')).toBe('Weekly review')
    expect(neighbourInList(order, title => title === 'Done once')).toBe('Switched off')
    expect(neighbourInList(['Only'], title => title === 'Only')).toBeNull()
  })
})

describe('what the routine is called', () => {
  it("drops the Bots plugin's [bot:name] namespace from the title", () => {
    expect(routineTitle(job({ name: '[bot:ops] Standup digest' }))).toBe('Standup digest')
  })
})

describe('send results to', () => {
  it('reads "local" (save only) as no destination, and an empty selection back as local', () => {
    expect(deliveryTargets('local')).toEqual([])
    expect(deliveryTargets('local,telegram')).toEqual(['telegram'])
    expect(toggleDelivery('local', 'telegram', true)).toBe('telegram')
    expect(toggleDelivery('telegram,bot-chat:ops', 'telegram', false)).toBe('bot-chat:ops')
    expect(toggleDelivery('telegram', 'telegram', false)).toBe('local')
  })
})

describe('remember last run', () => {
  it('is the job feeding its own output back (context_from "self"), keeping any other chain', () => {
    expect(remembersLastRun(job({ context_from: ['self'] }))).toBe(true)
    expect(remembersLastRun(job({ context_from: ['other-job'] }))).toBe(false)
    expect(withRememberLastRun(['other-job'], true)).toEqual(['other-job', 'self'])
    expect(withRememberLastRun(['other-job', 'self'], false)).toEqual(['other-job'])
  })
})

describe('run limit', () => {
  it('is the job’s lifetime run budget (repeat.times), not retries within a run; absent or ≤0 is unlimited', () => {
    expect(runLimit(job({ repeat: { completed: 2, times: 5 } }))).toBe(5)
    expect(runLimit(job({ repeat: { completed: 0, times: null } }))).toBeNull()
    expect(runLimit(job({ repeat: { completed: 0, times: 0 } }))).toBeNull()
    expect(runLimit(job({}))).toBeNull()
  })
})

describe('run chips', () => {
  const run = (patch: Partial<SessionInfo>) => ({ ended_at: 10, id: 'r', source: 'cron', ...patch }) as SessionInfo

  it('shows a live run as running and the newest attempt’s recorded outcome, never a guessed success', () => {
    const subject = job({ latest_execution: { delivery_outcome: 'failed', id: 'exec-2', status: 'completed' } })

    expect(runChip(run({ ended_at: null, scheduler_owned: true }), subject)).toBe('running')
    expect(runChip(run({ cron_execution_id: 'exec-2' }), subject)).toBe('delivery')
    expect(runChip(run({ cron_execution_id: 'exec-2' }), job({ latest_execution: { id: 'exec-2', status: 'completed' } }))).toBe(
      'succeeded'
    )
    expect(runChip(run({ cron_execution_id: 'exec-1' }), subject)).toBeNull()
    expect(runChip(run({}), job({}))).toBeNull()
  })

  it('reads the status the backend titled a session-less run with', () => {
    const output = run({ source: 'cron_output', title: 'FAILED · Script exited with code 1\nstderr: …' })

    expect(runChip(output, job({}))).toBe('failed')
    expect(runChip(run({ source: 'cron_output', title: 'CANCELLED · Stopped' }), job({}))).toBe('stopped')
    expect(runChip(run({ source: 'cron_output', title: 'COMPLETED · Script-only run' }), job({}))).toBe('succeeded')
  })
})
