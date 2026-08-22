import { describe, expect, it, vi } from 'vitest'

import type { CronJob, SessionInfo } from '@/hermes'

import { createCronCompletionNotifier } from './cron-completion-notifications'

const run = { id: 'cron_daily_1720000000' } as SessionInfo

function job(updates: Partial<CronJob> = {}): CronJob {
  return {
    deliver: 'local',
    enabled: true,
    id: 'daily',
    last_run_at: null,
    name: 'Daily briefing',
    ...updates
  }
}

describe('cron completion notifications', () => {
  it('notifies once when a locally delivered job completes after hydration', async () => {
    const getRuns = vi.fn().mockResolvedValue([run])
    const notify = vi.fn()
    const observer = createCronCompletionNotifier({ getRuns, notify })

    await observer.observe('local\0default', [job()])
    await observer.observe('local\0default', [job({ last_run_at: '2026-08-20T10:00:00+00:00' })])
    await observer.observe('local\0default', [job({ last_run_at: '2026-08-20T10:00:00+00:00' })])

    expect(getRuns).toHaveBeenCalledTimes(1)
    expect(getRuns).toHaveBeenCalledWith('daily', 5)
    expect(notify).toHaveBeenCalledOnce()
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({
        global: true,
        kind: 'backgroundDone',
        sessionId: run.id
      })
    )
  })

  it('uses the first accepted snapshot as a silent hydration baseline', async () => {
    const getRuns = vi.fn().mockResolvedValue([run])
    const notify = vi.fn()
    const observer = createCronCompletionNotifier({ getRuns, notify })

    await observer.observe('local\0default', [job({ last_run_at: '2026-08-20T10:00:00+00:00' })])

    expect(getRuns).not.toHaveBeenCalled()
    expect(notify).not.toHaveBeenCalled()
  })

  it('silently re-baselines when the connection or profile scope changes', async () => {
    const getRuns = vi.fn().mockResolvedValue([run])
    const notify = vi.fn()
    const observer = createCronCompletionNotifier({ getRuns, notify })

    await observer.observe('local\0default', [job()])
    await observer.observe('remote\0work', [job({ last_run_at: '2026-08-20T10:00:00+00:00' })])

    expect(getRuns).not.toHaveBeenCalled()
    expect(notify).not.toHaveBeenCalled()
  })

  it('does not notify for a remote-only delivery target', async () => {
    const getRuns = vi.fn().mockResolvedValue([run])
    const notify = vi.fn()
    const observer = createCronCompletionNotifier({ getRuns, notify })

    await observer.observe('local\0default', [job({ deliver: 'telegram' })])
    await observer.observe('local\0default', [
      job({ deliver: 'telegram', last_run_at: '2026-08-20T10:00:00+00:00' })
    ])

    expect(getRuns).not.toHaveBeenCalled()
    expect(notify).not.toHaveBeenCalled()
  })

  it('notifies once for a multi-target job that includes local delivery', async () => {
    const getRuns = vi.fn().mockResolvedValue([run])
    const notify = vi.fn()
    const observer = createCronCompletionNotifier({ getRuns, notify })

    await observer.observe('local\0default', [job({ deliver: 'local,telegram' })])
    await observer.observe('local\0default', [
      job({ deliver: 'local,telegram', last_run_at: '2026-08-20T10:00:00+00:00' })
    ])

    expect(notify).toHaveBeenCalledOnce()
  })

  it('labels failed runs and retains the exact completed run session target', async () => {
    const getRuns = vi.fn().mockResolvedValue([run])
    const notify = vi.fn()
    const observer = createCronCompletionNotifier({ getRuns, notify })

    await observer.observe('local\0default', [job()])
    await observer.observe('local\0default', [
      job({
        last_error: 'provider timed out',
        last_run_at: '2026-08-20T10:00:00+00:00',
        last_status: 'error'
      })
    ])

    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({
        body: 'provider timed out',
        sessionId: run.id,
        title: expect.stringMatching(/failed/i)
      })
    )
  })

  it('matches the notification to the run nearest the accepted completion timestamp', async () => {
    const completedRun = { ended_at: 1_777_000_000, id: 'cron_daily_completed' } as SessionInfo
    const newerRun = { ended_at: 1_777_000_120, id: 'cron_daily_newer' } as SessionInfo
    const getRuns = vi.fn().mockResolvedValue([newerRun, completedRun])
    const notify = vi.fn()
    const observer = createCronCompletionNotifier({ getRuns, notify })
    const completedAt = new Date(completedRun.ended_at! * 1000).toISOString()

    await observer.observe('local\0default', [job()])
    await observer.observe('local\0default', [job({ last_run_at: completedAt })])

    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ sessionId: completedRun.id }))
  })

  it('keeps the dedupe watermark monotonic when overlapping completions resolve out of order', async () => {
    let resolveFirst!: (runs: SessionInfo[]) => void
    let resolveSecond!: (runs: SessionInfo[]) => void

    const firstRuns = new Promise<SessionInfo[]>(resolve => {
      resolveFirst = resolve
    })

    const secondRuns = new Promise<SessionInfo[]>(resolve => {
      resolveSecond = resolve
    })

    const firstRun = { ended_at: 1_777_000_000, id: 'cron_daily_first' } as SessionInfo
    const secondRun = { ended_at: 1_777_000_120, id: 'cron_daily_second' } as SessionInfo
    const firstAt = new Date(firstRun.ended_at! * 1000).toISOString()
    const secondAt = new Date(secondRun.ended_at! * 1000).toISOString()
    const getRuns = vi.fn().mockReturnValueOnce(firstRuns).mockReturnValueOnce(secondRuns)
    const notify = vi.fn()
    const observer = createCronCompletionNotifier({ getRuns, notify })

    await observer.observe('local\0default', [job()])
    const firstObservation = observer.observe('local\0default', [job({ last_run_at: firstAt })])
    const secondObservation = observer.observe('local\0default', [job({ last_run_at: secondAt })])

    resolveSecond([secondRun, firstRun])
    await secondObservation
    resolveFirst([secondRun, firstRun])
    await firstObservation
    await observer.observe('local\0default', [job({ last_run_at: secondAt })])

    expect(getRuns).toHaveBeenCalledTimes(2)
    expect(notify).toHaveBeenCalledTimes(2)
    expect(notify.mock.calls.map(([input]) => input.sessionId)).toEqual([secondRun.id, firstRun.id])
  })

  it('retries run lookup after the run session is not yet visible', async () => {
    const getRuns = vi.fn().mockResolvedValueOnce([]).mockResolvedValueOnce([run])
    const notify = vi.fn()
    const observer = createCronCompletionNotifier({ getRuns, notify })
    const completed = job({ last_run_at: '2026-08-20T10:00:00+00:00' })

    await observer.observe('local\0default', [job()])
    await observer.observe('local\0default', [completed])
    await observer.observe('local\0default', [completed])

    expect(getRuns).toHaveBeenCalledTimes(2)
    expect(notify).toHaveBeenCalledOnce()
  })
})
