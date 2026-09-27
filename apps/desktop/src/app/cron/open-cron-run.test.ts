import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $readOnlyCronRuns, isCronRunReadOnly, isStoredTranscriptReadOnly } from '@/store/read-only-transcript'

import { isResumableCronRun, openCronRun } from './open-cron-run'

const run = (over: Partial<{ ended_at: null | number; id: string; is_active: boolean }> = {}) => ({
  ended_at: 1_700_000_600 as null | number,
  id: 'cron_job-1_1700000000',
  is_active: false,
  ...over
})

beforeEach(() => {
  $readOnlyCronRuns.set(new Set())
})

describe('isResumableCronRun', () => {
  it('treats a live run as resumable (the scheduler agent still owns it)', () => {
    expect(isResumableCronRun(run({ ended_at: null, is_active: true }))).toBe(true)
  })

  it('treats a properly closed run as resumable', () => {
    expect(isResumableCronRun(run({ ended_at: 1_700_000_600, is_active: false }))).toBe(true)
  })

  it('treats a never-closed, not-live run as a zombie (NOT resumable)', () => {
    // The incident shape: end_session never ran (watchdog kill / crash), so
    // ended_at is NULL while no agent owns the session any more.
    expect(isResumableCronRun(run({ ended_at: null, is_active: false }))).toBe(false)
  })

  it('fails SAFE when an older backend omits is_active', () => {
    expect(isResumableCronRun({ ended_at: null, is_active: undefined as never })).toBe(false)
    expect(isResumableCronRun({ ended_at: 1_700_000_600, is_active: undefined as never })).toBe(true)
  })
})

describe('openCronRun', () => {
  it('opens a live / completed run without latching it read-only', () => {
    const open = vi.fn()

    const live = run({ ended_at: null, is_active: true, id: 'live-run' })
    const closed = run({ ended_at: 1_700_000_600, id: 'closed-run' })

    openCronRun(live, open)
    openCronRun(closed, open)

    expect(open.mock.calls).toEqual([
      ['live-run', live],
      ['closed-run', closed]
    ])
    expect(isStoredTranscriptReadOnly('live-run')).toBe(false)
    expect(isStoredTranscriptReadOnly('closed-run')).toBe(false)
  })

  it('opens a zombie run but latches it read-only before the route flips', () => {
    const open = vi.fn(() => {
      // The route has flipped by the time the open runs: the latch must already
      // be in place, or the first send lands inside the cron session.
      expect(isStoredTranscriptReadOnly('zombie-run')).toBe(true)
    })

    const zombie = run({ ended_at: null, is_active: false, id: 'zombie-run' })

    openCronRun(zombie, open)

    expect(open).toHaveBeenCalledWith('zombie-run', zombie)
    expect(isCronRunReadOnly('zombie-run')).toBe(true)
  })
})
