import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $cronRunReadOnlyVerdicts, isStoredTranscriptReadOnly } from '@/store/read-only-transcript'

import { isResumableCronRun, openCronRun, reconcileCronRunVerdicts, refreshCronRunWriteGate } from './open-cron-run'

const NOW_MS = 1_800_000_000_000
const NOW_S = NOW_MS / 1000

// Backend row as the runs endpoint produces it: `is_active` is the 300s
// activity window, `scheduler_owned` the live-execution ownership (#88443).
const run = (
  over: Partial<{
    ended_at: null | number
    id: string
    is_active: boolean
    last_active: number
    scheduler_owned: boolean
    cron_finalized: boolean
  }> = {}
) => ({
  ended_at: null as null | number,
  id: 'cron_job-1_20260929_120000',
  is_active: false,
  last_active: NOW_S - 1800,
  ...over
})

beforeEach(() => {
  $cronRunReadOnlyVerdicts.set(new Map())
})

describe('isResumableCronRun', () => {
  it('keeps a scheduler-owned run live through a long tool call past the activity window', () => {
    // The review's blocker: 30 min without a tick → is_active=false, yet the
    // scheduler still runs it. Ownership, not freshness, decides.
    expect(isResumableCronRun(run({ is_active: false, scheduler_owned: true }), NOW_MS)).toBe(true)
  })

  it('treats a never-closed run the scheduler does not own as a zombie, even if it ticked recently', () => {
    expect(isResumableCronRun(run({ is_active: true, last_active: NOW_S - 5, scheduler_owned: false }), NOW_MS)).toBe(
      false
    )
  })

  it('treats a properly closed run as resumable', () => {
    expect(isResumableCronRun(run({ ended_at: NOW_S - 60, scheduler_owned: false }), NOW_MS)).toBe(true)
  })

  it('treats a resumed scheduler-finalized run as resumable', () => {
    expect(isResumableCronRun(run({ cron_finalized: true, scheduler_owned: false }), NOW_MS)).toBe(true)
    expect(isResumableCronRun(run({ cron_finalized: false, scheduler_owned: false }), NOW_MS)).toBe(false)
  })

  it('falls back to the activity window against a backend without scheduler_owned', () => {
    expect(isResumableCronRun(run({ is_active: true }), NOW_MS)).toBe(true)
    expect(isResumableCronRun(run({ is_active: false }), NOW_MS)).toBe(false)
    // Session-detail rows on an older backend carry neither flag: same formula over last_active.
    expect(isResumableCronRun({ ended_at: null, last_active: NOW_S - 10 }, NOW_MS)).toBe(true)
    expect(isResumableCronRun({ ended_at: null, last_active: NOW_S - 600 }, NOW_MS)).toBe(false)
  })
})

describe('openCronRun', () => {
  it('opens a live / closed run writable', () => {
    const open = vi.fn()
    const live = run({ id: 'cron_live_20260929_120000', scheduler_owned: true })
    const closed = run({ ended_at: NOW_S - 60, id: 'cron_closed_20260929_120000' })

    openCronRun(live, open)
    openCronRun(closed, open)

    expect(open.mock.calls).toEqual([
      [live.id, live],
      [closed.id, closed]
    ])
    expect(isStoredTranscriptReadOnly(live.id)).toBe(false)
    expect(isStoredTranscriptReadOnly(closed.id)).toBe(false)
  })

  it('records a zombie read-only before the route flips', () => {
    const zombie = run({ scheduler_owned: false })

    const open = vi.fn(() => {
      expect(isStoredTranscriptReadOnly(zombie.id)).toBe(true)
    })

    openCronRun(zombie, open)

    expect(open).toHaveBeenCalledWith(zombie.id, zombie)
  })
})

describe('the verdict re-evaluates instead of latching', () => {
  it('a later poll showing the run live or closed makes it writable again, and back', () => {
    const id = 'cron_job-1_20260929_120000'

    openCronRun(run({ id, is_active: false }), vi.fn()) // older backend, stale tick
    expect(isStoredTranscriptReadOnly(id)).toBe(true)

    reconcileCronRunVerdicts([run({ id, is_active: true })]) // it ticked
    expect(isStoredTranscriptReadOnly(id)).toBe(false)

    reconcileCronRunVerdicts([run({ id, scheduler_owned: false })]) // its process died
    expect(isStoredTranscriptReadOnly(id)).toBe(true)

    reconcileCronRunVerdicts([run({ ended_at: NOW_S, id })]) // it closed
    expect(isStoredTranscriptReadOnly(id)).toBe(false)
  })

  it('a poll never gates runs the user did not open', () => {
    reconcileCronRunVerdicts([run({ id: 'cron_other_20260929_120000', scheduler_owned: false })])

    expect(isStoredTranscriptReadOnly('cron_other_20260929_120000')).toBe(false)
  })

  it('the pre-send refresh reads the authoritative row and clears a stale zombie verdict', async () => {
    const id = 'cron_job-1_20260929_120000'
    openCronRun(run({ id, scheduler_owned: false }), vi.fn())

    const fetchRow = vi.fn(async () => ({ ended_at: NOW_S, last_active: NOW_S, source: 'cron' }))

    expect(await refreshCronRunWriteGate(id, fetchRow)).toBe(false)
    expect(fetchRow).toHaveBeenCalledWith(id)
    expect(isStoredTranscriptReadOnly(id)).toBe(false)
  })
})

describe('refreshCronRunWriteGate — restored tab after a restart (no verdict yet)', () => {
  const id = 'cron_job-1_20260929_120000'

  it('gates a restored zombie run by its cron run id', async () => {
    const fetchRow = vi.fn(async () => ({ ended_at: null, last_active: 0, scheduler_owned: false, source: 'cron' }))

    expect(await refreshCronRunWriteGate(id, fetchRow)).toBe(true)
    expect(isStoredTranscriptReadOnly(id)).toBe(true)
  })

  it('leaves a restored live run writable', async () => {
    const fetchRow = vi.fn(async () => ({ ended_at: null, last_active: 0, scheduler_owned: true, source: 'cron' }))

    expect(await refreshCronRunWriteGate(id, fetchRow)).toBe(false)
  })

  it('fails closed when an unknown cron run cannot be read, but keeps a known verdict', async () => {
    const failing = vi.fn(async () => {
      throw new Error('offline')
    })

    expect(await refreshCronRunWriteGate(id, failing)).toBe(true)

    openCronRun(run({ id, scheduler_owned: true }), vi.fn())
    expect(await refreshCronRunWriteGate(id, failing)).toBe(false)
  })

  it('never fetches for an ordinary session', async () => {
    const fetchRow = vi.fn()

    expect(await refreshCronRunWriteGate('20260929_120000_abc123', fetchRow)).toBe(false)
    expect(await refreshCronRunWriteGate(null, fetchRow)).toBe(false)
    expect(fetchRow).not.toHaveBeenCalled()
  })
})
