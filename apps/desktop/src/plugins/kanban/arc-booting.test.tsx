import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'

import type { KanbanTask } from './types'
import { arcState } from './ui'

let now = 0

beforeEach(() => {
  vi.useFakeTimers()
  now = Math.floor(new Date('2026-10-20T03:27:00Z').getTime() / 1000)
  vi.setSystemTime(new Date(now * 1000))
})

afterEach(() => {
  vi.useRealTimers()
})

function task(over: Partial<KanbanTask> = {}): KanbanTask {
  return { id: 't1', title: 'retry', status: 'running', ...over }
}

// A retried run inherits tasks.last_heartbeat_at from the PREVIOUS run until
// the backend reset lands; that stale beat used to read as "the worker is
// dead" for the whole boot window of the fresh run.
describe('arcState — per-run heartbeat window', () => {
  it('shows booting (not "no heartbeat") for a re-claimed run whose only beat is from the previous run, inside the boot window', () => {
    const t = task({
      started_at: now - 7200,
      current_run_started_at: now - 30, // re-claimed 30s ago
      last_heartbeat_at: now - 900 // previous run's last beat: 15 min old
    })

    expect(arcState(t, '')).toBe('booting')
  })

  it('"no heartbeat" once the boot window passes without any beat for this run', () => {
    const t = task({
      started_at: now - 7200,
      current_run_started_at: now - 180,
      last_heartbeat_at: now - 900
    })

    expect(arcState(t, '')).toBe('stale')
  })

  it('backends that DO reset the claim (hb = NULL) stay in booting for a fresh claim', () => {
    const t = task({
      started_at: now - 40,
      current_run_started_at: now - 40,
      last_heartbeat_at: null
    })

    expect(arcState(t, '')).toBe('booting')
  })

  it('flips to running as soon as the current run has a fresh beat', () => {
    const t = task({
      started_at: now - 7200,
      current_run_started_at: now - 1200, // run older than the grace window…
      last_heartbeat_at: now - 20 // …but this run's own beat is 20s old
    })

    expect(arcState(t, '')).toBe('running')
  })

  it("flags stale when the current run's own beat goes quiet for 2+ min", () => {
    const t = task({
      started_at: now - 7200,
      current_run_started_at: now - 1200,
      last_heartbeat_at: now - 150 // this run's last beat: 2.5 min old
    })

    expect(arcState(t, '')).toBe('stale')
  })

  it('falls back to the task first start when the backend predates current_run_started_at', () => {
    const t = task({
      started_at: now - 30,
      last_heartbeat_at: null
    })

    expect(arcState(t, '')).toBe('booting')
  })

  it('stays stale for an old-backend retry with no run-start signal', () => {
    const t = task({
      last_heartbeat_at: null
    })

    expect(arcState(t, '')).toBe('stale')
  })
})