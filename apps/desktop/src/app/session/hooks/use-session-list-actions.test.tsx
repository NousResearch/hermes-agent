import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useSessionListActions } from './use-session-list-actions'

// B4(b): every message.complete used to fire a full 4-request sidebar refresh
// (recents + cron sessions + cron jobs + messaging). scheduleSessionsRefresh
// must coalesce bursts into one trailing refresh, and the ancillary fan-out
// must be bounded to its own longer interval.

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobs: vi.fn(async () => []),
  listAllProfileSessions: vi.fn(async () => ({ sessions: [], total: 0, profile_totals: {} }))
}))

const { getCronJobs, listAllProfileSessions } = await import('@/hermes')

type Actions = ReturnType<typeof useSessionListActions>

let actions: Actions | null = null

function Harness() {
  actions = useSessionListActions({ profileScope: 'default' })

  return null
}

describe('useSessionListActions refresh coalescing', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.mocked(listAllProfileSessions).mockClear()
    vi.mocked(getCronJobs).mockClear()
    actions = null
    render(<Harness />)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('collapses a burst of scheduleSessionsRefresh calls into one trailing refresh', async () => {
    await act(async () => {
      actions!.scheduleSessionsRefresh()
      actions!.scheduleSessionsRefresh()
      actions!.scheduleSessionsRefresh()
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(listAllProfileSessions).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_500)
    })

    // One recents fetch + the first ancillary fan-out (cron sessions +
    // messaging slice), NOT one fetch per scheduled call.
    expect(vi.mocked(listAllProfileSessions).mock.calls).toHaveLength(3)
    expect(getCronJobs).toHaveBeenCalledTimes(1)
  })

  it('skips the ancillary fan-out for refreshes inside the throttle window', async () => {
    await act(async () => {
      actions!.scheduleSessionsRefresh()
      await vi.advanceTimersByTimeAsync(1_500)
    })

    expect(getCronJobs).toHaveBeenCalledTimes(1)
    const callsAfterFirst = vi.mocked(listAllProfileSessions).mock.calls.length

    // A second completed turn 5s later: recents refresh only.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
      actions!.scheduleSessionsRefresh()
      await vi.advanceTimersByTimeAsync(1_500)
    })

    expect(vi.mocked(listAllProfileSessions).mock.calls).toHaveLength(callsAfterFirst + 1)
    expect(getCronJobs).toHaveBeenCalledTimes(1)
  })

  it('re-runs the ancillary fan-out once the throttle interval has elapsed', async () => {
    await act(async () => {
      actions!.scheduleSessionsRefresh()
      await vi.advanceTimersByTimeAsync(1_500)
    })

    expect(getCronJobs).toHaveBeenCalledTimes(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(31_000)
      actions!.scheduleSessionsRefresh()
      await vi.advanceTimersByTimeAsync(1_500)
    })

    expect(getCronJobs).toHaveBeenCalledTimes(2)
  })
})
