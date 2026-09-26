import { QueryClient } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { browserFailure, browserKeys, browserRefresh, createBoardBrowserReads } from './browser-api'

const source = { connectionId: 'synthetic', profile: 'worker', label: 'Synthetic' }

afterEach(() => vi.useRealTimers())

describe('bounded board browser reads', () => {
  it('pins queued sources, bounds concurrent reads and discards cancelled or timed-out work', async () => {
    vi.useFakeTimers()
    const release: Array<(value: unknown) => void> = []
    const rest = vi.fn(() => new Promise(resolve => release.push(resolve)))
    const reads = createBoardBrowserReads(rest as never)
    const controllers = Array.from({ length: 5 }, () => new AbortController())
    const pins = controllers.map((_, index) => ({ ...source, connectionId: `synthetic-${index}` }))
    const pending = pins.map((pin, index) => reads.boards(pin, controllers[index].signal))
    const cancelled = expect(pending[3]).rejects.toThrow('Read cancelled')
    const timeout = expect(pending[4]).rejects.toThrow('timed out')

    await vi.advanceTimersByTimeAsync(0)
    expect(rest).toHaveBeenCalledTimes(3)
    controllers[3].abort()
    pins[4].connectionId = 'changed-after-request'
    release[0]({ boards: [], current: '' })
    await pending[0]
    await vi.advanceTimersByTimeAsync(0)
    expect(rest).toHaveBeenCalledTimes(4)
    expect(rest.mock.calls[3]).toEqual(['/boards', {
      method: 'GET', source: { connectionId: 'synthetic-4', profile: 'worker' }, timeoutMs: 10_000
    }])
    release[1]({ boards: [], current: '' })
    release[2]({ boards: [], current: '' })
    await Promise.all([pending[1], pending[2], cancelled])
    await vi.advanceTimersByTimeAsync(15_000)
    await timeout
    release[3]({ boards: [{ slug: 'too-late' }], current: 'too-late' })
    expect(rest).toHaveBeenCalledTimes(4)
  })

  it('stops failed retries, distinguishes failures, and does not cache malformed success as an empty board', async () => {
    vi.useFakeTimers()
    const qc = new QueryClient()
    const rest = vi.fn().mockRejectedValue(new Error('Network unavailable'))
    const reads = createBoardBrowserReads(rest)
    const query = () => qc.fetchQuery({
      ...browserRefresh, queryKey: browserKeys.boards(source), queryFn: ({ signal }) => reads.boards(source, signal)
    })
    const failed = expect(query()).rejects.toThrow('Network unavailable')

    await vi.advanceTimersByTimeAsync(10_000)
    await failed
    expect(rest).toHaveBeenCalledTimes(3)
    expect(browserRefresh.refetchInterval({ state: { status: 'error' } })).toBe(false)
    expect(browserRefresh.refetchInterval({ state: { status: 'success' } })).toBe(30_000)

    for (const message of ['401: unauthorized', '403: denied', '404: not found', '405: not allowed', '501: unsupported']) {
      rest.mockRejectedValueOnce(new Error(message))
      await expect(query()).rejects.toThrow(message)
      expect(browserRefresh.retry(0, new Error(message))).toBe(false)
    }

    expect(browserFailure(new Error('403: denied'))).toBe('auth')
    expect(browserFailure(new Error('404: not found'))).toBe('unsupported')
    expect(browserFailure(new Error('Network unavailable'))).toBe('unavailable')
    rest.mockResolvedValue({})
    await expect(reads.boards(source, new AbortController().signal)).rejects.toThrow('Invalid board inventory')
    expect(qc.getQueryData(browserKeys.boards(source))).toBeUndefined()
    qc.clear()
  })
})
