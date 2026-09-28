import { afterEach, describe, expect, it, vi } from 'vitest'

import type { AgentOverview, OverviewSource } from '../../electron/agent-overview'

import { createOverviewCache, mergeOverview } from './agent-overview'

export function source(overrides: Partial<OverviewSource> = {}): OverviewSource {
  return {
    connectionId: 'local',
    label: 'This computer',
    kind: 'local',
    state: 'ready',
    sessions: [],
    canonical: [],
    live: [],
    profiles: [],
    total: 0,
    offset: 0,
    complete: true,
    liveCoverage: 'process',
    errors: [],
    ...overrides
  }
}

export const snapshot = (...sources: OverviewSource[]): AgentOverview => ({ sources, fetchedAt: 1 })

describe('visible snapshot cache', () => {
  it('manual retry requests fresh history and a hung bridge has a bounded deadline', async () => {
    vi.useFakeTimers()
    const fetch = vi.fn(async (_options?: { force?: boolean }) => snapshot(source()))
    const cache = createOverviewCache(fetch)
    await cache.refresh({ force: true })
    expect(fetch).toHaveBeenLastCalledWith({ force: true })
    fetch.mockImplementation(() => new Promise(() => {}))
    const pending = cache.refresh()
    await vi.advanceTimersByTimeAsync(20_000)
    await pending
    expect(cache.state.get().error).toMatch(/timed out/i)
  })
  afterEach(() => vi.useRealTimers())
  it('only schedules visible reads, fences hidden responses, and self-heals after the fast retry budget', async () => {
    vi.useFakeTimers()
    let finish!: (value: AgentOverview) => void

    const fetch = vi.fn(
      () =>
        new Promise<AgentOverview>(resolve => {
          finish = resolve
        })
    )

    const cache = createOverviewCache(fetch)
    const hide = cache.watch()
    expect(fetch).toHaveBeenCalledTimes(1)
    hide()
    finish(snapshot(source({ sessions: [{ id: 'late', profile: 'default' }] })))
    await vi.advanceTimersByTimeAsync(20_000)
    expect(cache.state.get().data).toBeUndefined()
    expect(fetch).toHaveBeenCalledTimes(1)
    fetch.mockImplementation(async () => snapshot(source({ sessions: [{ id: 'kept', profile: 'default' }] })))
    const close = cache.watch()
    await vi.advanceTimersByTimeAsync(0)
    expect(cache.state.get().data?.rows[0]?.id).toBe('kept')
    fetch.mockRejectedValue(new Error('offline'))
    await vi.advanceTimersByTimeAsync(60_000)
    expect(fetch).toHaveBeenCalledTimes(5) // one cancelled, one success, three bounded failures
    expect(cache.state.get().data?.rows[0]?.stale).toBe(true)
    expect(cache.state.get().error).toBe('offline')
    fetch.mockResolvedValue(snapshot(source()))
    await vi.advanceTimersByTimeAsync(60_000)
    expect(cache.state.get().error).toBeNull()
    expect(fetch.mock.calls.length).toBeGreaterThan(5)
    close()
    const calls = fetch.mock.calls.length
    await vi.advanceTimersByTimeAsync(60_000)
    expect(fetch).toHaveBeenCalledTimes(calls)
    expect(vi.getTimerCount()).toBe(0)
  })

  it.each([
    ['rejection', (fetch: ReturnType<typeof vi.fn>) => fetch.mockRejectedValue(new Error('bridge down')), 0],
    [
      'deadline expiry',
      (fetch: ReturnType<typeof vi.fn>) => fetch.mockImplementation(() => new Promise(() => {})),
      20_000
    ]
  ] as const)(
    'stops claiming ready coverage after a bridge %s and restores it only after a successful read',
    async (_mode, fail, elapse) => {
      vi.useFakeTimers()

      const healthy = snapshot(
        source({ sessions: [{ id: 'one', profile: 'default' }], total: 7, offset: 7 }),
        source({ connectionId: 'remote', label: 'Remote', kind: 'remote', state: 'on-demand', complete: false })
      )

      const fetch = vi.fn(async (_options?: { force?: boolean }) => healthy)
      const cache = createOverviewCache(fetch)
      await cache.refresh()
      expect(cache.state.get().data?.sources[0]).toMatchObject({ state: 'ready', complete: true })

      fail(fetch)
      const failed = cache.refresh()
      await vi.advanceTimersByTimeAsync(elapse)
      await failed
      const outage = cache.state.get()
      expect(outage.error).toBeTruthy()
      expect(outage.data?.rows[0]).toMatchObject({ id: 'one', stale: true })
      // Last-known counts survive; the freshness claim does not.
      expect(outage.data?.sources[0]).toMatchObject({ connectionId: 'local', total: 7, offset: 7, complete: false })
      expect(outage.data?.sources[0]?.state).not.toBe('ready')
      expect(outage.data?.sources.every(source => !(source.state === 'ready' && source.complete))).toBe(true)
      expect(outage.data?.sources[1]?.state).toBe('on-demand')

      fetch.mockResolvedValue({ ...healthy, fetchedAt: 2 })
      await cache.refresh()
      expect(cache.state.get().error).toBeNull()
      expect(cache.state.get().data?.sources[0]).toMatchObject({ state: 'ready', complete: true, total: 7, offset: 7 })
      expect(cache.state.get().data?.rows[0]).toMatchObject({ id: 'one', stale: false })
    }
  )

  it('clears a hidden hung read deadline without accepting its late rejection or disturbing a new watch', async () => {
    vi.useFakeTimers()
    let reject!: (error: Error) => void

    const fetch = vi.fn(
      () =>
        new Promise<AgentOverview>((_, fail) => {
          reject = fail
        })
    )

    const cache = createOverviewCache(fetch)
    const hide = cache.watch()
    hide()
    await vi.advanceTimersByTimeAsync(0)
    expect(vi.getTimerCount()).toBe(0)
    fetch.mockResolvedValue(snapshot(source()))
    const close = cache.watch()
    await vi.advanceTimersByTimeAsync(0)
    const current = cache.state.get()
    reject(new Error('late rejection'))
    await vi.advanceTimersByTimeAsync(0)
    expect(cache.state.get()).toBe(current)
    expect(vi.getTimerCount()).toBe(1)
    close()
    expect(vi.getTimerCount()).toBe(0)
  })
})

describe('overview reconciliation', () => {
  it('does not attribute historical model/provider to the current profile configuration', () => {
    const data = mergeOverview(
      undefined,
      snapshot(
        source({
          sessions: [{ id: 'old', profile: 'coder' }],
          profiles: [{ name: 'coder', model: 'current-model', provider: 'current-provider' }]
        })
      )
    )

    expect(data.rows[0]).toMatchObject({ model: '', provider: '' })
  })
  it('retains stale rows after source failure without claiming a stale turn is live, and reuses unchanged references', () => {
    const data = snapshot(
      source({
        sessions: [{ id: 'one', profile: 'default' }],
        live: [{ id: 'one', profile: 'default', status: 'working' }],
        total: 1,
        offset: 1
      })
    )

    const first = mergeOverview(undefined, data)
    const unchanged = mergeOverview(first, { ...data, fetchedAt: 2 })
    expect(unchanged.rows).toBe(first.rows)
    expect(unchanged.sources).toBe(first.sources)
    const failed = mergeOverview(first, snapshot(source({ state: 'offline', error: 'read-failed', complete: false })))
    expect(failed.rows).toHaveLength(1)
    expect(failed.rows[0]).toMatchObject({ id: 'one', stale: true, attention: 'idle', runtimeId: undefined })
    expect(failed.sources[0]).toMatchObject({ state: 'offline', total: 1, offset: 1 })
    expect(mergeOverview(failed, data).rows[0]?.stale).toBe(false)
  })
  it('keeps same-id owners separate while collapsing canonical compression aliases and independent live rows', () => {
    const state = mergeOverview(
      undefined,
      snapshot(
        source({
          sessions: [{ id: 'tip', profile: 'coder', title: 'Bot Chat', unread: true }],
          canonical: [{ id: 'root', resolved_id: 'tip', profile: 'coder', title: 'Bot Chat' }],
          live: [
            {
              id: 'tip',
              runtime_id: 'runtime',
              profile: 'coder',
              status: 'working',
              model: 'model',
              provider: 'provider'
            },
            { id: 'old', runtime_id: 'old-run', profile: 'coder', status: 'waiting' }
          ],
          profiles: [
            {
              name: 'coder',
              display_name: 'Builder',
              description: 'Builds tools',
              model: 'model',
              provider: 'provider'
            }
          ],
          total: 1,
          offset: 1
        }),
        source({ connectionId: 'remote', sessions: [{ id: 'tip', profile: 'coder' }], total: 1, offset: 1 })
      )
    )

    expect(state.rows).toHaveLength(3)
    const bot = state.rows.find(row => row.canonical)!
    expect(bot).toMatchObject({
      key: JSON.stringify(['local', 'coder', 'root']),
      id: 'root',
      resolvedId: 'tip',
      runtimeId: 'runtime',
      attention: 'working',
      owner: 'Builder',
      description: 'Builds tools',
      model: 'model',
      provider: 'provider'
    })
    expect(state.rows.find(row => row.id === 'old')?.attention).toBe('needs-you')
    expect(state.rows.find(row => row.connectionId === 'remote')?.attention).toBe('idle')
  })
})
