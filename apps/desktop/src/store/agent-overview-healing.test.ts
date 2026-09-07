import { afterEach, describe, expect, it, vi } from 'vitest'

import { createAgentOverviewReader, gatherOverviewBackends } from '../../electron/agent-overview'

import { createOverviewCache } from './agent-overview'

const inventory = (id: string) => ({
  sessions: [{ id, profile: 'coder' }],
  total: 1,
  live: [{ id, profile: 'coder', runtime_id: `runtime-${id}`, status: 'working' }],
  live_coverage: 'process'
})

describe('collector / renderer source isolation', () => {
  afterEach(() => vi.useRealTimers())

  it.each(['descriptors', 'pagination', 'connect', 'legacy', 'parked'] as const)(
    'publishes healthy coverage before the bridge deadline despite slow %s, fencing late reads',
    async mode => {
      vi.useFakeTimers()
      const read = createAgentOverviewReader<string>()
      const late: Array<{ resolve: (value: unknown) => void; reject: (error: Error) => void }> = []
      let failing = false
      const wait = () => new Promise<unknown>((resolve, reject) => late.push({ resolve, reject }))
      const connect = vi.fn(async () => (await wait()) as string[])
      const discoverParked = vi.fn(async () => (await wait()) as Array<{ name: string }>)

      const fetch = vi.fn(async (descriptor: string, path: string): Promise<unknown> => {
        if (descriptor === 'healthy') {
          return inventory('healthy')
        }

        if (!failing) {
          return inventory('last-known')
        }

        if (mode === 'pagination' && path.endsWith('offset=0')) {
          return { ...inventory('partial'), total: 2 }
        }

        if (mode === 'legacy' && path.includes('agent-overview')) {
          throw Object.assign(new Error('missing'), { statusCode: 404 })
        }

        if (mode === 'descriptors') {
          return new Promise((_, reject) => setTimeout(() => reject(new Error('offline')), 8_000))
        }

        return wait()
      })

      const cache = createOverviewCache(async control => {
        // Exercise the actual main-process gathering seam, including its budget.
        const pooled = await gatherOverviewBackends(
          [Promise.resolve('healthy'), Promise.resolve('slow'), new Promise<string>(() => {})],
          descriptor => descriptor
        )

        if (failing && (mode === 'connect' || mode === 'parked')) {
          pooled.delete('slow')
        }

        if (mode === 'descriptors') {
          pooled.set('slow', ['slow', 'slow-2', 'slow-3'])
        }

        return read(
          {
            sources: [
              { id: 'slow', label: 'Slow', kind: mode === 'connect' ? 'remote' : 'ssh' },
              { id: 'healthy', label: 'Healthy', kind: 'ssh' }
            ],
            pooled,
            connect,
            discoverParked,
            fetch
          },
          control
        )
      })

      const seed = cache.refresh()
      await vi.advanceTimersByTimeAsync(2_000)
      await seed
      failing = true
      const refresh = cache.refresh({ force: true })
      await vi.advanceTimersByTimeAsync(19_000)
      expect(cache.state.get().error).toBeNull()
      expect(cache.state.get().data?.sources.find(source => source.connectionId === 'slow')?.complete).toBe(false)
      expect(cache.state.get().data?.rows.find(row => row.id === 'healthy')).toMatchObject({ stale: false })
      expect(cache.state.get().data?.rows.find(row => row.id === 'last-known')).toMatchObject({
        stale: true,
        runtimeId: undefined,
        attention: 'idle'
      })

      if (mode === 'pagination') {
        expect(cache.state.get().data?.rows.find(row => row.id === 'partial')).toMatchObject({ stale: true })
      }

      await refresh
      const settled = cache.state.get()
      const serialized = JSON.stringify(settled)
      const calls = fetch.mock.calls.length

      for (const pending of late) {
        pending.resolve(mode === 'connect' ? ['late-backend'] : inventory('late'))
      }

      await vi.advanceTimersByTimeAsync(30_000)
      expect(cache.state.get()).toBe(settled)
      expect(JSON.stringify(cache.state.get())).toBe(serialized)
      expect(fetch).toHaveBeenCalledTimes(calls)
      expect(vi.getTimerCount()).toBe(0)
      expect(connect).toHaveBeenCalledTimes(mode === 'connect' ? 1 : 0)
      expect(
        fetch.mock.calls.every(
          ([, path]) => path.startsWith('/api/profiles/agent-overview') || path === '/api/profiles'
        )
      ).toBe(true)

      failing = false
      const recovery = cache.refresh()
      await vi.advanceTimersByTimeAsync(2_000)
      await recovery
      expect(cache.state.get().data?.sources.every(source => source.complete)).toBe(true)
      expect(cache.state.get().data?.rows.every(row => !row.stale)).toBe(true)
      expect(
        cache.state
          .get()
          .data?.rows.map(row => row.id)
          .sort()
      ).toEqual(['healthy', 'last-known'])
      expect(vi.getTimerCount()).toBe(0)
    }
  )

  it('keeps cold-start healthy sources visible through repeated source timeouts and automatically recovers', async () => {
    vi.useFakeTimers()
    let failing = true
    const read = createAgentOverviewReader<string>()
    const connect = vi.fn()

    const fetch = vi.fn(async (descriptor: string) => {
      if (descriptor !== 'healthy' && failing) {
        return new Promise((_, reject) => setTimeout(() => reject(new Error('offline')), 8_000))
      }

      return inventory(descriptor === 'healthy' ? 'healthy' : 'recovered')
    })

    const cache = createOverviewCache(control =>
      read(
        {
          sources: [
            { id: 'slow', label: 'Slow', kind: 'ssh' },
            { id: 'healthy', label: 'Healthy', kind: 'ssh' },
            { id: 'parked', label: 'Parked', kind: 'ssh' }
          ],
          pooled: new Map([
            ['slow', ['slow', 'slow-2', 'slow-3']],
            ['healthy', ['healthy']]
          ]),
          connect,
          fetch
        },
        control
      )
    )

    const hide = cache.watch()

    try {
      await vi.advanceTimersByTimeAsync(12_000)

      for (let round = 0; round < 4; round++) {
        expect(cache.state.get().error).toBeNull()
        expect(cache.state.get().data?.sources.find(source => source.connectionId === 'slow')).toMatchObject({
          complete: false,
          state: 'offline',
          error: 'read-timeout'
        })
        expect(cache.state.get().data?.rows.find(row => row.id === 'healthy')).toMatchObject({ stale: false })
        await vi.advanceTimersByTimeAsync(17_000)
      }

      failing = false
      await vi.advanceTimersByTimeAsync(5_000)
      expect(cache.state.get().data?.sources.find(source => source.connectionId === 'slow')).toMatchObject({
        complete: true,
        state: 'ready'
      })
      expect(cache.state.get().data?.rows.find(row => row.id === 'recovered')).toMatchObject({
        stale: false,
        runtimeId: 'runtime-recovered',
        attention: 'working'
      })
      expect(connect).not.toHaveBeenCalled()
    } finally {
      hide()
    }

    expect(vi.getTimerCount()).toBe(0)
  })
})
