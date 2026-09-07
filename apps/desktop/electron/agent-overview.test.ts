import { describe, expect, it, vi } from 'vitest'

import { collectAgentOverview, createAgentOverviewReader, gatherOverviewBackends } from './agent-overview'

describe('read-only fleet inventory', () => {
  it.each([false, true])('keeps later-page profile failures incomplete when totals shrink (empty=%s)', async empty => {
    const cache = new Map()

    const fetch = vi.fn(async (_descriptor: string, _path: string) => ({
      sessions: [{ id: 'retained', profile: 'broken' }],
      total: 1,
      errors: [] as Array<{ profile: string; error: string; path?: string }>
    }))

    const options = {
      sources: [{ id: 'a', kind: 'local', label: 'A' }],
      pooled: new Map([['a', ['backend']]]),
      connect: vi.fn(),
      cache,
      fetch
    }

    await collectAgentOverview(options)
    const cached = cache.get('a')
    fetch.mockImplementation(async (_descriptor, path) => {
      const first = path.endsWith('offset=0')

      return {
        sessions: first ? [{ id: 'first', profile: 'healthy' }] : empty ? [] : [{ id: 'last', profile: 'healthy' }],
        total: first ? 3 : empty ? 1 : 2,
        errors: first ? [] : [{ profile: 'broken', error: 'SECRET_ERROR', path: 'SECRET_PATH' }]
      }
    })

    const partial = await collectAgentOverview({ ...options, force: true })
    expect(partial.sources[0]).toMatchObject({
      complete: false,
      state: 'partial',
      errors: [{ profile: 'broken', error: 'session-store-unavailable' }]
    })
    expect(JSON.stringify(partial)).not.toContain('SECRET_')
    expect(cache.get('a')).toBe(cached)
    cache.clear()
    await collectAgentOverview(options)
    expect(cache.size).toBe(0)
    fetch.mockClear()
    await collectAgentOverview(options)
    expect(fetch.mock.calls[0][1]).toContain('offset=0')
    expect(options.connect).not.toHaveBeenCalled()
  })
  it.each([false, true])('keeps live-only profile errors visible (cached=%s)', async cached => {
    const read = createAgentOverviewReader<string>()

    const fetch = vi.fn(async (_descriptor: string, path: string) => ({
      sessions: path.includes('live_only') ? [] : [{ id: 'history', profile: 'worker' }],
      total: path.includes('live_only') ? 0 : 1,
      errors: path.includes('live_only') ? [{ error: 'SECRET_ERROR' }] : []
    }))

    const options = {
      sources: [{ id: 'a', kind: 'local', label: 'A' }],
      pooled: new Map([['a', cached ? ['backend'] : ['backend', 'other']]]),
      connect: vi.fn(),
      fetch
    }

    const first = await read(options)
    const result = cached ? await read(options) : first
    expect(result.sources[0]).toMatchObject({
      complete: false,
      state: 'partial',
      sessions: [{ id: 'history', profile: 'worker' }],
      errors: [{ error: 'session-store-unavailable' }]
    })
    expect(JSON.stringify(result)).not.toContain('SECRET_')
  })
  it('discovers parked profile names without connecting a runtime', async () => {
    const connect = vi.fn()
    const discoverParked = vi.fn(async () => [{ name: 'coder', display_name: 'Builder' }])

    const result = await collectAgentOverview({
      sources: [{ id: 'sleep', kind: 'ssh', label: 'Sleep' }],
      pooled: new Map(),
      connect,
      fetch: vi.fn(),
      discoverParked
    })

    expect(discoverParked).toHaveBeenCalledWith('sleep')
    expect(connect).not.toHaveBeenCalled()
    expect(result.sources[0].profiles).toEqual([{ name: 'coder', display_name: 'Builder' }])
    expect(result.sources[0].state).toBe('on-demand')
    expect(result.sources[0].complete).toBe(false)
  })
  it('does not let a dead pooled descriptor hide later live runtimes', async () => {
    const result = await collectAgentOverview({
      sources: [{ id: 'a', kind: 'local', label: 'A' }],
      pooled: new Map([['a', ['dead', 'working']]]),
      connect: vi.fn(),
      fetch: async descriptor => {
        if (descriptor === 'dead') {
          throw new Error('offline')
        }

        return {
          sessions: [{ id: 'history', profile: 'coder' }],
          total: 1,
          profiles: [{ name: 'coder' }],
          live: [{ id: 'run', profile: 'coder', runtime_id: 'live-run', status: 'working' }],
          live_coverage: 'process'
        }
      }
    })

    expect(result.sources[0].live[0]?.runtime_id).toBe('live-run')
    expect(result.sources[0].sessions).toHaveLength(1)
    expect(result.sources[0].state).toBe('partial')
  })
  it('bounds pending pooled descriptors without blocking healthy owners', async () => {
    vi.useFakeTimers()

    try {
      const work = gatherOverviewBackends(
        [new Promise<{ owner: string }>(() => {}), Promise.resolve({ owner: 'healthy' })],
        row => row.owner
      )

      await vi.advanceTimersByTimeAsync(2_100)
      expect((await work).get('healthy')).toEqual([{ owner: 'healthy' }])
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      vi.useRealTimers()
    }
  })
  it('polls live-only against cached history and refreshes history on explicit retry', async () => {
    const read = createAgentOverviewReader<string>()

    const fetch = vi.fn(async (_descriptor: string, path: string) => ({
      sessions: path.includes('live_only') ? [] : [{ id: 'history', profile: 'coder' }],
      total: path.includes('live_only') ? 0 : 1,
      profiles: [{ name: 'coder' }],
      canonical: [],
      errors: [],
      live_coverage: 'process',
      live: [
        {
          id: 'live',
          runtime_id: 'runtime',
          profile: 'coder',
          status: path.includes('live_only') ? 'waiting' : 'working'
        }
      ]
    }))

    const options = {
      sources: [{ id: 'a', kind: 'local', label: 'A' }],
      pooled: new Map([['a', ['backend']]]),
      connect: vi.fn(),
      fetch
    }

    await read(options)
    const polled = await read(options)
    expect(fetch.mock.calls[1][1]).toContain('live_only=true')
    expect(polled.sources[0].sessions).toHaveLength(1)
    expect(polled.sources[0].live[0].status).toBe('waiting')
    await read(options, { force: true })
    expect(fetch.mock.calls[2][1]).not.toContain('live_only')
  })
  it('keeps legacy profiles visible without using mutating legacy session reads or claiming completeness', async () => {
    // Real old REST /api/profiles omits canonical_session. The RPC counterpart
    // can resurrect archived Bot Chats; REST histories can archive/heal stores.
    const profiles = ['default', 'analyst', 'shared'].map(name => ({
      name,
      path: 'SECRET_PATH',
      model: 'profile-model'
    }))

    const legacyProfiles = vi.fn(async () => ({
      profiles: profiles.map(profile => ({ ...profile, canonical_session: { id: 'e2e_bot', title: 'Bot Chat' } }))
    }))

    const fetch = vi.fn(async (_descriptor: string, path: string) => {
      if (path.startsWith('/api/profiles/agent-overview')) {
        throw Object.assign(new Error('missing'), { statusCode: 404 })
      }

      if (path === '/api/profiles') {
        return { profiles }
      }

      return { sessions: [{ id: 'e2e_ordinary', profile: 'default' }], total: 2 }
    })

    const connect = vi.fn()

    const options = {
      sources: [{ id: 'b', label: 'B', kind: 'remote' }],
      pooled: new Map([['b', ['backend', 'other-pooled-profile']]]),
      connect,
      legacyProfiles,
      fetch
    }

    const result = await collectAgentOverview(options)
    expect(legacyProfiles).not.toHaveBeenCalled()
    expect(
      fetch.mock.calls.every(([, path]) => path === '/api/profiles' || path.startsWith('/api/profiles/agent-overview'))
    ).toBe(true)
    expect(result.sources[0]).toMatchObject({
      state: 'unsupported',
      error: 'read-only-inventory-unavailable',
      complete: false,
      liveCoverage: 'unavailable',
      sessions: [],
      canonical: [],
      live: [],
      total: 0,
      offset: 0,
      profiles: profiles.map(({ name, model }) => ({ name, model }))
    })
    expect(connect).not.toHaveBeenCalled()
    expect(JSON.stringify(result)).not.toContain('SECRET_')
    fetch.mockClear()
    fetch.mockRejectedValue(new Error('network down'))
    const failed = await collectAgentOverview(options)
    expect(failed.sources[0].state).toBe('offline')
    expect(fetch.mock.calls.every(([, path]) => path.startsWith('/api/profiles/agent-overview'))).toBe(true)
  })
  it('does not report or cache a complete history when a shifted page repeats an already-read session', async () => {
    // Activity-ordered LIMIT/OFFSET: a session becoming active between pages
    // moves to the front, so page two re-serves page one's row and the newly
    // active session is never returned by any page.
    const cache = new Map()

    const shifted = vi.fn(async (_descriptor: string, _path: string) => ({
      sessions: [{ id: 'same', profile: 'coder' }],
      total: 2,
      errors: []
    }))

    const options = {
      sources: [{ id: 'a', kind: 'local', label: 'A' }],
      pooled: new Map([['a', ['backend']]]),
      connect: vi.fn(),
      cache,
      fetch: shifted
    }

    const result = await collectAgentOverview(options)
    const source = result.sources[0]
    const unique = new Set(source.sessions.map(row => `${row.profile}/${row.id}`))
    expect(unique.size).toBe(1)
    expect(source.sessions).toHaveLength(unique.size)
    expect(source.complete).toBe(false)
    expect(source.state).toBe('partial')
    expect(source.offset).toBeLessThan(source.total)
    expect(cache.size).toBe(0)
    // The next visible poll re-reads history instead of trusting a cached total.
    shifted.mockClear()
    await collectAgentOverview(options)
    expect(shifted.mock.calls[0][1]).toContain('offset=0')
  })
  it('recovers a shifted page by re-reading history once within the source budget', async () => {
    const cache = new Map()
    let reads = 0

    const fetch = vi.fn(async (_descriptor: string, path: string) => {
      const offset = Number(new URL(path, 'http://local').searchParams.get('offset'))
      reads += 1
      // First pass: 'fresh' becomes active between pages, so offset=1 re-serves
      // 'same'. Second pass sees a stable order.
      const rows = reads <= 2 ? ['same'] : offset === 0 ? ['fresh'] : ['same']

      return { sessions: rows.map(id => ({ id, profile: 'coder' })), total: 2, errors: [] }
    })

    const result = await collectAgentOverview({
      sources: [{ id: 'a', kind: 'local', label: 'A' }],
      pooled: new Map([['a', ['backend']]]),
      connect: vi.fn(),
      cache,
      fetch
    })

    const source = result.sources[0]
    expect(source.sessions.map(row => row.id).sort()).toEqual(['fresh', 'same'])
    expect(source).toMatchObject({ complete: true, state: 'ready', offset: 2, total: 2 })
    expect(cache.get('a')?.source.sessions).toHaveLength(2)
  })
  it('completes and caches honest non-overlapping pages', async () => {
    const cache = new Map()

    const fetch = vi.fn(async (_descriptor: string, path: string) => {
      const offset = Number(new URL(path, 'http://local').searchParams.get('offset'))

      return {
        sessions: offset === 0 ? [{ id: 'newest', profile: 'coder' }] : [{ id: 'older', profile: 'coder' }],
        total: 2,
        errors: []
      }
    })

    const options = {
      sources: [{ id: 'a', kind: 'local', label: 'A' }],
      pooled: new Map([['a', ['backend']]]),
      connect: vi.fn(),
      cache,
      fetch
    }

    const result = await collectAgentOverview(options)
    expect(result.sources[0]).toMatchObject({ complete: true, state: 'ready', offset: 2, total: 2 })
    expect(result.sources[0].sessions.map(row => row.id)).toEqual(['newest', 'older'])
    expect(cache.get('a')?.source.complete).toBe(true)
    fetch.mockClear()
    await collectAgentOverview(options)
    expect(fetch.mock.calls.map(([, path]) => path)).toEqual(['/api/profiles/agent-overview?live_only=true'])
  })
  it('gathers primary and every pooled descriptor by resolved owner without dialing', async () => {
    const primary = { owner: 'url', runtime: 'primary' }
    const child = { owner: 'local', runtime: 'coder' }

    const result = await gatherOverviewBackends(
      [Promise.resolve(primary), Promise.resolve(child), Promise.resolve(primary)],
      descriptor => descriptor.owner
    )

    expect(result.get('url')).toEqual([primary])
    expect(result.get('local')).toEqual([child])
  })
  it('reads every registry source and every pooled runtime without waking parked hosts', async () => {
    const connect = vi.fn(async (id: string) => [{ id }])

    const fetch = vi.fn(async (descriptor: { id: string }, path: string) => {
      const offset = Number(new URL(path, 'http://local').searchParams.get('offset'))

      return {
        sessions: offset === 0 ? [{ id: 'same', profile: 'coder' }] : [{ id: 'older', profile: 'coder' }],
        total: 2,
        profiles: [{ name: 'coder' }],
        canonical: [],
        errors: [],
        live_coverage: 'process',
        live: [{ id: descriptor.id, profile: 'coder', runtime_id: `runtime-${descriptor.id}`, status: 'working' }]
      }
    })

    const result = await collectAgentOverview({
      sources: [
        { id: 'a', label: 'A', kind: 'remote' },
        { id: 'b', label: 'B', kind: 'ssh' },
        { id: 'sleep', label: 'Sleep', kind: 'ssh' }
      ],
      pooled: new Map([['b', [{ id: 'b-default' }, { id: 'b-coder' }]]]),
      connect,
      fetch
    })

    expect(connect.mock.calls).toEqual([['a']])
    expect(result.sources.map(source => source.connectionId)).toEqual(['a', 'b', 'sleep'])
    expect(result.sources[0].sessions.map(row => row.id)).toEqual(['same', 'older'])
    expect(result.sources[1].live.map(row => row.runtime_id)).toEqual(['runtime-b-default', 'runtime-b-coder'])
    expect(result.sources[2].state).toBe('on-demand')
    expect(result.sources[0].total).toBe(2)
  })
})
