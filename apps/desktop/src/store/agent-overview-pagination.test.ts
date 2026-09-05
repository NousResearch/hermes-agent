import { describe, expect, it, vi } from 'vitest'

import { collectAgentOverview } from '../../electron/agent-overview'

import { mergeOverview } from './agent-overview'

describe('fleet pagination coverage integration', () => {
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

    const previous = mergeOverview(undefined, await collectAgentOverview(options))
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
    expect(mergeOverview(previous, partial).rows).toContainEqual(
      expect.objectContaining({ id: 'retained', profile: 'broken', stale: true })
    )
    cache.clear()
    await collectAgentOverview(options)
    expect(cache.size).toBe(0)
    fetch.mockClear()
    await collectAgentOverview(options)
    expect(fetch.mock.calls[0][1]).toContain('offset=0')
    expect(options.connect).not.toHaveBeenCalled()
  })
})
