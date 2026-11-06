import { describe, expect, it, vi } from 'vitest'

import { CACHE_TTL_MS, cacheKey, type CacheRoute, createRefinementCache } from './cache'
import type { RefineRequest, RefineResult } from './types'

const ok = (text: string): RefineResult => ({ ok: true, text, unchanged: false, analysis: null })

describe('refinement reuse cache', () => {
  it('answers the same request again without another model call', async () => {
    const cache = createRefinementCache()
    const compute = vi.fn(async () => ok('Refined.'))

    expect(await cache.run('k', 1, compute)).toEqual(ok('Refined.'))
    expect(await cache.run('k', 1, compute)).toEqual(ok('Refined.'))
    expect(compute).toHaveBeenCalledTimes(1)
  })

  it('Try again (attempt > 1) always asks afresh, and its answer replaces the old one', async () => {
    const cache = createRefinementCache()
    await cache.run('k', 1, async () => ok('First.'))

    expect(await cache.run('k', 2, async () => ok('Second.'))).toEqual(ok('Second.'))
    expect(await cache.run('k', 1, async () => ok('never'))).toEqual(ok('Second.'))
  })

  it('never keeps a failure', async () => {
    const cache = createRefinementCache()
    await cache.run('k', 1, async () => ({ ok: false, failure: 'timeout' }))
    const compute = vi.fn(async () => ok('Refined.'))

    await cache.run('k', 1, compute)
    expect(compute).toHaveBeenCalledTimes(1)
  })

  it('keys differ by request, and the least recently used entry goes first', async () => {
    const cache = createRefinementCache(2)
    await cache.run('a', 1, async () => ok('A'))
    await cache.run('b', 1, async () => ok('B'))
    await cache.run('a', 1, async () => ok('never')) // touch a
    await cache.run('c', 1, async () => ok('C')) // evicts b

    const b = vi.fn(async () => ok('B again'))
    expect(await cache.run('b', 1, b)).toEqual(ok('B again'))
    expect(b).toHaveBeenCalledTimes(1)
    expect(await cache.run('c', 1, async () => ok('never'))).toEqual(ok('C'))
  })

  it('clear forgets everything', async () => {
    const cache = createRefinementCache()
    await cache.run('k', 1, async () => ok('Old.'))
    cache.clear()

    expect(await cache.run('k', 1, async () => ok('New.'))).toEqual(ok('New.'))
  })
})

describe('cache keys — an answer is reused only for exactly the same request and route', () => {
  const REQUEST: RefineRequest = {
    text: 'refactor the parser',
    attachmentLabels: ['spec.md'],
    style: 'technical',
    depth: 'auto',
    customStyle: undefined,
    context: 'User: hi'
  }

  const ROUTE: CacheRoute = {
    connectionId: 'local',
    profile: 'default',
    modelSource: 'dedicated',
    sessionId: 'session-a',
    chatModel: 'gpt-6.1-sol'
  }

  const key = (request: Partial<RefineRequest> = {}, route: Partial<CacheRoute> = {}) =>
    cacheKey({ ...REQUEST, ...request }, { ...ROUTE, ...route })

  it.each<[string, Partial<RefineRequest>]>([
    ['prompt text', { text: 'refactor the lexer' }],
    ['attachments', { attachmentLabels: ['other.md'] }],
    ['depth', { depth: 'polish' }],
    ['style', { style: 'research' }],
    ['custom guidance', { style: 'custom', customStyle: 'Be terse.' }],
    ['chat excerpt', { context: 'User: something else' }],
    ['no chat excerpt', { context: undefined }]
  ])('differs by %s', (_name, change) => {
    expect(key(change)).not.toBe(key())
  })

  it.each<[string, Partial<CacheRoute>]>([
    ['profile', { profile: 'work' }],
    ['connection', { connectionId: 'ssh:devbox' }]
  ])('differs by %s, whichever model refines', (_name, change) => {
    expect(key({}, change)).not.toBe(key())
    expect(key({}, { ...change, modelSource: 'chat' })).not.toBe(key({}, { modelSource: 'chat' }))
  })

  it('differs by which model refines (the dedicated slot or the chat’s own)', () => {
    expect(key({}, { modelSource: 'chat' })).not.toBe(key())
  })

  it("with This chat's model, differs by chat and by that chat's model", () => {
    const chat = { modelSource: 'chat' as const }

    expect(key({}, { ...chat, sessionId: 'session-b' })).not.toBe(key({}, chat))
    expect(key({}, { ...chat, chatModel: 'claude-opus-5-5' })).not.toBe(key({}, chat))
  })

  it('with the dedicated model, the same draft in another chat of the same profile is reused', () => {
    expect(key({}, { sessionId: 'session-b', chatModel: 'other' })).toBe(key())
  })
})

describe('cache expiry — an old answer is not reused', () => {
  it(`asks again once an entry is ${CACHE_TTL_MS / 60_000} minutes old`, async () => {
    let clock = 0
    const cache = createRefinementCache(20, CACHE_TTL_MS, () => clock)
    await cache.run('k', 1, async () => ok('Old.'))

    clock = CACHE_TTL_MS - 1
    expect(await cache.run('k', 1, async () => ok('never'))).toEqual(ok('Old.'))

    clock = CACHE_TTL_MS
    expect(await cache.run('k', 1, async () => ok('New.'))).toEqual(ok('New.'))
  })

  it('reuse does not extend an entry’s life', async () => {
    let clock = 0
    const cache = createRefinementCache(20, 100, () => clock)
    await cache.run('k', 1, async () => ok('Old.'))

    clock = 90
    await cache.run('k', 1, async () => ok('never'))
    clock = 100
    expect(await cache.run('k', 1, async () => ok('New.'))).toEqual(ok('New.'))
  })

  it('a canceled call (a failure) is never stored, so the next Enter asks the model', async () => {
    const cache = createRefinementCache()
    await cache.run('k', 1, async () => ({ ok: false, failure: 'canceled' }))
    const compute = vi.fn(async () => ok('Fresh.'))

    expect(await cache.run('k', 1, compute)).toEqual(ok('Fresh.'))
    expect(compute).toHaveBeenCalledTimes(1)
  })
})
