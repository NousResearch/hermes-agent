import { describe, expect, it } from 'vitest'

import { createConditionalReadCache, normalizeConditionalResult } from './session-conditional-reads'

// Unit coverage for the main-process revalidation cache (C3 desktop half,
// t_5c4b52bf): validator lifecycle, 304 reuse, stale-validator refresh, the
// no-ETag no-op, the unasked-304 guard, derived-store behavior and bounds.

const envelope = (data: unknown, status: number, etag: string | null) => ({ data, status, etag })

describe('normalizeConditionalResult', () => {
  it('recognizes the internal conditional envelope', () => {
    expect(normalizeConditionalResult(envelope({ a: 1 }, 200, '"v1"'))).toEqual({ data: { a: 1 }, status: 200, etag: '"v1"' })
    expect(normalizeConditionalResult(envelope(null, 304, '"v1"'))).toEqual({ data: null, status: 304, etag: '"v1"' })
  })

  it('reads a plain body as an unconditional 200 with no validator', () => {
    expect(normalizeConditionalResult({ sessions: [] })).toEqual({ data: { sessions: [] }, status: 200, etag: null })
    expect(normalizeConditionalResult(null)).toEqual({ data: null, status: 200, etag: null })
  })

  it('does not mistake a payload for an envelope', () => {
    // A body that happens to carry data/status keys but no etag is plain data.
    expect(normalizeConditionalResult({ data: 1, status: 200 })).toEqual({ data: { data: 1, status: 200 }, status: 200, etag: null })
  })
})

describe('createConditionalReadCache.readPage', () => {
  it('stores the validator on a 200 and revalidates with it on the next read', async () => {
    const cache = createConditionalReadCache()
    const seenValidators: Array<string | undefined> = []

    const fetch = (ifNoneMatch?: string) => {
      seenValidators.push(ifNoneMatch)

      return Promise.resolve(ifNoneMatch === '"v1"' ? envelope(null, 304, '"v1"') : envelope({ n: 1 }, 200, '"v1"'))
    }

    const first = await cache.readPage('d1', '/api/sessions?a=1', fetch)

    expect(first).toEqual({ data: { n: 1 }, notModified: false, etag: '"v1"' })

    const second = await cache.readPage('d1', '/api/sessions?a=1', fetch)

    expect(second).toEqual({ data: { n: 1 }, notModified: true, etag: '"v1"' })
    expect(seenValidators).toEqual([undefined, '"v1"'])
  })

  it('refreshes the stored page when the validator goes stale', async () => {
    const cache = createConditionalReadCache()
    let version = 1

    const fetch = (ifNoneMatch?: string) => {
      // The server only honors the validator of the content it currently serves.
      return Promise.resolve(ifNoneMatch === `"v${version}"` ? envelope(null, 304, `"v${version}"`) : envelope({ n: version }, 200, `"v${version}"`))
    }

    expect(await cache.readPage('d1', '/p', fetch)).toMatchObject({ data: { n: 1 }, notModified: false })

    version = 2

    const second = await cache.readPage('d1', '/p', fetch)

    expect(second).toEqual({ data: { n: 2 }, notModified: false, etag: '"v2"' })

    // The refreshed validator now revalidates.
    expect(await cache.readPage('d1', '/p', fetch)).toMatchObject({ data: { n: 2 }, notModified: true })
  })

  it('never stores (or sends) a validator the server does not stamp', async () => {
    const cache = createConditionalReadCache()
    const seenValidators: Array<string | undefined> = []

    const fetch = (ifNoneMatch?: string) => {
      seenValidators.push(ifNoneMatch)

      return Promise.resolve({ sessions: [] }) // plain body: older server, no ETag
    }

    expect(await cache.readPage('d1', '/p', fetch)).toMatchObject({ notModified: false, etag: null })
    expect(await cache.readPage('d1', '/p', fetch)).toMatchObject({ notModified: false, etag: null })
    expect(seenValidators).toEqual([undefined, undefined])
  })

  it('scopes validators per descriptor: one is never sent to another backend', async () => {
    const cache = createConditionalReadCache()

    await cache.readPage('backend-a', '/p', () => Promise.resolve(envelope({ n: 1 }, 200, '"a"')))

    const seen: Array<string | undefined> = []

    const fetchB = (ifNoneMatch?: string) => {
      seen.push(ifNoneMatch)

      return Promise.resolve(envelope({ n: 2 }, 200, '"b"'))
    }

    await cache.readPage('backend-b', '/p', fetchB)

    expect(seen).toEqual([undefined])
  })

  it('re-reads unconditionally when a 304 arrives without a stored validator', async () => {
    const cache = createConditionalReadCache()
    let calls = 0

    const fetch = (ifNoneMatch?: string) => {
      calls += 1

      return Promise.resolve(calls === 1 ? envelope(null, 304, '"x"') : envelope({ n: 9 }, 200, '"x2"'))
    }

    const read = await cache.readPage('d1', '/p', fetch)

    expect(calls).toBe(2)
    expect(read).toEqual({ data: { n: 9 }, notModified: false, etag: '"x2"' })
  })
})

describe('createConditionalReadCache derived store', () => {
  it('returns the stored derived value under the same (descriptor, base, query)', () => {
    const cache = createConditionalReadCache()
    const value = { rows: [{ id: 's1' }], first: { total: 1 } }

    cache.derivedSet('d1', '/api/sessions', 'limit=10', value)

    expect(cache.derivedGet('d1', '/api/sessions', 'limit=10')).toBe(value)
    expect(cache.derivedGet('d1', '/api/sessions', 'limit=20')).toBeUndefined()
    expect(cache.derivedGet('d2', '/api/sessions', 'limit=10')).toBeUndefined()
  })

  it('evicts the oldest derived entry past the bound and resets cleanly', () => {
    const cache = createConditionalReadCache({ maxDerivedEntries: 2 })

    cache.derivedSet('d', '/b', 'q1', { rows: [1], first: null })
    cache.derivedSet('d', '/b', 'q2', { rows: [2], first: null })
    cache.derivedSet('d', '/b', 'q3', { rows: [3], first: null })

    expect(cache.derivedGet('d', '/b', 'q1')).toBeUndefined()
    expect(cache.derivedGet('d', '/b', 'q3')).toEqual({ rows: [3], first: null })

    cache.reset()

    expect(cache.derivedGet('d', '/b', 'q3')).toBeUndefined()
  })
})
