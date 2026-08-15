import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  fetchPrimaryProfileSessions,
  fetchRemoteProfileSessions,
  mergeProfileSessionWindow
} from './profile-session-routing'

test('primary session reads use the profile-aware request path', async () => {
  const calls: Array<{ profile: string | null; path: string }> = []
  const expected = { sessions: [{ id: 'session-1' }], total: 1, profile_totals: { default: 1 } }

  const result = await fetchPrimaryProfileSessions(
    new URLSearchParams({ profile: 'default', limit: '20' }),
    async (profile, path) => {
      calls.push({ profile, path })

      return expected
    }
  )

  assert.deepEqual(calls, [{ profile: null, path: '/api/profiles/sessions?profile=default&limit=20' }])
  assert.equal(result, expected)
})

test('primary session reads preserve the empty-list fallback', async () => {
  const result = await fetchPrimaryProfileSessions(new URLSearchParams({ profile: 'all' }), async () => {
    throw new Error('remote unavailable')
  })

  assert.deepEqual(result, { sessions: [], total: 0, profile_totals: {} })
})

test('remote session reads split oversized sidebar windows into API-safe pages', async () => {
  const calls: Array<{ profile: string | null; path: string }> = []
  const rows = Array.from({ length: 250 }, (_, index) => ({ id: `session-${index}` }))

  const result = await fetchRemoteProfileSessions(
    'remote-work',
    new URLSearchParams({ profile: 'remote-work', limit: '300', offset: '0', order: 'updated' }),
    async (profile, path) => {
      calls.push({ profile, path })
      const url = new URL(path, 'http://desktop.test')
      const limit = Number(url.searchParams.get('limit'))
      const offset = Number(url.searchParams.get('offset'))

      if (limit > 100) {
        throw new Error(`remote /api/sessions rejects limit ${limit}`)
      }

      return {
        sessions: rows.slice(offset, offset + limit),
        total: rows.length,
        limit,
        offset
      }
    }
  )

  assert.deepEqual(calls, [
    { profile: 'remote-work', path: '/api/sessions?limit=100&offset=0&order=updated' },
    { profile: 'remote-work', path: '/api/sessions?limit=100&offset=100&order=updated' },
    { profile: 'remote-work', path: '/api/sessions?limit=50&offset=200&order=updated' }
  ])
  assert.equal(result.sessions.length, 250)
  assert.equal(result.total, 250)
  assert.equal(result.limit, 300)
  assert.equal(result.offset, 0)
  assert.deepEqual(
    result.sessions.map(row => (row as { id: string }).id),
    rows.map(row => row.id)
  )
})

test('remote paging preserves offsets and deduplicates pinned backfill rows', async () => {
  const calls: string[] = []

  const rows = Array.from({ length: 240 }, (_, index) => ({
    id: `session-${index}`,
    pinned: index === 20 || index === 200
  }))

  const pinned = rows.filter(row => row.pinned)

  const result = await fetchRemoteProfileSessions(
    'remote-work',
    new URLSearchParams({ profile: 'remote-work', limit: '150', offset: '80' }),
    async (_profile, path) => {
      calls.push(path)
      const url = new URL(path, 'http://desktop.test')
      const limit = Number(url.searchParams.get('limit'))
      const offset = Number(url.searchParams.get('offset'))
      const window = rows.slice(offset, offset + limit)
      const windowIds = new Set(window.map(row => row.id))

      return {
        sessions: [...window, ...pinned.filter(row => !windowIds.has(row.id))],
        total: rows.length,
        limit,
        offset
      }
    }
  )

  assert.deepEqual(calls, ['/api/sessions?limit=100&offset=80', '/api/sessions?limit=50&offset=180'])
  assert.deepEqual(
    result.sessions.map(row => (row as { id: string }).id),
    [...rows.slice(80, 230).map(row => row.id), 'session-20']
  )
})

test('remote paging treats malformed totals as unknown instead of truncating the result', async () => {
  const rows = Array.from({ length: 250 }, (_, index) => ({ id: `session-${index}` }))

  for (const malformedTotal of [null, '', false, 100.5]) {
    const calls: string[] = []

    const result = await fetchRemoteProfileSessions(
      'remote-work',
      new URLSearchParams({ limit: '300', offset: '0' }),
      async (_profile, path) => {
        calls.push(path)
        const url = new URL(path, 'http://desktop.test')
        const limit = Number(url.searchParams.get('limit'))
        const offset = Number(url.searchParams.get('offset'))

        return {
          sessions: rows.slice(offset, offset + limit),
          total: malformedTotal,
          limit,
          offset
        }
      }
    )

    assert.deepEqual(calls, [
      '/api/sessions?limit=100&offset=0',
      '/api/sessions?limit=100&offset=100',
      '/api/sessions?limit=100&offset=200'
    ])
    assert.equal(result.sessions.length, 250)
    assert.equal(result.total, 250)
  }
})

test('merged profile windows retain pinned rows outside the recency window', () => {
  const rows = [
    { id: 'recent-default', profile: 'default', pinned: false },
    { id: 'shared-id', profile: 'default', pinned: false },
    { id: 'recent-remote', profile: 'remote-work', pinned: false },
    { id: 'shared-id', profile: 'remote-work', pinned: true },
    { id: 'old-remote', profile: 'remote-work', pinned: true },
    { id: 'old-unpinned', profile: 'remote-work', pinned: false }
  ]

  assert.deepEqual(mergeProfileSessionWindow(rows, 0, 3), [rows[0], rows[1], rows[2], rows[3], rows[4]])
})

test('remote session reads keep small requests on one call', async () => {
  const calls: Array<{ profile: string | null; path: string }> = []
  const expected = { sessions: [{ id: 'session-1' }], total: 1, limit: 20, offset: 0 }

  const result = await fetchRemoteProfileSessions(
    'remote-work',
    new URLSearchParams({ profile: 'remote-work', limit: '20', offset: '0' }),
    async (profile, path) => {
      calls.push({ profile, path })

      return expected
    }
  )

  assert.deepEqual(calls, [{ profile: 'remote-work', path: '/api/sessions?limit=20&offset=0' }])
  assert.equal(result, expected)
})
