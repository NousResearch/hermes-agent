import assert from 'node:assert/strict'

import { test } from 'vitest'

import { fetchPrimaryProfileSessions, resolveProfileSessionAggregateRoute } from './profile-session-routing'

test('mixed global remote and primary override reads the aggregate from the global backend', () => {
  assert.equal(
    resolveProfileSessionAggregateRoute({ globalRemote: true, primaryProfileRemoteOverride: true }),
    'global-remote'
  )
})

test('global remote without a primary override reuses the primary backend', () => {
  assert.equal(
    resolveProfileSessionAggregateRoute({ globalRemote: true, primaryProfileRemoteOverride: false }),
    'primary'
  )
})

test('mixed remote aggregate fetches bypass the primary profile override', async () => {
  const expected = { sessions: [{ id: 'default-telegram' }], total: 1, profile_totals: { default: 1 } }
  const calls: string[] = []

  const result = await fetchPrimaryProfileSessions(
    new URLSearchParams({ profile: 'all' }),
    async () => {
      calls.push('primary')
      throw new Error('the active profile override does not serve default')
    },
    {
      globalRemote: true,
      primaryProfileRemoteOverride: true,
      fetchJsonForGlobalRemote: async () => {
        calls.push('global-remote')

        return expected
      }
    }
  )

  assert.equal(result, expected)
  assert.deepEqual(calls, ['global-remote'])
})

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
