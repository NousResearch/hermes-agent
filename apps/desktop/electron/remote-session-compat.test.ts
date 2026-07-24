import assert from 'node:assert/strict'
import test from 'node:test'

import {
  fetchRemoteSessionListWithFallback,
  missingSessionListEndpoint,
  normalizeRemoteSessionList
} from './remote-session-compat'

test('recognizes only missing-endpoint failures as fallback candidates', () => {
  assert.equal(missingSessionListEndpoint(new Error('404: Not Found')), true)
  assert.equal(
    missingSessionListEndpoint(
      new Error(
        'Expected JSON from https://host/api/profiles/sessions but got HTML. The endpoint is likely missing on the Hermes backend.'
      )
    ),
    true
  )
  assert.equal(missingSessionListEndpoint(new Error('401: Unauthorized')), false)
  assert.equal(missingSessionListEndpoint(new Error('connect ECONNREFUSED')), false)
})

test('normalizes the legacy gateway list shape for the desktop sidebar', () => {
  const result = normalizeRemoteSessionList(
    {
      data: [{ id: 'remote-1', message_count: 2 }],
      has_more: true,
      limit: 40,
      offset: 40
    },
    'health'
  )

  assert.deepEqual(result.sessions, [
    {
      id: 'remote-1',
      is_default_profile: false,
      message_count: 2,
      profile: 'health'
    }
  ])
  assert.equal(result.total, 42)
  assert.equal(result.total_is_lower_bound, true)
  assert.deepEqual(result.profile_totals, { health: 42 })
})

test('falls back to /api/sessions only when the aggregate endpoint is missing', async () => {
  const paths: string[] = []
  const result = await fetchRemoteSessionListWithFallback(
    async path => {
      paths.push(path)

      if (path.startsWith('/api/profiles/sessions')) {
        throw new Error('404: Not Found')
      }

      return { sessions: [{ id: 'remote-1' }], total: 1 }
    },
    '/api/profiles/sessions?profile=health',
    '/api/sessions?profile=health',
    'health'
  )

  assert.deepEqual(paths, ['/api/profiles/sessions?profile=health', '/api/sessions?profile=health'])
  assert.equal(result.sessions[0].id, 'remote-1')
})

test('does not hide authentication or connectivity failures behind the fallback', async () => {
  let calls = 0

  await assert.rejects(
    fetchRemoteSessionListWithFallback(
      async () => {
        calls += 1
        throw new Error('401: Unauthorized')
      },
      '/api/profiles/sessions',
      '/api/sessions',
      'default'
    ),
    /401/
  )
  assert.equal(calls, 1)
})
