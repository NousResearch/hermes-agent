import assert from 'node:assert/strict'

import { test } from 'vitest'

import { normalizeRemoteSessionList } from './remote-session-list'

test('normalizeRemoteSessionList preserves pagination and exposes the owning profile total', () => {
  const response = normalizeRemoteSessionList('implementer', {
    limit: 40,
    offset: 0,
    sessions: [{ id: 'newest' }, { id: 'older' }],
    total: 107
  })

  assert.equal(response.limit, 40)
  assert.equal(response.offset, 0)
  assert.equal(response.total, 107)
  assert.deepEqual(response.profile_totals, { implementer: 107 })
  assert.deepEqual(response.sessions, [
    { id: 'newest', profile: 'implementer', is_default_profile: false },
    { id: 'older', profile: 'implementer', is_default_profile: false }
  ])
})

test('normalizeRemoteSessionList falls back to returned row count when a legacy backend omits total', () => {
  const response = normalizeRemoteSessionList('remote', {
    sessions: [{ id: 'only-row' }]
  })

  assert.equal(response.total, 1)
  assert.deepEqual(response.profile_totals, { remote: 1 })
})
