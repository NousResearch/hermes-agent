import assert from 'node:assert/strict'

import { test } from 'vitest'

import { fetchRemoteProfileSessions, findRemoteOwnerProfileForSession } from './profile-session-routing'

function remoteList(pinned: number, lineage = false) {
  const rows = Array.from({ length: 5000 }, (_, i) => ({
    id: 'session-' + i,
    ...(lineage && i === 150 ? { _lineage_root_id: 'wanted-root' } : {})
  }))

  const requests: Array<[number, number]> = []

  const list = (profile: string, params: URLSearchParams) =>
    fetchRemoteProfileSessions(profile, params, async (_profile, path) => {
      const query = new URL(path, 'http://test.local').searchParams
      const limit = Number(query.get('limit'))
      const offset = Number(query.get('offset'))
      requests.push([limit, offset])
      // Stop a regression from hanging the test; the final assertion also rejects a swallowed error.
      assert.ok(requests.length <= 2, 'owner lookup exceeded its two-page budget')
      assert.ok(limit <= 100, 'backend rejects pages above 100')
      const window = rows.slice(offset, offset + limit)
      const ids = new Set(window.map(row => row.id))

      return {
        total: rows.length,
        sessions: [...window, ...rows.slice(0, pinned).filter(row => !ids.has(row.id))]
      }
    })

  return { list, requests }
}

test.each([0, 120])('missing remote owner has bounded reads with %i pinned sessions', async pinned => {
  const { list, requests } = remoteList(pinned)

  assert.equal(await findRemoteOwnerProfileForSession('local-session', ['remote'], list), null)
  assert.deepEqual(requests, [
    [100, 0],
    [100, 100]
  ])
})

test('remote owner lookup finds a second-page lineage root through the real paging adapter', async () => {
  const { list, requests } = remoteList(120, true)

  assert.equal(await findRemoteOwnerProfileForSession('wanted-root', ['remote'], list), 'remote')
  assert.deepEqual(requests, [
    [100, 0],
    [100, 100]
  ])
})
