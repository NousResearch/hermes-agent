import { beforeEach, expect, it, vi } from 'vitest'

import type * as ClientApi from './client'

vi.mock('@/lib/legacy-session-owner-backfill', () => ({ maybeBackfillLegacySessionOwners: vi.fn() }))
vi.mock('@/store/transcript-tail', () => ({ recordTranscriptTail: vi.fn() }))
vi.mock('./client', async importOriginal => ({
  ...await importOriginal<typeof ClientApi>(), hermesApi: vi.fn()
}))
import { hermesApi, setApiRequestConnection, setApiRequestProfile } from './client'
import { deleteSession, renameSession, setSessionArchived, setSessionPinnedRemote, setSessionUnreadRemote } from './sessions'

beforeEach(() => { vi.mocked(hermesApi).mockReset(); setApiRequestConnection('owner'); setApiRequestProfile('work') })

it('concurrent equal sidebar edits share the missing-snapshot verdict on older standalone servers', async () => {
  vi.mocked(hermesApi).mockImplementation(async request => {
    if (!request.method || request.method === 'GET') {
      throw new Error('404: {"detail":"No such API endpoint: /api/sessions/shared-old/mutation-snapshot"}')
    }

    return { ok: true } as never
  })

  const results = await Promise.allSettled([
    renameSession('shared-old', 'same name', 'work'),
    renameSession('shared-old', 'same name', 'work')
  ])

  expect(results.map(result => result.status)).toEqual(['fulfilled', 'fulfilled'])
  expect(vi.mocked(hermesApi).mock.calls.filter(([request]) => !request.method || request.method === 'GET')).toHaveLength(1)
})

it('a missing-route response from a write never selects an unfenced retry', async () => {
  const missing = new Error('404: {"detail":"No such API endpoint: /api/sessions/write-only"}')
  const writes: unknown[] = []
  vi.mocked(hermesApi).mockImplementation(async request => {
    if (!request.method || request.method === 'GET') {
      return { exists: true, runtime_revision: 2, runtime_generation: 1 } as never
    }

    writes.push(request.body)
    throw missing
  })

  await expect(renameSession('write-only', 'same name', 'work')).rejects.toBe(missing)
  expect(writes).toHaveLength(1)
  expect(writes[0]).toMatchObject({ request_id: expect.any(String), expected_revision: 2 })
})

it('uses owner snapshots and preserves the exact delete identity after a lost reply', async () => {
  let lost = true
  const writes: unknown[] = []
  vi.mocked(hermesApi).mockImplementation(async request => {
    if (!request.method || request.method === 'GET') {
      return { exists: true, runtime_revision: 12, runtime_generation: 8 } as never
    }

    writes.push(request)

    if (lost) { lost = false; throw new Error('network disconnected after commit') }

    return { deleted_ids: ['delete-me'], revision: 13 } as never
  })
  const owner = { connectionId: 'server', profile: 'work' }
  await expect(deleteSession('delete-me', owner)).rejects.toThrow('network disconnected')
  setApiRequestConnection('unrelated')
  await deleteSession('delete-me', owner)
  expect(writes).toHaveLength(2)
  expect(writes[1]).toEqual(writes[0])
  const request = writes[0] as { path: string; connectionId: string }
  const url = new URL(request.path, 'http://localhost')
  expect(request.connectionId).toBe('server')
  expect(url.searchParams.get('expected_revision')).toBe('12')
  expect(url.searchParams.get('expected_generation')).toBe('8')
  expect(url.searchParams.get('request_id')).toBeTruthy()
})

it('all sidebar mutations use real counters and surface conflicts without a blind write', async () => {
  const calls = [() => renameSession('sidebar', 'name', 'work'),
    () => setSessionArchived('sidebar', true, 'work'), () => setSessionPinnedRemote('sidebar', false, 'work'),
    () => setSessionUnreadRemote('sidebar', true, 'work')]

  vi.mocked(hermesApi).mockImplementation(async request => {
    if (!request.method || request.method === 'GET') {
      return { exists: true, runtime_revision: 19, runtime_generation: 4 } as never
    }

    expect(request.body).toMatchObject({ expected_revision: 19, expected_generation: 4,
      request_id: expect.any(String), profile: 'work' })
    throw new Error('revision_conflict')
  })

  for (const invoke of calls) { await expect(invoke()).rejects.toThrow('revision_conflict') }
  vi.mocked(hermesApi).mockReset().mockResolvedValue({ exists: true } as never)
  await expect(renameSession('unknown-counter', 'name')).rejects.toThrow(/snapshot/)
  expect(hermesApi).toHaveBeenCalledTimes(1)
})

// R12: an older standalone runtime has no fencing route (its SPA catch-all answers the exact
// marker below); sidebar edits there keep the base write. Any other refusal of the snapshot read
// (a profile 404, permission, timeout, an unavailable authority) is never a downgrade.
it('sidebar edits fall back to the base write only when the backend lacks the fencing route', async () => {
  const routeMissing = '404: {"detail":"No such API endpoint: /api/sessions/old/mutation-snapshot"}'
  const writes: Array<{ path: string; method?: string; body?: unknown }> = []
  vi.mocked(hermesApi).mockImplementation(async request => {
    if (!request.method || request.method === 'GET') { throw new Error(routeMissing) }
    writes.push(request as never)

    return { ok: true } as never
  })

  await renameSession('old', 'name', 'work')
  await setSessionArchived('old', true, 'work')
  await setSessionPinnedRemote('old', true, 'work')
  await setSessionUnreadRemote('old', false, 'work')
  await deleteSession('old', { connectionId: 'server', profile: 'work' })
  expect(writes.map(write => [write.method, write.body])).toEqual([
    ['PATCH', { title: 'name', profile: 'work' }], ['PATCH', { archived: true, profile: 'work' }],
    ['PATCH', { pinned: true, profile: 'work' }], ['PATCH', { unread: false, profile: 'work' }], ['DELETE', undefined]])
  expect(new URL(writes[4].path, 'http://localhost').search).toBe('?profile=work')

  for (const refusal of ['404: {"detail":"Profile \'work\' does not exist."}', '403: permission_denied',
    '503: {"detail":"session_authority_unavailable"}', 'Request timed out']) {
    writes.length = 0
    vi.mocked(hermesApi).mockReset().mockImplementation(async request => {
      if (!request.method || request.method === 'GET') { throw new Error(refusal) }
      writes.push(request as never)

      return { ok: true } as never
    })
    await expect(renameSession('old', 'name', 'work')).rejects.toThrow(refusal)
    expect(writes).toEqual([])
  }
})
