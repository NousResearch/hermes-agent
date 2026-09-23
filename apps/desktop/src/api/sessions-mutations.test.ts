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
