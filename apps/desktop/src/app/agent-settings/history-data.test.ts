import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type * as ClientApi from '@/api/client'

vi.mock('@/lib/legacy-session-owner-backfill', () => ({ maybeBackfillLegacySessionOwners: vi.fn() }))
vi.mock('@/store/transcript-tail', () => ({ recordTranscriptTail: vi.fn() }))
vi.mock('@/api/client', async importOriginal => ({ ...(await importOriginal<typeof ClientApi>()), hermesApi: vi.fn() }))
vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: vi.fn() }))

import { hermesApi, setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { requestGatewayForAgent } from '@/store/gateway'
import { makeSessionInfo } from '@/test/session-info'
import type { SessionInfo } from '@/types/hermes'

import {
  cleanupMatches,
  DEFAULT_HISTORY_FILTERS,
  deleteHistorySelection,
  getHistoryPage,
  historyLastActive,
  matchesHistory,
  mutateHistorySession,
  previewHistoryCleanup
} from './history-data'

const owner = { connectionId: 'alpha', profile: 'research' }
const now = 2_000_000_000
beforeEach(() => {
  vi.mocked(hermesApi).mockReset()
  vi.mocked(requestGatewayForAgent).mockReset().mockResolvedValue({ sessions: [] })
  setApiRequestConnection('unrelated-gateway')
  setApiRequestProfile('unrelated-agent')
  vi.spyOn(Date, 'now').mockReturnValue(now * 1000)
})
afterEach(() => vi.restoreAllMocks())

it('sends real source mappings and archive scope while rejecting mismatched pinned backfills', async () => {
  const rows = [
    makeSessionInfo({ id: 'gui', source: 'gui', archived: true }),
    makeSessionInfo({ id: 'cron', source: 'cron', archived: true }),
    makeSessionInfo({ id: 'telegram', source: 'telegram', archived: true }),
    makeSessionInfo({ id: 'pin', source: 'desktop', pinned: true }),
    makeSessionInfo({ id: 'protected', title: 'Bot Chat', source: 'desktop', archived: true })
  ]

  vi.mocked(hermesApi).mockResolvedValue({ sessions: rows, limit: 25, offset: 525, total: 700 } as never)

  const filters = {
    ...DEFAULT_HISTORY_FILTERS,
    source: ['chat', 'schedule'] as Array<'chat' | 'schedule'>,
    archived: 'only' as const
  }

  const page = await getHistoryPage(owner, filters, 525)
  expect(page.sessions.filter(row => matchesHistory(row, filters)).map(row => row.id)).toEqual(['gui', 'cron'])
  const request = vi.mocked(hermesApi).mock.calls[0][0]
  expect(request).toMatchObject(owner)
  const query = new URL(request.path, 'https://fixture').searchParams
  expect(query.get('archived')).toBe('only')
  expect(query.get('offset')).toBe('525')
  expect(query.get('limit')).toBe('25')
  expect(query.get('sources')?.split(',')).toEqual(expect.arrayContaining(['gui', 'desktop', 'cli', 'tui', 'cron']))
  expect(query.get('sources')).not.toContain('telegram')
})

it('previews exact cleanup candidates without writes, then skips newly running, changed and protected sessions', async () => {
  const old = { started_at: now - 30 * 86_400, last_active: now - 8 * 86_400, source: 'desktop', message_count: 4 }

  const inventory = [
    makeSessionInfo({ ...old, id: 'safe', is_active: true }),
    makeSessionInfo({ ...old, id: 'race' }),
    makeSessionInfo({ ...old, id: 'changed' }),
    makeSessionInfo({ ...old, id: 'protected' }),
    makeSessionInfo({ ...old, id: 'failed' }),
    makeSessionInfo({ ...old, id: 'live' }),
    makeSessionInfo({ ...old, id: 'archived', archived: true }),
    makeSessionInfo({ ...old, id: 'boundary', last_active: now - 7 * 86_400 }),
    makeSessionInfo({ ...old, id: 'five', message_count: 5 }),
    makeSessionInfo({ ...old, id: 'canonical', title: 'Bot Chat' }),
    ...Array.from({ length: 20 }, (_, index) => makeSessionInfo({ ...old, id: `recent-${index}`, last_active: now }))
  ]

  const detail = new Map(inventory.map(row => [row.id, row]))
  vi.mocked(requestGatewayForAgent).mockResolvedValue({
    sessions: [{ id: 'runtime-live', session_key: 'live', status: 'working' }]
  })
  const writes: string[] = []
  const reads: string[] = []
  vi.mocked(hermesApi).mockImplementation(async request => {
    expect(request).toMatchObject(owner)
    const url = new URL(request.path, 'https://fixture')

    if (url.pathname === '/api/profiles/sessions') {
      const offset = Number(url.searchParams.get('offset'))

      return { sessions: inventory.slice(offset, offset + 25), offset, limit: 25, total: inventory.length } as never
    }

    const id = url.pathname.split('/')[3]

    if (request.method === 'DELETE') {
      writes.push(id)
      expect(url.searchParams.get('expected_generation')).toBe('4')

      if (id === 'race') {
        throw new Error('409: session_busy')
      }

      if (id === 'failed') {
        throw new Error('Network disconnected; retry')
      }

      return { ok: true } as never
    }

    reads.push(url.pathname)

    if (url.pathname.endsWith('/mutation-snapshot')) {
      return { exists: true, session_id: id, runtime_revision: 8, runtime_generation: 4 } as never
    }

    return detail.get(id) as never
  })

  const preview = await previewHistoryCleanup(
    owner,
    { days: 7, messages: 'under5', source: 'all', includeArchived: false },
    () => true
  )

  expect(preview.candidates.map(row => row.id)).toEqual(['safe', 'race', 'changed', 'protected', 'failed'])
  expect(preview.running.map(row => row.id)).toEqual(['live'])
  expect(writes).toEqual([])
  detail.set('changed', { ...detail.get('changed')!, message_count: 8 })
  detail.set('protected', { ...detail.get('protected')!, title: 'Bot Chat', hidden: true })
  // New matches after the dry run are never admitted to this confirmation.
  detail.set('new-match', makeSessionInfo({ ...old, id: 'new-match' }))
  const outcome = await deleteHistorySelection(owner, preview.candidates, () => true, 'Upgrade required', preview)
  expect(outcome).toEqual({
    deleted: ['safe'],
    running: ['race'],
    changed: ['changed', 'protected'],
    errors: [{ id: 'failed', error: 'Network disconnected; retry' }]
  })
  expect(writes).toEqual(['safe', 'race', 'failed'])
  expect(reads.slice(0, 2)).toEqual(['/api/sessions/safe/mutation-snapshot', '/api/sessions/safe'])
  expect(
    vi.mocked(requestGatewayForAgent).mock.calls.every(call => call[0] === 'alpha' && call[1] === 'research')
  ).toBe(true)
})

it('stops every write when the owner changes during post-snapshot revalidation', async () => {
  let current = true
  const reading = Promise.withResolvers<void>()
  const detail = Promise.withResolvers<SessionInfo>()
  vi.mocked(hermesApi).mockImplementation(async request => {
    if (request.path.includes('/mutation-snapshot')) {
      return { exists: true, runtime_revision: 2, runtime_generation: 1 } as never
    }

    if (!request.method) {
      reading.resolve()

      return (await detail.promise) as never
    }

    throw new Error('Must not write to a departed owner')
  })
  const pending = mutateHistorySession(owner, 'cancel-owner', 'DELETE', {}, () => current, 'Upgrade required')
  await reading.promise
  current = false
  setApiRequestConnection('beta')
  detail.resolve(makeSessionInfo({ id: 'cancel-owner' }))
  await expect(pending).rejects.toThrow('History owner changed')
  expect(vi.mocked(hermesApi).mock.calls.filter(([request]) => request.method)).toEqual([])
  expect(vi.mocked(hermesApi).mock.calls.every(([request]) => request.connectionId === 'alpha')).toBe(true)
})

it('refuses unsafe old-runtime mutation fallbacks and propagates authoritative rename failures', async () => {
  vi.mocked(hermesApi).mockImplementation(async request => {
    if (request.path.includes('/mutation-snapshot')) {
      throw new Error('404: {"detail":"No such API endpoint: /api/sessions/no-fence/mutation-snapshot"}')
    }

    return makeSessionInfo({ id: 'no-fence' }) as never
  })
  await expect(mutateHistorySession(owner, 'no-fence', 'DELETE', {}, () => true, 'Upgrade required')).rejects.toThrow(
    'Upgrade required'
  )
  expect(vi.mocked(hermesApi).mock.calls.filter(([request]) => request.method)).toEqual([])
  vi.mocked(hermesApi).mockImplementation(async request => {
    if (request.method === 'PATCH') {
      throw new Error('409: revision_conflict')
    }

    return (
      request.path.includes('/mutation-snapshot')
        ? { exists: true, runtime_revision: 7, runtime_generation: 1 }
        : makeSessionInfo({ id: 'rename-failed' })
    ) as never
  })
  await expect(
    mutateHistorySession(owner, 'rename-failed', 'PATCH', { title: 'New name' }, () => true, 'Upgrade required')
  ).rejects.toThrow('revision_conflict')
  expect(vi.mocked(hermesApi).mock.calls.filter(([request]) => request.method)).toHaveLength(1)
})

it.each([7, 30, 90] as const)('keeps the %s-day age, empty-message and source boundaries exact', days => {
  const criteria = { days, messages: 'empty' as const, source: 'schedule' as const, includeArchived: true }
  const row = makeSessionInfo({ source: 'cron', archived: true, message_count: 0, last_active: now - days * 86_400 })
  expect(cleanupMatches(row, criteria, now)).toBe(false)
  const older = { ...row, last_active: row.last_active - 1 }
  expect(cleanupMatches(older, criteria, now)).toBe(true)
  expect(cleanupMatches({ ...older, message_count: 1 }, criteria, now)).toBe(false)
  expect(cleanupMatches({ ...older, source: 'desktop' }, criteria, now)).toBe(false)
  expect(cleanupMatches(older, { ...criteria, includeArchived: false }, now)).toBe(false)
  expect(cleanupMatches({ ...older, last_active: 0, started_at: 0 }, criteria, now)).toBe(false)
})

it('reads recent activity relative and older activity as a date, with the exact time kept for hover', () => {
  // Wednesday 15:00 local; "yesterday" is any time after the previous local midnight.
  const nowMs = new Date(2026, 10, 11, 15, 0).getTime()
  const at = (date: Date) => historyLastActive(date.getTime() / 1000, 'en-US', nowMs).label

  expect(at(new Date(2026, 10, 11, 14, 56))).toBe('4 min. ago')
  expect(at(new Date(2026, 10, 11, 9, 15))).toBe('6 hr. ago')
  expect(at(new Date(2026, 10, 10, 0, 5))).toBe('Yesterday')
  expect(at(new Date(2026, 9, 30, 12, 0))).toBe('Oct 30')
  expect(at(new Date(2025, 9, 30, 12, 0))).toBe('Oct 30, 2025')
  expect(historyLastActive(new Date(2026, 9, 30, 12, 0).getTime() / 1000, 'en-US', nowMs).absolute).toMatch(
    /Oct 30, 2026.*12:00/
  )
})
