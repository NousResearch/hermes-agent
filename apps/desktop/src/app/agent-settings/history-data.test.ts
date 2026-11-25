import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type * as ClientApi from '@/api/client'

/** The group store and the gateway's room list, as `history-data` reads them: rooms live outside
 *  sessions, so History reads the store — never a room's hidden plumbing sessions. */
const plugin = vi.hoisted(() => ({
  rooms: {} as Record<string, unknown>,
  gateway: [] as unknown[],
  routes: [] as unknown[],
  disbanded: new Set<string>(),
  fail: false
}))

vi.mock('@/lib/legacy-session-owner-backfill', () => ({ maybeBackfillLegacySessionOwners: vi.fn() }))
vi.mock('@/store/transcript-tail', () => ({ recordTranscriptTail: vi.fn() }))
vi.mock('@/api/client', async importOriginal => ({ ...(await importOriginal<typeof ClientApi>()), hermesApi: vi.fn() }))
vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: vi.fn() }))
vi.mock('@/plugins/hermes-bots/group-chat', () => ({ $groupChats: { get: () => plugin.rooms } }))
vi.mock('@/plugins/hermes-bots/canonical-group-registry', () => ({
  // The key format is the registry's own (`canonical:<connection>:<profile>:<room_id>`), which the
  // Bots roster's disband set is keyed by.
  canonicalGroupKey: (route: { connectionId: string; profile: string }, roomId: string) =>
    `canonical:${encodeURIComponent(route.connectionId)}:${encodeURIComponent(route.profile)}:${roomId}`,
  $disbandedGatewayRooms: { get: () => plugin.disbanded }
}))
vi.mock('@/plugins/hermes-bots/canonical-groups', () => ({
  listCanonicalRooms: vi.fn(async (route: unknown) => {
    plugin.routes.push(route)

    if (plugin.fail) {
      throw new Error('groups.list is unavailable')
    }

    return plugin.gateway
  })
}))

import { hermesApi, setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import type { CanonicalRoom } from '@/plugins/hermes-bots/canonical-groups'
import type { GroupChat, GroupMember } from '@/plugins/hermes-bots/types'
import { requestGatewayForAgent } from '@/store/gateway'
import { makeSessionInfo } from '@/test/session-info'
import type { SessionInfo } from '@/types/hermes'

import {
  agentGroupChats,
  cleanupMatches,
  DEFAULT_HISTORY_FILTERS,
  deleteHistorySelection,
  getAgentGroupChats,
  getHistoryPage,
  historyLastActive,
  historyRows,
  matchesHistory,
  mutateHistorySession,
  previewHistoryCleanup,
  visibleHistory
} from './history-data'

const owner = { connectionId: 'alpha', profile: 'research' }
const now = 2_000_000_000

/** A room record with everything a stored one always carries, plus the test's own fields. */
const room = (overrides: Partial<GroupChat> = {}): GroupChat => ({ log: [], watermarks: {}, ...overrides })
const seat = (overrides: Partial<GroupMember> = {}): GroupMember => ({ name: 'research', ...overrides })
const canonical = (overrides: Partial<CanonicalRoom> = {}): CanonicalRoom => ({
  room_id: 'r1',
  name: 'Team',
  members: [{ member_id: 'm1', profile: 'research', handle: 'research' }],
  ...overrides
})

beforeEach(() => {
  plugin.rooms = {}
  plugin.gateway = []
  plugin.routes = []
  plugin.disbanded = new Set()
  plugin.fail = false
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
  expect(at(new Date(2026, 9, 30, 12, 0))).toBe('Oct 30, 2026')
  expect(at(new Date(2025, 9, 30, 12, 0))).toBe('Oct 30, 2025')
  expect(historyLastActive(new Date(2026, 9, 30, 12, 0).getTime() / 1000, 'en-US', nowMs).absolute).toMatch(
    /Oct 30, 2026.*12:00/
  )
})

it('renders a malformed timestamp as empty, never a RangeError', () => {
  const nowMs = new Date(2026, 10, 11, 15, 0).getTime()
  const gone = historyLastActive(Number.NaN, 'en-US', nowMs)

  expect(gone.absolute).toBe('')
  expect(gone.label).toBe('')
})

it("lists the agent's group chats in History, never another agent's room or a disbanded one", () => {
  const rooms = {
    Team: room({ members: [seat({ connectionId: 'alpha' }), seat({ connectionId: 'alpha', name: 'ops' })] }),
    // A room that seats only another profile of this connection, and one of another machine's
    // identically named profile: neither involves this agent.
    Ops: room({ members: [seat({ connectionId: 'alpha', name: 'ops' })] }),
    Shared: room({ members: [seat({ connectionId: 'beta' })] }),
    Gone: room({ members: [seat({ connectionId: 'alpha' })], tombstone: true }),
    Placeholder: room()
  }

  expect(agentGroupChats(owner, rooms, []).map(chat => [chat.kind, chat.name, chat.seats])).toEqual([
    ['local', 'Team', 2]
  ])
})

it('hides a gateway room this window saw disband, even from a list read before the disband', async () => {
  const gateway = [
    canonical({ room_id: 'r1', name: 'Live', updated_at: 900 }),
    // The room's own tombstone, and the disband set the Bots roster keeps for a `groups.list` that
    // started before the disband (or a failed re-read): neither gets a row in History.
    canonical({ room_id: 'r2', name: 'Tombstoned', updated_at: 800, disbanded_at: 1 }),
    canonical({ room_id: 'r3', name: 'Hidden', updated_at: 700 })
  ]
  // A stale Desktop record of a disbanded room must not stand in for it either.
  const rooms = { Stale: room({ roomId: 'r2', members: [seat({ connectionId: 'alpha' })] }) }
  plugin.gateway = gateway
  plugin.rooms = rooms
  plugin.disbanded = new Set(['canonical:alpha:research:r3'])

  expect((await getAgentGroupChats(owner)).map(chat => chat.name)).toEqual(['Live'])
  expect(agentGroupChats(owner, rooms, gateway).map(chat => chat.name)).toEqual(['Live'])
})

it('seats the local owner through the unsourced descriptors legacy rooms hold, and nobody else', () => {
  const rooms = { Legacy: room({ members: [{ name: 'research' }] }) }

  expect(agentGroupChats({ connectionId: 'local', profile: 'research' }, rooms, []).map(chat => chat.name)).toEqual([
    'Legacy'
  ])
  // A bare name is this machine's member only: it never seats a remote connection's profile.
  expect(agentGroupChats(owner, rooms, [])).toEqual([])
  expect(agentGroupChats(owner, { Remote: room({ members: [{ name: 'research', remoteSource: true }] }) }, [])).toEqual(
    []
  )
})

it('names a room once, keeps the gateway facts, and lists Desktop-only rooms beside it', () => {
  const message = (at: number) => ({ at, from: { kind: 'user' as const, name: 'You' }, text: 'hi' })
  const rooms = {
    // The local record of a room the gateway lists: one row, from the gateway.
    Team: room({ members: [seat({ connectionId: 'alpha' })], roomId: 'r1', log: [message(60_000)] }),
    // Recorded as this agent's gateway room before its members are read back.
    Desk: room({ engine: { kind: 'gateway', connectionId: 'alpha', profile: 'research', roomId: 'r2' } }),
    Mixed: room({ engine: 'desktop', members: [seat({ connectionId: 'alpha' })], log: [message(30_000)] })
  }

  expect(agentGroupChats(owner, rooms, [canonical({ updated_at: 900 })])).toEqual([
    { kind: 'gateway', name: 'Team', seats: 1, lastActive: 900, room: canonical({ updated_at: 900 }) },
    { kind: 'local', name: 'Mixed', seats: 1, lastActive: 30 },
    { kind: 'local', name: 'Desk', seats: 0, lastActive: 0 }
  ])
})

it("reads the agent's rooms from its route and the store, and survives a gateway that cannot answer", async () => {
  plugin.rooms = {
    Mine: room({ members: [seat({ connectionId: 'alpha' })] }),
    Theirs: room({ members: [seat({ connectionId: 'alpha', name: 'ops' })] })
  }
  plugin.gateway = [canonical({ room_id: 'r9', name: 'Remote', updated_at: 500 })]

  expect((await getAgentGroupChats(owner)).map(chat => chat.name)).toEqual(['Remote', 'Mine'])

  // An older gateway has no group chats at all, and an unreachable one must not take the agent's
  // own rooms (or its conversations) down with it.
  plugin.fail = true
  plugin.gateway = []
  expect((await getAgentGroupChats(owner)).map(chat => chat.name)).toEqual(['Mine'])
  // A room lives in one profile's store: only this agent's route is ever asked, so another agent's
  // rooms cannot arrive through the gateway either.
  expect(plugin.routes).toEqual([owner, owner])
})

it("keeps a room's hidden plumbing sessions out of History", () => {
  const pages = [
    {
      sessions: [
        makeSessionInfo({ id: 'chat' }),
        // The canonical hidden "Bot Chat" and every other internal child are not conversations.
        makeSessionInfo({ id: 'canonical', title: 'Bot Chat', hidden: true }),
        makeSessionInfo({ id: 'room-session', source: 'bot_room', hidden: true }),
        makeSessionInfo({ id: 'member-session', source: 'bot_room', is_internal_child: true })
      ],
      limit: 25,
      offset: 0,
      total: 4
    }
  ]

  expect(historyRows(pages).map(row => row.id)).toEqual(['chat'])
  expect(visibleHistory(makeSessionInfo({ id: 'hidden', hidden: true }))).toBe(false)
  expect(visibleHistory(makeSessionInfo({ id: 'child', is_internal_child: true }))).toBe(false)
})
