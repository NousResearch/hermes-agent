/**
 * Bot Mode sessions are ALWAYS hidden from the global Sessions sidebar.
 *
 * Canonical Bot Chats and group-chat member sessions alike ride the core
 * generic `hidden` session flag. There is no user pref: `session.create`
 * passes `hidden: true` unconditionally (pinned in
 * canonical-chat-creation.test.ts), and this reconciliation sweep pushes rows
 * born visible under the old pref — or minted outside the plugin entirely —
 * back to hidden on load and on every reconnect.
 *
 * The sweep has two halves and runs BOTH, id-based first:
 *   1. by id — every group room's member sessions, which the plugin recorded;
 *   2. by title — each roster bot's own profile listing, which is the only
 *      thing that reaches CLI-born "Agent Inbox" / extra "Bot Chat" rows AND
 *      the only thing that hides a canonical Bot Chat. Canonical chats are
 *      identified by NAME; no stored id pointer is consulted anywhere here.
 *
 * Ported from tests/hide-bot-chats.test.mjs, which sliced the sweep out of
 * the old plugin.js bundle and ran it under `vm`. The functions are module-
 * private, so this drives them through the scheduler the plugin actually
 * starts.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { GroupMember, ProfileRoute, RosterRow } from './types'

interface SweepRow {
  id: string
  started_at?: number
  title?: string
}

const { groupChats, hostMock, lastRoster, requestForBotMock } = vi.hoisted(() => ({
  groupChats: { value: {} as Record<string, unknown> },
  hostMock: {
    listPersistedSessions: vi.fn(),
    request: vi.fn(),
    setPersistedSessionHidden: vi.fn(),
    state: {
      gateway: { listen: vi.fn(() => () => undefined) },
      profile: { get: () => 'default' }
    }
  },
  lastRoster: { value: [] as RosterRow[] },
  requestForBotMock: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', () => ({ host: hostMock }))

vi.mock('./canonical-chat', () => ({ PROFILE_SESSION_LIST_LIMIT: 200 }))

vi.mock('./data', () => ({ $lastRoster: { get: () => lastRoster.value } }))

vi.mock('./group-chat', () => ({ $groupChats: { get: () => groupChats.value } }))

vi.mock('./group-membership', () => ({
  groupMemberKey: (member: GroupMember) =>
    member?.route?.connectionId ? `${member.route.connectionId}::${member.name}` : member?.name
}))

vi.mock('./routing', () => ({
  backendTargetProfile: (route: null | ProfileRoute, fallback: string) => route?.targetProfile || fallback,
  botConnectionRoute: (bot: RosterRow) =>
    bot?.route ||
    (bot?.remoteSource
      ? { connectionId: bot.connectionId, mode: 'remote', profile: bot.name, targetProfile: bot.name }
      : null),
  requestForBot: requestForBotMock
}))

/** Every (route, options) pair the sweep pushed through the REST setter. */
function hiddenCalls() {
  return hostMock.setPersistedSessionHidden.mock.calls as Array<
    [null | ProfileRoute, { hidden: boolean; profile: string; sessionId: string }]
  >
}

/** Start the scheduler the plugin starts and let its first pass settle. The
 *  sweep is scheduled on a macrotask, then runs as an unawaited promise
 *  chain — three turns is well past what its resolved awaits need. */
async function runSweep() {
  const { startHideSweepScheduler } = await import('./session-sweep')

  startHideSweepScheduler({})

  for (let turn = 0; turn < 3; turn += 1) {
    await new Promise(resolve => setTimeout(resolve, 0))
  }
}

beforeEach(() => {
  vi.resetModules()
  vi.clearAllMocks()
  // Date only: the scheduler's own setTimeout must stay real so runSweep can
  // simply wait for it. 1_000_000ms puts `Date.now() / 1000` at 1000s, which
  // is what the age fixtures below are written against.
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(1_000_000)
  groupChats.value = {}
  lastRoster.value = []
  hostMock.listPersistedSessions.mockResolvedValue({ sessions: [] })
  hostMock.setPersistedSessionHidden.mockResolvedValue(undefined)
  hostMock.request.mockResolvedValue({})
  requestForBotMock.mockResolvedValue({})
})

afterEach(() => {
  vi.useRealTimers()
})

describe('the id half: group room member sessions', () => {
  it('hides every room member session exactly once', async () => {
    groupChats.value = {
      Core: { sessions: { alpha: 'room-core-a', beta: 'room-core-b' } },
      // Duplicate id under the same key — must dedupe.
      Quiet: { sessions: { alpha: 'room-core-a' } },
      // Pre-sessions room shape.
      Legacy: {}
    }
    lastRoster.value = [{ name: 'alpha' } as RosterRow]

    await runSweep()

    expect(
      hiddenCalls()
        .map(([, options]) => options.sessionId)
        .sort()
    ).toEqual(['room-core-a', 'room-core-b'])
    expect(hiddenCalls().every(([, options]) => options.hidden)).toBe(true)
  })

  it('never consults a stored canonical pointer', async () => {
    // Canonical Bot Chats are hidden by the TITLE sweep below — they are
    // identified by name, not by pointer. The load-time reconciliation reads
    // no bot-meta chat id and issues no id-verification RPC.
    groupChats.value = { Core: { sessions: { alpha: 'room-core-a' } } }
    lastRoster.value = [{ name: 'alpha' } as RosterRow]

    await runSweep()

    expect(requestForBotMock).not.toHaveBeenCalled()
    expect(hostMock.request).not.toHaveBeenCalled()
  })

  it('routes a remote member session through its immutable persisted owner', async () => {
    const owner = {
      name: 'worker',
      route: { connectionId: 'source-a', mode: 'remote', profile: 'worker', targetProfile: 'backend-worker' },
      sourceScoped: true
    }

    groupChats.value = {
      Core: { members: [owner], sessions: { 'source-a::worker': 'remote-room-1' } }
    }
    requestForBotMock.mockRejectedValue(new Error('gateway RPC must not be used'))

    await runSweep()

    expect(hiddenCalls()).toHaveLength(1)

    const [route, options] = hiddenCalls()[0]

    expect(route).toMatchObject({ connectionId: 'source-a', targetProfile: 'backend-worker' })
    expect(options.sessionId).toBe('remote-room-1')
  })

  it('keeps two remote owners of the same session id on their own routes', async () => {
    const owner = (connectionId: string) => ({
      name: 'worker',
      route: { connectionId, mode: 'remote', profile: 'worker', targetProfile: 'backend-worker' },
      sourceScoped: true
    })

    groupChats.value = {
      A: { sessionOwners: { 'source-a::worker': owner('source-a') }, sessions: { 'source-a::worker': 'same-id' } },
      B: { sessionOwners: { 'source-b::worker': owner('source-b') }, sessions: { 'source-b::worker': 'same-id' } }
    }
    requestForBotMock.mockRejectedValue(new Error('gateway RPC must not be used'))

    await runSweep()

    expect(
      hiddenCalls()
        .map(([route]) => route?.connectionId)
        .sort()
    ).toEqual(['source-a', 'source-b'])
    expect(hiddenCalls().every(([, options]) => options.sessionId === 'same-id')).toBe(true)
  })

  it('fails closed on a source-qualified session whose persisted owner is malformed', async () => {
    // A source-qualified key without its immutable owner must never fall
    // through to ambient routing — that hides whichever session happens to
    // carry that id on the active connection.
    groupChats.value = {
      LegacyRemote: {
        sessionOwners: { 'source-a::worker': { name: 'worker' } },
        sessions: { 'source-a::worker': 'same-id' }
      }
    }
    lastRoster.value = [{ name: 'worker' } as RosterRow]

    await runSweep()

    expect(hiddenCalls()).toHaveLength(0)
    expect(requestForBotMock).not.toHaveBeenCalled()
  })
})

describe('the title half: each roster bot’s own profile listing', () => {
  const rowsByProfile: Record<string, SweepRow[]> = {
    alpha: [
      { id: 'a-1', started_at: 1, title: 'Bot Chat' },
      { id: 'a-2', started_at: 1, title: 'Agent Inbox' },
      { id: 'a-3', started_at: 1, title: 'Group: Core' },
      { id: 'a-4', started_at: 1, title: 'My real conversation' },
      // Not an exact title — kept.
      { id: 'a-5', started_at: 1, title: 'Bot Chat notes' },
      // Live draft inside the grace period — kept.
      { id: 'a-6', started_at: 701, title: 'Bot Chat' },
      // Missing age metadata — kept, fail closed.
      { id: 'a-7', title: 'Agent Inbox' },
      // Grace boundary reached — hidden.
      { id: 'a-8', started_at: 700, title: 'Bot Chat' }
    ],
    remy: [{ id: 'r-1', started_at: 1, title: 'Agent Inbox' }]
  }

  beforeEach(() => {
    lastRoster.value = [{ name: 'alpha' }, { connectionId: 'mini', name: 'remy', remoteSource: true }] as RosterRow[]
    hostMock.listPersistedSessions.mockImplementation(async (_route: unknown, options: { profile: string }) => ({
      sessions: rowsByProfile[options.profile] || []
    }))
  })

  it('hides Bot-Mode plumbing titles per roster bot, and only those', async () => {
    await runSweep()

    const lists = hostMock.listPersistedSessions.mock.calls as Array<
      [null | ProfileRoute, { include_hidden?: boolean; profile: string }]
    >

    expect(lists.map(([, options]) => options.profile).sort()).toEqual(['alpha', 'remy'])
    // Visible-rows-only listing keeps the sweep idempotent.
    expect(lists.every(([, options]) => !options.include_hidden)).toBe(true)

    // Exact plumbing titles only — user-titled and brand-new rows stay visible.
    expect(
      hiddenCalls()
        .map(([, options]) => options.sessionId)
        .sort()
    ).toEqual(['a-1', 'a-2', 'a-3', 'a-8', 'r-1'])
    expect(hiddenCalls().every(([, options]) => options.hidden)).toBe(true)
    // Remote-source rows keep their immutable source owner on the REST route.
    expect(hiddenCalls().find(([, options]) => options.sessionId === 'r-1')?.[0]?.connectionId).toBe('mini')
  })

  it('runs beside the id half, and a throwing title sweep never breaks it', async () => {
    // The load/reconnect entrypoint runs BOTH halves; the title sweep is
    // best-effort, so an unreachable source must not cost the known ids.
    groupChats.value = { Core: { sessions: { alpha: 'room-core-a' } } }
    hostMock.listPersistedSessions.mockRejectedValue(new Error('gateway not ready'))

    await runSweep()

    expect(hiddenCalls().map(([, options]) => options.sessionId)).toEqual(['room-core-a'])
  })
})
