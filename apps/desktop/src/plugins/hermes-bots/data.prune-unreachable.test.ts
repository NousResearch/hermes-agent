/**
 * pruneUnreachableRemoteRows — a remote roster row's contents are metadata of
 * an AUTHENTICATED gateway, learned while logged in and cached locally. While
 * that source is unreachable or removed, the row must not render: an offline
 * cache must not advertise another machine's profile inventory. Local rows,
 * connect-on-demand sources, and unknown-health sources stay visible.
 */

import { describe, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    host: { state: { connectionId: { get: () => 'local' }, profile: { get: () => 'default' } } },
    queryClient: {
      getQueriesData: () => [],
      getQueryData: () => null,
      invalidateQueries: vi.fn(),
      setQueryData: vi.fn()
    },
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

vi.mock('./routing', () => ({
  botRosterMeta: () => null,
  botRosterKey: (bot: { connectionId?: string; name?: string }) => `${bot?.connectionId || ''}::${bot?.name}`
}))

vi.mock('./hidden-bots', () => ({
  isBotHidden: () => false,
  isBotPinned: () => false
}))

vi.mock('./roster-sections', () => ({
  filterBotsByGateway: (rows: unknown[]) => rows,
  groupMatchesRosterFilters: () => true,
  rosterGatewaySections: () => ({ sectioned: false })
}))

vi.mock('./group-chat', () => ({ $groupChats: { get: () => ({}) } }))

vi.mock('./group-membership', () => ({
  groupChatNames: () => [],
  groupChatMemberBots: () => [],
  groupLastActivity: () => 0
}))

vi.mock('./group-order', () => ({ sortGroupRosterRows: (rows: unknown[]) => rows }))

vi.mock('./row-helpers', () => ({
  BOT_ROSTER_SEARCH_THRESHOLD: 10,
  ACTIVE_WINDOW_S: 300,
  rosterActivityMatches: () => true
}))

const { pruneUnreachableRemoteRows } = await import('./data')
const { deriveRosterRows } = await import('./roster-pane-derivation')

const row = (over: Record<string, unknown>) => ({ name: 'bot', ...over }) as never

describe('pruneUnreachableRemoteRows', () => {
  it('keeps every local row regardless of source health', () => {
    const rows = [row({ name: 'default' }), row({ name: 'writer' })]

    expect(pruneUnreachableRemoteRows(rows)).toHaveLength(2)
  })

  it('drops remote rows marked unreachable by the source-health layer', () => {
    const rows = [
      row({ remoteSource: true, connectionId: 'vm01', sourceReachable: false }),
      row({ remoteSource: true, connectionId: 'vm02', sourceReachable: true })
    ]

    const kept = pruneUnreachableRemoteRows(rows)

    expect(kept).toHaveLength(1)
    expect(kept[0]).toMatchObject({ connectionId: 'vm02' })
  })

  it('drops remote rows whose gateway was removed', () => {
    const rows = [row({ remoteSource: true, connectionId: 'gone', sourceMissing: true })]

    expect(pruneUnreachableRemoteRows(rows)).toHaveLength(0)
  })

  it('drops remote rows with a source error other than connect-on-demand', () => {
    const rows = [row({ remoteSource: true, connectionId: 'vm01', sourceError: 'connection refused' })]

    expect(pruneUnreachableRemoteRows(rows)).toHaveLength(0)
  })

  it('keeps connect-on-demand and unknown-health remote rows', () => {
    const rows = [
      row({ remoteSource: true, connectionId: 'ssh-box', sourceError: 'connect-on-demand' }),
      row({ remoteSource: true, connectionId: 'vm03' })
    ]

    expect(pruneUnreachableRemoteRows(rows)).toHaveLength(2)
  })

  it('treats null/undefined input as an empty list', () => {
    expect(pruneUnreachableRemoteRows(null)).toEqual([])
    expect(pruneUnreachableRemoteRows(undefined)).toEqual([])
  })
})

describe('deriveRosterRows display pruning', () => {
  const base = {
    allMeta: {},
    gatewayFilter: 'all',
    query: '',
    activityFilter: 'all' as never,
    rowKindFilter: 'all' as never,
    groupRooms: {},
    activeRosterKeys: new Set<string>(),
    gatewayOptions: [] as never[],
    activityOf: () => 0,
    isPinned: () => false
  }

  it('never paints a remote row whose gateway is unreachable or removed', () => {
    const roster = [
      row({ name: 'local-bot' }),
      row({ name: 'down', remoteSource: true, connectionId: 'vm01', sourceReachable: false }),
      row({ name: 'gone', remoteSource: true, connectionId: 'vm02', sourceMissing: true }),
      row({ name: 'up', remoteSource: true, connectionId: 'vm03', sourceReachable: true })
    ]

    const { rosterRows, visibleRoster } = deriveRosterRows({ ...base, roster })

    const names = rosterRows.filter(row => row.kind === 'bot').map(row => row.bot.name)

    expect(names).toEqual(['local-bot', 'up'])
    expect(visibleRoster).toHaveLength(2)
  })
})
