import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { host } = await import('@hermes/plugin-sdk')
const { $groupChatWorkspace } = await import('./group-chat')

const { isBrowsableBotConversation, loadBotConversationHistory, openBotConversation } = await import(
  './conversation-history'
)

type MutableHost = {
  focusOpenWorkspaceSession: typeof host.focusOpenWorkspaceSession
  listPersistedSessions: typeof host.listPersistedSessions
  openSession: typeof host.openSession
  setWorkspaceScope: typeof host.setWorkspaceScope
}

const mutableHost = host as unknown as MutableHost

const originals = {
  focusOpenWorkspaceSession: host.focusOpenWorkspaceSession,
  listPersistedSessions: host.listPersistedSessions,
  openSession: host.openSession,
  setWorkspaceScope: host.setWorkspaceScope
}

const bot = {
  canonical_session: { id: 'bot-chat-root', resolved_id: 'bot-chat-tip' },
  connectionId: 'source-a',
  connectionKind: 'remote',
  name: 'alpha',
  remoteSource: true,
  route: {
    connectionId: 'source-a',
    mode: 'remote',
    profile: 'alpha',
    targetProfile: 'backend-alpha'
  },
  sourceScoped: true
} as RosterRow

beforeEach(() => {
  vi.clearAllMocks()
  $groupChatWorkspace.set(null)
})

afterEach(() => {
  mutableHost.focusOpenWorkspaceSession = originals.focusOpenWorkspaceSession
  mutableHost.listPersistedSessions = originals.listPersistedSessions
  mutableHost.openSession = originals.openSession
  mutableHost.setWorkspaceScope = originals.setWorkspaceScope
})

describe('bot conversation history enumeration', () => {
  it('filters Bot Mode plumbing and recognizes the canonical compression lineage', () => {
    expect(
      isBrowsableBotConversation(bot, {
        id: 'bot-chat-middle',
        _lineage_ids: ['bot-chat-root', 'bot-chat-middle', 'bot-chat-tip'],
        message_count: 8,
        title: 'Renamed by compression'
      })
    ).toBe(false)
    expect(isBrowsableBotConversation(bot, { id: 'group', message_count: 4, title: 'Group: Build Council' })).toBe(
      false
    )
    expect(isBrowsableBotConversation(bot, { id: 'cron', message_count: 4, source: 'cron', title: 'Daily run' })).toBe(
      false
    )
    expect(
      isBrowsableBotConversation(bot, {
        id: 'side-thread',
        message_count: 4,
        title: 'Recovery thread'
      })
    ).toBe(true)
  })

  it('filters compressed Group and Agent Inbox plumbing by lineage root title', () => {
    expect(
      isBrowsableBotConversation(bot, {
        id: 'group-tip',
        _lineage_root_title: 'Group: Build Council',
        message_count: 4,
        title: null
      })
    ).toBe(false)
    expect(
      isBrowsableBotConversation(bot, {
        id: 'inbox-tip',
        _lineage_root_title: 'Agent Inbox',
        message_count: 4,
        title: 'Generated continuation title'
      })
    ).toBe(false)
  })

  it('reads every page, includes archived conversations, deduplicates, and returns recent-first', async () => {
    const listPersistedSessions = vi
      .fn()
      .mockResolvedValueOnce({
        sessions: [
          { id: 'older', last_active: 10, message_count: 2, title: 'Older' },
          { id: 'bot-chat-tip', last_active: 30, message_count: 8, title: 'Bot Chat' }
        ],
        total: 601
      })
      .mockResolvedValueOnce({
        sessions: [{ archived: true, id: 'archived', last_active: 20, message_count: 3, title: 'Archived work' }],
        total: 601
      })
      .mockResolvedValueOnce({
        sessions: [{ archived: true, id: 'archived', last_active: 20, message_count: 3, title: 'Archived work' }],
        total: 601
      })
      .mockResolvedValueOnce({
        sessions: [{ id: 'newest', last_active: 40, message_count: 2, title: 'Newest' }],
        total: 601
      })

    mutableHost.listPersistedSessions = listPersistedSessions

    await expect(loadBotConversationHistory(bot)).resolves.toMatchObject([
      { id: 'newest' },
      { id: 'archived', archived: true },
      { id: 'older' }
    ])
    expect(listPersistedSessions).toHaveBeenNthCalledWith(1, bot.route, {
      profile: 'backend-alpha',
      limit: 200,
      offset: 0,
      minMessages: 1,
      archived: 'include',
      order: 'recent',
      includeHidden: true
    })
    expect(listPersistedSessions).toHaveBeenNthCalledWith(2, bot.route, {
      profile: 'backend-alpha',
      limit: 200,
      offset: 200,
      minMessages: 1,
      archived: 'include',
      order: 'recent',
      includeHidden: true
    })
    expect(listPersistedSessions).toHaveBeenNthCalledWith(4, bot.route, {
      profile: 'backend-alpha',
      limit: 200,
      offset: 600,
      minMessages: 1,
      archived: 'include',
      order: 'recent',
      includeHidden: true
    })
  })
})

describe('opening a recovered bot conversation', () => {
  it('opens a source-routed normal tab in the existing Bot workspace', async () => {
    const focusOpenWorkspaceSession = vi.fn()

    const openSession = vi.fn(
      async (_storedSessionId: string, _options?: Parameters<typeof host.openSession>[1]) => undefined
    )

    const setWorkspaceScope = vi.fn()
    mutableHost.focusOpenWorkspaceSession = focusOpenWorkspaceSession
    mutableHost.openSession = openSession
    mutableHost.setWorkspaceScope = setWorkspaceScope

    await openBotConversation(bot, {
      id: 'side-thread',
      message_count: 5,
      preview: 'Recover this side conversation',
      title: 'Recovery thread'
    })

    expect(setWorkspaceScope).toHaveBeenCalledWith('bots', 'bot:source-a::alpha', {
      kind: 'route',
      route: bot.route
    })
    expect(openSession).toHaveBeenCalledWith(
      'side-thread',
      expect.objectContaining({
        route: bot.route,
        profile: 'alpha',
        intent: 'tab',
        keepAllProfilesScope: true,
        workspaceMode: 'bots',
        workspaceOwnerKey: 'bot:source-a::alpha',
        tabTitle: 'Recovery thread'
      })
    )

    const options = vi.mocked(openSession).mock.calls[0]?.[1]

    expect(options).not.toHaveProperty('awaitHydration')
    expect(options).not.toHaveProperty('forceResume')
    expect(options).not.toHaveProperty('retryHydrationTimeoutOnce')
    expect(focusOpenWorkspaceSession).toHaveBeenCalledWith('bot:source-a::alpha', undefined, ['side-thread'])
  })
})
