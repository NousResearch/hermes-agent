/**
 * The unread/toast poll every roster refresh feeds.
 *
 * This poll is the ONLY unread signal a canonical Bot Chat can have: Bot Chats
 * are unconditionally hidden, so they never reach the session list the
 * backend's own watermark iterates, and a delivery from the CLI, cron, another
 * bot, or another machine never touches this window's live turn edge either.
 * Hence the local watermark per source-qualified bot.
 *
 * The toasts themselves are opt-in and default OFF — a busy roster (cron runs,
 * bot-to-bot chatter) turns them into a firehose, and the dot already carries
 * the signal. The unread MARK is recorded either way, and it goes through
 * core's store (`markSessionUnreadFinished`) rather than a plugin-local map,
 * so the roster row's `SessionStatusDot` and the badge can never drift apart.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { hostMock, markUnreadMock, storageMock } = vi.hoisted(() => ({
  hostMock: {
    notify: vi.fn(),
    request: vi.fn(),
    state: { connectionId: { get: () => 'local' }, focusedSessionOwner: null, profile: { get: () => 'default' } }
  },
  markUnreadMock: vi.fn(),
  storageMock: { get: vi.fn(), set: vi.fn() }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    ackStoredSessionId: vi.fn(),
    atom,
    haptic: vi.fn(),
    host: hostMock,
    markSessionUnreadFinished: markUnreadMock,
    queryClient: { invalidateQueries: vi.fn() },
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({
  bumpBotOpenGeneration: vi.fn(),
  getBotOpenGeneration: vi.fn(),
  getPluginCtx: () => ({ storage: storageMock }),
  ID: 'hermes-bots'
}))

// The open path drags the whole group-chat surface in; the poll under test
// touches none of it.
vi.mock('./canonical-chat', () => ({
  notifyBotOpenFailure: vi.fn(),
  openBotCanonicalChat: vi.fn(),
  prepareBotSource: vi.fn()
}))
vi.mock('./group-chat', async () => {
  const { atom } = await import('nanostores')

  return { $groupChats: atom({}), $groupChatWorkspace: atom<null | string>(null) }
})
vi.mock('./group-chat-view', () => ({ openGroupChat: vi.fn() }))
vi.mock('./group-membership', () => ({ liveGroupChatNames: () => [] }))
vi.mock('./group-panes', () => ({ closeGroupChatMainTab: vi.fn() }))

/** A bot whose canonical Bot Chat carries the activity — the only shape that
 *  can be marked unread, since the marker is keyed by canonical session id. */
const chatting = (name: string, lastActive: number, preview = 'hello'): RosterRow =>
  ({ canonical_session: { id: `${name}-chat`, last_active: lastActive, preview }, name }) as RosterRow

async function loadActions() {
  vi.resetModules()

  const [actions, botState, data] = await Promise.all([
    import('./roster-actions'),
    import('./bot-state'),
    import('./data')
  ])

  return { ...actions, ...botState, $botMeta: data.$botMeta }
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('the first poll only seeds watermarks', () => {
  it('marks nothing unread for history that predates the mount', async () => {
    const { trackInboundActivity } = await loadActions()

    trackInboundActivity([chatting('researcher', 5000)])

    expect(markUnreadMock).not.toHaveBeenCalled()
    expect(hostMock.notify).not.toHaveBeenCalled()
  })
})

describe('new activity after the seed', () => {
  it('marks the canonical chat unread through core’s store', async () => {
    const { trackInboundActivity } = await loadActions()

    trackInboundActivity([chatting('researcher', 5000)])
    trackInboundActivity([chatting('researcher', 6000)])

    expect(markUnreadMock).toHaveBeenCalledWith('researcher-chat', 'researcher')
  })

  it('stays silent unless the user opted into toasts', async () => {
    const { $activityToasts, trackInboundActivity } = await loadActions()

    trackInboundActivity([chatting('researcher', 5000)])
    trackInboundActivity([chatting('researcher', 6000)])
    expect(hostMock.notify).not.toHaveBeenCalled()

    $activityToasts.set(true)
    trackInboundActivity([chatting('researcher', 7000, 'a plain update')])

    expect(hostMock.notify).toHaveBeenCalledTimes(1)
    expect(hostMock.notify.mock.calls[0][0]).toMatchObject({ kind: 'info', message: 'a plain update' })
  })

  it('titles a bot-to-bot delivery differently from ordinary activity', async () => {
    const { $activityToasts, trackInboundActivity } = await loadActions()

    $activityToasts.set(true)
    trackInboundActivity([chatting('researcher', 5000)])
    trackInboundActivity([chatting('researcher', 6000, 'Message from 🤖 manager (@manager): ship it')])

    expect(hostMock.notify.mock.calls[0][0].title).toMatch(/New message for/)

    trackInboundActivity([chatting('researcher', 7000, 'finished the run')])
    expect(hostMock.notify.mock.calls[1][0].title).toMatch(/has new activity/)
  })

  it('never badges the bot the user is currently looking at', async () => {
    const { $selectedBot, trackInboundActivity } = await loadActions()

    trackInboundActivity([chatting('researcher', 5000)])
    $selectedBot.set('researcher')
    trackInboundActivity([chatting('researcher', 6000)])

    expect(markUnreadMock).not.toHaveBeenCalled()
  })

  it('keeps marking a roster-hidden bot but never toasts it', async () => {
    const { $activityToasts, $botMeta, trackInboundActivity } = await loadActions()

    $activityToasts.set(true)
    $botMeta.set({ researcher: { hidden: true } })
    trackInboundActivity([chatting('researcher', 5000)])
    trackInboundActivity([chatting('researcher', 6000)])

    // Unhiding must reveal the dot, so the mark accumulates silently.
    expect(markUnreadMock).toHaveBeenCalledWith('researcher-chat', 'researcher')
    expect(hostMock.notify).not.toHaveBeenCalled()
  })

  it('sees a DM delivered into the hidden Bot Chat that last_session cannot', async () => {
    // The whole reason watermarks follow botActivitySession: a stale visible
    // session would otherwise hold the watermark and swallow the DM.
    const { trackInboundActivity } = await loadActions()

    const withStaleVisible = (lastActive: number) =>
      ({
        canonical_session: { id: 'dm', last_active: lastActive, preview: 'ping' },
        last_session: { id: 'scratch', last_active: 10 },
        name: 'dixie'
      }) as RosterRow

    trackInboundActivity([withStaleVisible(5000)])
    trackInboundActivity([withStaleVisible(6000)])

    expect(markUnreadMock).toHaveBeenCalledWith('dm', 'dixie')
  })

  it('tracks two same-named bots on different connections independently', async () => {
    const { trackInboundActivity } = await loadActions()

    const scoped = (connectionId: string, lastActive: number) =>
      ({
        canonical_session: { id: `${connectionId}-chat`, last_active: lastActive },
        connectionId,
        name: 'default',
        route: { connectionId, mode: 'remote', profile: 'default', targetProfile: 'default' },
        sourceScoped: true
      }) as RosterRow

    trackInboundActivity([scoped('a', 5000), scoped('b', 5000)])
    trackInboundActivity([scoped('a', 6000), scoped('b', 5000)])

    expect(markUnreadMock.mock.calls).toEqual([['a-chat', 'default']])
  })
})

describe('the toast preference', () => {
  it('persists through the plugin storage without throwing when it is absent', async () => {
    const { $activityToasts, setActivityToasts } = await loadActions()

    setActivityToasts(true)
    expect($activityToasts.get()).toBe(true)
    expect(storageMock.set).toHaveBeenCalledWith('activity-toasts', true)

    storageMock.set.mockImplementation(() => {
      throw new Error('storage unavailable')
    })
    expect(() => setActivityToasts(false)).not.toThrow()
    expect($activityToasts.get()).toBe(false)
  })
})
