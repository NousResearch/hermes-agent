import type * as HermesSdk from '@hermes/plugin-sdk'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { focusedStoredSessionId, invalidateQueries, openBotCanonicalChat, requestForBot } = vi.hoisted(() => {
  let focused = 'canonical-tip'

  return {
    focusedStoredSessionId: {
      get: () => focused,
      set: (value: string) => {
        focused = value
      }
    },
    invalidateQueries: vi.fn(),
    openBotCanonicalChat: vi.fn(),
    requestForBot: vi.fn()
  }
})

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: { ...sdk.host, state: { ...sdk.host.state, focusedStoredSessionId } },
    queryClient: { invalidateQueries }
  }
})

vi.mock('./canonical-chat', () => ({
  isCanonicalChatOnScreen: (bot: RosterRow, storedId: string) =>
    [bot.canonical_session?.id, bot.canonical_session?.resolved_id].filter(Boolean).includes(storedId),
  openBotCanonicalChat
}))

vi.mock('./routing', () => ({
  backendTargetProfile: (_route: unknown, fallback: string) => (fallback === 'worker' ? 'backend-worker' : fallback),
  botConnectionRoute: () => ({
    connectionId: 'remote-a',
    mode: 'remote',
    profile: 'worker',
    targetProfile: 'backend-worker'
  }),
  requestForBot
}))

const { clearBotCanonicalChat } = await import('./bot-clear')

const bot = {
  canonical_session: { id: 'canonical-root', resolved_id: 'canonical-tip' },
  name: 'worker'
} as RosterRow

beforeEach(() => {
  vi.clearAllMocks()
  focusedStoredSessionId.set('canonical-tip')
  requestForBot.mockResolvedValue({ cleared: true })
  invalidateQueries.mockResolvedValue(undefined)
  openBotCanonicalChat.mockResolvedValue('canonical-tip')
})

describe('clearBotCanonicalChat', () => {
  it('targets the owner profile without sending or storing a canonical session id, then refreshes the focused chat', async () => {
    await clearBotCanonicalChat(bot)

    expect(requestForBot).toHaveBeenCalledWith(bot, 'session.clear_bot_chat', { profile: 'backend-worker' })
    expect(requestForBot.mock.calls[0][2]).not.toHaveProperty('session_id')
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ['hermes-bots', 'roster'] })
    expect(openBotCanonicalChat).toHaveBeenCalledWith(bot, expect.any(Function))
  })

  it('refreshes roster truth without stealing focus when a different chat is on screen', async () => {
    focusedStoredSessionId.set('some-other-chat')

    await clearBotCanonicalChat(bot)

    expect(invalidateQueries).toHaveBeenCalledOnce()
    expect(openBotCanonicalChat).not.toHaveBeenCalled()
  })

  it('reports an older gateway as unavailable and never attempts a client-side fallback', async () => {
    requestForBot.mockRejectedValue(new Error('Method not found (-32601)'))

    await expect(clearBotCanonicalChat(bot)).rejects.toThrow(/Update the gateway/)

    expect(requestForBot).toHaveBeenCalledOnce()
    expect(invalidateQueries).not.toHaveBeenCalled()
    expect(openBotCanonicalChat).not.toHaveBeenCalled()
  })
})
