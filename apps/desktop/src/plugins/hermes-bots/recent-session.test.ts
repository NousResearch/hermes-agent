/**
 * "Open recent session" on a bot row (hermes-agent#93054): opens the profile's
 * newest LISTED session as a tab in the bot's workspace — never in place of
 * the canonical Bot Chat — and hands a bot with nothing listable back to the
 * ordinary row click.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

const { openRosterBot, openSession, prepareBotSource, notifyError, pluginCtx } = vi.hoisted(() => ({
  openRosterBot: vi.fn(async () => true),
  openSession: vi.fn(async () => undefined),
  prepareBotSource: vi.fn(async () => undefined),
  notifyError: vi.fn(),
  // Null unless a test installs one — the plugin ctx is genuinely absent until
  // register() runs, which is why the failure toast carries an English floor.
  pluginCtx: { current: null as null | { i18n?: { t: (key: string) => string } } }
}))

vi.mock('@hermes/plugin-sdk', () => ({
  BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS: 60_000,
  haptic: () => undefined,
  host: { openSession, notifyError }
}))
vi.mock('./bot-state', () => ({ saveSelectedRosterBot: () => undefined }))
vi.mock('./canonical-chat', () => ({
  prepareBotSource,
  resolveExpectHistory: (summary: { message_count?: number; live_message_count?: number } | null | undefined) =>
    (summary?.live_message_count ?? summary?.message_count ?? 1) > 0
}))
vi.mock('./roster-actions', () => ({ openRosterBot }))
vi.mock('./routing', () => ({
  botConnectionRoute: () => ({ connectionId: 'local', mode: 'local', profile: 'researcher' }),
  botWorkspaceOwnerKey: () => 'bot:local::researcher',
  setBotsWorkspaceOwner: () => undefined
}))
vi.mock('./shared', () => ({ getPluginCtx: () => pluginCtx.current }))

import { botRecentSession, openBotRecentSession } from './recent-session'
import type { RosterRow } from './types'

const bot = (last_session: RosterRow['last_session']) =>
  ({ name: 'researcher', connectionId: 'local', last_session, canonical_session: { id: 'bot-chat-1' } }) as RosterRow

beforeEach(() => {
  openRosterBot.mockClear()
  openSession.mockClear()
  notifyError.mockClear()
  pluginCtx.current = null
})

describe('open recent session', () => {
  it('opens the newest listed session as a tab in the bot workspace, titled after the session', async () => {
    const row = bot({ id: 'sess-9', title: 'Deploy notes', message_count: 4, last_active: 1 })
    expect(botRecentSession(row)).toEqual({ id: 'sess-9', title: 'Deploy notes' })

    await expect(openBotRecentSession(row)).resolves.toBe(true)

    expect(prepareBotSource).toHaveBeenCalledWith(row)
    expect(openSession).toHaveBeenCalledTimes(1)
    const [id, options] = openSession.mock.calls[0] as unknown as [string, Record<string, unknown>]
    expect(id).toBe('sess-9')
    expect(options).toMatchObject({
      intent: 'tab',
      profile: 'researcher',
      workspaceMode: 'bots',
      workspaceOwnerKey: 'bot:local::researcher',
      expectHistory: true
    })
    // The stored row titles the tab; a tabTitle equal to it would caption the
    // tab as the bot (the canonical-chat rule) and hide which session it is.
    expect(options).not.toHaveProperty('tabTitle')
    // The canonical chat is never the target of this item.
    expect(id).not.toBe('bot-chat-1')
    expect(openRosterBot).not.toHaveBeenCalled()
  })

  it('falls back to the ordinary row click when the profile has no listed session', async () => {
    const row = bot(null)
    expect(botRecentSession(row)).toBeNull()

    await expect(openBotRecentSession(row)).resolves.toBe(true)

    expect(openRosterBot).toHaveBeenCalledWith(row)
    expect(openSession).not.toHaveBeenCalled()
  })

  it('surfaces a failed open through notifyError with the English floor when no plugin ctx is installed', async () => {
    openSession.mockRejectedValueOnce(new Error('gateway gone'))

    await expect(openBotRecentSession(bot({ id: 'sess-9', title: 'Deploy notes', message_count: 4, last_active: 1 }))).resolves.toBe(false)

    expect(notifyError).toHaveBeenCalledTimes(1)
    const [, fallback] = notifyError.mock.calls[0] as unknown as [unknown, string]
    expect(fallback).toBe('Could not open the recent session')
  })

  it('localizes the failure toast through the plugin i18n context when one is installed', async () => {
    pluginCtx.current = { i18n: { t: (key: string) => (key === 'bot.openRecentSessionFailed' ? '无法打开最近会话' : key) } }
    openSession.mockRejectedValueOnce(new Error('gateway gone'))

    await expect(openBotRecentSession(bot({ id: 'sess-9', title: 'Deploy notes', message_count: 4, last_active: 1 }))).resolves.toBe(false)

    const [, message] = notifyError.mock.calls[0] as unknown as [unknown, string]
    expect(message).toBe('无法打开最近会话')
  })
})
