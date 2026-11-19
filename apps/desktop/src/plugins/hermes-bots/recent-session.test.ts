/**
 * "Open recent session" on a bot row (hermes-agent#93054): opens the profile's
 * newest LISTED session as a tab in the bot's workspace — never in place of
 * the canonical Bot Chat — and hands a bot with nothing listable back to the
 * ordinary row click.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

const { openRosterBot, openSession, prepareBotSource } = vi.hoisted(() => ({
  openRosterBot: vi.fn(async () => true),
  openSession: vi.fn(async () => undefined),
  prepareBotSource: vi.fn(async () => undefined)
}))

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, resolve, reject }
}

// The real shared module loads (recent-session reads $pendingBotOpen from it),
// so the sdk mock must supply the atom factory it binds.
vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS: 60_000,
    haptic: () => undefined,
    host: { openSession, notifyError: vi.fn() }
  }
})
vi.mock('./bot-state', () => ({ saveSelectedRosterBot: () => undefined }))
vi.mock('./canonical-chat', () => ({
  prepareBotSource,
  resolveExpectHistory: (summary: { message_count?: number; live_message_count?: number } | null | undefined) =>
    (summary?.live_message_count ?? summary?.message_count ?? 1) > 0
}))
// Same key contract as data.ts (connectionId::name) without pulling the live
// roster module into this minimal graph.
vi.mock('./data', () => ({ botRosterKey: (bot: { connectionId: string; name: string }) => `${bot.connectionId}::${bot.name}` }))
vi.mock('./roster-actions', () => ({ openRosterBot }))
vi.mock('./routing', () => ({
  botConnectionRoute: () => ({ connectionId: 'local', mode: 'local', profile: 'researcher' }),
  botWorkspaceOwnerKey: () => 'bot:local::researcher',
  setBotsWorkspaceOwner: () => undefined
}))

import { botRecentSession, openBotRecentSession } from './recent-session'
import { $pendingBotOpen } from './shared'
import type { RosterRow } from './types'

const bot = (last_session: RosterRow['last_session']) =>
  ({ name: 'researcher', connectionId: 'local', last_session, canonical_session: { id: 'bot-chat-1' } }) as RosterRow

beforeEach(() => {
  $pendingBotOpen.set(null)
  openRosterBot.mockClear()
  openSession.mockClear()
  prepareBotSource.mockClear()
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

  it('marks the deliberate pick in flight and settles the mark on success and failure', async () => {
    const row = bot({ id: 'sess-9', title: 'Deploy notes', message_count: 4, last_active: 1 })

    // In flight: the ambient reconcile reads $pendingBotOpen and defers, so
    // this door's publish survives a roster refresh the same way the row
    // click's does (#154651 review follow-up). The arm is synchronous —
    // openBotRecentSession sets it before its first await.
    const flight = deferred<undefined>()
    prepareBotSource.mockImplementationOnce(() => flight.promise)
    const opened = openBotRecentSession(row)

    expect($pendingBotOpen.get()).toMatchObject({ key: 'local::researcher' })

    flight.resolve(undefined)
    await expect(opened).resolves.toBe(true)
    expect($pendingBotOpen.get()).toBeNull()

    // A failed open settles too — the pick stands, the defer does not.
    const failure = deferred<undefined>()
    prepareBotSource.mockImplementationOnce(() => failure.promise)
    const failed = openBotRecentSession(row)

    expect($pendingBotOpen.get()).toMatchObject({ key: 'local::researcher' })

    failure.reject(new Error('unreachable'))
    await expect(failed).resolves.toBe(false)
    expect($pendingBotOpen.get()).toBeNull()
  })
})
