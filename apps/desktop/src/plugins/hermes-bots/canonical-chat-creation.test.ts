/**
 * Minting a bot's forever-chat.
 *
 * `session.create` is lazy: the stored row does not exist until something
 * writes to it. Creation therefore materializes and TITLES the row eagerly,
 * before it opens or prompts — until the row carries "Bot Chat" the registry
 * has no entry for this bot, and a second click during that window mints a
 * duplicate forever-chat. Older gateways that reject the eager title keep a
 * narrow compat kickoff, else the pruner reaps the empty lazy session and the
 * chat never survives its own creation.
 *
 * The other half of the contract is navigation: a create still completes
 * registry-side when the user has already moved on, but it must not steal the
 * workspace (#89834 family).
 *
 * Ported from tests/canonical-chat-creation.test.mjs, which sliced the
 * creation section out of the old plugin.js bundle and ran it under `vm`.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { hostMock, persistMock, pluginCtx, requestForBotMock, saveBotMetaMock } = vi.hoisted(() => ({
  hostMock: { openSession: vi.fn(), request: vi.fn() },
  persistMock: vi.fn(),
  // Null unless a test installs one — the plugin ctx is genuinely absent until
  // register() runs, which is why every read of it carries an English floor.
  pluginCtx: { current: null as null | { i18n?: { t: (key: string) => string } } },
  requestForBotMock: vi.fn(),
  saveBotMetaMock: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', () => ({
  BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS: 15_000,
  host: hostMock
}))

vi.mock('./routing', () => ({
  backendTargetProfile: (route: { targetProfile?: string } | null, name: string) => route?.targetProfile ?? name,
  botConnectionRoute: () => null,
  botRosterMeta: () => ({}),
  botWorkspaceOwnerKey: (bot: { connectionId?: string; name?: string } | null) =>
    `bot:${bot?.connectionId ? `${bot.connectionId}::` : ''}${bot?.name || 'default'}`,
  requestForBot: requestForBotMock
}))

vi.mock('./data', () => ({
  $botMeta: { get: () => ({}), set: vi.fn() },
  botMetaKey: (bot: { name?: string }) => bot?.name ?? '',
  botOwner: (owner: RosterRow | string) =>
    typeof owner === 'string'
      ? { bot: { name: owner }, key: owner, name: owner, route: null }
      : { bot: owner, key: owner?.name, name: owner?.name, route: null },
  persistBotMetaSnapshot: persistMock,
  saveBotMeta: saveBotMetaMock
}))

vi.mock('./shared', () => ({ getPluginCtx: () => pluginCtx.current }))

/** Ordered log of everything creation did — RPCs and navigations interleaved,
 *  because the ORDER between them is most of what this suite pins. */
let events: string[]

function respondWith(handler: (method: string, params: Record<string, unknown>) => unknown) {
  requestForBotMock.mockImplementation(async (_bot: unknown, method: string, params: Record<string, unknown>) =>
    handler(method, params ?? {})
  )
}

async function loadModule() {
  // Creation is single-flighted through a module-level map keyed by bot.
  vi.resetModules()

  return import('./canonical-chat')
}

beforeEach(() => {
  vi.clearAllMocks()
  events = []
  pluginCtx.current = null
  hostMock.openSession.mockImplementation(async (id: string) => {
    events.push(`open:${id}`)
  })
})

/** Runs a kickoff creation and hands back the text the intro turn submitted. */
async function kickoffTextSent(): Promise<string> {
  let sent = ''

  respondWith((method, params) => {
    if (method === 'session.create') {
      return { session_id: 'runtime-1', stored_session_id: 'stored-1' }
    }

    if (method === 'prompt.submit') {
      sent = String(params.text ?? '')
    }

    return {}
  })

  const { createCanonicalChat } = await loadModule()

  await createCanonicalChat('ops', { kickoff: true })

  return sent
}

describe('the lazy row is materialized before anything else touches it', () => {
  it('titles the created row, then opens it, and sends no intro', async () => {
    respondWith((method, params) => {
      events.push(method)

      if (method === 'session.create') {
        return { session_id: 'runtime-1', stored_session_id: 'stored-1' }
      }

      if (method === 'session.title') {
        // A throw here is NOT inert: createCanonicalChat reads it as "eager
        // title unsupported" and falls back to the compat kickoff.
        expect(params).toEqual({ session_id: 'runtime-1', title: 'Bot Chat' })
      }

      return {}
    })

    const { createCanonicalChat } = await loadModule()

    // No kickoff option: the click-path mint. The eager title write persists
    // the row, so NO intro turn fires — the user speaks first (ScottFive).
    expect(await createCanonicalChat('ops')).toBe('stored-1')
    expect(events).toEqual(['session.list', 'session.create', 'session.title', 'open:stored-1'])
  })

  it('always creates hidden — Bot Mode sessions have no visibility pref', async () => {
    // Canonical Bot Chats are plugin-owned forever-chats, never scratch
    // conversations, so `hidden` is unconditional. The `$hideBotChats` user
    // pref that used to gate it is gone.
    let created: Record<string, unknown> | null = null

    respondWith((method, params) => {
      if (method === 'session.create') {
        created = params

        return { session_id: 'rt-1', stored_session_id: 'sid-1' }
      }

      return {}
    })

    const { createCanonicalChat } = await loadModule()
    await createCanonicalChat('alpha')

    expect(created).toMatchObject({
      hidden: true,
      title: 'Bot Chat',
      // The PR #97008 contract: the canonical Bot Chat's runtime always
      // follows the profile's CURRENT config on resume — never the stored
      // model/provider pin. Dropping this param silently regresses bots to
      // the server's exact-title legacy fallback.
      follow_profile_config: true
    })
  })

  it('sends the one intro turn on New Bot creation (kickoff: true)', async () => {
    respondWith((method, params) => {
      events.push(method)

      if (method === 'session.create') {
        return { session_id: 'runtime-1', stored_session_id: 'stored-1' }
      }

      if (method === 'prompt.submit') {
        expect(params.session_id).toBe('runtime-1')
      }

      return {}
    })

    const { createCanonicalChat } = await loadModule()

    expect(await createCanonicalChat('ops', { kickoff: true })).toBe('stored-1')
    expect(events).toEqual(['session.list', 'session.create', 'session.title', 'open:stored-1', 'prompt.submit'])
  })

  it('speaks the intro in the active locale (#91827)', async () => {
    // The first line of the forever-chat, and the bot's reply follows its
    // language — so a hardcoded English intro biased the whole conversation.
    pluginCtx.current = { i18n: { t: key => (key === 'bot.kickoff' ? 'こんにちは、自己紹介をしてください！' : key) } }

    expect(await kickoffTextSent()).toBe('こんにちは、自己紹介をしてください！')
  })

  it('falls back to English when the bundle has not registered yet', async () => {
    // Creation can race plugin registration; an unresolved key must never
    // reach the model as the literal `bot.kickoff`.
    expect(await kickoffTextSent()).toBe('Hey, tell me about yourself!')
  })

  it('retries navigation after the compat kickoff when the eager title is unsupported', async () => {
    let attempts = 0

    hostMock.openSession.mockImplementation(async (id: string) => {
      events.push(`open:${id}`)
      attempts += 1

      if (attempts === 1) {
        throw new Error('stored row not persisted yet')
      }
    })
    respondWith(method => {
      if (method === 'session.create') {
        return { session_id: 'runtime-1', stored_session_id: 'stored-1' }
      }

      if (method === 'session.title') {
        throw new Error('unknown method')
      }

      if (method === 'prompt.submit') {
        events.push('kickoff:persisted')
      }

      return {}
    })

    const { createCanonicalChat } = await loadModule()

    expect(await createCanonicalChat('ops')).toBe('stored-1')
    expect(events).toEqual(['open:stored-1', 'kickoff:persisted', 'open:stored-1'])
  })

  it('still returns the created registry row when the intro fails', async () => {
    respondWith(method => {
      if (method === 'session.create') {
        return { session_id: 'rt-1', stored_session_id: 'new-bot-chat' }
      }

      if (method === 'prompt.submit') {
        throw new Error('gateway timeout')
      }

      return {}
    })

    const { createCanonicalChat } = await loadModule()

    // The chat exists under the canonical title — the next click finds it by
    // NAME (the registry), so a failed kickoff can never orphan or fork it.
    expect(await createCanonicalChat('newbie', { kickoff: true })).toBe('new-bot-chat')
  })
})

describe('a superseded click completes registry-side but never navigates', () => {
  it('creates the canonical row without stealing the workspace', async () => {
    let current = true

    respondWith(method => {
      if (method === 'session.create') {
        current = false

        return { session_id: 'new-runtime', stored_session_id: 'new-stored' }
      }

      return {}
    })

    const { createCanonicalChat } = await loadModule()

    expect(await createCanonicalChat('ops', { openingStillCurrent: () => current })).toBe('new-stored')
    expect(hostMock.openSession).not.toHaveBeenCalled()
  })

  it('lets a newer same-bot open take over the in-flight creation navigation', async () => {
    let firstCurrent = true
    let releaseCreate = () => undefined as void
    let markCreateStarted = () => undefined as void

    const createStarted = new Promise<void>(resolve => {
      markCreateStarted = resolve
    })

    respondWith(method => {
      if (method === 'session.create') {
        markCreateStarted()

        return new Promise(resolve => {
          releaseCreate = () => resolve({ session_id: 'shared-runtime', stored_session_id: 'shared-stored' })
        })
      }

      return {}
    })

    const { createCanonicalChat } = await loadModule()
    const first = createCanonicalChat('ops', { openingStillCurrent: () => firstCurrent })

    await createStarted
    firstCurrent = false

    const second = createCanonicalChat('ops', { openingStillCurrent: () => true })

    releaseCreate()

    expect(await first).toBe('shared-stored')
    expect(await second).toBe('shared-stored')
    // The newer current click owns the shared creation's one navigation.
    expect(hostMock.openSession).toHaveBeenCalledTimes(1)
  })
})
