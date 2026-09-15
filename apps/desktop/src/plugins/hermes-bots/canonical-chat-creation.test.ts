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
  hostMock: { agents: vi.fn(), openSession: vi.fn(), request: vi.fn() },
  persistMock: vi.fn(),
  // Null unless a test installs one — the plugin ctx is genuinely absent until
  // register() runs, which is why every read of it carries an English floor.
  pluginCtx: { current: null as null | { i18n?: { t: (key: string) => string } } },
  requestForBotMock: vi.fn(),
  saveBotMetaMock: vi.fn()
}))

// test-only: loads the REAL shared resolver module and re-exports its own
// exact behavior as the `vi.mock('@hermes/plugin-sdk', ...)` fixture below,
// so the mock can never silently drift from the module the plugin fence
// forbids importing at runtime.
// eslint-disable-next-line no-restricted-imports
import { CANONICAL_AGENT_CHAT_TITLE, isAuthorizedCanonicalChatTarget, isCanonicalAgentChatRow, isTitleConflictError, resolveCanonicalAgentChat } from '../../lib/canonical-agent-chat'

vi.mock('@hermes/plugin-sdk', () => ({
  CANONICAL_AGENT_CHAT_TITLE,
  isAuthorizedCanonicalChatTarget,
  isCanonicalAgentChatRow,
  isTitleConflictError,
  resolveCanonicalAgentChat,
  BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS: 15_000,
  host: hostMock
}))

vi.mock('./routing', () => ({
  aliasIdentityFor: () => null,
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
  // Roster admission (Architect corrective, 2026-09-02) runs before every
  // canonical resolution — authorize every profile string this suite uses.
  hostMock.agents.mockImplementation(async () => ({
    agents: [
      { connectionId: null, profile: 'ops', targetProfile: 'ops' },
      { connectionId: null, profile: 'alpha', targetProfile: 'alpha' },
      { connectionId: null, profile: 'newbie', targetProfile: 'newbie' }
    ]
  }))
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

  it('scopes the open to the bots workspace even with no staleness probe', async () => {
    // The create path is the one caller that passes no probe. Gating the
    // workspace fields on the probe left ITS chat unscoped, so the composer
    // could not tell a bot chat from a working session and kept the branch
    // rail up until the next (probed) click reopened the same row.
    respondWith(method =>
      method === 'session.create' ? { session_id: 'runtime-1', stored_session_id: 'stored-1' } : {}
    )

    const { createCanonicalChat } = await loadModule()

    await createCanonicalChat('ops', { kickoff: true })

    expect(hostMock.openSession).toHaveBeenCalledWith(
      'stored-1',
      expect.objectContaining({ tabTitle: 'Bot Chat', workspaceMode: 'bots', workspaceOwnerKey: 'bot:ops' })
    )
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
    // Post-corrective (Architect, 2026-09-02, fourth pass): every caller now
    // performs its OWN current-roster admission read before reaching the
    // shared resolver — there is no bypass for a caller joining an
    // already-in-flight creation. That async admission hop means a second
    // caller arriving shortly after the first can legitimately land AFTER
    // the first flight has already fully settled and removed itself from
    // the in-process singleflight map — so it independently issues its own
    // `session.create`/`session.title`, exactly like two genuinely
    // sequential opens would. The pre-existing, separately-tested
    // adopt-before-mint conflict path (a title-uniqueness rejection means
    // someone else won the registry) is what reconciles the two into ONE
    // canonical identity — the in-memory map is no longer relied on to
    // prevent this from happening across an admission-check-sized gap; the
    // backend's UNIQUE(title) index is the real, cross-process guarantee,
    // and this test now exercises that path instead of assuming a shared
    // flight the new corrective made non-guaranteed.
    let firstCurrent = true
    let createCalls = 0
    let titled: { id: string } | null = null
    const idBySessionId = new Map<string, string>()

    respondWith((method, params) => {
      if (method === 'session.list') {
        // The registry read every caller performs first (adoption check)
        // and the re-lookup after a title conflict.
        return titled ? { sessions: [{ id: titled.id, resolved_id: titled.id, message_count: 0, title: 'Bot Chat' }] } : { sessions: [] }
      }

      if (method === 'session.create') {
        createCalls += 1
        const runtimeId = `runtime-${createCalls}`
        const storedId = `stored-${createCalls}`

        idBySessionId.set(runtimeId, storedId)

        return { session_id: runtimeId, stored_session_id: storedId }
      }

      if (method === 'session.title') {
        if (titled) {
          throw new Error(`Title 'Bot Chat' is already in use by session ${titled.id}`)
        }

        const storedId = idBySessionId.get(String(params.session_id))!

        titled = { id: storedId }

        return {}
      }

      return {}
    })

    const { createCanonicalChat } = await loadModule()
    const first = createCanonicalChat('ops', { openingStillCurrent: () => firstCurrent })

    // Once the first call's own async admission read resolves, flip current
    // — this models the click having moved on while the first open was
    // still in progress — then fire the second, newer-and-current call.
    firstCurrent = false
    const second = createCanonicalChat('ops', { openingStillCurrent: () => true })

    const [firstId, secondId] = await Promise.all([first, second])

    // Both callers resolve to the SAME canonical identity — whichever one's
    // session.title call actually won the race, the other adopted it via
    // the conflict path, exactly as two genuinely sequential opens would.
    expect(firstId).toBe(secondId)
    // Only the CURRENT call navigates; the stale first click does not.
    expect(hostMock.openSession).toHaveBeenCalledTimes(1)
  })

  it('evaluates roster admission PER CALLER at the time that caller requests entry — Caller 2 revoked mid-flight is rejected on its OWN fresh read, before any session RPC (Architect corrective, 2026-09-02, fifth pass)', async () => {

    // CALLER 1: fresh roster read authorizes 'ops'; enters canonical
    // creation; creation is held in flight via a controlled session.create.
    let releaseCreate = () => undefined as void
    let markCreateStarted = () => undefined as void

    const createStarted = new Promise<void>(resolve => {
      markCreateStarted = resolve
    })

    let rosterReads = 0
    hostMock.agents.mockImplementation(async () => {
      rosterReads += 1

      // CALLER 1's fresh read (the 1st call): 'ops' is authorized.
      // CALLER 2's fresh read (every subsequent call): 'ops' has been
      // revoked from the current roster — this is the exact "authorized
      // then revoked mid-flight" scenario the corrective requires.
      return rosterReads === 1
        ? { agents: [{ connectionId: null, profile: 'ops', targetProfile: 'ops' }] }
        : { agents: [] }
    })

    const rpcCalls: string[] = []

    respondWith(method => {
      rpcCalls.push(method)

      if (method === 'session.create') {
        markCreateStarted()

        return new Promise(resolve => {
          releaseCreate = () => resolve({ session_id: 'runtime-1', stored_session_id: 'stored-1' })
        })
      }

      return {}
    })

    const { createCanonicalChat } = await loadModule()
    const first = createCanonicalChat('ops')

    // CALLER 1 is now in flight (past its own admission, inside
    // session.create). CALLER 2 starts NOW, while Caller 1 is still
    // in flight, and performs its OWN fresh roster read.
    await createStarted
    const rpcCallsBeforeSecond = rpcCalls.length
    const second = createCanonicalChat('ops')

    // CALLER 2 is rejected on its own fresh roster state — BEFORE reaching
    // the shared resolver or issuing any session RPC. No admission
    // shortcut based on Caller 1's in-flight state exists: Caller 2 does
    // NOT inherit Caller 1's now-stale authorization.
    await expect(second).rejects.toThrow(/not present in the current authorized agent roster/)

    // The roster capability was called separately for each caller — not
    // once and cached/shared.
    expect(rosterReads).toBeGreaterThanOrEqual(2)

    // Caller 2 performed NO session RPC of any kind — the RPC log gained
    // zero new entries between Caller 2 starting and Caller 2's rejection.
    expect(rpcCalls.length).toBe(rpcCallsBeforeSecond)
    expect(rpcCalls).not.toContain('session.title')
    expect(hostMock.openSession).not.toHaveBeenCalled()

    // CALLER 1 completes normally — its own in-flight creation is
    // unaffected by Caller 2's independent rejection.
    releaseCreate()
    await expect(first).resolves.toBe('stored-1')
    expect(hostMock.openSession).toHaveBeenCalledTimes(1)
  })
})
