/**
 * The canonical-chat REGISTRY contract.
 *
 * A bot's forever-chat has exactly ONE identity: the session titled "Bot Chat"
 * on that bot's profile. The core UNIQUE(title) index makes (profile, "Bot
 * Chat") an exact registry — at most one row, resolved fresh on every open via
 * `session.list { title: 'Bot Chat', include_hidden: true }`.
 *
 * There is NO session-id pin. The previous design stored a pointer in
 * ui_meta['hermes-bots'].chat and spent five hardening waves (#88690, #90732,
 * #90751, the #91791 revert, #92042) guarding its failure modes: rows[0]
 * steals, last_session adoptions, transient clears, drifted-title welds. Every
 * "lost canonical chat" incident traced to that pointer dangling and a later
 * guard welding the wrong session in. Name-as-identity removes the failure
 * class instead of guarding it: a name cannot dangle.
 *
 * This suite pins the whole contract:
 *   1. open = registry lookup → open the row (lineage tip)
 *   2. no row → create (adopt-before-mint lives inside creation)
 *   3. no pointer is ever read or written on the open path
 *
 * It drives the real module. Its predecessor sliced `plugin.js` out of the
 * bundle with string offsets and ran it under `vm`, which meant the tripwire
 * below could only be a regex over source text; here it is an assertion about
 * what the code DOES — no write reaches the metadata store, and no RPC carries
 * an id to verify.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { hostMock, persistMock, requestForBotMock, saveBotMetaMock } = vi.hoisted(() => ({
  hostMock: { openSession: vi.fn(), request: vi.fn() },
  persistMock: vi.fn(),
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
  // Local bots carry no route, so everything resolves onto the ambient
  // request — the single-connection shape these tests pin.
  botOwner: (owner: RosterRow | string) =>
    typeof owner === 'string'
      ? { bot: { name: owner }, key: owner, name: owner, route: null }
      : { bot: owner, key: owner?.name, name: owner?.name, route: null },
  persistBotMetaSnapshot: persistMock,
  saveBotMeta: saveBotMetaMock
}))

vi.mock('./shared', () => ({ getPluginCtx: () => null }))

/** Route every RPC through one table, recording what was asked. */
function respondWith(handler: (method: string, params: Record<string, unknown>) => unknown) {
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []

  requestForBotMock.mockImplementation(async (_bot: unknown, method: string, params: Record<string, unknown>) => {
    calls.push({ method, params: structuredClone(params ?? {}) })

    return handler(method, params)
  })

  return calls
}

async function loadModule() {
  vi.resetModules()

  return import('./canonical-chat')
}

beforeEach(() => {
  vi.clearAllMocks()
  hostMock.openSession.mockResolvedValue(undefined)
})

describe('the registry row wins, always', () => {
  it('resolves the profile\u2019s "Bot Chat" row by exact title and opens it', async () => {
    const calls = respondWith(method => {
      if (method === 'session.list') {
        return { sessions: [{ id: 'forever-chat', message_count: 930, title: 'Bot Chat' }] }
      }

      if (method === 'session.create') {
        throw new Error('must not create: the registry row exists')
      }

      return {}
    })

    const { openBotCanonicalChat } = await loadModule()
    const opened = await openBotCanonicalChat('ops')

    expect(opened).toEqual({ openedId: 'forever-chat', registryId: 'forever-chat' })
    expect(hostMock.openSession).toHaveBeenCalledTimes(1)

    const [id, options] = hostMock.openSession.mock.calls[0]

    expect(id).toBe('forever-chat')
    expect(options).toMatchObject({
      profile: 'ops',
      // Opening a bot leaves the Sessions workspace on its current gateway.
      keepAllProfilesScope: true,
      tabTitle: 'Bot Chat',
      workspaceMode: 'bots',
      workspaceOwnerKey: 'bot:ops'
    })
    // Same intent a session row click uses. `tab` stacked a fresh tile on every
    // miss, so bot chats piled up beside each other and beside the untouched
    // "New session" draft.
    expect(options.intent).toBe('in-place')

    const list = calls.find(call => call.method === 'session.list')

    expect(list?.params).toMatchObject({
      profile: 'ops',
      // Canonical chats are always hidden — the lookup must see hidden rows.
      include_hidden: true,
      title: 'Bot Chat'
    })
  })

  it('opens the lineage tip of a compression-rotated registry row', async () => {
    respondWith(method =>
      method === 'session.list'
        ? {
            sessions: [
              { id: 'root-1', message_count: 400, resolved_id: 'tip-9', root_title: 'Bot Chat', title: 'Bot Chat' }
            ]
          }
        : {}
    )

    const { openBotCanonicalChat } = await loadModule()
    const opened = await openBotCanonicalChat('ops')

    // The durable registry id names the chat; the tip is what takes focus.
    expect(opened).toEqual({ openedId: 'tip-9', registryId: 'root-1' })
    expect(hostMock.openSession.mock.calls[0][0]).toBe('tip-9')
  })

  it('never reads or writes a stored pointer while opening', async () => {
    const calls = respondWith(method =>
      method === 'session.list' ? { sessions: [{ id: 'forever-chat', title: 'Bot Chat' }] } : {}
    )

    const { openBotCanonicalChat } = await loadModule()
    await openBotCanonicalChat('ops')

    expect(saveBotMetaMock).not.toHaveBeenCalled()
    expect(persistMock).not.toHaveBeenCalled()
    // No id-verification RPC: the name IS the identity, so there is nothing to
    // verify a stored id against.
    expect(calls.every(call => !('preferred_session_ids' in call.params))).toBe(true)
    expect(calls.map(call => call.method)).toEqual(['session.list'])
  })
})

describe('no registry row → create', () => {
  it('mints a hidden "Bot Chat" WITHOUT an intro kickoff', async () => {
    // Click-path resolution mints silently. The intro turn fires only from New
    // Bot creation (kickoff: true) — re-firing it on a resolution miss burned a
    // model turn and stamped a user-attributed prompt into the chat.
    const calls = respondWith(method => {
      if (method === 'session.list') {
        return { sessions: [] }
      }

      if (method === 'session.create') {
        return { session_id: 'rt-1', stored_session_id: 'fresh-1' }
      }

      return {}
    })

    const { openBotCanonicalChat } = await loadModule()
    const opened = await openBotCanonicalChat('newbie')

    expect(opened).toEqual({ openedId: 'fresh-1', registryId: 'fresh-1' })
    expect(calls.find(call => call.method === 'session.create')?.params).toMatchObject({
      hidden: true,
      title: 'Bot Chat'
    })
    // The eager title write persists the row; no user-attributed intro.
    expect(calls.find(call => call.method === 'session.title')?.params).toMatchObject({ session_id: 'rt-1' })
    expect(calls.find(call => call.method === 'prompt.submit')).toBeUndefined()
  })

  it('never claims an ordinary titled session', async () => {
    respondWith(method => {
      if (method === 'session.list') {
        // An older gateway ignores the title param and returns a windowed
        // listing — the local exact-title scan still applies.
        return {
          sessions: [
            { id: 'scratch', message_count: 40, title: 'help me with x' },
            { id: 'draft', message_count: 0, title: '' }
          ]
        }
      }

      return method === 'session.create' ? { session_id: 'rt-2', stored_session_id: 'fresh-2' } : {}
    })

    const { openBotCanonicalChat } = await loadModule()
    const opened = await openBotCanonicalChat('ops')

    expect(opened).toEqual({ openedId: 'fresh-2', registryId: 'fresh-2' })
    expect(hostMock.openSession.mock.calls.every(([id]) => id !== 'scratch')).toBe(true)
  })

  it('surfaces a failed open of the registry row instead of forking a replacement', async () => {
    respondWith(method => {
      if (method === 'session.list') {
        return { sessions: [{ id: 'forever-chat', message_count: 12, title: 'Bot Chat' }] }
      }

      if (method === 'session.create') {
        throw new Error('must not create: a transient open failure is not ownership loss')
      }

      return {}
    })
    hostMock.openSession.mockRejectedValue(new Error('backend restarting'))

    const { openBotCanonicalChat } = await loadModule()

    await expect(openBotCanonicalChat('ops')).rejects.toThrow('backend restarting')
  })
})

describe('a failed lookup fails CLOSED — never "no chat exists"', () => {
  // The post-update window: the desktop restarts every profile backend, the
  // first bot click races the warm-up, and the lookup RPC fails transiently.
  // Swallowing that made the failure indistinguishable from "this bot has no
  // Bot Chat yet", so create minted a fresh forever-chat while the real one
  // (data intact, hidden) still held the canonical title — read by users as
  // "my bot lost everything after the update".
  const refuseToMint = (method: string) => {
    if (method === 'session.list') {
      throw new Error('gateway not ready')
    }

    if (method === 'session.create') {
      throw new Error('must not create: a failed lookup is not "no chat exists"')
    }

    return {}
  }

  it('rejects instead of minting a replacement chat', async () => {
    const calls = respondWith(refuseToMint)
    const { openBotCanonicalChat } = await loadModule()

    await expect(openBotCanonicalChat('ops')).rejects.toThrow(/Bot Chat registry/)
    expect(calls.some(call => call.method === 'session.create')).toBe(false)
    expect(hostMock.openSession).not.toHaveBeenCalled()
  })

  it('refuses to mint when the adoption lookup inside creation fails', async () => {
    const calls = respondWith(refuseToMint)
    const { createCanonicalChat } = await loadModule()

    await expect(createCanonicalChat('ops')).rejects.toThrow(/Bot Chat registry/)
    expect(calls.some(call => call.method === 'session.create')).toBe(false)
  })
})
