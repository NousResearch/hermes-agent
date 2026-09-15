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
  hostMock: { agents: vi.fn(), openSession: vi.fn(), request: vi.fn() },
  persistMock: vi.fn(),
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
  // Existing tests exercise registry/lookup/open behavior, not authorization
  // — default the roster to authorize whatever profile string the test uses
  // ('ops', 'newbie', ...) so the new roster-admission gate (Architect
  // corrective, 2026-09-02) doesn't fail closed on unrelated assertions.
  hostMock.agents.mockImplementation(async () => ({
    agents: [
      { connectionId: null, profile: 'ops', targetProfile: 'ops' },
      { connectionId: null, profile: 'newbie', targetProfile: 'newbie' }
    ]
  }))
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

  // #98383: a profile backend mid-restart can answer `session.list`
  // SUCCESSFULLY with an empty list instead of throwing. `rows.find(...) ||
  // null` used to read that identically to "this bot never had a chat",
  // which minted a replacement and re-fired the kickoff on every click.
  it('refuses to mint on an empty lookup when the roster already confirmed a canonical chat', async () => {
    const calls = respondWith(method => {
      if (method === 'session.list') {
        return { sessions: [] }
      }

      if (method === 'session.create') {
        throw new Error('must not create: an empty result is not confirmed absence')
      }

      return {}
    })

    const bot = { canonical_session: { id: 'forever-chat' }, name: 'ops' } as RosterRow
    const { openBotCanonicalChat } = await loadModule()

    await expect(openBotCanonicalChat(bot)).rejects.toThrow(/Bot Chat registry/)
    expect(calls.some(call => call.method === 'session.create')).toBe(false)
    expect(hostMock.openSession).not.toHaveBeenCalled()
  })
})

describe('roster admission runs BEFORE any canonical resolution (Architect corrective, 2026-09-02)', () => {
  it('performs current-roster admission before any session.list/create/title/open RPC', async () => {
    const calls = respondWith(() => ({}))
    let rosterCheckedAt = -1

    hostMock.agents.mockImplementation(async () => {
      rosterCheckedAt = calls.length

      return { agents: [{ connectionId: null, profile: 'ops', targetProfile: 'ops' }] }
    })

    const { openBotCanonicalChat } = await loadModule()
    await openBotCanonicalChat('ops')

    // The roster check ran before ANY RPC was recorded — i.e. before the
    // canonical lookup, not interleaved with or after it.
    expect(rosterCheckedAt).toBe(0)
    expect(hostMock.agents).toHaveBeenCalledTimes(1)
  })

  it('an arbitrary profile string absent from the roster reaches no session RPC and does not open', async () => {
    const calls = respondWith(() => ({}))
    hostMock.agents.mockResolvedValue({ agents: [{ connectionId: null, profile: 'someone-else', targetProfile: 'someone-else' }] })

    const { openBotCanonicalChat } = await loadModule()

    await expect(openBotCanonicalChat('ops')).rejects.toThrow(/authorized agent roster/)
    expect(calls.some(call => ['session.list', 'session.create', 'session.title'].includes(call.method))).toBe(false)
    expect(hostMock.openSession).not.toHaveBeenCalled()
  })

  it('rejects a mismatched backend targetProfile even when connectionId/profile look valid', async () => {
    const calls = respondWith(() => ({}))
    hostMock.agents.mockResolvedValue({
      agents: [{ connectionId: null, profile: 'ops', targetProfile: 'some-other-backend-identity' }]
    })

    const { openBotCanonicalChat } = await loadModule()

    await expect(openBotCanonicalChat('ops')).rejects.toThrow(/authorized agent roster/)
    expect(calls.length).toBe(0)
  })

  it('rejects a mismatched displayed profile even when connectionId/targetProfile look valid', async () => {
    const calls = respondWith(() => ({}))
    // The roster entry's OWN reported profile diverges from the target's
    // displayed profile despite a matching targetProfile — must still fail
    // closed; targetProfile alone is never sufficient authorization.
    hostMock.agents.mockResolvedValue({
      agents: [{ connectionId: null, profile: 'some-other-display-name', targetProfile: 'ops' }]
    })

    const { openBotCanonicalChat } = await loadModule()

    await expect(openBotCanonicalChat('ops')).rejects.toThrow(/authorized agent roster/)
    expect(calls.length).toBe(0)
  })

  it('fails closed before any session operation when the roster fetch itself fails', async () => {
    const calls = respondWith(() => ({}))
    hostMock.agents.mockRejectedValue(new Error('roster RPC timed out'))

    const { openBotCanonicalChat } = await loadModule()

    await expect(openBotCanonicalChat('ops')).rejects.toThrow(/roster/i)
    expect(calls.length).toBe(0)
  })

  it('newAgentBirth polls the SAME live host.agents() roster until it admits the target — never trusts a caller-supplied claim (Architect corrective, 2026-09-02, second pass)', async () => {
    respondWith(method => {
      if (method === 'session.create') {
        return { session_id: 'runtime-1', stored_session_id: 'stored-1' }
      }

      return {}
    })

    // The FIRST few live roster reads genuinely do not have 'newbie' yet
    // (profiles.create just fired and enumeration hasn't caught up); the
    // Nth read does. newAgentBirth must observe the REAL roster transition,
    // not a caller-asserted shortcut.
    let rosterReads = 0
    hostMock.agents.mockImplementation(async () => {
      rosterReads += 1

      return {
        agents:
          rosterReads < 3
            ? [{ connectionId: null, profile: 'ops', targetProfile: 'ops' }]
            : [
                { connectionId: null, profile: 'ops', targetProfile: 'ops' },
                { connectionId: null, profile: 'newbie', targetProfile: 'newbie' }
              ]
      }
    })

    const { createCanonicalChat } = await loadModule()

    await expect(createCanonicalChat('newbie', { newAgentBirth: true, kickoff: true, rosterAdmissionPoll: { attempts: 5, delayMs: 1 } })).resolves.toBe('stored-1')
    // At least the polling reads plus the subsequent assertAuthorizedTarget
    // read — every admission came from a REAL host.agents() call, never a
    // client-side object.
    expect(rosterReads).toBeGreaterThanOrEqual(3)
  })

  it('newAgentBirth fails closed when the live roster never admits the target within the bound — no bypass, no silent open', async () => {
    const calls = respondWith(() => ({}))
    // The roster NEVER admits 'newbie' — a genuinely stuck/failed creation.
    hostMock.agents.mockResolvedValue({ agents: [{ connectionId: null, profile: 'ops', targetProfile: 'ops' }] })

    const { createCanonicalChat } = await loadModule()

    await expect(
      createCanonicalChat('newbie', { newAgentBirth: true, kickoff: true, rosterAdmissionPoll: { attempts: 3, delayMs: 1 } })
    ).rejects.toThrow(/did not appear in the current authorized agent roster/)
    expect(calls.some(call => ['session.list', 'session.create', 'session.title'].includes(call.method))).toBe(false)
  })

  it('a target absent from the roster is rejected even with kickoff/newAgentBirth unset — no other flag substitutes for admission', async () => {
    const calls = respondWith(() => ({}))
    hostMock.agents.mockResolvedValue({ agents: [] })

    const { createCanonicalChat } = await loadModule()

    await expect(createCanonicalChat('newbie')).rejects.toThrow(/authorized agent roster/)
    expect(calls.length).toBe(0)
  })
})
