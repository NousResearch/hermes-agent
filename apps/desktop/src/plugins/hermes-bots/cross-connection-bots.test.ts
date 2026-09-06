/**
 * Cross-connection Bot Mode: a roster row can belong to another registered
 * connection's backend, and a group chat can seat members from several
 * machines at once.
 *
 * Everything hangs off one rule — a source-scoped row carries an IMMUTABLE
 * route descriptor and every RPC for it goes through `host.requestProfile`
 * with that descriptor, never through the active gateway. The alias case is
 * the sharp edge: Desktop's identity for the row (`route.profile`) and the
 * backend's own name for it (`route.targetProfile`) differ, so the params of
 * every profile-shaped RPC have to be translated on the way out while the
 * route keeps the Desktop identity.
 *
 * Ported from tests/cross-connection-bots.test.mjs, which ran the whole
 * plugin.js bundle under `vm`.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { GroupMember, ProfileRoute, RosterRow } from './types'

const { overrides, request, requestProfile } = vi.hoisted(() => ({
  overrides: {} as Record<string, unknown>,
  request: vi.fn(),
  requestProfile: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  overrides.request = request
  overrides.requestProfile = requestProfile

  return {
    ...sdk,
    host: new Proxy(sdk.host, {
      get: (target, prop) => (prop in overrides ? overrides[prop as string] : Reflect.get(target, prop))
    })
  }
})

const { botBackendProfileScope, botConnectionRoute, requestForBot } = await import('./routing')
const { groupMemberKey } = await import('./group-membership')

const { buildGroupChatTurnPrompt, formatGroupChatLine, parseGroupChatMentions, resolveGroupResponders } =
  await import('./group-rounds')

const aliasBot = {
  name: 'worker',
  route: {
    connectionId: 'remote-a',
    mode: 'remote',
    profile: 'worker',
    targetProfile: 'backend-worker'
  } satisfies ProfileRoute,
  sourceScoped: true
} as RosterRow

beforeEach(() => {
  vi.clearAllMocks()
  request.mockResolvedValue({})
  requestProfile.mockResolvedValue({})
})

describe('every source-scoped row resolves to an immutable owner', () => {
  it('routes only source-annotated rows, and freezes the descriptor', () => {
    expect(botConnectionRoute({ connectionId: 'local', name: 'writer' })).toBeNull()
    expect(botConnectionRoute({ name: 'writer' })).toBeNull()
    expect(Object.isFrozen(botConnectionRoute({ connectionId: 'spark', name: 'writer', sourceScoped: true }))).toBe(
      true
    )
    expect(botConnectionRoute({ connectionId: 'mac-mini', name: 'dixie', remoteSource: true })).toEqual({
      connectionId: 'mac-mini',
      mode: 'remote',
      profile: 'dixie',
      targetProfile: 'dixie'
    })
  })

  it('keeps Desktop identity separate from the backend target in the capability scope', () => {
    expect(
      botBackendProfileScope({
        connectionId: 'remote-a',
        mode: 'remote',
        profile: 'worker',
        targetProfile: 'backend-worker'
      })
    ).toEqual({ connectionId: 'remote-a', profile: 'backend-worker' })
  })
})

describe('requestForBot dispatches on the owner, not the active gateway', () => {
  it('sends source-scoped rows through requestProfile and local rows through request', async () => {
    await requestForBot({ name: 'local-bot' }, 'session.create', { title: 'x' })
    await requestForBot({ connectionId: 'local', name: 'local-bot', sourceScoped: true }, 'session.create', {
      title: 'x'
    })
    await requestForBot({ connectionId: 'mac-mini', name: 'dixie', remoteSource: true }, 'prompt.submit', {
      text: 'hi'
    })

    expect(request.mock.calls.map(([method]) => method)).toEqual(['session.create'])
    // A registered LOCAL source still uses the explicit descriptor.
    expect(requestProfile.mock.calls.map(([route, method]) => [route.connectionId, method])).toEqual([
      ['local', 'session.create'],
      ['mac-mini', 'prompt.submit']
    ])
  })

  it('translates every backend-profile RPC shape for a non-identity alias', async () => {
    await requestForBot(aliasBot, 'profiles.describe', { name: 'worker' })
    await requestForBot(aliasBot, 'profiles.configure', { name: 'worker', soul: 'x' })
    await requestForBot(aliasBot, 'profiles.create', { clone_from: 'worker', name: 'worker-2' })
    await requestForBot(aliasBot, 'session.create', { profile: 'worker', title: 'Bot Chat' })
    await requestForBot(aliasBot, 'cli.exec', { argv: ['--profile', 'worker', 'config', 'unset', 'model'] })
    await requestForBot(aliasBot, 'cli.exec', { argv: ['profile', 'describe', 'worker', '--text', 'worker'] })

    expect(requestProfile.mock.calls.map(([, , params]) => params)).toEqual([
      { name: 'backend-worker' },
      { name: 'backend-worker', soul: 'x' },
      { clone_from: 'backend-worker', name: 'worker-2' },
      { profile: 'backend-worker', title: 'Bot Chat' },
      { argv: ['--profile', 'backend-worker', 'config', 'unset', 'model'] },
      // Only the profile operand is rewritten — the trailing --text value is
      // the user's own argument and stays untouched.
      { argv: ['profile', 'describe', 'backend-worker', '--text', 'worker'] }
    ])
    // The route keeps Desktop's identity for the row throughout.
    expect(requestProfile.mock.calls.every(([route]) => route.profile === 'worker')).toBe(true)
  })
})

describe('a group room seats members from several machines', () => {
  it('source-qualifies remote member keys and leaves local ones bare', () => {
    // Bare names keep persisted-room compatibility for single-source rooms.
    expect(groupMemberKey({ name: 'ops' } as GroupMember)).toBe('ops')
    expect(groupMemberKey({ connectionId: 'mac-mini', name: 'dixie', remoteSource: true } as GroupMember)).toBe(
      'mac-mini::dixie'
    )
  })

  it('resolves @name-device to the remote member, keeping same-named agents distinct', () => {
    const members = [
      { name: 'dixie', title: '' },
      { connectionId: 'mac-mini', handle: 'dixie-mac-mini', name: 'dixie', remoteSource: true }
    ] as GroupMember[]

    const parsed = parseGroupChatMentions('hey @dixie-mac-mini can you check disk space', members)

    expect([...parsed.mentioned]).toEqual(['mac-mini::dixie'])

    const responders = resolveGroupResponders(
      [{ at: 1, from: { kind: 'user', name: 'You' }, text: '@dixie-mac-mini ping' }],
      members
    )

    expect(responders).toHaveLength(1)
    expect(responders[0].remoteSource).toBe(true)
  })

  it('badges cross-connection speakers with their device in lines and turn prompts', () => {
    expect(
      formatGroupChatLine({ at: 1, from: { kind: 'member', name: 'dixie', source: 'Mac Mini' }, text: 'done' }, 'ops')
    ).toBe('dixie [Mac Mini]: done')
    expect(
      buildGroupChatTurnPrompt({
        deltaLines: ['You (user): hi'],
        groupName: 'infra',
        members: [
          { name: 'ops', title: '' },
          { connectionId: 'mac-mini', connectionLabel: 'Mac Mini', name: 'dixie', remoteSource: true }
        ] as GroupMember[],
        viewer: { name: 'ops' } as GroupMember
      })
    ).toMatch(/@dixie \[on Mac Mini\]/)
  })
})
