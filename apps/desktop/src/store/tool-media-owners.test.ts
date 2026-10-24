import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionOwnerScope } from '@/store/session-request-router'

import type * as AgentPlugins from './agent-plugins'
import type * as ToolMediaOwners from './tool-media-owners'

type OwnersModule = typeof ToolMediaOwners
type PluginsModule = typeof AgentPlugins

interface Pending {
  owner: SessionOwnerScope
  params: Record<string, unknown>
  resolve: (value: unknown) => void
  reject: (error: Error) => void
}

const env = vi.hoisted(() => ({
  connectionId: 'local' as null | string,
  epoch: 1,
  gateway: { request: () => Promise.resolve({}) } as null | { request: () => Promise<unknown> },
  owners: new Map<string, unknown>(),
  pending: [] as unknown[],
  profileKey: 'default'
}))

vi.mock('@/store/gateway', () => ({
  activeGateway: () => env.gateway,
  activeGatewayConnectionId: () => env.connectionId,
  activeGatewayProfileKey: () => env.profileKey,
  gatewayActivationEpoch: () => env.epoch
}))

vi.mock('@/store/session-states', () => ({
  knownOwnerForSession: (sessionId: null | string | undefined) => (sessionId ? env.owners.get(sessionId) : undefined)
}))

vi.mock('@/store/session-request-router', () => ({
  requestForSessionProfile: vi.fn(
    (owner: SessionOwnerScope, _ambient: unknown, method: string, params: Record<string, unknown>) =>
      new Promise((resolve, reject) => {
        expect(method).toBe('plugins.manage')
        env.pending.push({ owner, params, reject, resolve })
      })
  )
}))

vi.mock('@/contrib/plugins-store', async () => {
  const { atom } = await import('nanostores')

  return { $pluginRecords: atom({}) }
})

let owners: OwnersModule
let plugins: PluginsModule

const pending = () => env.pending as Pending[]

const flush = async () => {
  for (let i = 0; i < 6; i += 1) {
    await Promise.resolve()
  }
}

const listWith = (name: string, tools: string[]) => ({
  plugins: [{ description: '', name, source: 'user', status: 'enabled', tools, version: '1' }]
})

const entryFor = (sessionId: null | string) => owners.$toolMediaOwners.get()[owners.toolMediaOwnerKey(sessionId)]

const ownedTools = (sessionId: null | string, pkg: string) => {
  const entry = entryFor(sessionId)

  if (entry?.status !== 'ready') {
    return undefined
  }

  const tools = entry.owners.get(pkg)?.tools

  return tools ? [...tools] : undefined
}

const invalidatePlugins = () => plugins.$agentPlugins.set([...plugins.$agentPlugins.get()])

beforeEach(async () => {
  vi.useFakeTimers()
  vi.resetModules()
  env.connectionId = 'local'
  env.epoch = 1
  env.gateway = { request: () => Promise.resolve({}) }
  env.owners.clear()
  env.pending.length = 0
  env.profileKey = 'default'
  owners = await import('./tool-media-owners')
  plugins = await import('./agent-plugins')
})

afterEach(() => {
  vi.useRealTimers()
})

describe('stale-answer guard', () => {
  it('drops an answer orphaned by a plugin-list change and lets the newer request win', async () => {
    owners.ensureToolMediaOwners('s1')
    expect(pending()).toHaveLength(1)
    expect(entryFor('s1')?.status).toBe('loading')

    invalidatePlugins()
    expect(entryFor('s1')).toBeUndefined()

    owners.ensureToolMediaOwners('s1')
    expect(pending()).toHaveLength(2)

    const [older, newer] = pending()

    newer.resolve(listWith('fresh', ['new_tool']))
    await flush()
    older.resolve(listWith('stale', ['old_tool']))
    await flush()

    expect(ownedTools('s1', 'fresh')).toEqual(['new_tool'])
    expect(ownedTools('s1', 'stale')).toBeUndefined()
  })

  it('an orphaned answer never resurrects the entry it lost, even when it lands first', async () => {
    owners.ensureToolMediaOwners('s1')
    invalidatePlugins()
    pending()[0].resolve(listWith('stale', ['old_tool']))
    await flush()

    expect(entryFor('s1')).toBeUndefined()

    owners.ensureToolMediaOwners('s1')
    expect(pending()).toHaveLength(2)
  })

  it('an orphaned failure neither marks the newer request as error nor schedules its retry', async () => {
    owners.ensureToolMediaOwners('s1')
    invalidatePlugins()
    owners.ensureToolMediaOwners('s1')
    pending()[0].reject(new Error('socket closed'))
    await flush()

    expect(entryFor('s1')?.status).toBe('loading')

    await vi.advanceTimersByTimeAsync(60_000)
    owners.ensureToolMediaOwners('s1')
    expect(entryFor('s1')?.status).toBe('loading')
    expect(pending()).toHaveLength(2)
  })

  it("a superseded failure's retry timer leaves the newer in-flight request alone", async () => {
    owners.ensureToolMediaOwners('s1')
    pending()[0].reject(new Error('socket closed'))
    await flush()
    expect(entryFor('s1')?.status).toBe('error')

    invalidatePlugins()
    owners.ensureToolMediaOwners('s1')
    await vi.advanceTimersByTimeAsync(2_000)

    expect(entryFor('s1')?.status).toBe('loading')
    owners.ensureToolMediaOwners('s1')
    expect(pending()).toHaveLength(2)
  })
})

describe('per-owner cache keys', () => {
  it('keeps one entry and one correctly scoped request per session owner', async () => {
    const research = { connectionId: 'remote-a', profile: 'desk-research', targetProfile: 'research' }

    env.owners.set('s-coder', 'coder')
    env.owners.set('s-research', research)

    owners.ensureToolMediaOwners('s-coder')
    owners.ensureToolMediaOwners('s-research')
    owners.ensureToolMediaOwners('s-coder')

    expect(owners.toolMediaOwnerKey('s-coder')).not.toBe(owners.toolMediaOwnerKey('s-research'))
    expect(pending().map(p => [p.owner, p.params])).toEqual([
      ['coder', { action: 'list', profile: 'coder' }],
      [research, { action: 'list', profile: 'research' }]
    ])

    pending()[0].resolve(listWith('coder-pkg', ['draw']))
    pending()[1].resolve(listWith('research-pkg', ['plot']))
    await flush()

    expect(ownedTools('s-coder', 'coder-pkg')).toEqual(['draw'])
    expect(ownedTools('s-coder', 'research-pkg')).toBeUndefined()
    expect(ownedTools('s-research', 'research-pkg')).toEqual(['plot'])
  })

  it('separates the same profile name served by two connections', () => {
    env.owners.set('s-a', { connectionId: 'conn-a', profile: 'work' })
    env.owners.set('s-b', { connectionId: 'conn-b', profile: 'work' })

    owners.ensureToolMediaOwners('s-a')
    owners.ensureToolMediaOwners('s-b')

    expect(owners.toolMediaOwnerKey('s-a')).not.toBe(owners.toolMediaOwnerKey('s-b'))
    expect(pending()).toHaveLength(2)
  })

  it('keys an unknown owner by the active route, apart from a known owner of the same profile', () => {
    env.owners.set('s-known', 'default')

    owners.ensureToolMediaOwners('s-known')
    owners.ensureToolMediaOwners('draft')

    expect(owners.toolMediaOwnerKey('draft')).not.toBe(owners.toolMediaOwnerKey('s-known'))
    expect(owners.toolMediaOwnerKey('draft')).toBe(owners.toolMediaOwnerKey(null))
    expect(pending().map(p => p.params)).toEqual([{ action: 'list', profile: 'default' }, { action: 'list' }])

    // A gateway re-activation is a new active route: the old list is not reused.
    const before = owners.toolMediaOwnerKey(null)

    env.epoch = 2
    expect(owners.toolMediaOwnerKey(null)).not.toBe(before)
    owners.ensureToolMediaOwners(null)
    expect(pending()).toHaveLength(3)
  })
})

describe('retry backoff', () => {
  const failNext = async () => {
    pending().at(-1)?.reject(new Error('unreachable'))
    await flush()
  }

  it('waits 2s, then doubles up to a 60s cap before refetching a failed list', async () => {
    owners.ensureToolMediaOwners('s1')
    await failNext()

    for (const delay of [2_000, 4_000, 8_000, 16_000, 32_000, 60_000, 60_000]) {
      const calls = pending().length

      expect(entryFor('s1')?.status).toBe('error')
      await vi.advanceTimersByTimeAsync(delay - 1)
      owners.ensureToolMediaOwners('s1')
      expect(pending()).toHaveLength(calls)
      expect(entryFor('s1')?.status).toBe('error')

      await vi.advanceTimersByTimeAsync(1)
      expect(entryFor('s1')).toBeUndefined()
      owners.ensureToolMediaOwners('s1')
      expect(pending()).toHaveLength(calls + 1)
      await failNext()
    }
  })

  it('a success resets the backoff to 2s', async () => {
    owners.ensureToolMediaOwners('s1')
    await failNext()
    await vi.advanceTimersByTimeAsync(2_000)
    owners.ensureToolMediaOwners('s1')
    await failNext()
    await vi.advanceTimersByTimeAsync(4_000)
    owners.ensureToolMediaOwners('s1')
    pending().at(-1)?.resolve(listWith('pkg', ['draw']))
    await flush()
    expect(ownedTools('s1', 'pkg')).toEqual(['draw'])

    // The entry is evicted from outside (the atom is shared state) and the next fetch fails.
    owners.$toolMediaOwners.set({})
    owners.ensureToolMediaOwners('s1')
    await failNext()

    const calls = pending().length

    await vi.advanceTimersByTimeAsync(2_000)
    owners.ensureToolMediaOwners('s1')
    expect(pending()).toHaveLength(calls + 1)
  })

  it('a plugin-list change refetches at once and restarts the backoff', async () => {
    owners.ensureToolMediaOwners('s1')
    await failNext()
    await vi.advanceTimersByTimeAsync(2_000)
    owners.ensureToolMediaOwners('s1')
    await failNext()

    invalidatePlugins()
    owners.ensureToolMediaOwners('s1')
    await failNext()

    const calls = pending().length

    await vi.advanceTimersByTimeAsync(2_000)
    owners.ensureToolMediaOwners('s1')
    expect(pending()).toHaveLength(calls + 1)
  })

  it('does not fetch without an active gateway', () => {
    env.gateway = null
    owners.ensureToolMediaOwners('s1')

    expect(pending()).toHaveLength(0)
    expect(entryFor('s1')).toBeUndefined()
  })
})
