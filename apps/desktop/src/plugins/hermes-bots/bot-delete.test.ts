/**
 * Deleting a bot deletes its Hermes profile, then everything plugin-local
 * that would otherwise leave stale appearance/unread data behind.
 *
 * `host.deleteProfile` is preferred whenever the Desktop build ships it: it
 * routes through the Electron-intercepted REST delete, which tears the bot's
 * pool backend down FIRST. The older `cli.exec` path bypasses that
 * interception, so a backend the roster's hover pre-warm just woke (a
 * right-click hovers the row!) holds the profile dir open and the CLI's rmtree
 * races it — that is the "can't delete a bot" error (hermes-agent#52279).
 *
 * Ported from tests/bot-delete.test.mjs, which ran the whole plugin.js bundle
 * under `vm`.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileRoute, RosterRow } from './types'

const { forgetSessionUnread, invalidateQueries, newChat, overrides, request, stateAtoms } = vi.hoisted(() => {
  const makeAtom = <T>(initial: T) => {
    let value = initial

    return {
      get: () => value,
      listen: () => () => undefined,
      set: (next: T) => {
        value = next
      }
    }
  }

  return {
    forgetSessionUnread: vi.fn(),
    invalidateQueries: vi.fn(),
    newChat: vi.fn(),
    // Read through a Proxy below so a test can make a host verb *absent* —
    // `typeof host.deleteProfile === 'function'` is the whole feature gate.
    overrides: {} as Record<string, unknown>,
    request: vi.fn(),
    stateAtoms: {
      connectionId: makeAtom('local'),
      focusedSessionOwner: makeAtom<{ connectionId: string; profile: string }>({
        connectionId: 'local',
        profile: 'default'
      }),
      profile: makeAtom('default')
    }
  }
})

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  overrides.newChat = newChat
  overrides.request = request
  overrides.state = { ...sdk.host.state, ...stateAtoms }

  return {
    ...sdk,
    forgetSessionUnread,
    host: new Proxy(sdk.host, {
      get: (target, prop) => (prop in overrides ? overrides[prop as string] : Reflect.get(target, prop))
    }),
    queryClient: { invalidateQueries }
  }
})

interface StorageOp {
  key: string
  op: 'remove' | 'set'
  value?: unknown
}

const remoteRoute: ProfileRoute = {
  connectionId: 'source-a',
  mode: 'remote',
  profile: 'worker',
  targetProfile: 'backend-worker'
}

async function loadModules() {
  vi.resetModules()

  const ops: StorageOp[] = []
  const [data, profileOps, shared] = await Promise.all([import('./data'), import('./profile-ops'), import('./shared')])

  shared.setPluginCtx({
    storage: {
      get: () => null,
      remove: (key: string) => {
        ops.push({ key, op: 'remove' })
      },
      set: (key: string, value: unknown) => {
        ops.push({ key, op: 'set', value })
      }
    }
  } as unknown as Parameters<typeof shared.setPluginCtx>[0])

  return { data, ops, profileOps }
}

// Pay the graph's cold transform once, up front. `loadModules` re-imports on
// every test, and charging that one-time cost to whichever test happens to run
// first makes it time out under a loaded runner.
beforeAll(async () => {
  await loadModules()
}, 60_000)

beforeEach(() => {
  vi.clearAllMocks()
  // Absent by default: the older-desktop shape. Tests opt in to the SDK verb.
  overrides.deleteProfile = undefined
  request.mockResolvedValue({ blocked: false, code: 0, output: 'deleted' })
  stateAtoms.connectionId.set('local')
  stateAtoms.focusedSessionOwner.set({ connectionId: 'local', profile: 'default' })
  stateAtoms.profile.set('default')
})

describe('the SDK delete verb is preferred over the CLI', () => {
  it('deletes through host.deleteProfile and never shells out', async () => {
    const deleteProfile = vi.fn()

    overrides.deleteProfile = deleteProfile

    const { profileOps } = await loadModules()

    await profileOps.deleteBot({ name: 'researcher' } as RosterRow)

    expect(deleteProfile.mock.calls).toEqual([['researcher']])
    expect(request.mock.calls.filter(([method]) => method === 'cli.exec')).toHaveLength(0)
  })

  it('hands a source-scoped row its whole route, targetProfile included', async () => {
    const deleteProfile = vi.fn()

    overrides.deleteProfile = deleteProfile

    const { profileOps } = await loadModules()

    await profileOps.deleteBot({ name: 'worker', route: remoteRoute, sourceScoped: true } as RosterRow)

    expect(deleteProfile.mock.calls[0][0]).toMatchObject(remoteRoute)
  })

  it('falls back to the non-interactive profile CLI on an older desktop', async () => {
    const { profileOps } = await loadModules()

    await profileOps.deleteBot({ name: 'researcher' } as RosterRow)

    const exec = request.mock.calls.filter(([method]) => method === 'cli.exec')

    expect(exec[0][1]).toEqual({ argv: ['profile', 'delete', 'researcher', '--yes'] })
  })
})

describe('deletes that must not happen', () => {
  it('rejects an alias whose backend target is the default profile', async () => {
    const deleteProfile = vi.fn()

    overrides.deleteProfile = deleteProfile

    const { profileOps } = await loadModules()

    await expect(
      profileOps.deleteBot({
        name: 'worker',
        route: { ...remoteRoute, targetProfile: 'default' },
        sourceScoped: true
      } as RosterRow)
    ).rejects.toThrow(/default profile cannot be deleted/i)

    expect(deleteProfile).not.toHaveBeenCalled()
    expect(request.mock.calls.some(([method]) => method === 'cli.exec')).toBe(false)
  })

  it('fails a source-scoped delete closed rather than shelling out', async () => {
    const { profileOps } = await loadModules()

    await expect(
      profileOps.deleteBot({
        name: 'worker',
        route: { ...remoteRoute, targetProfile: 'worker' },
        sourceScoped: true
      } as RosterRow)
    ).rejects.toThrow(/source-scoped profile deletion requires host\.deleteProfile/i)

    expect(request.mock.calls.some(([method]) => method === 'cli.exec')).toBe(false)
  })
})

describe('plugin-local state is cleaned up behind the delete', () => {
  it('commits the source-scoped bot-meta-v2 snapshot atomically', async () => {
    overrides.deleteProfile = vi.fn()

    const { data, ops, profileOps } = await loadModules()

    data.$botMeta.set({
      'source-a::worker': { title: 'Worker' },
      'source-a::writer': { title: 'Writer' }
    })

    await profileOps.deleteBot({ name: 'worker', route: remoteRoute, sourceScoped: true } as RosterRow)

    expect(ops.slice(-3)).toEqual([
      { key: 'bot-meta-v2-migrated', op: 'remove' },
      { key: 'bot-meta-v2', op: 'set', value: { 'source-a::writer': { title: 'Writer' } } },
      { key: 'bot-meta-v2-migrated', op: 'set', value: true }
    ])
  })

  it('drops meta, unread and selection, then refreshes the roster', async () => {
    const { data, ops, profileOps } = await loadModules()

    data.$botMeta.set({ researcher: { title: 'Research' }, writer: { title: 'Writer' } })
    data.$lastRoster.set([{ name: 'researcher' }, { name: 'writer' }] as RosterRow[])

    const { $selectedBot } = await import('./bot-state')

    $selectedBot.set('researcher')

    await profileOps.deleteBot({
      canonical_session: { id: 'reg-1', resolved_id: 'tip-9' },
      name: 'researcher'
    } as RosterRow)

    expect(data.$botMeta.get().researcher).toBeUndefined()
    // Unread lives in core's store now; both ids because the marker may have
    // been filed under either tip of the chat's compression lineage.
    expect(forgetSessionUnread).toHaveBeenCalledWith(['reg-1', 'tip-9'], 'researcher')
    expect($selectedBot.get()).toBe('default')
    expect(ops.at(-1)?.key).toBe('bot-meta')
    expect((ops.at(-1)?.value as Record<string, unknown>).researcher).toBeUndefined()
    expect(invalidateQueries).toHaveBeenCalledTimes(1)
  })

  it('keeps the active chat when the deleted bot is a same-name twin elsewhere', async () => {
    overrides.deleteProfile = vi.fn()
    stateAtoms.connectionId.set('source-b')
    stateAtoms.focusedSessionOwner.set({ connectionId: 'source-b', profile: 'worker' })
    stateAtoms.profile.set('worker')

    const { profileOps } = await loadModules()

    await profileOps.deleteBot({
      name: 'worker',
      route: { ...remoteRoute, targetProfile: 'worker' },
      sourceScoped: true
    } as RosterRow)

    expect(newChat).not.toHaveBeenCalled()
  })
})
