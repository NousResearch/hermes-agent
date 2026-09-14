import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createGroupGateway, deferTimers, drain, scriptedStorage } from './group-test-utils'
import type { GroupChat } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

const HASH = 'ab'.repeat(32)
const OTHER = 'cd'.repeat(32)

const room = (extra: Partial<GroupChat> = {}): GroupChat => ({
  roomId: 'classic-1',
  log: [{ at: 1, from: { kind: 'user', name: 'You' }, text: 'hello', thread: 'thread-1' }],
  watermarks: {},
  ...extra
})

async function setup() {
  vi.resetModules()
  const gateway = createGroupGateway()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)
  const chat = await import('./group-chat')
  const shared = await import('./shared')
  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, gateway, shared }
}

beforeEach(() => {
  deferTimers()
})

afterEach(async () => {
  const chat = await import('./group-chat')
  chat.setGroupChatSyncDisposed(true)
  chat.stopGroupChatServerSync()
  await drain(() => false)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('classic Desktop authority commitment', () => {
  it('rechecks authority after an async read-back before publishing the earlier snapshot', async () => {
    const { chat, gateway, shared } = await setup()
    const context = scriptedStorage(gateway.storage)
    const get = context.storage.get
    let changed = false
    context.storage.get = (async (key: string, fallback: unknown) => {
      const saved = await get(key, fallback)

      if (!changed && key === 'group-chats') {
        changed = true
        chat.updateGroupChat('Team', current => ({ ...current, desktopAuthorityConflict: true }), { sync: false })
      }

      return saved
    }) as typeof context.storage.get
    shared.setPluginCtx(context)
    chat.$groupChats.set({ Team: room() })
    await expect(chat.activateClassicGroupAuthorities()).rejects.toThrow('could not be saved')
    await drain(() => true, 30)
    expect(gateway.rpcFor('profiles.configure')).toHaveLength(0)
    expect(chat.$groupChats.get().Team.desktopAuthorityConflict).toBe(true)
  })

  it('never publishes after a silently dropped write and retries on the next activation', async () => {
    const { chat, gateway, shared } = await setup()
    const context = scriptedStorage(gateway.storage)
    const workingSet = context.storage.set
    context.storage.set = vi.fn(() => undefined)
    shared.setPluginCtx(context)
    chat.$groupChats.set({ Team: room() })

    await expect(chat.activateClassicGroupAuthorities()).rejects.toThrow('could not be saved')
    const token = chat.$groupChats.get().Team.desktopAuthorityToken
    await drain(() => true, 30)
    expect(gateway.storage.get('group-chats')).toBeUndefined()
    expect(gateway.rpcFor('profiles.configure')).toHaveLength(0)

    context.storage.set = workingSet
    await expect(chat.activateClassicGroupAuthorities()).resolves.toBe(true)
    await drain(() => gateway.rpcFor('profiles.configure').length === 0)
    expect(gateway.rpcFor('profiles.configure')).toHaveLength(1)
    expect(chat.$groupChats.get().Team.desktopAuthorityToken).toBe(token)
  })

  it.each([false, true])('opening a projection-only room stays passive with local sessions=%s', async local => {
    const { chat } = await setup()
    const view = await import('./group-chat-view')
    const original = chat.groupChatSyncSnapshot({ Team: room({ desktopAuthorityHash: HASH }) })
    const imported = chat.mergeRemoteGroupChatSnapshotIntoRooms(original, {})

    if (local) {
      imported.Team.sessions = { bot: 'old-session' }
    }

    chat.$groupChats.set(imported)
    view.openGroupChat('Team')
    await chat.activateClassicGroupAuthorities(['Team'])
    const merged = chat.mergeGroupChatSyncSnapshots(original, chat.groupChatSyncSnapshot())

    expect(merged.rooms['id:classic-1'].desktopAuthorityConflict).toBeUndefined()
    expect(merged.rooms['id:classic-1'].desktopAuthorityHash).toBe(HASH)
    expect(chat.$groupChats.get().Team.desktopAuthorityToken).toBeUndefined()
  })

  it.each(['roomId', 'desktopAuthorityHash', 'desktopAuthorityToken', 'desktopAuthorityConflict'] as const)(
    'rejects storage that does not preserve the exact %s',
    async field => {
      const { chat, gateway, shared } = await setup()
      chat.updateGroupChat('Team', () => room(), { sync: false })
      const context = scriptedStorage(gateway.storage)
      context.storage.set = vi.fn((_key, value) => {
        const saved = structuredClone(value) as Record<string, Record<string, unknown>>
        saved.Team[field] = field === 'desktopAuthorityConflict' ? true : 'changed'
        gateway.storage.set('group-chats', saved)
      })
      shared.setPluginCtx(context)
      await expect(chat.persistGroupChatRoomsRequired()).rejects.toThrow('could not be saved')
      chat.scheduleGroupChatServerSync()
      await drain(() => true, 30)
      expect(gateway.rpcFor('profiles.configure')).toHaveLength(0)
    }
  )

  it('mints a new private token when a copied record receives a new room ID', async () => {
    const { chat } = await setup()
    const initial = chat.updateGroupChat('Team', () => room(), { sync: false })
    const recreated = chat.updateGroupChat('Team', current => ({ ...current, roomId: 'new-incarnation' }), { sync: false })
    expect(recreated.desktopAuthorityToken).not.toBe(initial.desktopAuthorityToken)
    expect(recreated.desktopAuthorityHash).not.toBe(initial.desktopAuthorityHash)
  })

  it('matches the gateway SHA-256 commitment contract', async () => {
    const { chat } = await setup()
    expect(chat).toBeTruthy()
    const authority = await import('./group-desktop-authority')
    expect(authority.classicAuthorityHash('authority:test')).toBe(
      'a4dbcde6957af558cf02d48d8057168a059f9fd659eb721c1b4a7b92fb1775f9'
    )
  })

  it('recovers an existing room on its next local update and persists the commitment', async () => {
    const { chat, gateway } = await setup()
    chat.$groupChats.set({ Team: room() })
    const recovered = chat.updateGroupChat('Team', current => current, { sync: false })
    expect(recovered.desktopAuthorityHash).toMatch(/^[a-f0-9]{64}$/)
    const saved = gateway.storage.get('group-chats') as Record<string, GroupChat>
    expect(saved.Team.desktopAuthorityHash).toBe(recovered.desktopAuthorityHash)
    expect(chat.groupChatSyncSnapshot().rooms['id:classic-1'].desktopAuthorityHash).toBe(recovered.desktopAuthorityHash)
  })

  it('keeps the commitment through reload, rename and either persistence path', async () => {
    const { chat } = await setup()
    const saved = chat.durableGroupChatRooms({ Old: room({ desktopAuthorityHash: HASH }) })
    expect(saved.Old.desktopAuthorityHash).toBe(HASH)
    chat.$groupChats.set({ Renamed: structuredClone(saved.Old) })
    const next = chat.updateGroupChat('Renamed', current => current, { sync: false })
    expect(next.desktopAuthorityHash).toBe(HASH)
    const projected = chat.groupChatSyncSnapshot()
    expect(projected.rooms['id:classic-1'].desktopAuthorityHash).toBe(HASH)
    expect(chat.mergeRemoteGroupChatSnapshotIntoRooms(projected, {}).Renamed.desktopAuthorityHash).toBe(HASH)
  })

  it('does not clear an established value when an older projection omits it', async () => {
    const { chat } = await setup()
    const local = chat.groupChatSyncSnapshot({ Team: room({ desktopAuthorityHash: HASH }) })
    const old = chat.groupChatSyncSnapshot({ Team: room({ syncRevision: 100 }) })

    for (const [left, right] of [
      [local, old],
      [old, local]
    ]) {
      expect(chat.mergeGroupChatSyncSnapshots(left, right).rooms['id:classic-1'].desktopAuthorityHash).toBe(HASH)
    }
  })

  it('fails closed and converges when two clients minted conflicting commitments', async () => {
    const { chat } = await setup()
    const one = chat.groupChatSyncSnapshot({ Team: room({ desktopAuthorityHash: HASH }) })
    const two = chat.groupChatSyncSnapshot({ Team: room({ desktopAuthorityHash: OTHER }) })
    const merged = chat.mergeGroupChatSyncSnapshots(one, two)
    expect(merged.rooms['id:classic-1'].desktopAuthorityHash).toBeUndefined()
    expect(merged.rooms['id:classic-1'].desktopAuthorityConflict).toBe(true)
    expect(chat.mergeGroupChatSyncSnapshots(two, one).rooms).toEqual(merged.rooms)
    expect(chat.mergeGroupChatSyncSnapshots(merged, one).rooms).toEqual(merged.rooms)
  })

  it('never resurrects the old commitment when the same name is recreated', async () => {
    const { chat } = await setup()
    const old = chat.groupChatSyncSnapshot({ Team: room({ desktopAuthorityHash: HASH }) })
    chat.$groupChats.set({})
    const fresh = chat.updateGroupChat('Team', () => room({ roomId: 'classic-2' }), { sync: false })
    expect(fresh.desktopAuthorityHash).toMatch(/^[a-f0-9]{64}$/)
    expect(fresh.desktopAuthorityHash).not.toBe(HASH)

    const merged = chat.mergeGroupChatSyncSnapshots(old, chat.groupChatSyncSnapshot(), {
      deletedRooms: ['id:classic-1'],
      writeRevision: 10
    })

    expect(merged.rooms['id:classic-1']).toBeUndefined()
    expect(merged.rooms['id:classic-2'].desktopAuthorityHash).toBe(fresh.desktopAuthorityHash)
  })

  it('retains commitments on included rooms under the projection byte limit', async () => {
    const { chat } = await setup()

    const many = Object.fromEntries(
      Array.from({ length: 100 }, (_, index) => [
        String(index),
        room({
          roomId: `classic-${index}`,
          desktopAuthorityHash: HASH,
          log: [{ at: index, from: { kind: 'user', name: 'You' }, text: '文'.repeat(3000) }]
        })
      ])
    )

    const projection = chat.groupChatSyncSnapshot(many)
    expect(chat.groupChatGatewayJsonSize(projection)).toBeLessThanOrEqual(48000)
    expect(Object.keys(projection.rooms).length).toBeGreaterThan(0)

    for (const projected of Object.values(projection.rooms)) {
      expect(projected.desktopAuthorityHash).toBe(HASH)
    }
  })

  it.each(['not-hex', 'a'.repeat(63), 'a'.repeat(65), 123, {}, ['ab'.repeat(32)]])(
    'does not import malformed remote authority %j',
    async value => {
      const { chat } = await setup()
      const remote = { version: 3, rooms: { 'id:classic-1': { ...room(), name: 'Team', desktopAuthorityHash: value } } }

      const merged = chat.mergeRemoteGroupChatSnapshotIntoRooms(remote as never, {
        Team: room({ desktopAuthorityHash: HASH })
      })

      expect(merged.Team.desktopAuthorityHash).toBe(HASH)
      const imported = chat.mergeRemoteGroupChatSnapshotIntoRooms(remote as never, {})
      expect(imported.Team.desktopAuthorityHash).toBeUndefined()
    }
  )

  it('does not publish classic authority for a hosted room', async () => {
    const { chat } = await setup()
    const hosted = room({ hosted: 'gateway-1', desktopAuthorityHash: HASH })
    expect(chat.groupChatSyncSnapshot({ Team: hosted }).rooms['id:classic-1'].desktopAuthorityHash).toBeUndefined()
  })

  it('keeps projection-only commitments passive through repeated activation', async () => {
    const { chat } = await setup()
    chat.$groupChats.set({ Team: room({ desktopAuthorityHash: HASH }) })
    chat.backfillClassicGroupAuthorities()
    expect(chat.$groupChats.get().Team.desktopAuthorityToken).toBeUndefined()
    chat.backfillClassicGroupAuthorities(['Team'])
    expect(chat.$groupChats.get().Team.desktopAuthorityToken).toBeUndefined()
    expect(chat.$groupChats.get().Team.desktopAuthorityHash).toBe(HASH)
  })

  it('persists the private token but never publishes it in ui_meta', async () => {
    const { chat } = await setup()
    chat.$groupChats.set({ Team: room() })
    chat.backfillClassicGroupAuthorities(['Team'])
    const current = chat.$groupChats.get().Team
    const durable = chat.durableGroupChatRooms()
    const projection = chat.groupChatSyncSnapshot()
    expect(durable.Team.desktopAuthorityToken).toBe(current.desktopAuthorityToken)
    expect(projection.rooms['id:classic-1'].desktopAuthorityHash).toBe(current.desktopAuthorityHash)
    expect(JSON.stringify(projection)).not.toContain(String(current.desktopAuthorityToken))
  })

  it('backfills a hydrated room exactly once and publishes the same value after reopening', async () => {
    const { chat, gateway } = await setup()
    chat.$groupChats.set({ Team: room() })
    await chat.activateClassicGroupAuthorities()
    await drain(() => gateway.rpcFor('profiles.configure').length < 1)
    const first = chat.$groupChats.get().Team.desktopAuthorityHash
    expect(first).toMatch(/^[a-f0-9]{64}$/)
    await chat.activateClassicGroupAuthorities(['Team'])
    await drain(() => true, 15)
    expect(gateway.rpcFor('profiles.configure')).toHaveLength(1)
    expect(chat.$groupChats.get().Team.desktopAuthorityHash).toBe(first)
    expect((gateway.storage.get('group-chats') as Record<string, GroupChat>).Team.desktopAuthorityHash).toBe(first)
  })

  it('does not regenerate a conflicted identity after persistence and old-client replay', async () => {
    const { chat } = await setup()
    chat.$groupChats.set(chat.durableGroupChatRooms({ Team: room({ desktopAuthorityConflict: true }) }))
    chat.backfillClassicGroupAuthorities()

    const merged = chat.mergeRemoteGroupChatSnapshotIntoRooms(
      chat.groupChatSyncSnapshot({ Team: room({ desktopAuthorityHash: HASH, syncRevision: 99 }) })
    )

    expect(merged.Team.desktopAuthorityHash).toBeUndefined()
    expect(merged.Team.desktopAuthorityConflict).toBe(true)
  })

  it('cannot inherit authority from a different room with the same display name', async () => {
    const { chat } = await setup()
    const remote = chat.groupChatSyncSnapshot({ Team: room({ desktopAuthorityHash: HASH, roomId: 'old-room' }) })
    const merged = chat.mergeRemoteGroupChatSnapshotIntoRooms(remote, { Team: room({ roomId: 'new-room' }) })
    expect(merged.Team.desktopAuthorityHash).toBeUndefined()
  })

  it('rejects a commitment whose projection key and embedded room ID disagree', async () => {
    const { chat } = await setup()

    const remote = {
      version: 3,
      rooms: {
        'id:classic-1': { log: room().log, roomId: 'other', name: 'Team', revision: 100, desktopAuthorityHash: OTHER }
      }
    }

    const merged = chat.mergeGroupChatSyncSnapshots(
      remote,
      chat.groupChatSyncSnapshot({ Team: room({ desktopAuthorityHash: HASH }) })
    )

    expect(merged.rooms['id:classic-1'].desktopAuthorityHash).toBe(HASH)
    expect(merged.rooms['id:classic-1'].roomId).toBe('classic-1')
    expect(chat.mergeRemoteGroupChatSnapshotIntoRooms(remote, {}).Team.desktopAuthorityHash).toBeUndefined()
  })

  it('does not copy the old commitment into a replacement incarnation', async () => {
    const { chat } = await setup()
    chat.$groupChats.set({ Team: room({ desktopAuthorityHash: HASH }) })

    const replacement = chat.updateGroupChat('Team', current => ({ ...current, roomId: 'replacement' }), {
      sync: false
    })

    expect(replacement.desktopAuthorityHash).toMatch(/^[a-f0-9]{64}$/)
    expect(replacement.desktopAuthorityHash).not.toBe(HASH)
  })

  it('real open, rename and disband paths preserve other rooms and retire this incarnation', async () => {
    const { chat, gateway } = await setup()
    const view = await import('./group-chat-view')
    chat.$groupChats.set({ Team: room(), Other: room({ roomId: 'other', desktopAuthorityHash: OTHER }) })
    view.openGroupChat('Team')
    const hash = chat.$groupChats.get().Team.desktopAuthorityHash
    expect(hash).toMatch(/^[a-f0-9]{64}$/)
    await view.renameGroupChat('Team', 'Renamed', [])
    expect(chat.$groupChats.get().Renamed.desktopAuthorityHash).toBe(hash)
    await view.disbandGroupChat('Renamed', [])
    const saved = gateway.storage.get('group-chats') as Record<string, GroupChat>
    expect(saved.Renamed).toBeUndefined()
    expect(saved.Other.desktopAuthorityHash).toBe(OTHER)
    chat.updateGroupChat('Team', () => room({ roomId: chat.mintGroupRoomId() }), { sync: false })
    expect(chat.$groupChats.get().Team.desktopAuthorityHash).not.toBe(hash)
  })

  it('does not advertise a new commitment if durable storage fails', async () => {
    const { chat, gateway, shared } = await setup()
    const context = scriptedStorage(gateway.storage)
    context.storage.set = vi.fn(async () => {
      throw new Error('disk full')
    })
    shared.setPluginCtx(context)
    chat.$groupChats.set({ Team: room() })
    await expect(chat.activateClassicGroupAuthorities()).rejects.toThrow('disk full')
    await drain(() => true, 30)
    expect(context.storage.set).toHaveBeenCalled()
    expect(gateway.rpcFor('profiles.configure')).toHaveLength(0)
  })

  it('preserves omitted commitments on local mutation and fences attempted rotation', async () => {
    const { chat } = await setup()
    chat.$groupChats.set({ Team: room({ desktopAuthorityHash: HASH }) })
    const retained = chat.updateGroupChat('Team', () => room(), { sync: false })
    expect(retained.desktopAuthorityHash).toBe(HASH)

    const conflict = chat.updateGroupChat('Team', current => ({ ...current, desktopAuthorityHash: OTHER }), {
      sync: false
    })

    expect(conflict.desktopAuthorityHash).toBeUndefined()
    expect(conflict.desktopAuthorityConflict).toBe(true)
  })

  it('propagates conflicts across gateways and then stops writing', async () => {
    const { chat, gateway } = await setup()
    const peer = createGroupGateway()
    const metaKey = 'hermes-bots-groups'
    gateway.uiMeta[metaKey] = chat.groupChatSyncSnapshot({ Team: room({ desktopAuthorityHash: HASH }) })
    peer.uiMeta[metaKey] = chat.groupChatSyncSnapshot({ Team: room({ desktopAuthorityHash: OTHER }) })
    chat.$groupChats.set({ Team: room({ desktopAuthorityHash: HASH }) })
    host.profileRoutes = async () => [{ connectionId: 'peer', profile: 'default' }]

    host.requestProfile = async (route: { connectionId: string }, method: string, params: Record<string, unknown>) => {
      const target = route.connectionId === 'peer' ? peer : gateway

      return (target.host.request as (method: string, params: Record<string, unknown>) => Promise<unknown>)(
        method,
        params
      )
    }

    chat.scheduleGroupChatServerSync()
    await drain(() => true, 50)

    for (const target of [gateway, peer]) {
      const projected = target.uiMeta[metaKey] as ReturnType<typeof chat.groupChatSyncSnapshot>
      expect(projected.rooms['id:classic-1'].desktopAuthorityHash).toBeUndefined()
      expect(projected.rooms['id:classic-1'].desktopAuthorityConflict).toBe(true)
    }

    const writes = gateway.rpcFor('profiles.configure').length + peer.rpcFor('profiles.configure').length
    chat.scheduleGroupChatServerSync()
    await drain(() => true, 30)
    expect(gateway.rpcFor('profiles.configure').length + peer.rpcFor('profiles.configure').length).toBe(writes)
  })

  it('generates entropy without time or Math.random and refuses a missing secure RNG', async () => {
    const { chat } = await setup()
    vi.spyOn(Math, 'random').mockReturnValue(0)
    vi.spyOn(Date, 'now').mockReturnValue(1)
    const ids = Array.from({ length: 20 }, () => chat.mintGroupRoomId())
    expect(new Set(ids).size).toBe(ids.length)

    const hashes = ids.map(
      roomId => chat.updateGroupChat(roomId, () => room({ roomId }), { sync: false }).desktopAuthorityHash
    )

    const tokens = Object.values(chat.$groupChats.get()).map(value => value.desktopAuthorityToken)

    expect(new Set(hashes).size).toBe(ids.length)
    expect(tokens.every(token => /^authority:[a-f0-9]{64}$/.test(String(token)))).toBe(true)
    vi.stubGlobal('crypto', {})
    expect(() => chat.updateGroupChat('unavailable', () => room(), { sync: false })).toThrow()
    expect(chat.$groupChats.get().unavailable).toBeUndefined()
  })
})
