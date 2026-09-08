import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { pluginSdkMock } from './group-test-utils'
const { host } = vi.hoisted(() => ({ host: {} as Record<string, any> }))
vi.mock('@hermes/plugin-sdk', async () => pluginSdkMock(host))

const state = vi.hoisted(() => ({ lifecycle: 1, rooms: {} as any, capabilities: {} as any }))
vi.mock('./group-chat', () => ({
  $groupChats: { get: () => state.rooms },
  groupChatHostedGateway: (room: any) => room?.hosted
}))
vi.mock('./i18n', () => ({
  botsText: () => ({ group: { hostRouteMissing: 'missing route', hostUpdateNeeded: () => 'update needed' } })
}))
vi.mock('./hosted-room-runtime', async () => {
  const transport = await import('./hosted-room-transport')

  return {
    $hostedRoomCapabilities: {
      get: () => state.capabilities,
      set: (v: any) => {
        state.capabilities = v
      }
    },
    hostedRoomLifecycleIsCurrent: (v: number) => v === state.lifecycle,
    hostedRoomLifecycleToken: () => state.lifecycle,
    invalidateHostedRoomPoll: vi.fn(),
    refreshHostedRooms: vi.fn(async () => {}),
    requestHostedConnection: transport.requestHostedConnection
  }
})
import { createHash } from 'node:crypto'

import {
  $hostedRoomCleanup,
  dispatchHostedRoomCleanup,
  resetHostedRoomCleanupForTests,
  startHostedRoomCleanup,
  stopHostedRoomCleanup
} from './hosted-room-cleanup'
import { reconnectHostedGroupChatPeer } from './hosted-room-reauthorization'

const home = { connectionId: 'home', mode: 'remote', profile: 'default', targetProfile: 'default' }
const peer = { connectionId: 'peer', mode: 'remote', profile: 'desktop-alias', targetProfile: 'ops' }
const decoy = { connectionId: 'decoy', mode: 'remote', profile: 'desktop-alias', targetProfile: 'ops' }

const catalog = {
  installation_id: 'install:peer',
  catalog_digest: 'digest:peer',
  persistent_process: true,
  link_modes: ['direct'],
  protocol_versions: [2],
  text: true,
  attachments: false
}

const oldHash = 'a'.repeat(64)
const grant = 'synthetic-fresh-grant'
const freshHash = createHash('sha256').update(grant).digest('hex')
let calls: Array<{ route: any; method: string; params: any }>
let liveHash: string
let registeredControl: boolean
let mode: string
let persisted: Map<string, any>

const storage = {
  get: async (key: string, fallback: any) => (persisted.has(key) ? structuredClone(persisted.get(key)) : fallback),
  set: async (key: string, value: any) => {
    persisted.set(key, structuredClone(value))
  }
}

beforeEach(async () => {
  resetHostedRoomCleanupForTests()
  state.lifecycle = 1
  mode = 'success'
  liveHash = oldHash
  registeredControl = false
  calls = []
  state.capabilities = {
    home: {
      authorityId: 'install:home',
      peerGrantRenewal: true,
      routeGrantFingerprint: true,
      reciprocalRoomControl: true,
      roomLink: { endpoint: 'https://home.example.test' }
    },
    peer: { authorityId: 'install:peer' },
    decoy: { authorityId: 'install:decoy' }
  }
  state.rooms = {
    Review: {
      hosted: 'install:home',
      hostedConnectionId: 'home',
      roomId: 'room-1',
      members: [decoy, peer].map(route => ({
        name: 'desktop-alias',
        handle: 'ops',
        targetProfile: 'ops',
        connectionId: route.connectionId,
        sourceScoped: true,
        route
      }))
    }
  }
  host.profileRoutes = async () => [home, decoy, peer]

  host.request = () => {
    throw new Error('ambient routing must never be used')
  }

  host.requestProfile = async (route: any, method: string, params: any) => {
    calls.push({ route, method, params })

    if (method === 'groups.state') {
      return {
        room: {
          room_id: 'room-1',
          authority_gateway_id: 'install:home',
          authority_epoch: mode === 'epoch-changed' ? 2 : 1,
          members:
            mode === 'member-removed'
              ? []
              : [
                  {
                    member_id: 'member-ops',
                    profile: 'ops',
                    handle: 'ops',
                    target: { kind: 'peer', installation_id: 'install:peer' }
                  }
                ]
        },
        driver_status: {
          peer_routes: [
            {
              member_id: 'member-ops',
              grant_sha256: liveHash,
              status: liveHash === oldHash ? 'needs_reauthorization' : 'ready'
            }
          ]
        }
      }
    }

    if (method === 'groups.capabilities') {
      return {
        authority_gateway_id:
          route.connectionId === 'home' ? (mode === 'wrong-home' ? 'install:new' : 'install:home') : 'install:peer',
        features: mode === 'old-capability' ? [] : ['reciprocal_room_control', 'reciprocal_room_control_setup'],
        methods: ['groups.peer.revoke_exact'],
        driver: true,
        persistent_process: true,
        room_link: {
          enabled: true,
          endpoint: {
            available: true,
            url: route.connectionId === 'home' ? 'https://home.example.test' : 'https://peer.example.test'
          },
          catalog: { ...catalog, installation_id: route.connectionId === 'home' ? 'install:home' : 'install:peer' }
        }
      }
    }

    if (method === 'groups.peer.invite') {
      return { grant, target_profile: 'ops', catalog, expires_at: 3601, status_expires_at: 2592001 }
    }

    if (method === 'groups.peer.register') {
      liveHash = freshHash

      return { registered: true }
    }

    if (method === 'groups.control.invite') {
      if (mode === 'stale-after-invite') {
        state.lifecycle++
      }

      return {
        room_id: 'room-1',
        member_id: 'member-ops',
        authority_gateway_id: 'install:home',
        authority_epoch: mode === 'stale-epoch' ? 2 : 1,
        control_token: 'c'.repeat(43),
        home_url: 'https://home.example.test',
        room_name: 'Review',
        member_count: 2,
        expires_at: 253402300799
      }
    }

    if (method === 'groups.control.register') {
      if (mode === 'control-offline') {
        throw new Error('control registration offline')
      }

      registeredControl = true

      if (mode === 'lost-control-ack') {
        throw new Error('control ACK lost')
      }

      if (mode === 'stale-after-register') {
        state.lifecycle++
      }

      return { registered: true, room_id: 'room-1', member_id: 'member-ops' }
    }

    if (method === 'groups.peer.revoke_exact') {
      return { revoked: true }
    }

    throw new Error(`unexpected ${method}`)
  }

  persisted = new Map()
  await startHostedRoomCleanup(storage as any)
})
afterEach(() => resetHostedRoomCleanupForTests())

it.each(['same-runtime', 'restart', 'rotated-grant'])(
  'recovers missing participant control after %s',
  async recovery => {
    mode = 'control-offline'
    await expect(reconnectHostedGroupChatPeer('Review', 'member-ops')).rejects.toThrow()
    expect(liveHash).toBe(freshHash)
    expect(registeredControl).toBe(false)
    expect($hostedRoomCleanup.get().operations).toEqual([
      expect.objectContaining({
        reciprocalControl: true,
        controlAuthorityId: 'install:home',
        controlAuthorityEpoch: 1,
        roomId: 'room-1',
        memberId: 'member-ops',
        profile: 'ops'
      })
    ])
    mode = 'success'

    if (recovery === 'rotated-grant') {
      liveHash = 'd'.repeat(64)
    }

    if (recovery === 'restart') {
      stopHostedRoomCleanup()
      await startHostedRoomCleanup(storage as any)
    } else {
      await dispatchHostedRoomCleanup()
    }

    expect(registeredControl).toBe(true)
    expect($hostedRoomCleanup.get().operations).toEqual([])
    expect(calls.some(call => call.route.connectionId === 'decoy')).toBe(false)

    if (recovery === 'rotated-grant') {
      expect(liveHash).toBe('d'.repeat(64))
      expect(calls.filter(call => call.method.includes('revoke'))).toEqual([
        expect.objectContaining({ method: 'groups.peer.revoke_exact', params: { grant, profile: 'ops' } })
      ])
    } else {
      expect(calls.some(call => call.method.includes('revoke'))).toBe(false)
    }
  }
)

it('keeps the obligation across a lost reciprocal acknowledgement', async () => {
  mode = 'lost-control-ack'
  await expect(reconnectHostedGroupChatPeer('Review', 'member-ops')).rejects.toThrow()
  expect(registeredControl).toBe(true)
  expect($hostedRoomCleanup.get().operations).toHaveLength(1)
  mode = 'success'
  await dispatchHostedRoomCleanup()
  expect($hostedRoomCleanup.get().operations).toEqual([])
  expect(liveHash).toBe(freshHash)
})

it.each(['epoch-changed', 'member-removed', 'wrong-home'])(
  'does not forward old credentials after %s',
  async change => {
    mode = 'control-offline'
    await expect(reconnectHostedGroupChatPeer('Review', 'member-ops')).rejects.toThrow()
    const before = calls.length
    mode = change
    await dispatchHostedRoomCleanup()
    const recoveryCalls = calls.slice(before)
    expect(recoveryCalls.some(call => call.method.includes('register') || call.method.includes('invite'))).toBe(false)
    expect(registeredControl).toBe(false)
    expect(liveHash).toBe(freshHash)

    if (change === 'wrong-home') {
      expect(recoveryCalls.some(call => call.method.includes('revoke'))).toBe(false)
      expect($hostedRoomCleanup.get().operations).toHaveLength(1)
    } else {
      expect($hostedRoomCleanup.get().operations).toEqual([])
      expect(recoveryCalls.filter(call => call.method.includes('revoke'))).toEqual([
        expect.objectContaining({ method: 'groups.peer.revoke_exact', params: { grant, profile: 'ops' } })
      ])
    }
  }
)
