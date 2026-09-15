import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { pluginSdkMock, scriptedStorage } from './group-test-utils'
import type { GroupChat, GroupMember } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => pluginSdkMock(host))

const members: GroupMember[] = ['writer', 'reviewer'].map(name => ({
  connectionId: 'local',
  name,
  remoteSource: true,
  sourceScoped: true,
  targetProfile: name
}))

// Exact shape observed through the packaged1d gateway during classic UAT.
const nonpersistent = {
  protocol_version: 2,
  driver: true,
  persistent_process: false,
  authority_gateway_id: 'install:home',
  room_link: { enabled: false, reason: 'gateway_roomlink_secret_unavailable' },
  features: [
    'authority_epoch',
    'coordinator_fencing',
    'room_identity',
    'monotonic_log',
    'idempotent_send',
    'replayable_disband',
    'typed_events',
    'actor_identity',
    'log_replication',
    'authority_takeover'
  ],
  methods: [
    'groups.capabilities',
    'groups.list',
    'groups.create',
    'groups.state',
    'groups.send',
    'groups.rename',
    'groups.log',
    'groups.disband',
    'groups.replicate',
    'groups.replica_state',
    'groups.promote',
    'groups.demote',
    'groups.stop',
    'groups.retry',
    'groups.approve',
    'groups.peer.invite',
    'groups.peer.revoke',
    'groups.peer.register'
  ],
  max_log_limit: 500
}

const stored = {
  room_id: 'room-classic',
  name: 'Classic',
  authority_gateway_id: 'install:home',
  authority_epoch: 1,
  members: members.map(member => ({ member_id: member.name, profile: member.name, handle: member.name })),
  latest_seq: 1,
  disbanded_at: null
}

function projected(overrides: Partial<GroupChat> = {}): GroupChat {
  return { roomId: stored.room_id, log: [], watermarks: {}, members, continuityMode: 'desktop', ...overrides }
}

async function load(handler: (method: string, params: Record<string, unknown>) => unknown) {
  vi.resetModules()
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  const values = new Map<string, unknown>()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  const state = (value: string) => ({ get: () => value, listen: () => () => undefined })
  Object.assign(host, {
    activeConnectionId: () => 'local',
    notify: vi.fn(),
    profileRoutes: async () => [{ connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' }],
    requestProfile: async (_route: unknown, method: string, params: Record<string, unknown>) => {
      calls.push({ method, params })

      return handler(method, params)
    },
    state: { connectionId: state('local'), gateway: state('open'), profile: state('default') }
  })
  const chat = await import('./group-chat')
  const rounds = await import('./group-rounds')
  const runtime = await import('./hosted-room-runtime')
  const shared = await import('./shared')
  const storage = scriptedStorage(values).storage
  shared.setPluginCtx(scriptedStorage(values))
  chat.$groupChats.set({ Classic: projected() })

  return { chat, rounds, runtime, calls, storage }
}

function event() {
  return {
    room_id: stored.room_id,
    seq: 1,
    event_id: 'member-1',
    kind: 'message.member',
    actor: { kind: 'member', id: 'writer', profile: 'writer' },
    payload: { text: 'Stored reply', thread_id: 'thread-1' },
    created_at: 1
  }
}

beforeEach(() => vi.useFakeTimers())
afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
})

function heldResponse() {
  let entered!: () => void
  let release!: (value: unknown) => void

  const started = new Promise<void>(resolve => {
    entered = resolve
  })

  const response = new Promise<unknown>(resolve => {
    release = resolve
  })

  return {
    started,
    release,
    wait: () => {
      entered()

      return response
    }
  }
}

describe('capability reply freshness', () => {
  it('waits for the first accepted capability without throwing or granting classic absence', async () => {
    const older = heldResponse()
    const newer = heldResponse()
    let phase = 'older'

    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        if (phase === 'older') {
          return older.wait()
        }

        if (phase === 'newer') {
          return newer.wait()
        }

        return nonpersistent
      }

      if (method === 'groups.list') {
        return { rooms: [], next_offset: null }
      }

      throw new Error(`Unexpected ${method}`)
    })

    const routes = host.profileRoutes
    host.profileRoutes = async () => []
    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    host.profileRoutes = routes

    try {
      const refresh = loaded.runtime.refreshHostedRooms()

      const result = refresh.then(
        () => null,
        error => String(error)
      )

      await older.started
      phase = 'newer'
      const probe = loaded.runtime.probeHostedRoomMembers(members)
      await newer.started
      older.release(nonpersistent)
      expect(await result).toBeNull()
      expect(loaded.runtime.$hostedRoomCapabilities.get().local).toBeUndefined()
      expect(loaded.runtime.groupChatContinuityReady(projected({ roomId: 'unrelated' }))).toBe(false)
      phase = 'complete'
      newer.release(nonpersistent)
      await probe
      expect(loaded.runtime.$hostedRoomCapabilities.get().local.kind).toBe('driver-capable')
      await loaded.runtime.refreshHostedRooms()
      expect(loaded.runtime.groupChatContinuityReady(projected({ roomId: 'unrelated' }))).toBe(true)
    } finally {
      older.release(nonpersistent)
      newer.release(nonpersistent)
      loaded.runtime.stopHostedRoomRuntime()
    }
  })

  it.each(['none', 'member-probe', 'refresh'])(
    'does not cancel a real recovery probe after an unsupported-cache %s shortcut',
    async shortcut => {
      const held = heldResponse()
      let recovering = false

      const loaded = await load(method => {
        if (method === 'groups.capabilities') {
          if (!recovering) {
            throw Object.assign(new Error('Method not found'), { code: -32601 })
          }

          return held.wait()
        }

        if (method === 'groups.list') {
          return { rooms: [], next_offset: null }
        }

        if (method === 'groups.disband') {
          return {}
        }

        throw new Error(`Unexpected ${method}`)
      })

      try {
        await loaded.runtime.startHostedRoomRuntime(loaded.storage)
        loaded.chat.$groupChats.set({
          ...loaded.chat.$groupChats.get(),
          Hosted: projected({
            roomId: 'hosted-copy',
            hosted: 'install:home',
            hostedEpoch: 1,
            hostedConnectionId: 'local'
          })
        })
        recovering = true

        const recovery = loaded.runtime.disbandHostedGroupChat('Hosted').then(
          value => ({ value, error: null }),
          error => ({ value: false, error: String(error) })
        )

        await held.started
        const requests = loaded.calls.filter(call => call.method === 'groups.capabilities').length

        if (shortcut === 'member-probe') {
          await loaded.runtime.probeHostedRoomMembers(members)
        }

        if (shortcut === 'refresh') {
          await loaded.runtime.refreshHostedRooms()
        }

        expect(loaded.calls.filter(call => call.method === 'groups.capabilities')).toHaveLength(requests)
        held.release({ ...nonpersistent, persistent_process: true })
        expect(await recovery).toEqual({ value: true, error: null })
        expect(loaded.calls.filter(call => call.method === 'groups.disband')).toHaveLength(1)
        expect(loaded.runtime.$hostedRoomCapabilities.get().local.persistentProcess).toBe(true)
      } finally {
        held.release({ ...nonpersistent, persistent_process: true })
        loaded.runtime.stopHostedRoomRuntime()
      }
    }
  )

  for (const reader of ['refresh', 'member-probe']) {
    for (const baseline of [false, true]) {
      for (const olderFailure of [false, true]) {
        it(`keeps a newer confirmation over ${reader}, persistent=${baseline}, oldFailure=${olderFailure}`, async () => {
          const held = heldResponse()
          let hold = false
          let persistent = baseline

          const loaded = await load(method => {
            if (method === 'groups.capabilities') {
              const captured = { ...nonpersistent, persistent_process: persistent }

              if (hold) {
                hold = false

                return held.wait().then(() => {
                  if (olderFailure) {
                    throw new Error('Old capability request failed')
                  }

                  return captured
                })
              }

              return captured
            }

            if (method === 'groups.list') {
              return { rooms: [stored], next_offset: null }
            }

            if (method === 'groups.state') {
              return { room: stored }
            }

            if (method === 'groups.log') {
              return { events: [event()], latest_seq: 1, has_more: false }
            }

            throw new Error(`Unexpected ${method}`)
          })

          try {
            await loaded.runtime.startHostedRoomRuntime(loaded.storage)
            persistent = !baseline
            hold = true

            const older =
              reader === 'refresh'
                ? loaded.runtime.refreshHostedRooms()
                : loaded.runtime.probeHostedRoomMembers(members)

            await held.started
            persistent = baseline
            await loaded.runtime.probeHostedRoomMembers(members)
            const confirmed = loaded.runtime.$hostedRoomCapabilities.get().local
            expect(confirmed.persistentProcess).toBe(baseline)
            held.release({})
            await older
            expect(loaded.runtime.$hostedRoomCapabilities.get().local).toEqual(confirmed)

            if (!baseline) {
              expect(loaded.chat.$groupChats.get().Classic.hostedStatus?.state).toBe('read-only')
            }
          } finally {
            held.release({})
            loaded.runtime.stopHostedRoomRuntime()
          }
        })
      }
    }
  }
})

describe('inventory observation invalidation', () => {
  it('an authority-route probe failure invalidates a held list before recovery', async () => {
    const held = heldResponse()
    let hold = false
    let offline = false

    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        if (offline) {
          throw new Error('offline')
        }

        return nonpersistent
      }

      if (method === 'groups.list') {
        return hold ? held.wait() : { rooms: [], next_offset: null }
      }

      throw new Error(`Unexpected ${method}`)
    })

    try {
      await loaded.runtime.startHostedRoomRuntime(loaded.storage)
      loaded.chat.$groupChats.set({
        ...loaded.chat.$groupChats.get(),
        Hosted: projected({
          roomId: 'hosted-copy',
          hosted: 'install:home',
          hostedConnectionId: 'local',
          hostedEpoch: 1
        })
      })
      hold = true
      const refresh = loaded.runtime.refreshHostedRooms()
      await held.started
      offline = true
      await expect(loaded.runtime.disbandHostedGroupChat('Hosted')).rejects.toThrow()
      const afterFailure = loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)
      offline = false
      await loaded.runtime.probeHostedRoomMembers(members)
      held.release({ rooms: [], next_offset: null })
      await refresh
      expect(afterFailure).toBe(false)
      expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
      expect(loaded.calls.some(call => call.method === 'groups.disband')).toBe(false)
    } finally {
      held.release({ rooms: [], next_offset: null })
      loaded.runtime.stopHostedRoomRuntime()
    }
  })

  it('does not publish an expired-history error from a revoked authority read', async () => {
    const held = heldResponse()
    let missing = false
    let persistent = true

    const loaded = await load(async method => {
      if (method === 'groups.capabilities') {
        return { ...nonpersistent, persistent_process: persistent }
      }

      if (method === 'groups.list') {
        return { rooms: missing ? [] : [stored], next_offset: null }
      }

      if (method === 'groups.state') {
        if (missing) {
          await held.wait()
          throw new Error('history expired')
        }

        return { room: stored }
      }

      if (method === 'groups.log') {
        return { events: [event()], latest_seq: 1, has_more: false }
      }

      throw new Error(`Unexpected ${method}`)
    })

    try {
      await loaded.runtime.startHostedRoomRuntime(loaded.storage)
      missing = true
      const refresh = loaded.runtime.refreshHostedRooms()
      await held.started
      persistent = false
      await loaded.runtime.probeHostedRoomMembers(members)
      held.release({})
      await refresh
      expect(loaded.chat.$groupChats.get().Classic.hostedStatus?.state).toBe('read-only')
      expect(loaded.chat.$groupChats.get().Classic.log.map(entry => entry.text)).toContain('Stored reply')
      missing = false
      await loaded.runtime.refreshHostedRooms()
      expect(loaded.chat.$groupChats.get().Classic.hostedStatus?.state).toBe('read-only')
    } finally {
      held.release({})
      loaded.runtime.stopHostedRoomRuntime()
    }
  })

  for (const retired of [false, true]) {
    for (const provenance of ['legacy', 'mixed']) {
      it.each(['list-error', 'offline', 'authority', 'old-gateway'])(
        `retains known IDs after %s, retired=${retired}, members=${provenance}`,
        async failure => {
          let phase = 'initial'

          const legacy = members.map((member, index) => ({
            ...member,
            remoteSource: provenance === 'mixed' && index === 0 ? true : undefined
          }))

          const loaded = await load(method => {
            if (method === 'groups.capabilities') {
              if (phase === 'invalid' && failure === 'offline') {
                throw new Error('offline')
              }

              if (phase === 'invalid' && failure === 'old-gateway') {
                throw Object.assign(new Error('method not found'), { code: -32601 })
              }

              return {
                ...nonpersistent,
                authority_gateway_id: phase === 'invalid' && failure === 'authority' ? 'other' : 'install:home'
              }
            }

            if (method === 'groups.list') {
              if (phase === 'invalid') {
                throw new Error('list failed')
              }

              return {
                rooms: phase === 'initial' ? [{ ...stored, disbanded_at: retired ? 1 : null }] : [],
                next_offset: null
              }
            }

            throw new Error('state unavailable')
          })

          try {
            loaded.chat.$groupChats.set(retired ? {} : { Classic: projected({ members: legacy }) })
            await loaded.runtime.startHostedRoomRuntime(loaded.storage)
            loaded.chat.$groupChats.set({ Classic: projected({ members: legacy }) })
            expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
            phase = 'invalid'
            await loaded.runtime.refreshHostedRooms()
            expect(loaded.rounds.sendToGroupChat('Classic', legacy, 'Must not run')).toBeNull()
            phase = 'recovered'
            loaded.runtime.stopHostedRoomRuntime()
            await loaded.runtime.startHostedRoomRuntime(loaded.storage)
            expect(loaded.rounds.sendToGroupChat('Classic', legacy, 'Still owned')).toBeNull()
            expect(loaded.chat.$groupChats.get().Classic.log).toEqual([])
            expect(loaded.calls.some(call => call.method === 'prompt.submit')).toBe(false)
          } finally {
            loaded.runtime.stopHostedRoomRuntime()
          }
        }
      )
    }
  }

  it.each(['authority-aba', 'offline-recovery'])(
    'rejects old pages after %s and requires a fresh complete list',
    async change => {
      const held = heldResponse()
      let phase = 'initial'
      let capability: unknown = nonpersistent

      const loaded = await load((method, params) => {
        if (method === 'groups.capabilities') {
          if (capability instanceof Error) {
            throw capability
          }

          return capability
        }

        if (method === 'groups.list') {
          if (phase !== 'held') {
            return { rooms: [], next_offset: null }
          }

          if (params.offset === 0) {
            return { rooms: [{ ...stored, room_id: 'other' }], next_offset: 1 }
          }

          return held.wait()
        }

        throw new Error('Unexpected read')
      })

      try {
        await loaded.runtime.startHostedRoomRuntime(loaded.storage)
        phase = 'held'
        const refresh = loaded.runtime.refreshHostedRooms()
        await held.started
        capability =
          change === 'authority-aba' ? { ...nonpersistent, authority_gateway_id: 'replacement' } : new Error('offline')
        await loaded.runtime.probeHostedRoomMembers(members)
        capability = nonpersistent
        await loaded.runtime.probeHostedRoomMembers(members)
        expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
        held.release({ rooms: [{ ...stored, room_id: 'stale-page-only' }], next_offset: null })
        await refresh
        expect(loaded.rounds.sendToGroupChat('Classic', members, 'Not yet')).toBeNull()
        phase = 'fresh'
        await loaded.runtime.refreshHostedRooms()
        expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(true)
        expect(loaded.runtime.groupChatContinuityReady(projected({ roomId: 'stale-page-only' }))).toBe(true)
        expect(loaded.runtime.groupChatContinuityReady(projected({ roomId: 'other' }))).toBe(false)
      } finally {
        held.release({ rooms: [], next_offset: null })
        loaded.runtime.stopHostedRoomRuntime()
      }
    }
  )

  it.each(['groups.state', 'groups.log'])(
    'rejects stale %s UI, approvals and cache then recovers with fresh reads',
    async window => {
      const held = heldResponse()
      let persistent = false
      let hold = false

      const pending = {
        kind: 'approval',
        member_id: 'writer',
        task_id: 'task',
        request_id: 'request',
        execution_generation: 1
      }

      const response = { room: stored, driver_status: { pending_actions: [pending] } }
      const log = { events: [event()], latest_seq: 1, has_more: false }

      const loaded = await load(method => {
        if (method === 'groups.capabilities') {
          return { ...nonpersistent, persistent_process: persistent }
        }

        if (method === 'groups.list') {
          return { rooms: [stored], next_offset: null }
        }

        if (hold && method === window) {
          return held.wait()
        }

        if (method === 'groups.state') {
          return response
        }

        if (method === 'groups.log') {
          return log
        }

        throw new Error(`Unexpected ${method}`)
      })

      try {
        await loaded.runtime.startHostedRoomRuntime(loaded.storage)
        expect(loaded.chat.$groupChats.get().Classic.hostedStatus?.state).toBe('read-only')
        hold = true
        persistent = true
        const refresh = loaded.runtime.refreshHostedRooms()
        await held.started
        persistent = false
        await loaded.runtime.probeHostedRoomMembers(members)
        held.release(window === 'groups.state' ? response : log)
        await refresh
        expect(loaded.chat.$groupChats.get().Classic.hostedStatus?.state).toBe('read-only')
        expect(loaded.chat.$groupClarify.get()).toEqual({})
        expect(loaded.runtime.shouldRefreshHostedRoom(loaded.chat.$groupChats.get().Classic, stored)).toBe(true)
        const reads = loaded.calls.filter(call => call.method === 'groups.state').length
        hold = false
        await loaded.runtime.refreshHostedRooms()
        expect(loaded.calls.filter(call => call.method === 'groups.state')).toHaveLength(reads + 1)
        expect(loaded.chat.$groupChats.get().Classic.hostedStatus?.state).toBe('read-only')
        persistent = true
        await loaded.runtime.refreshHostedRooms()
        expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(true)
        expect(Object.values(loaded.chat.$groupClarify.get())).toHaveLength(1)
        persistent = false
        await loaded.runtime.probeHostedRoomMembers(members)
        expect(loaded.chat.$groupChats.get().Classic.hostedStatus?.state).toBe('read-only')
        expect(loaded.chat.$groupClarify.get()).toEqual({})
      } finally {
        held.release({})
        loaded.runtime.stopHostedRoomRuntime()
      }
    }
  )

  it.each(['refresh', 'member-probe'])(
    'does not let an old %s capability response undo a newer downgrade',
    async reader => {
      const held = heldResponse()
      let hold = false
      let persistent = true

      const loaded = await load(method => {
        if (method === 'groups.capabilities') {
          if (hold) {
            hold = false

            return held.wait()
          }

          return { ...nonpersistent, persistent_process: persistent }
        }

        if (method === 'groups.list') {
          return { rooms: [stored], next_offset: null }
        }

        if (method === 'groups.state') {
          return { room: stored }
        }

        if (method === 'groups.log') {
          return { events: [event()], latest_seq: 1, has_more: false }
        }

        throw new Error(`Unexpected ${method}`)
      })

      try {
        await loaded.runtime.startHostedRoomRuntime(loaded.storage)
        hold = true

        const old =
          reader === 'refresh' ? loaded.runtime.refreshHostedRooms() : loaded.runtime.probeHostedRoomMembers(members)

        await held.started
        persistent = false
        await loaded.runtime.probeHostedRoomMembers(members)
        held.release({ ...nonpersistent, persistent_process: true })
        await old
        expect(loaded.runtime.$hostedRoomCapabilities.get().local.persistentProcess).toBe(false)
        expect(loaded.chat.$groupChats.get().Classic.hostedStatus?.state).toBe('read-only')
      } finally {
        held.release({})
        loaded.runtime.stopHostedRoomRuntime()
      }
    }
  )
})

describe('classic inventory on readable gateways', () => {
  it('fences newly listed X during a slow later page without withdrawing prior evidence for Y', async () => {
    let paging = false
    let waiting!: () => void
    let release!: (value: unknown) => void

    const started = new Promise<void>(resolve => {
      waiting = resolve
    })

    const held = new Promise(resolve => {
      release = resolve
    })

    const loaded = await load((method, params) => {
      if (method === 'groups.capabilities') {
        return nonpersistent
      }

      if (method === 'groups.list') {
        if (!paging) {
          return { rooms: [], next_offset: null }
        }

        if (params.offset === 0) {
          return { rooms: [stored], next_offset: 1 }
        }

        waiting()

        return held
      }

      if (method === 'groups.state') {
        throw new Error('state unavailable')
      }

      throw new Error(`unexpected ${method}`)
    })

    loaded.chat.$groupChats.set({ Classic: projected(), Unrelated: projected({ roomId: 'room-y' }) })
    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    paging = true
    const refresh = loaded.runtime.refreshHostedRooms()
    await started
    const xReady = loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)
    const yReady = loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Unrelated)
    release({ rooms: [], next_offset: null })
    await refresh
    expect([xReady, yReady]).toEqual([false, true])
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('a concurrent member probe invalidates old-authority inventory and fences its late list', async () => {
    let stage = 'initial'
    let waiting!: () => void
    let release!: (value: unknown) => void

    const started = new Promise<void>(resolve => {
      waiting = resolve
    })

    const held = new Promise(resolve => {
      release = resolve
    })

    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        return { ...nonpersistent, authority_gateway_id: stage === 'new' ? 'install:new' : 'install:home' }
      }

      if (method === 'groups.list') {
        if (stage === 'initial') {
          return { rooms: [], next_offset: null }
        }

        waiting()

        return held
      }

      throw new Error(`unexpected ${method}`)
    })

    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    stage = 'held'
    const refresh = loaded.runtime.refreshHostedRooms()
    await started
    stage = 'new'
    await loaded.runtime.probeHostedRoomMembers(members)
    const readyAfterNewAuthority = loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)
    release({ rooms: [], next_offset: null })
    await refresh
    expect(readyAfterNewAuthority).toBe(false)
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('keeps completed inventory usable during a healthy slow reprobe and list refresh', async () => {
    let hold = false
    let probing!: () => void
    let listing!: () => void
    let releaseProbe!: (value: unknown) => void
    let releaseList!: (value: unknown) => void

    const probeStarted = new Promise<void>(resolve => {
      probing = resolve
    })

    const listStarted = new Promise<void>(resolve => {
      listing = resolve
    })

    const probe = new Promise(resolve => {
      releaseProbe = resolve
    })

    const list = new Promise(resolve => {
      releaseList = resolve
    })

    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        if (hold) {
          probing()

          return probe
        }

        return nonpersistent
      }

      if (method === 'groups.list') {
        if (hold) {
          listing()

          return list
        }

        return { rooms: [], next_offset: null }
      }

      throw new Error(`unexpected ${method}`)
    })

    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    hold = true
    const refresh = loaded.runtime.refreshHostedRooms()
    await probeStarted
    const readyDuringProbe = loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)
    releaseProbe(nonpersistent)
    await listStarted
    const readyDuringList = loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)
    releaseList({ rooms: [], next_offset: null })
    await refresh
    expect([readyDuringProbe, readyDuringList]).toEqual([true, true])
    loaded.runtime.stopHostedRoomRuntime()
  })

  it.each(['state-error', 'state-invalid', 'log-error', 'backfill'])(
    'a complete ID inventory permits absent Y while listed X has %s',
    async failure => {
      let replayStarted!: () => void
      let release!: (value: unknown) => void

      const started = new Promise<void>(resolve => {
        replayStarted = resolve
      })

      const held = new Promise(resolve => {
        release = resolve
      })

      const loaded = await load(method => {
        if (method === 'groups.capabilities') {
          return nonpersistent
        }

        if (method === 'groups.list') {
          return { rooms: [stored], next_offset: null }
        }

        if (method === 'groups.state') {
          if (failure === 'state-error') {
            throw new Error('state unavailable')
          }

          return failure === 'state-invalid' ? {} : { room: stored }
        }

        if (method === 'groups.log') {
          if (failure === 'log-error') {
            throw new Error('log unavailable')
          }

          replayStarted()

          return held
        }

        throw new Error(`unexpected ${method}`)
      })

      loaded.chat.$groupChats.set({ Classic: projected(), Unrelated: projected({ roomId: 'room-y' }) })
      const refresh = loaded.runtime.startHostedRoomRuntime(loaded.storage)
      let readyWhileHeld = true

      if (failure === 'backfill') {
        await started
        readyWhileHeld = loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Unrelated)
        release({ events: [], latest_seq: 1, has_more: true })
      }

      await refresh
      expect(readyWhileHeld).toBe(true)
      expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Unrelated)).toBe(true)
      expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
      expect(loaded.rounds.sendToGroupChat('Classic', members, 'Must not drive X')).toBeNull()
      loaded.runtime.stopHostedRoomRuntime()
    }
  )

  it.each([true, false])(
    'inventories an empty nonpersistent gateway with driver=%s before allowing classic',
    async driver => {
      const loaded = await load(method => {
        if (method === 'groups.capabilities') {
          return { ...nonpersistent, driver }
        }

        if (method === 'groups.list') {
          return { rooms: [], next_offset: null }
        }

        throw new Error(`unexpected ${method}`)
      })

      expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
      await loaded.runtime.startHostedRoomRuntime(loaded.storage)
      expect(loaded.calls.filter(call => call.method === 'groups.list')).toHaveLength(1)
      expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(true)
      expect((await loaded.runtime.probeHostedRoomMembers(members)).eligible).toBe(false)
      expect(loaded.calls.some(call => call.method === 'groups.create' || call.method === 'prompt.submit')).toBe(false)
      loaded.runtime.stopHostedRoomRuntime()
    }
  )

  it.each([true, false])(
    'replays stored hosted rooms without handing them to the classic driver, driver=%s',
    async driver => {
      const loaded = await load(method => {
        if (method === 'groups.capabilities') {
          return { ...nonpersistent, driver }
        }

        if (method === 'groups.list') {
          return { rooms: [stored], next_offset: null }
        }

        if (method === 'groups.state') {
          return { room: stored }
        }

        if (method === 'groups.log') {
          return { events: [event()], has_more: false, latest_seq: 1 }
        }

        throw new Error(`unexpected ${method}`)
      })

      await loaded.runtime.startHostedRoomRuntime(loaded.storage)
      const room = loaded.chat.$groupChats.get().Classic
      expect(room).toMatchObject({
        hosted: 'install:home',
        hostedSeq: 1,
        hostedStatus: { state: 'read-only', canStop: false, canRetry: false },
        running: false
      })
      expect(room.log.map(message => message.text)).toEqual(['Stored reply'])
      expect(loaded.runtime.groupChatContinuityReady(room)).toBe(false)
      expect(loaded.rounds.sendToGroupChat('Classic', members, 'Must not drive')).toBeNull()
      expect(loaded.calls.some(call => call.method === 'session.create' || call.method === 'prompt.submit')).toBe(false)
      loaded.runtime.stopHostedRoomRuntime()
    }
  )

  it.each([false, true])('does not declare absence after a state failure, persistent=%s', async persistent => {
    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        return { ...nonpersistent, persistent_process: persistent }
      }

      if (method === 'groups.list') {
        return { rooms: [stored], next_offset: null }
      }

      if (method === 'groups.state') {
        throw new Error('state unavailable')
      }

      throw new Error(`unexpected ${method}`)
    })

    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    expect(loaded.calls.some(call => call.method === 'groups.state')).toBe(true)
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
    expect(loaded.rounds.sendToGroupChat('Classic', members, 'Must not drive')).toBeNull()
    loaded.runtime.stopHostedRoomRuntime()
  })

  it.each([
    {},
    { rooms: null },
    { rooms: [null] },
    { rooms: [], next_offset: '500' },
    { ok: false, rooms: [] },
    { error: { code: 503 }, rooms: [] },
    { rooms: [{ ...stored, room_id: ` ${stored.room_id} ` }] }
  ])('rejects malformed list inventory %j', async listing => {
    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        return { ...nonpersistent, persistent_process: true }
      }

      if (method === 'groups.list') {
        return listing
      }

      throw new Error(`unexpected ${method}`)
    })

    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it.each([
    null,
    {},
    { room: {} },
    { room: { ...stored, room_id: 'wrong' } },
    { room: { ...stored, authority_gateway_id: '' } },
    { room: { ...stored, authority_epoch: 0 } },
    { room: { ...stored, members: null } }
  ])('rejects malformed state instead of trusting a list projection %j', async state => {
    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        return nonpersistent
      }

      if (method === 'groups.list') {
        return { rooms: [stored] }
      }

      if (method === 'groups.state') {
        return state
      }

      throw new Error(`unexpected ${method}`)
    })

    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    expect(loaded.calls.some(call => call.method === 'groups.state')).toBe(true)
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
    expect(loaded.calls.some(call => call.method === 'groups.log')).toBe(false)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it.each([null, {}, { driver: true }, { driver: false }, new Error('connection lost')])(
    'never treats an invalid/transient probe as absence %j',
    async capability => {
      const loaded = await load(() => {
        if (capability instanceof Error) {
          throw capability
        }

        return capability
      })

      await loaded.runtime.startHostedRoomRuntime(loaded.storage)
      expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
      expect(loaded.calls.some(call => call.method === 'groups.list')).toBe(false)
      loaded.runtime.stopHostedRoomRuntime()
    }
  )

  it('preserves genuine old-gateway classic fallback without converting cached hosted ownership', async () => {
    const loaded = await load(() => {
      throw Object.assign(new Error('method not found'), { code: -32601 })
    })

    loaded.chat.$groupChats.set({
      Classic: projected(),
      Hosted: projected({ roomId: 'hosted-old', hosted: 'install:home', hostedConnectionId: 'local', hostedEpoch: 1 })
    })
    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(true)
    expect(loaded.chat.$groupChats.get().Hosted).toMatchObject({
      hosted: 'install:home',
      hostedStatus: { state: 'unsupported' }
    })
    expect(loaded.rounds.sendToGroupChat('Hosted', members, 'Do not drive')).toBeNull()
    loaded.runtime.stopHostedRoomRuntime()
  })

  it.each(['transient', 'malformed', 'partial', 'new-authority'])(
    'revokes a completed absence observation after %s',
    async failure => {
      let healthy = true

      const loaded = await load(method => {
        if (method === 'groups.capabilities') {
          if (!healthy && failure === 'transient') {
            throw new Error('offline')
          }

          if (!healthy && failure === 'malformed') {
            return {}
          }

          return {
            ...nonpersistent,
            authority_gateway_id: healthy || failure !== 'new-authority' ? 'install:home' : 'install:new'
          }
        }

        if (method === 'groups.list') {
          if (!healthy) {
            throw new Error('listing interrupted')
          }

          return { rooms: [], next_offset: null }
        }

        throw new Error(`unexpected ${method}`)
      })

      await loaded.runtime.startHostedRoomRuntime(loaded.storage)
      expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(true)
      healthy = false
      await loaded.runtime.refreshHostedRooms()
      expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
      loaded.runtime.stopHostedRoomRuntime()
    }
  )

  it.each([0, -1, 5, '1'])('rejects a nonadvancing or inconsistent page cursor %s', async next => {
    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        return nonpersistent
      }

      if (method === 'groups.list') {
        return { rooms: [], next_offset: next }
      }

      throw new Error(`unexpected ${method}`)
    })

    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    expect(loaded.calls.filter(call => call.method === 'groups.list')).toHaveLength(1)
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('waits for the actual last page and reads a hosted room found there', async () => {
    let release!: (page: unknown) => void
    let lastPage!: () => void

    const started = new Promise<void>(resolve => {
      lastPage = resolve
    })

    const pending = new Promise(resolve => {
      release = resolve
    })

    const loaded = await load((method, params) => {
      if (method === 'groups.capabilities') {
        return nonpersistent
      }

      if (method === 'groups.list' && params.offset === 0) {
        return { rooms: [{ ...stored, room_id: 'retired-other', disbanded_at: 2 }], next_offset: 1 }
      }

      if (method === 'groups.list') {
        lastPage()

        return pending
      }

      if (method === 'groups.state') {
        return { room: stored }
      }

      if (method === 'groups.log') {
        return { events: [event()], latest_seq: 1, has_more: false }
      }

      throw new Error(`unexpected ${method}`)
    })

    const start = loaded.runtime.startHostedRoomRuntime(loaded.storage)
    await started
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
    release({ rooms: [stored], next_offset: null })
    await start
    expect(loaded.chat.$groupChats.get().Classic).toMatchObject({
      hosted: 'install:home',
      hostedStatus: { state: 'read-only' }
    })
    expect(loaded.calls.filter(call => call.method === 'groups.list').map(call => call.params.offset)).toEqual([0, 1])
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('bounds incomplete inventory to four500-room pages without proving absence', async () => {
    const loaded = await load((method, params) => {
      if (method === 'groups.capabilities') {
        return nonpersistent
      }

      if (method === 'groups.list') {
        const offset = Number(params.offset)

        return {
          rooms: Array.from({ length: 500 }, (_, index) => ({
            ...stored,
            room_id: `retired-${offset + index}`,
            disbanded_at: 2
          })),
          next_offset: offset + 500
        }
      }

      throw new Error(`unexpected ${method}`)
    })

    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    expect(loaded.calls.filter(call => call.method === 'groups.list').map(call => call.params.offset)).toEqual([
      0, 500, 1000, 1500
    ])
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('retains a listed tombstone against a later classic projection and replays before marking deleted', async () => {
    const deleted = { ...stored, disbanded_at: 2 }

    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        return nonpersistent
      }

      if (method === 'groups.list') {
        return { rooms: [deleted], next_offset: null }
      }

      if (method === 'groups.state') {
        return { room: deleted }
      }

      if (method === 'groups.log') {
        return { events: [event()], latest_seq: 1, has_more: false }
      }

      throw new Error(`unexpected ${method}`)
    })

    loaded.chat.$groupChats.set({})
    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    expect(loaded.chat.$groupChats.get()).toEqual({})
    loaded.chat.$groupChats.set({ Classic: projected() })
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
    await loaded.runtime.refreshHostedRooms()
    expect(loaded.chat.$groupChats.get().Classic).toMatchObject({
      hosted: 'install:home',
      hostedSeq: 1,
      hostedStatus: { state: 'deleted' }
    })
    expect(loaded.chat.$groupChats.get().Classic.log.map(message => message.text)).toEqual(['Stored reply'])
    expect(loaded.rounds.sendToGroupChat('Classic', members, 'Do not resurrect')).toBeNull()
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('does not transfer authority from an unproven state claim', async () => {
    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        return nonpersistent
      }

      if (method === 'groups.list') {
        return { rooms: [stored] }
      }

      if (method === 'groups.state') {
        return { room: { ...stored, authority_epoch: 2, authority_gateway_id: 'install:other' } }
      }

      throw new Error(`unexpected ${method}`)
    })

    loaded.chat.$groupChats.set({
      Classic: projected({ hosted: 'install:home', hostedEpoch: 1, hostedConnectionId: 'local' })
    })
    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    expect(loaded.chat.$groupChats.get().Classic).toMatchObject({ hosted: 'install:home', hostedEpoch: 1, log: [] })
    expect(loaded.calls.some(call => call.method === 'groups.log')).toBe(false)
    expect(loaded.rounds.sendToGroupChat('Classic', members, 'Do not drive')).toBeNull()
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('keeps a passive copy read-only even if its storage host can run other rooms', async () => {
    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        return { ...nonpersistent, persistent_process: true, authority_gateway_id: 'install:copy' }
      }

      if (method === 'groups.list') {
        return { rooms: [stored] }
      }

      if (method === 'groups.state') {
        return { room: stored }
      }

      if (method === 'groups.log') {
        return { events: [event()], latest_seq: 1, has_more: false }
      }

      throw new Error(`unexpected ${method}`)
    })

    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    expect(loaded.chat.$groupChats.get().Classic).toMatchObject({
      hosted: 'install:home',
      hostedStatus: { state: 'read-only' }
    })
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('does not promise continuity or accept new work after a read-only state failure', async () => {
    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        return nonpersistent
      }

      if (method === 'groups.list') {
        return { rooms: [stored] }
      }

      if (method === 'groups.state') {
        throw new Error('state unavailable')
      }

      throw new Error(`unexpected ${method}`)
    })

    loaded.chat.$groupChats.set({
      Classic: projected({ hosted: 'install:home', hostedEpoch: 1, hostedConnectionId: 'local' })
    })
    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    expect(loaded.chat.$groupChats.get().Classic).toMatchObject({
      hosted: 'install:home',
      hostedStatus: { state: 'read-only' }
    })
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('invalidates prior absence when a later paginated list fails', async () => {
    let interrupted = false

    const loaded = await load((method, params) => {
      if (method === 'groups.capabilities') {
        return nonpersistent
      }

      if (method === 'groups.list') {
        if (!interrupted) {
          return { rooms: [], next_offset: null }
        }

        if (params.offset === 0) {
          return { rooms: [stored], next_offset: 1 }
        }

        throw new Error('second page unavailable')
      }

      throw new Error(`unexpected ${method}`)
    })

    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(true)
    interrupted = true
    await loaded.runtime.refreshHostedRooms()
    expect(loaded.calls.filter(call => call.method === 'groups.list').map(call => call.params.offset)).toEqual([
      0, 0, 1
    ])
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it.each(['duplicate', 'oversized'])('does not accept a %s list page as absence', async failure => {
    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        return nonpersistent
      }

      if (method === 'groups.list') {
        return {
          rooms:
            failure === 'duplicate'
              ? [stored, stored]
              : Array.from({ length: 501 }, (_, index) => ({ ...stored, room_id: `room-${index}` })),
          next_offset: null
        }
      }

      throw new Error(`unexpected ${method}`)
    })

    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    expect(loaded.runtime.groupChatContinuityReady(loaded.chat.$groupChats.get().Classic)).toBe(false)
    expect(loaded.calls.filter(call => call.method === 'groups.list')).toHaveLength(1)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('requires successful state verification to leave a cached read-only state', async () => {
    let persistent = false
    let stateFails = false

    const loaded = await load(method => {
      if (method === 'groups.capabilities') {
        return { ...nonpersistent, persistent_process: persistent }
      }

      if (method === 'groups.list') {
        return { rooms: [stored] }
      }

      if (method === 'groups.state') {
        if (stateFails) {
          throw new Error('state unavailable')
        }

        return { room: stored }
      }

      if (method === 'groups.log') {
        return { events: [event()], latest_seq: 1, has_more: false }
      }

      throw new Error(`unexpected ${method}`)
    })

    await loaded.runtime.startHostedRoomRuntime(loaded.storage)
    expect(loaded.chat.$groupChats.get().Classic.hostedStatus?.state).toBe('read-only')
    persistent = true
    stateFails = true
    await loaded.runtime.refreshHostedRooms()
    expect(loaded.chat.$groupChats.get().Classic.hostedStatus?.state).toBe('read-only')
    stateFails = false
    await loaded.runtime.refreshHostedRooms()
    expect(loaded.chat.$groupChats.get().Classic.hostedStatus?.state).toBe('ready')
    expect(loaded.chat.$groupChats.get().Classic.hosted).toBe('install:home')
    loaded.runtime.stopHostedRoomRuntime()
  })
})
