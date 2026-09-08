import { describe, expect, it, vi } from 'vitest'

import {
  desktopRoomDescriptors,
  desktopRoomIdentity,
  refreshDesktopRoomPresence,
  runDesktopRoomCommandCycle
} from './desktop-room-command-client'
import { classicAuthorityHash } from './group-desktop-authority'
import type { GroupChat, ProfileRoute } from './types'

const route = (connectionId: string): ProfileRoute => ({
  connectionId,
  mode: 'remote',
  profile: 'default',
  targetProfile: 'default'
})

const classic = (overrides: Partial<GroupChat> = {}): GroupChat => ({
  desktopAuthorityHash: classicAuthorityHash('authority:test'),
  desktopAuthorityToken: 'authority:test',
  log: [],
  roomId: 'room-1',
  watermarks: {},
  ...overrides
})

describe('classic Group Chat command client', () => {
  it('caps a wake at 64 immediate single-command claims', async () => {
    const limits: unknown[] = []

    const result = await runDesktopRoomCommandCycle({
      consumerId: 'desktop:test',
      rooms: { Classic: classic() },
      routes: [route('current')],
      execute: async () => ({}),
      request: async (_route, method, params) => {
        if (method !== 'groups.desktop.claim') {
          return {}
        }

        limits.push(params.limit)

        return {
          commands: [
            {
              action: 'send',
              room_id: 'room-1',
              lease_token: 'lease:1',
              command_id: `command:${limits.length}`,
              payload: {}
            }
          ]
        }
      }
    })

    expect(result).toHaveLength(64)
    expect(limits).toEqual(Array(64).fill(1))
  })

  it.each(['send', 'stop'])(
    'leaves wrong-lane work reclaimable in the %s pump and visits healthy routes',
    async action => {
      const executed: string[] = []
      const completed: string[] = []
      const claimed = new Set<string>()
      await runDesktopRoomCommandCycle({
        actions: [action],
        consumerId: 'desktop:test',
        rooms: { Classic: classic() },
        routes: [route('wrong-lane'), route('healthy')],
        execute: async command => {
          executed.push(String(command.command_id))
        },
        request: async (route, method, params) => {
          if (method === 'groups.desktop.complete') {
            completed.push(String(params.command_id))
          }

          if (method !== 'groups.desktop.claim' || claimed.has(route.connectionId)) {
            return { commands: [] }
          }

          claimed.add(route.connectionId)

          return {
            commands: [
              {
                room_id: 'room-1',
                command_id: route.connectionId,
                lease_token: 'lease:1',
                payload: {},
                action: route.connectionId === 'healthy' ? action : action === 'send' ? 'stop' : 'send'
              }
            ]
          }
        }
      })
      expect(executed).toEqual(['healthy'])
      expect(completed).toEqual(['healthy'])
    }
  )

  it('abandons a lease at its local deadline when renewal never answers', async () => {
    vi.useFakeTimers()
    const methods: string[] = []

    try {
      const cycle = runDesktopRoomCommandCycle({
        consumerId: 'desktop:test',
        rooms: { Classic: classic() },
        routes: [route('current')],
        execute: async (_command, _rooms, context) =>
          new Promise((_resolve, reject) => {
            context.signal!.addEventListener('abort', () => reject(new Error('cancelled')))
          }),
        request: async (_route, method) => {
          methods.push(method)

          if (method === 'groups.desktop.renew') {
            return new Promise(() => undefined)
          }

          return {
            commands: [
              { action: 'send', room_id: 'room-1', command_id: 'command:1', lease_token: 'lease:1', payload: {} }
            ]
          }
        }
      })

      await vi.advanceTimersByTimeAsync(45_000)
      expect(await cycle).toEqual([expect.objectContaining({ leaseLost: true, retryable: true })])
      expect(methods).not.toContain('groups.desktop.complete')
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      vi.useRealTimers()
    }
  })

  it('does not execute an overlong in-flight claim response', async () => {
    vi.useFakeTimers()
    const execute = vi.fn(async () => ({}))

    try {
      const cycle = runDesktopRoomCommandCycle({
        consumerId: 'desktop:test',
        rooms: { Classic: classic() },
        routes: [route('current')],
        execute,
        request: async () => {
          await new Promise(resolve => setTimeout(resolve, 50_000))

          return {
            commands: [
              { action: 'send', room_id: 'room-1', command_id: 'command:1', lease_token: 'lease:1', payload: {} }
            ]
          }
        }
      })

      await vi.advanceTimersByTimeAsync(50_000)
      expect(await cycle).toEqual([])
      expect(execute).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })

  it('advertises only classic rooms with local authority tokens', () => {
    const rooms = {
      Classic: classic(),
      Legacy: classic({ roomId: null }),
      Hosted: classic({ hosted: 'gateway-a' }),
      Deleted: classic({ tombstone: true }),
      DuplicateA: classic({ roomId: 'duplicate' }),
      DuplicateB: classic({ roomId: 'duplicate' })
    }

    expect(desktopRoomIdentity('Legacy', rooms.Legacy)).toBe('name:Legacy')
    expect(desktopRoomDescriptors(rooms)).toEqual([
      {
        authorityHash: classicAuthorityHash('authority:test'),
        authorityToken: 'authority:test',
        name: 'Classic',
        roomId: 'room-1'
      },
      {
        authorityHash: classicAuthorityHash('authority:test'),
        authorityToken: 'authority:test',
        name: 'Legacy',
        roomId: 'name:Legacy'
      }
    ])
  })

  it('claims, executes, and completes once per gateway', async () => {
    const calls: Array<{ connectionId: string; method: string; params: Record<string, unknown> }> = []

    const request = vi.fn(async (target: ProfileRoute, method: string, params: Record<string, unknown>) => {
      calls.push({
        connectionId: target.connectionId,
        method,
        params
      })

      if (target.connectionId === 'old') {
        throw new Error('method not found')
      }

      return method === 'groups.desktop.claim'
        ? {
            commands: [
              {
                action: 'send',
                command_id: 'messaging:1',
                lease_token: 'lease:one',
                payload: { message: 'hello' },
                room_id: 'room-1'
              }
            ]
          }
        : {}
    })

    const outcomes = await runDesktopRoomCommandCycle({
      consumerId: 'desktop:test',
      execute: async command => ({
        thread_id: `thread:${command.command_id}`
      }),
      request,
      rooms: {
        Classic: classic()
      },
      routes: [route('old'), route('current'), route('current')]
    })

    expect(outcomes).toEqual([
      {
        commandId: 'messaging:1',
        connectionId: 'current',
        success: true
      }
    ])
    expect(calls.filter(call => call.method === 'groups.desktop.claim').map(call => call.connectionId)).toEqual([
      'old',
      'current',
      'current'
    ])
    expect(calls.find(call => call.method === 'groups.desktop.complete')?.params).toMatchObject({
      result: {
        thread_id: 'thread:messaging:1'
      },
      success: true
    })
  })

  it('reads attachments through the gateway that issued the claim', async () => {
    const calls: string[] = []

    await runDesktopRoomCommandCycle({
      consumerId: 'desktop:test',
      execute: async (_command, _rooms, context) => {
        await context.request('groups.attachment.read', {
          attachment_id: 'att_1'
        })

        return {
          settled: true
        }
      },
      request: async (target, method) => {
        calls.push(`${target.connectionId}:${method}`)

        return method === 'groups.desktop.claim'
          ? {
              commands: [
                {
                  action: 'send',
                  command_id: 'messaging:file',
                  lease_token: 'lease:one',
                  payload: { message: 'file' },
                  room_id: 'room-1'
                }
              ]
            }
          : {}
      },
      rooms: {
        Classic: classic()
      },
      routes: [route('gateway-b')]
    })

    expect(calls).toContain('gateway-b:groups.attachment.read')
  })

  it('leaves retryable work unacknowledged', async () => {
    const methods: string[] = []

    const outcomes = await runDesktopRoomCommandCycle({
      consumerId: 'desktop:test',
      execute: async () => {
        throw Object.assign(new Error('member offline'), {
          retryable: true
        })
      },
      request: async (_target, method) => {
        methods.push(method)

        return method === 'groups.desktop.claim'
          ? {
              commands: [
                {
                  action: 'send',
                  command_id: 'messaging:later',
                  lease_token: 'lease:later',
                  payload: { message: 'later' },
                  room_id: 'room-1'
                }
              ]
            }
          : {}
      },
      rooms: {
        Classic: classic()
      },
      routes: [route('current')]
    })

    expect(methods).toEqual(['groups.desktop.claim'])
    expect(outcomes).toEqual([
      {
        commandId: 'messaging:later',
        connectionId: 'current',
        retryable: true,
        success: false
      }
    ])
  })

  it('bounds large room claims', async () => {
    const claimSizes: number[] = []

    const rooms = Object.fromEntries(
      Array.from({ length: 260 }, (_, index) => [
        `Room ${index}`,
        classic({
          desktopAuthorityHash: classicAuthorityHash(`authority:${index}`),
          desktopAuthorityToken: `authority:${index}`,
          roomId: `room-${index}`
        })
      ])
    )

    await runDesktopRoomCommandCycle({
      consumerId: 'desktop:test',
      execute: async () => ({}),
      request: async (_target, method, params) => {
        if (method === 'groups.desktop.claim') {
          claimSizes.push((params.room_authorities as unknown[]).length)
        }

        return {
          commands: []
        }
      },
      rooms,
      routes: [route('current')]
    })

    expect(claimSizes).toEqual([128, 128, 4])
  })

  it('keeps the unscoped local gateway compatibility path', async () => {
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []

    await runDesktopRoomCommandCycle({
      consumerId: 'desktop:test',
      execute: async () => ({}),
      request: async (_target, method, params) => {
        calls.push({ method, params })

        return {
          commands: []
        }
      },
      rooms: {
        Local: classic({
          roomId: 'room-local'
        })
      },
      routes: [route('')]
    })

    expect(calls[0]).toMatchObject({
      method: 'groups.desktop.claim',
      params: {
        room_authorities: [
          {
            authority_token: 'authority:test',
            room_id: 'room-local'
          }
        ]
      }
    })
  })

  it('renews long work and abandons it without completion when the lease is lost', async () => {
    vi.useFakeTimers()
    const methods: string[] = []

    try {
      const cycle = runDesktopRoomCommandCycle({
        consumerId: 'desktop:test',
        execute: async (_command, _rooms, context) =>
          new Promise((_resolve, reject) =>
            context.signal?.addEventListener('abort', () => reject(new Error('aborted')))
          ),
        request: async (_target, method) => {
          methods.push(method)

          if (method === 'groups.desktop.claim') {
            return {
              commands: [
                {
                  action: 'send',
                  command_id: 'messaging:lease',
                  lease_token: 'lease:one',
                  payload: { message: 'work' },
                  room_id: 'room-1'
                }
              ]
            }
          }

          if (method === 'groups.desktop.renew') {
            throw new Error('lease lost')
          }

          return {}
        },
        rooms: { Classic: classic() },
        routes: [route('current')]
      })

      await vi.advanceTimersByTimeAsync(15_000)
      const outcomes = await cycle
      expect(outcomes).toEqual([
        {
          commandId: 'messaging:lease',
          connectionId: 'current',
          leaseLost: true,
          retryable: true,
          success: false
        }
      ])
      expect(methods).toEqual(['groups.desktop.claim', 'groups.desktop.renew'])
    } finally {
      vi.useRealTimers()
    }
  })

  it('retries the idempotent completion once after a transient ACK failure', async () => {
    let completions = 0

    const outcomes = await runDesktopRoomCommandCycle({
      consumerId: 'desktop:test',
      execute: async () => ({ thread_id: 'thread-1' }),
      request: async (_target, method) => {
        if (method === 'groups.desktop.claim') {
          return {
            commands: [
              {
                action: 'send',
                command_id: 'messaging:ack',
                lease_token: 'lease:one',
                payload: { message: 'work' },
                room_id: 'room-1'
              }
            ]
          }
        }

        if (method === 'groups.desktop.complete' && completions++ === 0) {
          throw new Error('connection reset')
        }

        return {}
      },
      rooms: { Classic: classic() },
      routes: [route('current')]
    })

    expect(completions).toBe(2)
    expect(outcomes[0].success).toBe(true)
  })

  it('uses presence without claiming commands and isolates a failed gateway', async () => {
    const methods: string[] = []
    await refreshDesktopRoomPresence({
      consumerId: 'desktop:test',
      request: async (target, method) => {
        methods.push(`${target.connectionId}:${method}`)

        if (target.connectionId === 'old') {
          throw new Error('method missing')
        }

        return { room_ids: ['room-1'] }
      },
      rooms: { Classic: classic() },
      routes: [route('old'), route('current')]
    })
    expect(methods).toEqual(['old:groups.desktop.presence', 'current:groups.desktop.presence'])
  })

  it('fails an unsupported mailbox action without executing it or blocking the next command', async () => {
    const execute = vi.fn(async () => ({ thread_id: 'thread-1' }))
    const completions: Record<string, unknown>[] = []
    let claimed = 0

    const outcomes = await runDesktopRoomCommandCycle({
      consumerId: 'desktop:test',
      execute,
      request: async (_target, method, params) => {
        if (method === 'groups.desktop.claim' && claimed < 2) {
          const offset = claimed
          claimed += Number(params.limit)

          return {
            commands: [
              {
                action: 'disband',
                command_id: 'messaging:bad',
                lease_token: 'lease:bad',
                payload: {},
                room_id: 'room-1'
              },
              {
                action: 'send',
                command_id: 'messaging:good',
                lease_token: 'lease:good',
                payload: { message: 'continue' },
                room_id: 'room-1'
              }
            ].slice(offset, claimed)
          }
        }

        if (method === 'groups.desktop.complete') {
          completions.push(params)
        }

        return { commands: [] }
      },
      rooms: { Classic: classic() },
      routes: [route('current')]
    })

    expect(execute).toHaveBeenCalledTimes(1)
    expect(completions.map(value => value.success)).toEqual([false, true])
    expect(outcomes.map(value => value.success)).toEqual([false, true])
  })

  it('review repro: isolates a null command and still executes the healthy command', async () => {
    const execute = vi.fn(async () => ({ thread_id: 'thread-1' }))
    let claimed = false

    await expect(
      runDesktopRoomCommandCycle({
        consumerId: 'desktop:test',
        execute,
        request: async (_target, method) => {
          if (method === 'groups.desktop.claim' && !claimed) {
            claimed = true

            return {
              commands: [
                null,
                false,
                [],
                17,
                'invalid',
                {
                  action: 'send',
                  command_id: 'messaging:good',
                  lease_token: 'lease:good',
                  payload: { message: 'continue' },
                  room_id: 'room-1'
                }
              ]
            }
          }

          return { commands: [] }
        },
        rooms: { Classic: classic() },
        routes: [route('current')]
      })
    ).resolves.toEqual([expect.objectContaining({ commandId: 'messaging:good', success: true })])
    expect(execute).toHaveBeenCalledTimes(1)
  })

  it('claims the next command only when it can start, after a turn longer than the lease TTL', async () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    let release!: () => void

    const blocked = new Promise<void>(resolve => {
      release = resolve
    })

    const started = new Map<string, number>()
    const leased = new Map<string, number>()

    const queue = [
      { action: 'send', command_id: 'messaging:slow', lease_token: 'lease:slow', payload: {}, room_id: 'room-1' },
      { action: 'send', command_id: 'messaging:next', lease_token: 'lease:next', payload: {}, room_id: 'room-2' }
    ]

    try {
      const cycle = runDesktopRoomCommandCycle({
        consumerId: 'desktop:test',
        execute: async command => {
          started.set(String(command.command_id), Date.now())

          if (command.command_id === 'messaging:slow') {
            await blocked
          }

          return {}
        },
        request: async (_target, method, params) => {
          if (method === 'groups.desktop.claim') {
            const commands = queue.splice(0, Number(params.limit))
            commands.forEach(command => leased.set(command.command_id, Date.now()))

            return { commands }
          }

          return { commands: [] }
        },
        rooms: { One: classic(), Two: classic({ roomId: 'room-2' }) },
        routes: [route('current')]
      })

      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(50_000)
      release()
      await cycle
      expect(started.size).toBe(2)
      expect(started.get('messaging:next')! - leased.get('messaging:next')!).toBeLessThan(45_000)
      expect(leased.get('messaging:next')).toBe(50_000)
    } finally {
      vi.useRealTimers()
    }
  })

  it('review repro: never executes an action excluded from this pump', async () => {
    const execute = vi.fn(async () => ({}))
    const completed = vi.fn()
    let claimed = false

    await runDesktopRoomCommandCycle({
      actions: ['send'],
      consumerId: 'desktop:test',
      execute,
      request: async (_target, method) => {
        if (method === 'groups.desktop.complete') {
          completed()
        }

        if (method === 'groups.desktop.claim' && !claimed) {
          claimed = true

          return {
            commands: [
              {
                action: 'stop',
                command_id: 'messaging:wrong-lane',
                lease_token: 'lease:wrong',
                payload: { target_message_id: 'message-1' },
                room_id: 'room-1'
              }
            ]
          }
        }

        return { commands: [] }
      },
      rooms: { Classic: classic() },
      routes: [route('current')]
    })

    expect(execute).not.toHaveBeenCalled()
    expect(completed).not.toHaveBeenCalled()
  })
})
