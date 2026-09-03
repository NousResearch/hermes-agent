import { describe, expect, it, vi } from 'vitest'

import { desktopRoomDescriptors, desktopRoomIdentity, runDesktopRoomCommandCycle } from './desktop-room-command-client'
import type { GroupChat, ProfileRoute } from './types'

const route = (connectionId: string): ProfileRoute => ({
  connectionId,
  mode: 'remote',
  profile: 'default',
  targetProfile: 'default'
})

const classic = (overrides: Partial<GroupChat> = {}): GroupChat => ({
  desktopAuthorityToken: 'authority:test',
  log: [],
  roomId: 'room-1',
  watermarks: {},
  ...overrides
})

describe('classic Group Chat command client', () => {
  it('advertises only classic rooms with local authority tokens', () => {
    const rooms = {
      Classic: classic(),
      Legacy: classic({ roomId: null }),
      Hosted: classic({ hosted: 'gateway-a' }),
      Deleted: classic({ tombstone: true })
    }

    expect(desktopRoomIdentity('Legacy', rooms.Legacy)).toBe('name:Legacy')
    expect(desktopRoomDescriptors(rooms)).toEqual([
      {
        authorityToken: 'authority:test',
        name: 'Classic',
        roomId: 'room-1'
      },
      {
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
                  command_id: 'messaging:later',
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
})
