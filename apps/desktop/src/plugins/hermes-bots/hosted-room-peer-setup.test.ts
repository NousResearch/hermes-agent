import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { HostedRoomCapability } from './hosted-room-client'
import { classifyHostedRoomCapability } from './hosted-room-client'
import { registerHostedPeerControl } from './hosted-room-peer-setup'
import type { ProfileRoute } from './types'

const { requestHome } = vi.hoisted(() => ({ requestHome: vi.fn() }))
vi.mock('./hosted-room-transport', () => ({ requestHostedConnection: requestHome }))

function setup() {
  const invitation = {
    room_id: 'room-1',
    member_id: 'reviewer',
    authority_gateway_id: 'install:home',
    authority_epoch: 2,
    control_token: 'c'.repeat(43),
    home_url: 'https://home.example.test',
    expires_at: 253402300799,
    room_name: 'Workshop',
    member_count: 2
  }

  requestHome.mockResolvedValue(invitation)
  const requestPeer = vi.fn().mockResolvedValue({ registered: true, room_id: 'room-1' })

  const input = {
    homeRoute: { connectionId: 'home', targetProfile: 'default', profile: 'default', mode: 'remote' } as ProfileRoute,
    homeCapability: {
      reciprocalRoomControl: true,
      authorityId: 'install:home',
      roomLink: { endpoint: 'https://home.example.test' }
    } as HostedRoomCapability,
    peerCapability: { reciprocalRoomControl: true, authorityId: 'install:peer' } as HostedRoomCapability,
    roomId: 'room-1',
    memberId: 'reviewer',
    authorityId: 'install:home',
    authorityEpoch: 2,
    targetProfile: 'reviewer',
    targetAuthority: 'install:peer',
    requestId: 'setup-room-1-reviewer',
    requestPeer
  }

  return { input, invitation, requestPeer }
}

beforeEach(() => vi.clearAllMocks())

describe('peer messaging return path', () => {
  it.each([false, true])('requires safe registration recovery support: %s', supported => {
    const capability = classifyHostedRoomCapability({
      driver: true,
      persistent_process: true,
      authority_gateway_id: 'install:home',
      features: ['reciprocal_room_control', ...(supported ? ['reciprocal_room_control_setup'] : [])]
    })

    expect(capability.reciprocalRoomControl).toBe(supported)
  })
  it('registers only the frozen room/member/profile and requests non-rotating reuse', async () => {
    const { input, invitation, requestPeer } = setup()
    await registerHostedPeerControl(input)
    expect(requestHome).toHaveBeenCalledWith(input.homeRoute, 'groups.control.invite', {
      room_id: 'room-1',
      member_id: 'reviewer',
      caller_install_id: 'install:peer',
      request_id: expect.stringMatching(/^desktop-control:[a-f0-9]{64}$/),
      reuse_existing: true
    })
    expect(requestPeer).toHaveBeenCalledWith('groups.control.register', { ...invitation, profile: 'reviewer' })
    expect(requestPeer.mock.invocationCallOrder[0]).toBeGreaterThan(requestHome.mock.invocationCallOrder[0])
    const firstId = requestHome.mock.calls[0][2].request_id
    await registerHostedPeerControl(input)
    expect(requestHome.mock.calls[1][2].request_id).toBe(firstId)
  })

  it.each(['home', 'peer', 'absent-peer'])('does not call unsupported %s gateways', async mode => {
    const { input, requestPeer } = setup()

    if (mode === 'home') {
      input.homeCapability.reciprocalRoomControl = false
    } else if (mode === 'peer') {
      input.peerCapability.reciprocalRoomControl = false
    } else {
      input.peerCapability = undefined as unknown as HostedRoomCapability
    }

    await registerHostedPeerControl(input)
    expect(requestHome).not.toHaveBeenCalled()
    expect(requestPeer).not.toHaveBeenCalled()
  })

  it.each([
    ['room_id', 'other'],
    ['member_id', 'other'],
    ['authority_gateway_id', 'install:other'],
    ['authority_epoch', 3],
    ['home_url', 'https://other.example.test'],
    ['control_token', 'invalid'],
    ['expires_at', Number.NaN],
    ['expires_at', Number.POSITIVE_INFINITY],
    ['expires_at', 0]
  ])('rejects mismatched or invalid %s without sending credentials to the peer', async (key, value) => {
    const { input, invitation, requestPeer } = setup()
    requestHome.mockResolvedValue({ ...invitation, [key as string]: value })
    await expect(registerHostedPeerControl(input)).rejects.toThrow('Messaging could not connect')
    expect(requestPeer).not.toHaveBeenCalled()
  })

  it.each(['home', 'peer'])('checks the %s installation before issuing control access', async which => {
    const { input, requestPeer } = setup()

    if (which === 'home') {
      input.homeCapability.authorityId = 'install:other'
    } else {
      input.peerCapability.authorityId = 'install:other'
    }

    await expect(registerHostedPeerControl(input)).rejects.toThrow('Messaging could not connect')
    expect(requestHome).not.toHaveBeenCalled()
    expect(requestPeer).not.toHaveBeenCalled()
  })

  it('does not register a link after the calling room lifecycle has changed', async () => {
    const { input, requestPeer } = setup()
    let checks = 0
    await expect(
      registerHostedPeerControl({
        ...input,
        assertCurrent: () => {
          if (++checks === 2) {
            throw new Error('room closed')
          }
        }
      })
    ).rejects.toThrow('room closed')
    expect(requestPeer).not.toHaveBeenCalled()
  })

  it.each([
    { registered: false, room_id: 'room-1' },
    { registered: true, room_id: 'other' }
  ])('requires an exact acknowledged registration', async result => {
    const { input, requestPeer } = setup()
    requestPeer.mockResolvedValue(result)
    await expect(registerHostedPeerControl(input)).rejects.toThrow('Messaging could not connect')
  })
})
