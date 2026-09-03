import { beforeEach, describe, expect, it, vi } from 'vitest'

import { pluginSdkMock } from './group-test-utils'

const EXPECTED_GRANT_SHA256 = 'c'.repeat(64)

const mocks = vi.hoisted(() => {
  const capabilities = {
    value: {} as Record<string, unknown>,
    get() {
      return this.value
    },
    set(value: Record<string, unknown>) {
      this.value = value
    }
  }

  return {
    addCleanup: vi.fn(async () => undefined),
    armCleanup: vi.fn(async () => undefined),
    capabilities,
    dispatchCleanup: vi.fn(async () => undefined),
    host: {} as Record<string, unknown>,
    lifecycle: { value: 1 },
    refresh: vi.fn(async () => undefined),
    invalidate: vi.fn(),
    requestForBot: vi.fn(),
    requestHosted: vi.fn(),
    releaseCleanup: vi.fn(async () => undefined)
  }
})

vi.mock('@hermes/plugin-sdk', async () => pluginSdkMock(mocks.host))

vi.mock('./hosted-room-runtime', () => ({
  $hostedRoomCapabilities: mocks.capabilities,
  invalidateHostedRoomPoll: mocks.invalidate,
  hostedRoomLifecycleIsCurrent: (token: number) => token === mocks.lifecycle.value,
  hostedRoomLifecycleToken: () => mocks.lifecycle.value,
  refreshHostedRooms: mocks.refresh,
  requestHostedConnection: mocks.requestHosted
}))

vi.mock('./hosted-room-cleanup', () => ({
  addHostedRoomCleanup: mocks.addCleanup,
  armHostedRoomCleanup: mocks.armCleanup,
  dispatchHostedRoomCleanup: mocks.dispatchCleanup,
  releaseHostedRoomCleanup: mocks.releaseCleanup
}))

vi.mock('./routing', () => ({
  requestForBot: mocks.requestForBot
}))

beforeEach(() => {
  vi.clearAllMocks()
  mocks.lifecycle.value = 1
  mocks.capabilities.value = {
    home: {
      authorityId: 'install:home',
      routeGrantFingerprint: true
    }
  }
  Object.assign(mocks.host, {
    profileRoutes: async () => [
      { connectionId: 'home', mode: 'remote', profile: 'default', targetProfile: 'default' },
      { connectionId: 'peer', mode: 'remote', profile: 'builder', targetProfile: 'builder' }
    ]
  })
})

describe('hosted Group Chat peer reauthorization', () => {
  it('issues a fresh peer grant through the source-qualified gateway', async () => {
    const { $groupChats } = await import('./group-chat')
    const { reconnectHostedGroupChatPeer } = await import('./hosted-room-reauthorization')

    mocks.capabilities.value = {
      ...mocks.capabilities.value,
      decoy: { authorityId: 'install:decoy' },
      peer: { authorityId: 'install:peer' }
    }
    mocks.host.profileRoutes = async () => [
      { connectionId: 'home', mode: 'remote', profile: 'default', targetProfile: 'default' },
      { connectionId: 'decoy', mode: 'remote', profile: 'builder', targetProfile: 'builder' },
      { connectionId: 'peer', mode: 'remote', profile: 'builder', targetProfile: 'builder' }
    ]
    $groupChats.set({
      Release: {
        continuityMode: 'distributed',
        hosted: 'install:home',
        hostedConnectionId: 'home',
        hostedEpoch: 1,
        log: [],
        members: [
          {
            connectionId: 'decoy',
            handle: 'builder',
            name: 'builder',
            route: {
              connectionId: 'decoy',
              mode: 'remote',
              profile: 'builder',
              targetProfile: 'builder'
            },
            sourceScoped: true,
            targetProfile: 'builder'
          },
          {
            connectionId: 'peer',
            handle: 'builder',
            name: 'builder',
            route: {
              connectionId: 'peer',
              mode: 'remote',
              profile: 'builder',
              targetProfile: 'builder'
            },
            sourceScoped: true,
            targetProfile: 'builder'
          }
        ],
        roomId: 'room-1',
        watermarks: {}
      }
    })

    mocks.requestHosted.mockImplementation(async (_route, method) => {
      if (method === 'groups.state') {
        return {
          driver_status: {
            peer_routes: [
              {
                grant_sha256: EXPECTED_GRANT_SHA256,
                member_id: 'member-builder',
                status: 'needs_reauthorization'
              }
            ]
          },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:home',
            members: [
              {
                display_name: 'Builder',
                handle: 'builder',
                member_id: 'member-builder',
                profile: 'builder',
                target: {
                  installation_id: 'install:peer',
                  kind: 'peer',
                  peer_id: 'install:peer'
                }
              }
            ],
            room_id: 'room-1'
          }
        }
      }

      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: 'install:peer',
          methods: ['groups.peer.revoke_exact'],
          driver: true,
          persistent_process: true,
          room_link: {
            enabled: true,
            endpoint: { available: true, url: 'https://peer.example.test:19445' },
            catalog: {
              attachments: true,
              catalog_digest: 'digest:peer',
              installation_id: 'install:peer',
              link_modes: ['direct'],
              persistent_process: true,
              protocol_versions: [2],
              text: true
            }
          }
        }
      }

      if (method === 'groups.peer.register') {
        return { registered: true }
      }

      throw new Error(`unexpected hosted method: ${method}`)
    })
    mocks.requestForBot.mockResolvedValue({
      catalog: {
        attachments: true,
        catalog_digest: 'digest:peer',
        installation_id: 'install:peer',
        link_modes: ['direct'],
        persistent_process: true,
        protocol_versions: [2],
        text: true
      },
      grant: 'private-grant',
      target_profile: 'builder'
    })

    await reconnectHostedGroupChatPeer('Release', 'member-builder')

    expect(mocks.requestForBot).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'peer', targetProfile: 'builder' }),
      'groups.peer.invite',
      expect.objectContaining({
        authority_epoch: 1,
        authority_gateway_id: 'install:home',
        member_id: 'member-builder',
        room_id: 'room-1'
      })
    )
    expect(mocks.requestHosted).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'home' }),
      'groups.peer.register',
      expect.objectContaining({
        grant: 'private-grant',
        member_id: 'member-builder',
        room_id: 'room-1',
        target_profile: 'builder',
        target_url: 'https://peer.example.test:19445/p/builder'
      })
    )
    expect(mocks.refresh).toHaveBeenCalledOnce()
    expect(mocks.invalidate).toHaveBeenCalledWith('room-1')
    expect(mocks.addCleanup).toHaveBeenCalledTimes(2)
    expect(mocks.releaseCleanup).toHaveBeenCalledOnce()
  })

  it('fails closed when duplicate members have no matching cached authority', async () => {
    const { $groupChats } = await import('./group-chat')
    const { reconnectHostedGroupChatPeer } = await import('./hosted-room-reauthorization')

    mocks.host.profileRoutes = async () => [
      { connectionId: 'home', mode: 'remote', profile: 'default', targetProfile: 'default' },
      { connectionId: 'peer-a', mode: 'remote', profile: 'builder', targetProfile: 'builder' },
      { connectionId: 'peer-b', mode: 'remote', profile: 'builder', targetProfile: 'builder' }
    ]
    $groupChats.set({
      Release: {
        continuityMode: 'distributed',
        hosted: 'install:home',
        hostedConnectionId: 'home',
        hostedEpoch: 1,
        log: [],
        members: ['peer-a', 'peer-b'].map(connectionId => ({
          connectionId,
          handle: 'builder',
          name: 'builder',
          route: { connectionId, mode: 'remote' as const, profile: 'builder', targetProfile: 'builder' },
          sourceScoped: true,
          targetProfile: 'builder'
        })),
        roomId: 'room-1',
        watermarks: {}
      }
    })
    mocks.requestHosted.mockImplementation(async (_route, method) => {
      if (method === 'groups.state') {
        return {
          driver_status: {
            peer_routes: [
              {
                grant_sha256: EXPECTED_GRANT_SHA256,
                member_id: 'member-builder',
                status: 'needs_reauthorization'
              }
            ]
          },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:home',
            members: [
              {
                handle: 'builder',
                member_id: 'member-builder',
                profile: 'builder',
                target: { installation_id: 'install:peer-b', kind: 'peer' }
              }
            ]
          }
        }
      }

      throw new Error(`unexpected hosted method: ${method}`)
    })

    await expect(reconnectHostedGroupChatPeer('Release', 'member-builder')).rejects.toThrow(
      'Reconnect the Bot gateway in Sessions'
    )
    expect(mocks.requestForBot).not.toHaveBeenCalled()
    expect(mocks.requestHosted).not.toHaveBeenCalledWith(expect.anything(), 'groups.capabilities', expect.anything())
    expect(mocks.requestHosted).not.toHaveBeenCalledWith(expect.anything(), 'groups.peer.register', expect.anything())
    expect(mocks.addCleanup).not.toHaveBeenCalled()
    expect(mocks.refresh).not.toHaveBeenCalled()
  })

  it('fails closed when a cached authority points at the wrong live gateway', async () => {
    const { $groupChats } = await import('./group-chat')
    const { reconnectHostedGroupChatPeer } = await import('./hosted-room-reauthorization')

    mocks.capabilities.value = {
      ...mocks.capabilities.value,
      'peer-a': { authorityId: 'install:peer-b' },
      'peer-b': { authorityId: 'install:other' }
    }
    mocks.host.profileRoutes = async () => [
      { connectionId: 'home', mode: 'remote', profile: 'default', targetProfile: 'default' },
      { connectionId: 'peer-a', mode: 'remote', profile: 'builder', targetProfile: 'builder' },
      { connectionId: 'peer-b', mode: 'remote', profile: 'builder', targetProfile: 'builder' }
    ]
    $groupChats.set({
      Release: {
        continuityMode: 'distributed',
        hosted: 'install:home',
        hostedConnectionId: 'home',
        hostedEpoch: 1,
        log: [],
        members: ['peer-a', 'peer-b'].map(connectionId => ({
          connectionId,
          handle: 'builder',
          name: 'builder',
          route: { connectionId, mode: 'remote' as const, profile: 'builder', targetProfile: 'builder' },
          sourceScoped: true,
          targetProfile: 'builder'
        })),
        roomId: 'room-1',
        watermarks: {}
      }
    })
    mocks.requestHosted.mockImplementation(async (_route, method) => {
      if (method === 'groups.state') {
        return {
          driver_status: {
            peer_routes: [
              {
                grant_sha256: EXPECTED_GRANT_SHA256,
                member_id: 'member-builder',
                status: 'needs_reauthorization'
              }
            ]
          },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:home',
            members: [
              {
                handle: 'builder',
                member_id: 'member-builder',
                profile: 'builder',
                target: { installation_id: 'install:peer-b', kind: 'peer' }
              }
            ]
          }
        }
      }

      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: 'install:peer-a',
          methods: ['groups.peer.revoke_exact'],
          driver: true,
          persistent_process: true,
          room_link: {
            enabled: true,
            endpoint: { available: true, url: 'https://peer-a.example.test:19445' },
            catalog: {
              attachments: true,
              catalog_digest: 'digest:peer-a',
              installation_id: 'install:peer-a',
              link_modes: ['direct'],
              persistent_process: true,
              protocol_versions: [2],
              text: true
            }
          }
        }
      }

      throw new Error(`unexpected hosted method: ${method}`)
    })

    await expect(reconnectHostedGroupChatPeer('Release', 'member-builder')).rejects.toThrow(
      'cannot reconnect to this Group Chat yet'
    )
    expect(mocks.requestForBot).not.toHaveBeenCalled()
    expect(mocks.requestHosted).not.toHaveBeenCalledWith(expect.anything(), 'groups.peer.register', expect.anything())
    expect(mocks.addCleanup).not.toHaveBeenCalled()
    expect(mocks.refresh).not.toHaveBeenCalled()
  })

  it('revokes the fresh peer grant when home registration fails', async () => {
    const { $groupChats } = await import('./group-chat')
    const { reconnectHostedGroupChatPeer } = await import('./hosted-room-reauthorization')

    $groupChats.set({
      Release: {
        continuityMode: 'distributed',
        hosted: 'install:home',
        hostedConnectionId: 'home',
        hostedEpoch: 1,
        log: [],
        members: [
          {
            connectionId: 'peer',
            handle: 'builder',
            name: 'builder',
            route: { connectionId: 'peer', mode: 'remote', profile: 'builder', targetProfile: 'builder' },
            sourceScoped: true,
            targetProfile: 'builder'
          }
        ],
        roomId: 'room-1',
        watermarks: {}
      }
    })
    let stateCalls = 0
    mocks.requestHosted.mockImplementation(async (_route, method) => {
      if (method === 'groups.state') {
        stateCalls += 1
        return {
          driver_status: {
            peer_routes: [
              {
                grant_sha256: EXPECTED_GRANT_SHA256,
                member_id: 'member-builder',
                status: 'needs_reauthorization'
              }
            ]
          },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:home',
            members: [
              {
                handle: 'builder',
                member_id: 'member-builder',
                profile: 'builder',
                target: { installation_id: 'install:peer', kind: 'peer' }
              }
            ]
          }
        }
      }

      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: 'install:peer',
          methods: ['groups.peer.revoke_exact'],
          driver: true,
          persistent_process: true,
          room_link: {
            enabled: true,
            endpoint: { available: true, url: 'https://peer.example.test:19445' },
            catalog: {
              attachments: true,
              catalog_digest: 'digest:peer',
              installation_id: 'install:peer',
              link_modes: ['direct'],
              persistent_process: true,
              protocol_versions: [2],
              text: true
            }
          }
        }
      }

      if (method === 'groups.peer.register') {
        throw new Error('register failed')
      }

      throw new Error(`unexpected hosted method: ${method}`)
    })
    mocks.requestForBot.mockResolvedValue({
      catalog: {
        catalog_digest: 'digest:peer',
        installation_id: 'install:peer'
      },
      grant: 'private-grant',
      target_profile: 'builder'
    })
    await expect(reconnectHostedGroupChatPeer('Release', 'member-builder')).rejects.toThrow('register failed')
    expect(mocks.armCleanup).toHaveBeenCalledOnce()
    expect(mocks.dispatchCleanup).toHaveBeenCalledOnce()
    expect(mocks.refresh).not.toHaveBeenCalled()
    expect(mocks.invalidate).not.toHaveBeenCalled()
  })

  it('journals the fresh grant before rejecting an invalid invitation', async () => {
    const { $groupChats } = await import('./group-chat')
    const { reconnectHostedGroupChatPeer } = await import('./hosted-room-reauthorization')

    $groupChats.set({
      Release: {
        continuityMode: 'distributed',
        hosted: 'install:home',
        hostedConnectionId: 'home',
        hostedEpoch: 1,
        log: [],
        members: [
          {
            connectionId: 'peer',
            handle: 'builder',
            name: 'builder',
            route: { connectionId: 'peer', mode: 'remote', profile: 'builder', targetProfile: 'builder' },
            sourceScoped: true,
            targetProfile: 'builder'
          }
        ],
        roomId: 'room-1',
        watermarks: {}
      }
    })
    mocks.requestHosted.mockImplementation(async (_route, method) => {
      if (method === 'groups.state') {
        return {
          driver_status: {
            peer_routes: [
              {
                grant_sha256: EXPECTED_GRANT_SHA256,
                member_id: 'member-builder',
                status: 'needs_reauthorization'
              }
            ]
          },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:home',
            members: [
              {
                handle: 'builder',
                member_id: 'member-builder',
                profile: 'builder',
                target: { installation_id: 'install:peer', kind: 'peer' }
              }
            ]
          }
        }
      }

      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: 'install:peer',
          methods: ['groups.peer.revoke_exact'],
          driver: true,
          persistent_process: true,
          room_link: {
            enabled: true,
            endpoint: { available: true, url: 'https://peer.example.test:19445' },
            catalog: {
              attachments: true,
              catalog_digest: 'digest:peer',
              installation_id: 'install:peer',
              link_modes: ['direct'],
              persistent_process: true,
              protocol_versions: [2],
              text: true
            }
          }
        }
      }

      throw new Error(`unexpected hosted method: ${method}`)
    })
    mocks.requestForBot.mockResolvedValue({
      catalog: {
        installation_id: 'install:peer'
      },
      grant: 'private-grant',
      target_profile: 'builder'
    })

    await expect(reconnectHostedGroupChatPeer('Release', 'member-builder')).rejects.toThrow(
      'could not prepare a secure connection'
    )
    expect(mocks.addCleanup).toHaveBeenCalledOnce()
    expect(mocks.addCleanup).toHaveBeenCalledWith(
      expect.objectContaining({
        connectionId: 'peer',
        grant: 'private-grant',
        kind: 'peer-revoke-exact',
        profile: 'builder'
      })
    )
    expect(mocks.armCleanup).toHaveBeenCalledOnce()
    expect(mocks.dispatchCleanup).toHaveBeenCalledOnce()
    expect(mocks.requestHosted).not.toHaveBeenCalledWith(expect.anything(), 'groups.peer.register', expect.anything())
  })

  it('does not report success when a registration reply is lost', async () => {
    const { $groupChats } = await import('./group-chat')
    const { reconnectHostedGroupChatPeer } = await import('./hosted-room-reauthorization')

    $groupChats.set({
      Release: {
        continuityMode: 'distributed',
        hosted: 'install:home',
        hostedConnectionId: 'home',
        hostedEpoch: 1,
        log: [],
        members: [
          {
            connectionId: 'peer',
            handle: 'builder',
            name: 'builder',
            route: { connectionId: 'peer', mode: 'remote', profile: 'builder', targetProfile: 'builder' },
            sourceScoped: true,
            targetProfile: 'builder'
          }
        ],
        roomId: 'room-1',
        watermarks: {}
      }
    })
    let stateCalls = 0
    mocks.requestHosted.mockImplementation(async (_route, method) => {
      if (method === 'groups.state') {
        stateCalls += 1
        return {
          driver_status:
            stateCalls > 1
              ? {
                  peer_routes: [
                    {
                      member_id: 'member-builder',
                      status: 'ready',
                      grant_sha256: '73238410238d13fffbccfb5ba0142555042d7153fd8196fcf6bba1c1ead06c5a'
                    }
                  ]
                }
              : {
                  peer_routes: [
                    {
                      grant_sha256: EXPECTED_GRANT_SHA256,
                      member_id: 'member-builder',
                      status: 'needs_reauthorization'
                    }
                  ]
                },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:home',
            members: [
              {
                handle: 'builder',
                member_id: 'member-builder',
                profile: 'builder',
                target: { installation_id: 'install:peer', kind: 'peer' }
              }
            ]
          }
        }
      }

      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: 'install:peer',
          methods: ['groups.peer.revoke_exact'],
          driver: true,
          persistent_process: true,
          room_link: {
            enabled: true,
            endpoint: { available: true, url: 'https://peer.example.test:19445' },
            catalog: {
              attachments: true,
              catalog_digest: 'digest:peer',
              installation_id: 'install:peer',
              link_modes: ['direct'],
              persistent_process: true,
              protocol_versions: [2],
              text: true
            }
          }
        }
      }

      if (method === 'groups.peer.register') {
        throw new Error('response lost')
      }

      throw new Error(`unexpected hosted method: ${method}`)
    })
    mocks.requestForBot.mockResolvedValue({
      catalog: {
        catalog_digest: 'digest:peer',
        installation_id: 'install:peer'
      },
      grant: 'private-grant',
      target_profile: 'builder'
    })

    await expect(reconnectHostedGroupChatPeer('Release', 'member-builder')).rejects.toThrow('response lost')
    expect(mocks.armCleanup).toHaveBeenCalledOnce()
    expect(mocks.dispatchCleanup).toHaveBeenCalledOnce()
    expect(mocks.invalidate).not.toHaveBeenCalled()
    expect(mocks.refresh).not.toHaveBeenCalled()
  })

  it('coalesces clicks within one lifecycle but lets a restarted runtime retry independently', async () => {
    const { $groupChats } = await import('./group-chat')
    const { reconnectHostedGroupChatPeer } = await import('./hosted-room-reauthorization')

    $groupChats.set({
      Release: {
        continuityMode: 'distributed',
        hosted: 'install:home',
        hostedConnectionId: 'home',
        hostedEpoch: 1,
        log: [],
        members: [
          {
            connectionId: 'peer',
            handle: 'builder',
            name: 'builder',
            route: { connectionId: 'peer', mode: 'remote', profile: 'builder', targetProfile: 'builder' },
            sourceScoped: true,
            targetProfile: 'builder'
          }
        ],
        roomId: 'room-1',
        watermarks: {}
      }
    })
    mocks.requestHosted.mockImplementation(async (_route, method) => {
      if (method === 'groups.state') {
        return {
          driver_status: {
            peer_routes: [
              {
                grant_sha256: EXPECTED_GRANT_SHA256,
                member_id: 'member-builder',
                status: 'needs_reauthorization'
              }
            ]
          },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:home',
            members: [
              {
                handle: 'builder',
                member_id: 'member-builder',
                profile: 'builder',
                target: { installation_id: 'install:peer', kind: 'peer' }
              }
            ]
          }
        }
      }

      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: 'install:peer',
          methods: ['groups.peer.revoke_exact'],
          driver: true,
          persistent_process: true,
          room_link: {
            enabled: true,
            endpoint: { available: true, url: 'https://peer.example.test:19445' },
            catalog: {
              attachments: true,
              catalog_digest: 'digest:peer',
              installation_id: 'install:peer',
              link_modes: ['direct'],
              persistent_process: true,
              protocol_versions: [2],
              text: true
            }
          }
        }
      }

      if (method === 'groups.peer.register') {
        return { registered: true }
      }

      throw new Error(`unexpected hosted method: ${method}`)
    })
    const releaseInvites: Array<(grant: string) => void> = []
    mocks.requestForBot.mockImplementation(
      () =>
        new Promise(resolve => {
          releaseInvites.push(grant =>
            resolve({
              catalog: {
                catalog_digest: 'digest:peer',
                installation_id: 'install:peer'
              },
              grant,
              target_profile: 'builder'
            })
          )
        })
    )

    const first = reconnectHostedGroupChatPeer('Release', 'member-builder')
    const second = reconnectHostedGroupChatPeer('Release', 'member-builder')

    expect(first).toBe(second)
    await vi.waitFor(() => expect(mocks.requestForBot).toHaveBeenCalledOnce())
    mocks.lifecycle.value = 2
    const third = reconnectHostedGroupChatPeer('Release', 'member-builder')

    expect(third).not.toBe(first)
    await vi.waitFor(() => expect(mocks.requestForBot).toHaveBeenCalledTimes(2))
    releaseInvites[0]('old-private-grant')
    releaseInvites[1]('new-private-grant')

    await expect(first).rejects.toThrow('connections changed')
    await expect(second).rejects.toThrow('connections changed')
    await expect(third).resolves.toBeUndefined()
    expect(mocks.armCleanup).toHaveBeenCalled()
    expect(mocks.dispatchCleanup).toHaveBeenCalled()
  })
})
