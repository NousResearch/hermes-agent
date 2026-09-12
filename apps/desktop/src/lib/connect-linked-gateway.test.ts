import { describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry, DesktopRegistryConnection } from '@/global'

import { connectLinkedGateway } from './connect-linked-gateway'

const request = { url: 'https://actual.inc/api/hermes/clusters/c-1', name: 'Studio' }
const local: DesktopRegistryConnection = {
  id: 'local',
  kind: 'local',
  label: 'Studio',
  tokenSet: false,
  tokenPreview: null
}
const remote: DesktopRegistryConnection = {
  id: 'remote',
  kind: 'remote',
  label: 'Studio (2)',
  url: request.url,
  authMode: 'oauth',
  tokenSet: false,
  tokenPreview: null
}

function fixture(connected = true) {
  let registry: DesktopConnectionsRegistry = {
    version: 2,
    primary: 'local',
    secureTokenStorage: true,
    connections: [local]
  }

  const save = vi.fn(async () => {
    registry = { ...registry, connections: [...registry.connections, remote] }

    return { ok: true, connection: remote, registry }
  })

  const bridge = {
    connections: {
      list: vi.fn(async () => registry),
      save,
      remove: vi.fn(),
      setPrimary: vi.fn(),
      test: vi.fn()
    },
    oauthLoginConnectionConfig: vi.fn(async () => ({ connected, ok: true, baseUrl: request.url }))
  }

  return { bridge, registry: () => registry }
}

describe('connect a linked gateway', () => {
  it('signs in, preserves other connections, and reuses the saved target on repeated handoffs', async () => {
    const { bridge, registry } = fixture()
    const activate = vi.fn(async () => {})

    await connectLinkedGateway(request, bridge, activate, 'Sign-in incomplete')
    await connectLinkedGateway(request, bridge, activate, 'Sign-in incomplete')
    expect(bridge.connections.save).toHaveBeenCalledTimes(1)
    expect(bridge.connections.save).toHaveBeenCalledWith({
      kind: 'remote',
      label: 'Studio (2)',
      url: request.url,
      authMode: 'oauth'
    })
    expect(activate).toHaveBeenLastCalledWith(registry(), 'remote')
    expect(registry().connections[0]).toEqual(local)
    expect(bridge.connections.setPrimary).not.toHaveBeenCalled()
  })

  it('does not save or switch after canceled sign-in', async () => {
    const { bridge } = fixture(false)
    const activate = vi.fn()

    await expect(connectLinkedGateway(request, bridge, activate, 'Sign-in incomplete')).rejects.toThrow(
      'Sign-in incomplete'
    )
    expect(bridge.connections.save).not.toHaveBeenCalled()
    expect(activate).not.toHaveBeenCalled()
  })

  it('surfaces an authenticated WebSocket switch failure and keeps the connection for retry', async () => {
    const { bridge } = fixture()
    const activate = vi.fn(async () => {
      throw new Error('WebSocket rejected')
    })

    await expect(connectLinkedGateway(request, bridge, activate, 'Sign-in incomplete')).rejects.toThrow(
      'WebSocket rejected'
    )
    expect(bridge.connections.remove).not.toHaveBeenCalled()
    expect(bridge.connections.setPrimary).not.toHaveBeenCalled()
  })
})
