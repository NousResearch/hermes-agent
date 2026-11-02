import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { setPrimaryGateway, setPrimaryGatewayConnectionId } from '@/store/gateway'
import { $gatewayProjectLooks } from '@/store/gateway-project-looks'
import { makeSessionInfo } from '@/test/session-info'

import { useGatewayProjectTree } from './use-gateway-project-tree'

// The window is on This device; Studio is another registered gateway with a chat in /code/app.
const studioChat = makeSessionInfo({
  id: 'studio-chat',
  connection_id: 'studio',
  profile: 'research',
  cwd: '/code/app',
  git_repo_root: '/code/app'
})

const getConnectionFor = vi.fn(() => new Promise(() => {}))

beforeEach(() => {
  $gatewayProjectLooks.set({})
  $connectionsRegistry.set({
    version: 2,
    primary: 'local',
    secureTokenStorage: true,
    connections: [
      { id: 'local', label: 'This computer', kind: 'local', tokenSet: false, tokenPreview: null },
      { id: 'studio', label: 'Studio', kind: 'remote', tokenSet: true, tokenPreview: null }
    ]
  } as NonNullable<typeof $connectionsRegistry.value>)
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = { getConnectionFor }
})

afterEach(() => {
  vi.clearAllMocks()
  setPrimaryGateway(null)
  $connectionsRegistry.set(null)
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

const settle = () => new Promise(resolve => setTimeout(resolve, 0))

describe("another gateway's project colours", () => {
  it('never dial an owner the window has no open socket to', async () => {
    const { result } = renderHook(() => useGatewayProjectTree([], [studioChat]))

    await settle()

    expect(result.current.tree.map(node => node.connectionId)).toEqual(['studio'])
    expect(getConnectionFor).not.toHaveBeenCalled()
  })

  it('are read over a socket already open to the owner', async () => {
    const request = vi.fn(async () => ({
      projects: [{ archived: false, color: '#f43f5e', folders: [], icon: null, id: 'p_app', primary_path: '/code/app' }]
    }))

    const socket = { connectionState: 'open', onState: () => () => {}, request }

    setPrimaryGateway(socket as unknown as HermesGateway, 'research')
    setPrimaryGatewayConnectionId('studio', 'remote')

    const { result } = renderHook(() => useGatewayProjectTree([], [studioChat]))

    await waitFor(() => expect(result.current.tree[0]?.color).toBe('#f43f5e'))
    expect(request).toHaveBeenCalledWith('projects.list', { profile: 'research' })
    expect(getConnectionFor).not.toHaveBeenCalled()
  })
})
