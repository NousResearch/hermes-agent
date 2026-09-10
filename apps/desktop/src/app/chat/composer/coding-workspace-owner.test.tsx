import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { HermesGateway } from '@/hermes'
import { $codingWorkspaceDrafts, listCodingWorkspaceProjects } from '@/store/coding-workspaces'
import { activeGateway, closeSecondaryGateways, configureGatewayRegistry, ensureGatewayForProfile, retainGatewayForAgent, setPrimaryGateway, setPrimaryGatewayConnectionId } from '@/store/gateway'
import { $activeGatewayProfile, $newChatConnectionId, $newChatProfile, $newChatRoute, ensureGatewayAgent, ensureGatewayProfile, pinNewChatProfile } from '@/store/profile'

import { useCodingWorkspace } from './use-coding-workspace'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  HermesGateway: class {
    connectionState = 'closed'
    connect = vi.fn(async () => { this.connectionState = 'open' })
    close = vi.fn(() => { this.connectionState = 'closed' })
    request = vi.fn(async () => ({ projects: [] }))
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
  }
}))

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

afterEach(() => {
  cleanup()
  closeSecondaryGateways()
  $codingWorkspaceDrafts.set({})
  $newChatProfile.set(null)
  $newChatConnectionId.set(null)
  $newChatRoute.set(null)
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

it.each(['local', 'remote'] as const)('keeps a named %s legacy profile on its own socket, not local or a same-name registry socket', async mode => {
  const api = vi.fn(async () => ({ desktop: { coding: { show_controls: true } } }))
  const getConnection = vi.fn(async (profile?: string) => ({ mode, profile, port: 5151, token: 'test', connectionId: 'descriptor-not-a-socket-key' }))
  const getConnectionFor = vi.fn(async ({ connectionId, profile }) => ({ mode: 'remote', connectionId, profile, port: 5152, token: 'test' }))
  Object.assign(window, { hermesDesktop: { api, getConnection, getConnectionFor } })
  configureGatewayRegistry({ onEvent: vi.fn(), onActiveRouteChanged: profile => $activeGatewayProfile.set(profile) })
  setPrimaryGateway(new HermesGateway() as never, 'default')
  await ensureGatewayForProfile('default')
  setPrimaryGatewayConnectionId('local')
  pinNewChatProfile('coder')
  await ensureGatewayProfile('coder')
  const legacySocket = activeGateway()!
  expect($newChatConnectionId.get()).toBeNull()
  const { result } = renderHook(() => useCodingWorkspace(null, true), { wrapper })
  await waitFor(() => expect(result.current.visible).toBe(true))
  const owner = result.current.owner!
  expect(owner).toEqual({ connectionId: null, profile: 'coder', draftKey: '__new__' })
  expect(api).toHaveBeenCalledWith({ path: '/api/config', profile: 'coder' })
  const release = await retainGatewayForAgent(owner.connectionId, owner.profile)
  await act(async () => { await ensureGatewayAgent('another-source', 'coder') })
  expect(activeGateway()).not.toBe(legacySocket)
  expect(result.current.owner).toEqual(owner)
  await listCodingWorkspaceProjects(owner)
  expect(legacySocket.request).toHaveBeenCalledWith('projects.list', { profile: 'coder' })
  expect(activeGateway()!.request).not.toHaveBeenCalled()
  release()
})
