import { renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'
import { $activeGatewayProfile } from '@/store/profile'

import { useStatusbarItems } from './use-statusbar-items'

// The approval-mode chip reads `approvals.mode` for the route the window is
// on. A profile served only by a remote gateway (`marina` on Homelab) has no
// backend on this machine: resolving its bare name through the profile pool
// made main try to start a LOCAL backend for it, refused with
// `Profile "marina" no longer exists`, twice on every switch.

const sockets = vi.hoisted(() => [] as Array<{ requests: Array<[string, unknown]>; wsUrl: string }>)

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesModule>()),
  HermesGateway: class {
    connectionState = 'closed'
    record = { requests: [] as Array<[string, unknown]>, wsUrl: '' }
    constructor() {
      sockets.push(this.record)
    }
    connect = async (wsUrl: string): Promise<void> => {
      this.record.wsUrl = wsUrl
      this.connectionState = 'open'
    }
    request = async (method: string, params: unknown) => {
      this.record.requests.push([method, params])

      return { value: 'off' }
    }
    close = (): void => {
      this.connectionState = 'closed'
    }
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
  }
}))

const { closeSecondaryGateways, configureGatewayRegistry, ensureGatewayForAgent, setPrimaryGateway } =
  await import('@/store/gateway')

const homelabMarina = {
  authMode: 'token',
  baseUrl: 'http://homelab.invalid',
  connectionId: 'homelab',
  mode: 'remote',
  profile: 'marina',
  token: 'fake-test-token',
  wsUrl: 'ws://homelab.invalid/api/ws?token=fake-test-token'
}

function renderStatusbar() {
  return renderHook(
    () =>
      useStatusbarItems({
        agentsOpen: false,
        changeSessionCwd: async () => {},
        chatOpen: true,
        commandCenterOpen: false,
        extraLeftItems: [],
        extraRightItems: [],
        freshDraftReady: false,
        gatewayState: 'open',
        inferenceStatus: null,
        openAgents: () => {},
        openCommandCenterSection: () => {},
        requestGateway: async () => undefined as never,
        statusSnapshot: null,
        toggleCommandCenter: () => {}
      }),
    { wrapper }
  )
}

const wrapper = ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter>

afterEach(() => {
  closeSecondaryGateways()
  sockets.length = 0
  $activeGatewayProfile.set('default')
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('approval-mode chip on a remote-only profile', () => {
  it('reads the mode from the active gateway and never asks main for a local backend of that name', async () => {
    const getConnection = vi.fn(async (profile?: string) => {
      if (profile) {
        throw new Error(`Profile "${profile}" no longer exists.`)
      }

      return { mode: 'local', port: 4242, token: 'primary-token' }
    })

    const getConnectionFor = vi.fn(async () => homelabMarina)

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { getConnection, getConnectionFor }

    configureGatewayRegistry({ onActiveRouteChanged: profile => $activeGatewayProfile.set(profile), onEvent: vi.fn() })
    setPrimaryGateway({ connectionState: 'open', request: vi.fn() } as never, 'default')
    await expect(ensureGatewayForAgent('homelab', 'marina')).resolves.toBe(true)
    expect($activeGatewayProfile.get()).toBe('marina')

    renderStatusbar()

    const homelab = sockets.find(socket => socket.wsUrl === homelabMarina.wsUrl)

    await waitFor(() =>
      expect(homelab?.requests).toContainEqual(['config.get', { key: 'approvals.mode', profile: 'marina' }])
    )
    expect(getConnection).not.toHaveBeenCalledWith('marina')
    expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'homelab', profile: 'marina' })
  })

  it('re-reads the mode when the window moves between two gateways that both serve `default`', async () => {
    const boxDefault = {
      ...homelabMarina,
      baseUrl: 'http://box.invalid',
      connectionId: 'box',
      profile: 'default',
      wsUrl: 'ws://box.invalid/api/ws'
    }

    const primaryRequest = vi.fn(async () => ({ value: 'manual' }))

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(async () => ({ mode: 'local', port: 4242, token: 'primary-token' })),
      getConnectionFor: vi.fn(async () => boxDefault)
    }

    configureGatewayRegistry({ onActiveRouteChanged: profile => $activeGatewayProfile.set(profile), onEvent: vi.fn() })
    setPrimaryGateway({ connectionState: 'open', request: primaryRequest } as never, 'default')
    const { rerender } = renderStatusbar()
    await waitFor(() =>
      expect(primaryRequest).toHaveBeenCalledWith('config.get', { key: 'approvals.mode', profile: 'default' })
    )

    await expect(ensureGatewayForAgent('box', 'default')).resolves.toBe(true)
    rerender()

    const box = sockets.find(socket => socket.wsUrl === boxDefault.wsUrl)

    await waitFor(() =>
      expect(box?.requests).toContainEqual(['config.get', { key: 'approvals.mode', profile: 'default' }])
    )
    expect(primaryRequest).toHaveBeenCalledTimes(1)
  })
})
