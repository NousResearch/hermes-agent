import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

const { ambientRequest, sockets } = vi.hoisted(() => ({
  ambientRequest: vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({ plugins: [] }) as unknown),
  sockets: [] as {
    request: ReturnType<
      typeof vi.fn<(method: string, params: Record<string, unknown>, timeoutMs?: number) => Promise<unknown>>
    >
  }[]
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: ambientRequest })
}))
vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getProfiles: async () => ({ profiles: [] }),
  HermesGateway: class {
    connectionState = 'closed'
    request = vi.fn(async (_method: string, params: Record<string, unknown>) => {
      if (params.action === 'presence') {
        return {
          presence: [{ name: 'weather-plugin', reason: presenceReason, app_title: 'Weather App' }]
        }
      }

      if (params.action === 'install') {
        return { ok: true, plugin_name: 'weather-plugin' }
      }

      return { plugins: [] }
    })
    constructor() {
      sockets.push(this)
    }
    async connect() {
      this.connectionState = 'open'
    }
    close() {
      this.connectionState = 'closed'
    }
    onEvent() {
      return () => undefined
    }
    onState() {
      return () => undefined
    }
  }
}))

import { queryClient } from '@/lib/query-client'
import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import * as gateways from '@/store/gateway'
import {
  $pendingPluginHelpDraft,
  $pluginInstallRequest,
  closePluginInstallRequest
} from '@/store/plugin-install-request'
import { $activeGatewayProfile, $newChatRoute, $profiles } from '@/store/profile'
import { $gatewayState, setConnection } from '@/store/session'

import { parseCatalog } from '../capabilities/catalog/catalog-data'
import { PluginsTab } from '../capabilities/plugins/plugins-tab'

import { PluginInstallModal } from './plugin-install-modal'

const entry = {
  name: 'weather-plugin',
  repo: 'https://example.org/weather',
  sha: 'a'.repeat(40),
  subdir: '',
  tier: 'community',
  category: 'general',
  description: 'A weather plugin'
}

let presenceReason: 'available' | 'missing_app' = 'available'
let probeHasDesktop = false
const reconcileDesktopPlugins = vi.fn(async () => ['weather-plugin'])
const installDesktopPlugin = vi.fn(async () => ({ ok: true, pluginName: 'weather-plugin' }))

function descriptor(connectionId: string, profile = 'default'): HermesConnection {
  return {
    connectionId,
    profile,
    registryScoped: true,
    mode: connectionId === 'local' ? 'local' : 'remote',
    authMode: 'token',
    token: 'test',
    wsUrl: `ws://${connectionId}.invalid/ws`,
    baseUrl: `http://${connectionId}.invalid`
  } as HermesConnection
}

beforeEach(() => {
  sockets.length = 0
  presenceReason = 'available'
  probeHasDesktop = false
  reconcileDesktopPlugins.mockClear()
  installDesktopPlugin.mockClear()
  ambientRequest.mockClear()
  $agentPlugins.set([])
  $agentPluginsStatus.set('ready')
  $gatewayState.set('idle')
  $activeGatewayProfile.set('default')
  $profiles.set([{ name: 'default' }, { name: 'workbot' }] as ReturnType<typeof $profiles.get>)
  $pendingPluginHelpDraft.set(null)
  $newChatRoute.set(null)
  closePluginInstallRequest()
  queryClient.clear()
  queryClient.setQueryData(['public-catalog', 'plugins'], parseCatalog('plugins', [entry]))
  Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      probePluginRepo: vi.fn(async () => ({
        ok: true,
        agent: true,
        agentName: 'weather-plugin',
        desktop: probeHasDesktop,
        warnings: []
      })),
      reconcileDesktopPlugins,
      installDesktopPlugin,
      getConnectionFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
        descriptor(connectionId, profile)
      ),
      getGatewayWsUrlFor: vi.fn(async ({ connectionId }: { connectionId: string }) => ({
        ok: true,
        wsUrl: `ws://${connectionId}.invalid/ws`
      }))
    }
  })
  gateways.configureGatewayRegistry({ onEvent: vi.fn(), onActiveRouteChanged: p => $activeGatewayProfile.set(p) })
  gateways.setPrimaryGateway({ connectionState: 'open', request: ambientRequest } as never, 'default')
  gateways.setPrimaryGatewayConnection(descriptor('cloud'))
  setConnection(descriptor('cloud'))
})

afterEach(() => {
  cleanup()
  gateways.closeSecondaryGateways()
  gateways.setPrimaryGateway(null)
  setConnection(null)
  closePluginInstallRequest()
  queryClient.clear()
})

function mount() {
  return render(
    <MemoryRouter initialEntries={['/capabilities?tab=plugins']}>
      <QueryClientProvider client={queryClient}>
        <PluginsTab profile={{ connectionId: 'local', profile: 'workbot' }} />
        <PluginInstallModal />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

async function chooseCatalog() {
  fireEvent.click((await screen.findAllByRole('button', { name: 'weather-plugin' }))[0])
  fireEvent.click(screen.getByRole('button', { name: 'Install weather-plugin' }))
  await screen.findByRole('dialog')
}

it('routes catalog preflight, recheck, install and inventory to the exact local::workbot socket despite a remote primary', async () => {
  mount()
  await chooseCatalog()
  await waitFor(() =>
    expect(sockets.some(s => s.request.mock.calls.some(([, p]) => p.action === 'presence'))).toBe(true)
  )
  fireEvent.click(screen.getByRole('button', { name: /^Install/ }))
  await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
  const secondary = sockets.find(s => s.request.mock.calls.some(([, p]) => p.action === 'install'))
  expect(secondary).toBeTruthy()
  expect(sockets.flatMap(s => s.request.mock.calls).filter(([, p]) => p.action === 'presence')).toHaveLength(2)
  expect(secondary?.request).toHaveBeenCalledWith(
    'plugins.manage',
    expect.objectContaining({ action: 'install', profile: 'workbot', catalog_name: 'weather-plugin' }),
    120_000,
    undefined
  )
  expect(sockets.flatMap(s => s.request.mock.calls)).toContainEqual([
    'plugins.manage',
    { action: 'list', profile: 'workbot' }
  ])
  expect(window.hermesDesktop?.getConnectionFor).toHaveBeenCalledWith(
    expect.objectContaining({ connectionId: 'local', profile: 'workbot' })
  )
  expect(ambientRequest.mock.calls.some(([, p]) => p?.action === 'presence' || p?.action === 'install')).toBe(false)
})

it('reconciles a unified desktop half from its pinned local owner, not the remote primary', async () => {
  probeHasDesktop = true
  mount()
  await chooseCatalog()
  await waitFor(() =>
    expect((screen.getByRole('button', { name: 'Install' }) as HTMLButtonElement).disabled).toBe(false)
  )
  fireEvent.click(screen.getByRole('button', { name: 'Install' }))
  await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
  expect(reconcileDesktopPlugins).toHaveBeenCalled()
  expect(installDesktopPlugin).not.toHaveBeenCalled()
})

it('drafts Ask Hermes on the catalog owner without submitting on the remote primary', async () => {
  presenceReason = 'missing_app'
  mount()
  await chooseCatalog()
  await screen.findByRole('button', { name: 'Ask Hermes for help' })
  expect(sockets.some(s => s.request.mock.calls.some(([, p]) => p.action === 'presence'))).toBe(true)
  fireEvent.click(screen.getByRole('button', { name: 'Ask Hermes for help' }))
  expect($newChatRoute.get()).toMatchObject({ connectionId: 'local', profile: 'workbot' })
  expect($pendingPluginHelpDraft.get()).toContain('Weather App')
  expect(ambientRequest.mock.calls.some(([, p]) => p?.action === 'presence')).toBe(false)
})
