import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The host tab lists installed plugins on mount; only an `install` action counts as installing.
const { requestGateway } = vi.hoisted(() => ({
  requestGateway: vi.fn(async (_method: string, _params?: Record<string, unknown>): Promise<unknown> => ({
    plugins: []
  }))
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: (_connection: unknown, _profile: unknown, method: string, params: Record<string, unknown>) =>
    requestGateway(method, params)
}))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getProfiles: async () => ({ profiles: [] })
}))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

import { setApiRequestConnection } from '@/api/client'
import { queryClient } from '@/lib/query-client'
import {
  $pluginInstallRequest,
  closePluginInstallRequest,
  openPluginInstallRequest
} from '@/store/plugin-install-request'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { PluginsTab } from '../capabilities/plugins/plugins-tab'

import { PluginInstallModal } from './plugin-install-modal'

const probePluginRepo = vi.fn()
const installDesktopPlugin = vi.fn()

const renderFlow = () =>
  render(
    <MemoryRouter initialEntries={['/capabilities?tab=plugins']}>
      <QueryClientProvider client={queryClient}>
        <PluginsTab profile={null} />
        <PluginInstallModal />
      </QueryClientProvider>
    </MemoryRouter>
  )

beforeEach(() => {
  vi.clearAllMocks()
  queryClient.clear()
  closePluginInstallRequest()
  $gatewayState.set('idle')
  $activeGatewayProfile.set('default')
  probePluginRepo.mockResolvedValue({ ok: true, agent: true, desktop: true, warnings: [] })
  vi.stubGlobal('hermesDesktop', { probePluginRepo, installDesktopPlugin })
})
afterEach(() => {
  cleanup()
  closePluginInstallRequest()
  vi.unstubAllGlobals()
  setApiRequestConnection(null)
})

describe('Install from Git entry flow', () => {
  it.each(['local', 'remote'] as const)(
    'opens repository entry and reviews without installing in %s mode',
    async mode => {
      $connection.set({ mode } as NonNullable<ReturnType<typeof $connection.get>>)
      renderFlow()
      fireEvent.click(screen.getByRole('button', { name: 'Install from Git' }))
      const input = await screen.findByRole('textbox', { name: 'Repository' })
      const review = screen.getByRole('button', { name: 'Review repository' })
      expect((review as HTMLButtonElement).disabled).toBe(true)
      fireEvent.change(input, { target: { value: '   ' } })
      fireEvent.submit(input.closest('form')!)
      expect(probePluginRepo).not.toHaveBeenCalled()
      fireEvent.change(input, { target: { value: 'https://github.com/example/plugin' } })
      fireEvent.click(review)
      await waitFor(() =>
        expect(probePluginRepo).toHaveBeenCalledWith({ identifier: 'https://github.com/example/plugin' })
      )
      expect(await screen.findByText('This package includes')).toBeTruthy()
      expect(
        screen.getByText(
          mode === 'remote'
            ? 'Installs into the connected default backend'
            : 'Installs into the default backend (~/.hermes/plugins/)'
        )
      ).toBeTruthy()
      // Local backend: the desktop half is copied out of the installed package
      // (one source of truth). Remote backend: cloned separately, as before.
      expect(
        screen.getByText(
          mode === 'remote'
            ? "Installs into this app's local desktop-plugins folder"
            : 'Loaded into this app from the package above — same for every profile'
        )
      ).toBeTruthy()
      expect(requestGateway).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'install' }))
      expect(installDesktopPlugin).not.toHaveBeenCalled()
      fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
      expect($pluginInstallRequest.get()).toBeNull()
    }
  )

  it('cancels repository entry and starts fresh when reopened', async () => {
    renderFlow()
    fireEvent.click(screen.getByRole('button', { name: 'Install from Git' }))
    fireEvent.change(await screen.findByRole('textbox', { name: 'Repository' }), { target: { value: 'unfinished' } })
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect($pluginInstallRequest.get()).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Install from Git' }))
    expect(((await screen.findByRole('textbox', { name: 'Repository' })) as HTMLInputElement).value).toBe('')
    expect(probePluginRepo).not.toHaveBeenCalled()
    expect(installDesktopPlugin).not.toHaveBeenCalled()
  })

  it('preserves prefilled deep-link inspection and legacy selection without auto-install', async () => {
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/plugin', legacyHint: 'desktop' }))
    expect(await screen.findByText('This package includes')).toBeTruthy()
    expect(screen.queryByRole('textbox', { name: 'Repository' })).toBeNull()
    const boxes = screen.getAllByRole('checkbox')
    expect(boxes.map(box => box.getAttribute('aria-checked'))).toEqual(['false', 'true'])
    expect(probePluginRepo).toHaveBeenCalledTimes(1)
    expect(requestGateway).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'install' }))
    expect(installDesktopPlugin).not.toHaveBeenCalled()
  })

  it('Review keeps the owner the request was opened for, not the foreground profile', async () => {
    $activeGatewayProfile.set('beta')
    renderFlow()
    act(() => openPluginInstallRequest({ repo: '', profile: { connectionId: null, profile: 'alpha' } }))
    fireEvent.change(await screen.findByRole('textbox', { name: 'Repository' }), {
      target: { value: 'example/plugin' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Review repository' }))
    expect($pluginInstallRequest.get()).toMatchObject({
      repo: 'example/plugin',
      target: { connectionId: null, profile: 'alpha' }
    })
  })

  it('a finished catalog install opens Capabilities on the profile it installed into', async () => {
    $activeGatewayProfile.set('beta')
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, desktop: false, warnings: [] })
    requestGateway.mockImplementation(async method =>
      method === 'plugins.manage' ? { ok: true, plugin_name: 'fixture' } : { plugins: [] }
    )
    render(
      <MemoryRouter initialEntries={['/']}>
        <QueryClientProvider client={queryClient}>
          <LocationProbe />
          <PluginInstallModal />
        </QueryClientProvider>
      </MemoryRouter>
    )
    act(() =>
      openPluginInstallRequest({
        repo: 'fixture/plugin',
        catalogName: 'fixture',
        profile: { connectionId: null, profile: 'alpha' }
      })
    )
    const install = (await screen.findByRole('button', { name: 'Install' })) as HTMLButtonElement
    await waitFor(() => expect(install.disabled).toBe(false))
    fireEvent.click(install)
    await waitFor(() => expect(screen.getByTestId('location').textContent).toBe('/capabilities?tab=plugins'))
    expect(screen.getByTestId('state').textContent).toBe(
      JSON.stringify({ capabilityScope: { connectionId: null, profile: 'alpha' } })
    )
  })

  it('pins a custom install to a full commit SHA and refuses anything shorter', async () => {
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, desktop: false, warnings: [] })
    requestGateway.mockImplementation(async method =>
      method === 'plugins.manage' ? { ok: true, plugin_name: 'plugin', plugins: [] } : { plugins: [] }
    )
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/plugin' }))
    const pin = await screen.findByRole('textbox', { name: 'Pin to commit (optional)' })
    const install = screen.getByRole('button', { name: 'Install' }) as HTMLButtonElement
    fireEvent.change(pin, { target: { value: 'main' } })
    expect(install.disabled).toBe(true)
    const sha = 'ABCDEF0123456789abcdef0123456789abcdef01'
    fireEvent.change(pin, { target: { value: ` ${sha} ` } })
    expect(install.disabled).toBe(false)
    fireEvent.click(install)
    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'install', ref: sha.toLowerCase() })
      )
    )
  })
})

function LocationProbe() {
  const location = useLocation()
  const navigate = useNavigate()

  return (
    <>
      <button onClick={() => navigate('/elsewhere')}>Leave</button>
      <output data-testid="location">{location.pathname + location.search}</output>
      <output data-testid="state">{JSON.stringify(location.state)}</output>
    </>
  )
}

describe('Install started from Memory settings', () => {
  const api = vi.fn()

  beforeEach(() => {
    $activeGatewayProfile.set('beta')
    $connection.set({ connectionId: 'source-a', mode: 'remote' } as never)
    setApiRequestConnection('source-a')
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, desktop: false, warnings: [] })
    api.mockResolvedValue({
      active: 'builtin',
      providers: [{ name: 'fixture', description: '', status: 'needs_config' }]
    })
    vi.stubGlobal('hermesDesktop', { probePluginRepo, api })
    requestGateway.mockImplementation(async (_method, params) =>
      params?.action === 'install' ? { ok: true, plugin_name: 'fixture' } : { plugins: [] }
    )
    openPluginInstallRequest({ repo: 'fixture/plugin', origin: { kind: 'memory', providerId: 'fixture' } })
    render(
      <MemoryRouter initialEntries={['/settings?tab=config:memory']}>
        <QueryClientProvider client={queryClient}>
          <LocationProbe />
          <PluginInstallModal />
        </QueryClientProvider>
      </MemoryRouter>
    )
  })

  async function install() {
    const button = await screen.findByRole('button', { name: 'Install' })
    await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(false))
    fireEvent.click(button)
  }

  const installed = () => waitFor(() => expect($pluginInstallRequest.get()).toBeNull())

  it('returns to Memory settings for the installed provider; the read goes to the owner captured when the request opened', async () => {
    await install()
    setApiRequestConnection('source-b')
    await installed()
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/api/memory', profile: 'beta', connectionId: 'source-a' })
    )
    expect(screen.getByTestId('location').textContent).toBe('/settings?tab=config:memory&provider=fixture')
  })

  it('a completion that lands after the view moved on does not navigate', async () => {
    await install()
    // The open dialog hides its siblings from the accessibility tree.
    fireEvent.click(screen.getByText('Leave'))
    await installed()
    expect(screen.getByTestId('location').textContent).toBe('/elsewhere')
  })
})
