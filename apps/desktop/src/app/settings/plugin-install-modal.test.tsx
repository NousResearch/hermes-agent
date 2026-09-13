import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
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
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getProfiles: async () => ({ profiles: [] })
}))

import { queryClient } from '@/lib/query-client'
import {
  $pluginInstallRequest,
  closePluginInstallRequest,
  openPluginInstallRequest
} from '@/store/plugin-install-request'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { PluginsTab } from '../skills/plugins-tab'

import { PluginInstallModal } from './plugin-install-modal'

const probePluginRepo = vi.fn()
const installDesktopPlugin = vi.fn()
const reconcileDesktopPlugins = vi.fn()

function CurrentLocation() {
  const location = useLocation()

  return <output data-testid="location">{location.pathname + location.search}</output>
}

const renderFlow = () =>
  render(
    <MemoryRouter initialEntries={['/skills?tab=plugins']}>
      <QueryClientProvider client={queryClient}>
        <PluginsTab profile={null} />
        <PluginInstallModal />
        <CurrentLocation />
      </QueryClientProvider>
    </MemoryRouter>
  )

beforeEach(() => {
  vi.clearAllMocks()
  requestGateway.mockResolvedValue({ plugins: [] })
  queryClient.clear()
  closePluginInstallRequest()
  $gatewayState.set('idle')
  $activeGatewayProfile.set('default')
  $connection.set(null)
  probePluginRepo.mockResolvedValue({ ok: true, agent: true, desktop: true, warnings: [] })
  reconcileDesktopPlugins.mockResolvedValue([])
  vi.stubGlobal('hermesDesktop', { probePluginRepo, installDesktopPlugin, reconcileDesktopPlugins })
})
afterEach(() => {
  cleanup()
  closePluginInstallRequest()
  vi.unstubAllGlobals()
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

it('offers setup review after files install without recloning a unified package', async () => {
  requestGateway.mockImplementation(async (_method, params) => {
    if (params?.action === 'list') {
      return { plugins: [] }
    }

    throw Object.assign(new Error('Review setup'), {
      data: { status: 'consent_required', installed: true, plugin_name: 'native-fixture', error: 'Review setup' }
    })
  })
  renderFlow()
  act(() => openPluginInstallRequest({ repo: 'owner/native-fixture', profile: 'work' }))
  const install = (await screen.findByRole('button', { name: 'Install' })) as HTMLButtonElement
  await waitFor(() => expect(install.disabled).toBe(false))
  fireEvent.click(install)
  const review = await screen.findByRole('button', { name: 'Review setup in Plugins' })
  expect(screen.getByText(/Files installed; enablement was not changed/)).toBeTruthy()
  expect(install.disabled).toBe(true)
  expect(reconcileDesktopPlugins).toHaveBeenCalledTimes(1)
  expect(installDesktopPlugin).not.toHaveBeenCalled()
  expect(requestGateway).toHaveBeenCalledWith('plugins.manage', { action: 'list', profile: 'work' })
  fireEvent.click(review)
  await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
  expect(screen.getByTestId('location').textContent).toBe('/skills?tab=plugins&plugin=native-fixture')
  expect(requestGateway.mock.calls.filter(([, params]) => params?.action === 'install')).toHaveLength(1)
})
