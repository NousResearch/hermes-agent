import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The host tab lists installed plugins on mount; only an `install` action counts as installing.
const { requestGateway, gateway } = vi.hoisted(() => ({
  gateway: { request: vi.fn() },
  requestGateway: vi.fn(async (_method: string, _params?: Record<string, unknown>): Promise<unknown> => ({
    plugins: []
  }))
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  activeGateway: () => gateway
}))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getProfiles: async () => ({ profiles: [] })
}))

import { $apiRequestScope } from '@/api/client'
import { queryClient } from '@/lib/query-client'
import { requestPluginCatalogInstallFromDeepLink } from '@/store/plugin-catalog-install'
import {
  $pluginInstallRequest,
  closePluginInstallRequest,
  openPluginInstallRequest
} from '@/store/plugin-install-request'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'
import { $settingsScopeOverride, $settingsScopeProfile } from '@/store/settings-scope'

import { PluginsTab } from '../capabilities/plugins/plugins-tab'

import { PluginInstallModal } from './plugin-install-modal'

import { SettingsPage } from './index'

const probePluginRepo = vi.fn()
const installDesktopPlugin = vi.fn()

function LocationProbe() {
  const location = useLocation()

  return (
    <output data-testid="location">
      {location.pathname}
      {location.search}
    </output>
  )
}

const renderFlow = (realSettings = false) =>
  render(
    <MemoryRouter initialEntries={['/capabilities?tab=plugins']}>
      <QueryClientProvider client={queryClient}>
        {realSettings ? (
          <Routes>
            <Route element={<PluginsTab profile={null} />} path="/capabilities" />
            <Route element={<SettingsPage onClose={() => {}} />} path="/settings" />
          </Routes>
        ) : (
          <PluginsTab profile={null} />
        )}
        <PluginInstallModal />
        <LocationProbe />
      </QueryClientProvider>
    </MemoryRouter>
  )

beforeEach(() => {
  vi.clearAllMocks()
  $apiRequestScope.set({ connectionId: 'a', profile: 'default' })
  Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
  queryClient.clear()
  closePluginInstallRequest()
  $gatewayState.set('idle')
  $activeGatewayProfile.set('default')
  $profiles.set([
    {
      has_env: false,
      is_default: true,
      model: null,
      name: 'default',
      path: '/profiles/default',
      provider: null,
      skill_count: 0
    },
    {
      display_name: 'Research Bot',
      has_env: false,
      is_default: false,
      model: null,
      name: 'research',
      path: '/profiles/research',
      provider: null,
      skill_count: 0
    }
  ])
  probePluginRepo.mockResolvedValue({ ok: true, agent: true, desktop: true, warnings: [] })
  vi.stubGlobal('hermesDesktop', { probePluginRepo, installDesktopPlugin })
})
afterEach(() => {
  cleanup()
  closePluginInstallRequest()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  $settingsScopeOverride.set(null)
})

describe('Memory catalog ownership', () => {
  const memoryRequest = () =>
    openPluginInstallRequest({
      repo: 'https://github.com/example/memory',
      catalogName: 'memory',
      sha: 'a'.repeat(40),
      legacyHint: 'agent',
      enable: true,
      profile: 'default',
      memory: { name: 'memory', owner: { connectionId: 'a', profile: 'default' } }
    })

  it('opens marketplace memory review over the owning Memory settings without selecting a provider', async () => {
    $settingsScopeOverride.set('research')

    const entry = {
      name: 'memory',
      repo: 'https://github.com/example/memory',
      sha: 'a'.repeat(40),
      category: 'memory',
      featured: true
    }

    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify([entry])))

    const api = vi.fn(async ({ path }: { path: string; method?: string }) => {
      if (path.startsWith('/api/model/')) {
        throw new Error('Model API is outside this memory fixture')
      }

      if (path === '/api/memory') {
        return { active: 'builtin', providers: [{ name: 'memory', status: 'needs_config' }], builtin_files: {} }
      }

      if (path === '/api/config/schema') {
        return { fields: {} }
      }

      if (path === '/api/config') {
        return { memory: { provider: 'builtin', memory_enabled: true } }
      }

      return { available: false, logged_in: false }
    })

    vi.stubGlobal('hermesDesktop', { probePluginRepo, installDesktopPlugin, api })
    gateway.request.mockResolvedValue({ ok: true, plugin_name: 'memory' })
    renderFlow(true)
    await act(() => requestPluginCatalogInstallFromDeepLink('memory'))
    expect(await screen.findByText('This package includes')).toBeTruthy()
    // The actual SettingsPage must consume the route, not silently fall back to Model.
    expect(await screen.findByRole('combobox', { name: 'Provider settings', hidden: true })).toBeTruthy()
    expect(screen.getByTestId('location').textContent).toBe('/settings?tab=config:memory&page=persistent')
    expect($settingsScopeProfile.get()).toBe('default')
    expect(screen.getByRole('dialog')).toBeTruthy()
    expect(probePluginRepo).toHaveBeenCalledTimes(1)
    expect(gateway.request).not.toHaveBeenCalled()
    expect(installDesktopPlugin).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    expect(await screen.findByRole('button', { name: 'Back to Memory settings' })).toBeTruthy()
    expect(api).toHaveBeenCalledWith({
      path: '/api/memory',
      connectionId: 'a',
      profile: 'default',
      priority: 'foreground'
    })
    expect(gateway.request).toHaveBeenCalledExactlyOnceWith(
      'plugins.manage',
      expect.objectContaining({ action: 'install', catalog_name: 'memory', profile: 'default', enable: true }),
      120000
    )
    expect(installDesktopPlugin).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Back to Memory settings' }))
    expect(await screen.findByRole('combobox', { name: 'Provider settings' })).toBeTruthy()
    expect(screen.getByTestId('location').textContent).toBe('/settings?tab=config:memory&page=persistent')
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(api.mock.calls.every(([request]) => !request.method || request.method === 'GET')).toBe(true)
  })

  it.each(['completed', 'in flight'])(
    'reviews and installs a replacement request independently of the %s install',
    async timing => {
      let finish!: (value: unknown) => void

      const pending = new Promise(resolve => {
        finish = resolve
      })

      gateway.request.mockReturnValueOnce(pending).mockResolvedValue({ ok: true })

      const api = vi.fn(async () => ({
        active: 'builtin',
        providers: [
          { name: 'memory', status: 'needs_config' },
          { name: 'second-memory', status: 'missing' }
        ],
        builtin_files: {}
      }))

      vi.stubGlobal('hermesDesktop', { probePluginRepo, api })
      renderFlow()
      act(memoryRequest)
      expect(await screen.findByText('This package includes')).toBeTruthy()
      fireEvent.click(screen.getByRole('button', { name: 'Install' }))

      const complete = async () => {
        await act(async () => {
          finish({ ok: true })
          await pending
        })
      }

      if (timing === 'completed') {
        await complete()
        expect(await screen.findByRole('button', { name: 'Back to Memory settings' })).toBeTruthy()
      }

      act(() =>
        openPluginInstallRequest({
          repo: 'https://github.com/example/second-memory',
          catalogName: 'second-memory',
          legacyHint: 'agent',
          memory: { name: 'second-memory', owner: { connectionId: 'a', profile: 'default' } }
        })
      )
      expect(await screen.findByText('This package includes')).toBeTruthy()
      expect(screen.queryByRole('button', { name: 'Back to Memory settings' })).toBeNull()

      if (timing === 'in flight') {
        expect((screen.getByRole('button', { name: 'Installing…' }) as HTMLButtonElement).disabled).toBe(true)
        expect(gateway.request).toHaveBeenCalledTimes(1)
        await complete()
      }

      const install = (await screen.findByRole('button', { name: 'Install' })) as HTMLButtonElement
      expect(install.disabled).toBe(false)
      expect(screen.queryByRole('button', { name: 'Back to Memory settings' })).toBeNull()
      fireEvent.click(install)
      expect(await screen.findByText(/not discovered yet/)).toBeTruthy()
      expect(screen.queryByRole('button', { name: 'Back to Memory settings' })).toBeNull()
      expect(gateway.request).toHaveBeenCalledTimes(2)
      expect(gateway.request).toHaveBeenLastCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'install', catalog_name: 'second-memory', profile: 'default' }),
        120000
      )
    }
  )

  it.each([
    { connectionId: 'b', profile: 'default' },
    { connectionId: 'a', profile: 'research' }
  ])('refuses an owner changed during catalog lookup: %j', async owner => {
    let finish!: (value: Response) => void

    const pending = new Promise<Response>(resolve => {
      finish = resolve
    })

    vi.spyOn(globalThis, 'fetch').mockReturnValue(pending)
    renderFlow()
    const lookup = requestPluginCatalogInstallFromDeepLink('memory')
    act(() => $apiRequestScope.set(owner))
    await act(async () => {
      finish(
        new Response(
          JSON.stringify([
            { name: 'memory', repo: 'https://github.com/example/memory', category: 'memory', featured: true }
          ])
        )
      )
      await lookup
    })
    expect(await screen.findByText('This package includes')).toBeTruthy()
    expect(screen.getByTestId('location').textContent).toBe('/capabilities?tab=plugins')
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    expect(await screen.findByText(/Switch back to the connection and profile/)).toBeTruthy()
    expect(gateway.request).not.toHaveBeenCalled()
    expect(requestGateway.mock.calls.some(([, params]) => params?.action === 'install')).toBe(false)
    expect(installDesktopPlugin).not.toHaveBeenCalled()
  })

  it('keeps an in-flight install and its rediscovery on A after switching to B, without a B handoff', async () => {
    let finish!: (value: unknown) => void

    const pending = new Promise(resolve => {
      finish = resolve
    })

    gateway.request.mockReturnValueOnce(pending)
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, desktop: false, warnings: [] })

    const response = {
      active: 'builtin',
      providers: [{ name: 'memory', description: '', configured: false, status: 'needs_config' }],
      builtin_files: { memory: 0, user: 0 }
    }

    const api = vi.fn(async () => response)
    vi.stubGlobal('hermesDesktop', { probePluginRepo, api })
    renderFlow()
    act(memoryRequest)
    expect(await screen.findByText('This package includes')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    act(() => $apiRequestScope.set({ connectionId: 'b', profile: 'other' }))
    await act(async () => {
      finish({ ok: true })
      await pending
    })
    await waitFor(() => expect(queryClient.getQueryData(['memory-discovery', 'a', 'default'])).toEqual(response))
    expect(api).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'a', profile: 'default' }))
    expect(queryClient.getQueryData(['memory-discovery', 'b', 'other'])).toBeUndefined()
    expect(screen.queryByRole('button', { name: 'Back to Memory settings' })).toBeNull()
    expect(gateway.request).toHaveBeenCalledTimes(1)
  })
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

  it('installs a deep-linked agent plugin into the selected profile', async () => {
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, desktop: false, warnings: [] })
    requestGateway.mockImplementation(async method =>
      method === 'plugins.manage' ? { ok: true, plugin_name: 'plugin', plugins: [] } : { plugins: [] }
    )
    renderFlow()
    act(() => openPluginInstallRequest({ catalogName: 'plugin', repo: 'https://github.com/example/plugin' }))

    const profile = await screen.findByRole('combobox', { name: 'Install for profile' })

    fireEvent.click(profile)
    fireEvent.click(await screen.findByRole('option', { name: 'Research Bot' }))
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'install', catalog_name: 'plugin', profile: 'research' }),
        expect.any(Number)
      )
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
        expect.objectContaining({ action: 'install', ref: sha.toLowerCase() }),
        expect.any(Number)
      )
    )
  })
})

describe('Unified package desktop half on a local backend', () => {
  const alreadyExists = "Plugin 'pkg' already exists. Use force reinstall to replace it."
  const reconcileDesktopPlugins = vi.fn(async (): Promise<string[]> => [])

  const installHybrid = async (mode: 'local' | 'remote') => {
    $connection.set({ mode } as NonNullable<ReturnType<typeof $connection.get>>)
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, agentName: 'pkg', desktop: true, warnings: [] })
    requestGateway.mockImplementation(async (method, params) =>
      method === 'plugins.manage' && params?.action === 'install'
        ? { ok: false, error: alreadyExists }
        : { plugins: [] }
    )
    installDesktopPlugin.mockResolvedValue({ ok: true, pluginName: 'pkg' })
    vi.stubGlobal('hermesDesktop', { installDesktopPlugin, probePluginRepo, reconcileDesktopPlugins })
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/pkg' }))
    expect(await screen.findByText('This package includes')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'install' }),
        expect.any(Number)
      )
    )
    expect(await screen.findByText(alreadyExists)).toBeTruthy()
  }

  it('never clones the desktop half standalone when the agent install is refused', async () => {
    // A no-Force retry of a package already on disk: the backend refuses the
    // agent half, and the desktop half is still served from that package.
    // Cloning it separately here is what left desktop-plugins/<git-name>/
    // beside the package copy (#100412).
    await installHybrid('local')

    expect(reconcileDesktopPlugins).toHaveBeenCalled()
    expect(installDesktopPlugin).not.toHaveBeenCalled()
  })

  it('still clones the desktop half for a remote backend', async () => {
    // A remote backend's plugins/ folder is unreadable from this machine, so
    // the separate clone remains the only door for its desktop half.
    await installHybrid('remote')

    expect(installDesktopPlugin).toHaveBeenCalledWith({ identifier: 'https://github.com/example/pkg', force: false })
    expect(reconcileDesktopPlugins).not.toHaveBeenCalled()
  })

  it('does not start the desktop half or offer a retry when the agent install outcome is unknown', async () => {
    $connection.set({ mode: 'remote' } as NonNullable<ReturnType<typeof $connection.get>>)
    requestGateway.mockImplementation(async (method, params) => {
      if (method === 'plugins.manage' && params?.action === 'install') {
        throw new Error('request timed out after 120s: plugins.manage')
      }

      return { plugins: [] }
    })
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/pkg' }))
    expect(await screen.findByText('This package includes')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))

    const status = await screen.findByRole('status')
    expect(status.textContent).toContain('may still be installing')
    expect(installDesktopPlugin).not.toHaveBeenCalled()
    expect((screen.getByRole('button', { name: 'Install' }) as HTMLButtonElement).disabled).toBe(true)
  })
})
