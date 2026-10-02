import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestLocalMode } from '@/api/client'

// The host tab lists installed plugins on mount; only an `install` action counts as installing.
const { requestGateway, scopedGateway, targetProfilesRead } = vi.hoisted(() => ({
  scopedGateway: vi.fn(),
  targetProfilesRead: vi.fn(),
  requestGateway: vi.fn(async (_method: string, _params?: Record<string, unknown>): Promise<unknown> => ({
    plugins: []
  }))
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: (...args: unknown[]) => scopedGateway(...args)
}))
vi.mock('@/api/profiles', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getProfiles: (...args: unknown[]) => targetProfilesRead(...args)
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getProfiles: async () => ({ profiles: [] })
}))

import { queryClient } from '@/lib/query-client'
import { $notifications } from '@/store/notifications'
import {
  $pluginInstallRequest,
  closePluginInstallRequest,
  openPluginInstallRequest
} from '@/store/plugin-install-request'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'
import { $settingsScopeProfile } from '@/store/settings-scope'

import { PluginActions } from '../capabilities/plugins/plugins-tab'

import { PluginInstallModal } from './plugin-install-modal'

const probePluginRepo = vi.fn()
const installDesktopPlugin = vi.fn()

function publishConnection(connection: ReturnType<typeof $connection.get>) {
  $connection.set(connection)
  setApiRequestConnection(connection?.connectionId ?? null)
  setApiRequestLocalMode(connection?.mode === 'local')
}

function LocationProbe() {
  return (
    <output data-testid="install-location">
      {useLocation().pathname}
      {useLocation().search}
    </output>
  )
}

const renderFlow = () =>
  render(
    <MemoryRouter initialEntries={['/capabilities?tab=plugins']}>
      <LocationProbe />
      <QueryClientProvider client={queryClient}>
        <PluginActions profile={null} />
        <PluginInstallModal />
      </QueryClientProvider>
    </MemoryRouter>
  )

beforeEach(() => {
  publishConnection({ mode: 'local', connectionId: 'local', baseUrl: 'http://127.0.0.1' } as NonNullable<
    ReturnType<typeof $connection.get>
  >)
  vi.clearAllMocks()
  $notifications.set([])
  scopedGateway.mockReset().mockResolvedValue({ ok: true, plugin_name: 'remote-plugin', plugins: [] })
  targetProfilesRead.mockReset().mockResolvedValue({ profiles: [{ name: 'remote-only' }] })
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
})

describe('Install from Git entry flow', () => {
  it.each(['Cancel', 'Close'] as const)(
    'releases replacement B controls and %s after pending A completes without publishing A results',
    async close => {
      const repoA = 'https://github.com/example/pending-a'
      const repoB = 'https://github.com/example/replacement-b'
      let finishA!: (value: unknown) => void
      probePluginRepo.mockResolvedValue({ ok: true, agent: true, desktop: false, warnings: [] })
      scopedGateway.mockImplementation(async (_connection, _profile, _method, params) => {
        if (params.action === 'install') {
          return new Promise(resolve => {
            finishA = resolve
          })
        }

        return { plugins: [] }
      })
      renderFlow()
      act(() => openPluginInstallRequest({ repo: repoA, profile: { connectionId: 'pinned-A', profile: 'default' } }))
      await screen.findByText('This package includes')
      fireEvent.click(screen.getByRole('button', { name: 'Install' }))
      await waitFor(() =>
        expect(scopedGateway).toHaveBeenCalledWith(
          'pinned-A',
          'default',
          'plugins.manage',
          expect.objectContaining({ action: 'install', identifier: repoA }),
          120000,
          undefined,
          { spawnPriority: 'foreground' }
        )
      )
      act(() => openPluginInstallRequest({ repo: repoB, profile: { connectionId: 'pinned-B', profile: 'default' } }))
      await screen.findByText('This package includes')
      expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Install' }).disabled).toBe(false)
      expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Cancel' }).disabled).toBe(false)
      await act(async () => {
        finishA({ ok: true, plugin_name: 'old-A', missing_env: ['SYNTHETIC_A_KEY'], warnings: ['old-A-warning'] })
      })
      expect($pluginInstallRequest.get()?.repo).toBe(repoB)
      expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Install' }).disabled).toBe(false)
      expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Cancel' }).disabled).toBe(false)
      expect($notifications.get()).toEqual([])
      expect(installDesktopPlugin).not.toHaveBeenCalled()
      fireEvent.click(screen.getByRole('button', { name: close }))
      await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
    }
  )

  it('keeps replacement B busy when A finishes and publishes only the B install result', async () => {
    const repoA = 'https://github.com/example/pending-a'
    const repoB = 'https://github.com/example/replacement-b'
    let finishA!: (value: unknown) => void
    let finishB!: (value: unknown) => void
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, desktop: false, warnings: [] })
    scopedGateway.mockImplementation(async (connection, _profile, _method, params) => {
      if (params.action === 'install') {
        return new Promise(resolve => {
          if (connection === 'pinned-A') {
            finishA = resolve
          } else {
            finishB = resolve
          }
        })
      }

      return { plugins: [] }
    })
    renderFlow()
    act(() => openPluginInstallRequest({ repo: repoA, profile: { connectionId: 'pinned-A', profile: 'default' } }))
    await screen.findByText('This package includes')
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    await waitFor(() => expect(finishA).toBeTypeOf('function'))
    act(() => openPluginInstallRequest({ repo: repoB, profile: { connectionId: 'pinned-B', profile: 'default' } }))
    await screen.findByText('This package includes')
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Install' }).disabled).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    await waitFor(() =>
      expect(scopedGateway).toHaveBeenCalledWith(
        'pinned-B',
        'default',
        'plugins.manage',
        expect.objectContaining({ action: 'install', identifier: repoB }),
        120000,
        undefined,
        { spawnPriority: 'foreground' }
      )
    )
    await act(async () => {
      finishA({ ok: false, error: 'old-A-error' })
    })
    expect($pluginInstallRequest.get()?.repo).toBe(repoB)
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Cancel' }).disabled).toBe(true)
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    expect($pluginInstallRequest.get()?.repo).toBe(repoB)
    expect(screen.queryByText('old-A-error')).toBeNull()
    expect($notifications.get()).toEqual([])
    await act(async () => {
      finishB({ ok: false, error: 'current-B-error' })
    })
    expect(await screen.findByText('current-B-error')).toBeTruthy()
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Install' }).disabled).toBe(false)
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Cancel' }).disabled).toBe(false)
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
  })

  it('keeps the same request busy when only its initial profile changes', async () => {
    let finish!: (value: unknown) => void
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, desktop: false, warnings: [] })
    requestGateway.mockImplementation(async (_method, params) => {
      if (params?.action === 'install') {
        return new Promise(resolve => {
          finish = resolve
        })
      }

      return { plugins: [] }
    })
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/same-request' }))
    await screen.findByText('This package includes')
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    await waitFor(() => expect(finish).toBeTypeOf('function'))
    const request = $pluginInstallRequest.get()
    act(() => $activeGatewayProfile.set('research'))
    await waitFor(() => expect(probePluginRepo).toHaveBeenCalledTimes(2))
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Cancel' }).disabled).toBe(true)
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    expect($pluginInstallRequest.get()).toBe(request)
    expect(requestGateway.mock.calls.filter(([, params]) => params?.action === 'install')).toHaveLength(1)
    await act(async () => {
      finish({ ok: false, error: 'same-request-error' })
    })
    expect(await screen.findByText('same-request-error')).toBeTruthy()
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Cancel' }).disabled).toBe(false)
  })

  it('does not redirect a pinned remote missing credential into the active local Keys page', async () => {
    publishConnection({ mode: 'local', connectionId: 'local' } as NonNullable<ReturnType<typeof $connection.get>>)
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, desktop: false, warnings: [] })
    scopedGateway.mockResolvedValue({
      ok: true,
      plugin_name: 'remote-plugin',
      missing_env: ['DEMO_API_KEY'],
      plugins: []
    })
    renderFlow()
    act(() =>
      openPluginInstallRequest({
        repo: 'https://github.com/example/plugin',
        profile: { connectionId: 'pinned-remote', profile: 'default' }
      })
    )
    await screen.findByText('This package includes')
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    await waitFor(() => expect($notifications.get().some(item => item.action)).toBe(true))
    act(() =>
      $notifications
        .get()
        .find(item => item.action)!
        .action!.onClick()
    )
    expect(screen.getByTestId('install-location').textContent).not.toContain('tab=keys')
    expect($notifications.get().at(-1)).toMatchObject({ kind: 'warning' })
  })

  it('keeps current-backend Keys navigation on the installed target profile', async () => {
    publishConnection({ mode: 'local', connectionId: 'local' } as NonNullable<ReturnType<typeof $connection.get>>)
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, desktop: false, warnings: [] })
    requestGateway.mockResolvedValue({ ok: true, plugin_name: 'plugin', missing_env: ['DEMO_API_KEY'], plugins: [] })
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/plugin', profile: 'research' }))
    await screen.findByText('This package includes')
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    await waitFor(() => expect($notifications.get().some(item => item.action)).toBe(true))
    act(() =>
      $notifications
        .get()
        .find(item => item.action)!
        .action!.onClick()
    )
    expect(screen.getByTestId('install-location').textContent).toContain('tab=keys&key=DEMO_API_KEY')
    expect($settingsScopeProfile.get()).toBe('research')
  })
  it('keeps remote install, roster, and Desktop-half topology pinned while the active backend changes', async () => {
    publishConnection({ mode: 'local', connectionId: 'local' } as NonNullable<ReturnType<typeof $connection.get>>)
    installDesktopPlugin.mockResolvedValue({ ok: true, pluginName: 'remote-plugin' })
    const reconcileDesktopPlugins = vi.fn(async () => [])
    vi.stubGlobal('hermesDesktop', { probePluginRepo, installDesktopPlugin, reconcileDesktopPlugins })
    renderFlow()
    const profile = { connectionId: 'pinned-remote', profile: 'default' }
    act(() =>
      openPluginInstallRequest({ catalogName: 'remote-plugin', repo: 'https://github.com/example/plugin', profile })
    )
    await screen.findByText('This package includes')
    await waitFor(() => expect(targetProfilesRead).toHaveBeenCalledWith(profile))
    fireEvent.click(screen.getByRole('combobox', { name: 'Install for profile' }))
    expect(screen.queryByRole('option', { name: 'Research Bot' })).toBeNull()
    fireEvent.click(await screen.findByRole('option', { name: 'remote-only' }))
    act(() => {
      $activeGatewayProfile.set('other-active')
      publishConnection({ mode: 'remote', connectionId: 'active-other' } as NonNullable<
        ReturnType<typeof $connection.get>
      >)
    })
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    await waitFor(() =>
      expect(scopedGateway).toHaveBeenCalledWith(
        'pinned-remote',
        'remote-only',
        'plugins.manage',
        expect.objectContaining({ action: 'install', catalog_name: 'remote-plugin', profile: 'remote-only' }),
        120000,
        undefined,
        { spawnPriority: 'foreground' }
      )
    )
    expect(requestGateway).not.toHaveBeenCalled()
    await waitFor(() => expect(installDesktopPlugin).toHaveBeenCalled())
    await waitFor(() =>
      expect(scopedGateway).toHaveBeenCalledWith(
        'pinned-remote',
        'remote-only',
        'plugins.manage',
        expect.objectContaining({ action: 'list', profile: 'remote-only' }),
        undefined,
        undefined,
        { spawnPriority: 'foreground' }
      )
    )
    expect(reconcileDesktopPlugins).not.toHaveBeenCalled()
    expect(probePluginRepo).toHaveBeenCalledTimes(1)
  })
  it.each(['local', 'remote'] as const)(
    'opens repository entry and reviews without installing in %s mode',
    async mode => {
      publishConnection({ mode } as NonNullable<ReturnType<typeof $connection.get>>)
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
    publishConnection({ mode, baseUrl: 'https://gateway.example/hermes' } as NonNullable<
      ReturnType<typeof $connection.get>
    >)
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
    publishConnection({ mode: 'remote', baseUrl: 'https://gateway.example/hermes' } as NonNullable<
      ReturnType<typeof $connection.get>
    >)
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
