import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
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

import { $pluginDecisions, $pluginRecords, dropPlugin, patchPlugin, publishPlugin } from '@/contrib/plugins-store'
import { queryClient } from '@/lib/query-client'
import { $notifications } from '@/store/notifications'
import {
  $pluginInstallRequest,
  closePluginInstallRequest,
  openPluginInstallRequest
} from '@/store/plugin-install-request'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { PluginsTab } from '../capabilities/plugins/plugins-tab'

import { PluginInstallModal } from './plugin-install-modal'

const probePluginRepo = vi.fn()
const installDesktopPlugin = vi.fn()

const BLENDER = {
  catalogName: 'blender',
  repo: 'https://github.com/NousResearch/hermes-official-plugins#blender',
  sha: 'abec9e822a23089943dd1d476a104bf746f19f86',
  summary: 'Build and render 3D scenes in your Blender.',
  title: 'Blender'
}

const installCalls = () =>
  requestGateway.mock.calls.filter(([method, params]) => method === 'plugins.manage' && params?.action === 'install')

const installButton = () => screen.getByRole('button', { name: /^Install/ }) as HTMLButtonElement

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => (resolve = done))

  return { promise, resolve }
}

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
  Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
  queryClient.clear()
  closePluginInstallRequest()
  $gatewayState.set('idle')
  $connection.set({ mode: 'local' } as NonNullable<ReturnType<typeof $connection.get>>)
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
  requestGateway.mockImplementation(async () => ({ plugins: [] }))
  probePluginRepo.mockResolvedValue({ ok: true, agent: true, agentName: 'blender', desktop: false, warnings: [] })
  vi.stubGlobal('hermesDesktop', { probePluginRepo, installDesktopPlugin })
})
afterEach(() => {
  cleanup()
  closePluginInstallRequest()
  vi.unstubAllGlobals()
})

describe('A reviewed catalog pick', () => {
  it('shows only name, one line and Verified until Advanced reveals source, version and profile', async () => {
    renderFlow()
    act(() => openPluginInstallRequest(BLENDER))

    const dialog = await screen.findByRole('dialog', { name: 'Blender' })
    expect(within(dialog).getByText('Verified')).toBeTruthy()
    expect(within(dialog).getByText(BLENDER.summary)).toBeTruthy()
    expect(within(dialog).getByText('Reviewed by Nous. Installs for default and turns on right away.')).toBeTruthy()
    expect(within(dialog).queryByText(BLENDER.repo)).toBeNull()
    expect(within(dialog).queryByRole('combobox')).toBeNull()
    expect(within(dialog).queryByRole('checkbox')).toBeNull()
    expect(within(dialog).queryByRole('switch')).toBeNull()

    fireEvent.click(within(dialog).getByRole('button', { name: 'Advanced' }))

    expect(within(dialog).getByText(BLENDER.repo)).toBeTruthy()
    expect(within(dialog).getByText('Reviewed pin abec9e82')).toBeTruthy()
    expect(await within(dialog).findByText('Agent plugin · blender')).toBeTruthy()
    expect(within(dialog).getByRole('combobox', { name: 'Install for profile' })).toBeTruthy()
    // Contents are read-only: the package decides its halves, not a checklist.
    expect(within(dialog).queryByRole('checkbox')).toBeNull()
  })

  it('shows one loader, installs once with enable on, then closes', async () => {
    const reply = deferred<unknown>()
    requestGateway.mockImplementation(async (method, params) =>
      method === 'plugins.manage' && params?.action === 'install' ? reply.promise : { plugins: [] }
    )
    renderFlow()
    act(() => openPluginInstallRequest(BLENDER))
    await waitFor(() => expect(probePluginRepo).toHaveBeenCalled())

    fireEvent.click(installButton())

    await waitFor(() => expect(installButton().textContent).toBe('Installing…'))
    expect(installButton().disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Cancel' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(installButton())
    reply.resolve({ ok: true, plugin_name: 'blender', enabled: true })

    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
    expect(installCalls()).toHaveLength(1)
    expect(installCalls()[0]?.[1]).toMatchObject({ catalog_name: 'blender', enable: true, force: false })
  })

  it('waits under the loader for an inspection still running, then installs once', async () => {
    const probe = deferred<unknown>()
    probePluginRepo.mockReturnValue(probe.promise)
    requestGateway.mockImplementation(async (method, params) =>
      method === 'plugins.manage' && params?.action === 'install'
        ? { ok: true, plugin_name: 'blender' }
        : { plugins: [] }
    )
    renderFlow()
    act(() => openPluginInstallRequest(BLENDER))
    await screen.findByRole('dialog', { name: 'Blender' })

    fireEvent.click(installButton())
    await waitFor(() => expect(installButton().textContent).toBe('Installing…'))
    expect(installCalls()).toHaveLength(0)

    probe.resolve({ ok: true, agent: true, agentName: 'blender', desktop: false, warnings: [] })

    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
    expect(installCalls()).toHaveLength(1)
  })

  it('installs into the profile picked under Advanced', async () => {
    requestGateway.mockImplementation(async (method, params) =>
      method === 'plugins.manage' && params?.action === 'install'
        ? { ok: true, plugin_name: 'blender' }
        : { plugins: [] }
    )
    renderFlow()
    act(() => openPluginInstallRequest(BLENDER))
    fireEvent.click(await screen.findByRole('button', { name: 'Advanced' }))
    fireEvent.click(screen.getByRole('combobox', { name: 'Install for profile' }))
    fireEvent.click(await screen.findByRole('option', { name: 'Research Bot' }))

    expect(screen.getByText('Reviewed by Nous. Installs for Research Bot and turns on right away.')).toBeTruthy()
    fireEvent.click(installButton())

    await waitFor(() => expect(installCalls()[0]?.[1]).toMatchObject({ catalog_name: 'blender', profile: 'research' }))
  })

  it('states a failure in one line and retries the install', async () => {
    requestGateway.mockImplementation(async (method, params) =>
      method === 'plugins.manage' && params?.action === 'install'
        ? { ok: false, error: 'Blender was not found on this computer.' }
        : { plugins: [] }
    )
    renderFlow()
    act(() => openPluginInstallRequest(BLENDER))
    await waitFor(() => expect(probePluginRepo).toHaveBeenCalled())
    fireEvent.click(installButton())

    expect((await screen.findByRole('alert')).textContent).toBe('Blender was not found on this computer.')
    expect($pluginInstallRequest.get()).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    await waitFor(() => expect(installCalls()).toHaveLength(2))
  })
})

describe('A raw git URL', () => {
  it.each(['local', 'remote'] as const)(
    'opens repository entry, then the same dialog marked Unverified with its details open (%s)',
    async mode => {
      $connection.set({ mode } as NonNullable<ReturnType<typeof $connection.get>>)
      probePluginRepo.mockResolvedValue({ ok: true, agent: true, agentName: 'plugin', desktop: false, warnings: [] })
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

      const dialog = await screen.findByRole('dialog', { name: 'plugin' })
      expect(within(dialog).getByText('Unverified')).toBeTruthy()
      expect(within(dialog).getByRole('button', { name: 'Advanced' }).getAttribute('aria-expanded')).toBe('true')
      expect(within(dialog).getByText('https://github.com/example/plugin')).toBeTruthy()
      expect(within(dialog).getByText('This package includes')).toBeTruthy()
      expect(installCalls()).toHaveLength(0)
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

  it('installs only once its contents are on screen, and keeps the security warning in view', async () => {
    const probe = deferred<unknown>()
    probePluginRepo.mockReturnValue(probe.promise)
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'http://example.test/plugin' }))
    await screen.findByText('Inspecting repository…')
    expect(installButton().disabled).toBe(true)

    probe.resolve({ ok: true, agent: true, agentName: 'plugin', desktop: false, warnings: [], insecure: true })

    await waitFor(() => expect(installButton().disabled).toBe(false))
    expect(screen.getByText(/insecure or local scheme/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Advanced' }))
    expect(screen.getByText(/insecure or local scheme/)).toBeTruthy()
  })

  it('keeps a legacy deep link to the half it names, read-only and without auto-install', async () => {
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, agentName: 'plugin', desktop: true, warnings: [] })
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/plugin', legacyHint: 'desktop' }))
    expect(await screen.findByText(/^Desktop UI/)).toBeTruthy()
    expect(screen.queryByText(/^Agent plugin/)).toBeNull()
    expect(screen.queryByRole('checkbox')).toBeNull()
    expect(screen.queryByRole('textbox', { name: 'Repository' })).toBeNull()
    expect(probePluginRepo).toHaveBeenCalledTimes(1)
    expect(installCalls()).toHaveLength(0)
    expect(installDesktopPlugin).not.toHaveBeenCalled()
  })

  it('pins a custom install to a full commit SHA and refuses anything shorter', async () => {
    requestGateway.mockImplementation(async method =>
      method === 'plugins.manage' ? { ok: true, plugin_name: 'plugin', plugins: [] } : { plugins: [] }
    )
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/plugin' }))
    const pin = await screen.findByRole('textbox', { name: 'Pin to commit (optional)' })
    await waitFor(() => expect(installButton().disabled).toBe(false))
    fireEvent.change(pin, { target: { value: 'main' } })
    expect(installButton().disabled).toBe(true)
    const sha = 'ABCDEF0123456789abcdef0123456789abcdef01'
    fireEvent.change(pin, { target: { value: ` ${sha} ` } })
    expect(installButton().disabled).toBe(false)
    fireEvent.click(installButton())
    await waitFor(() => expect(installCalls()[0]?.[1]).toMatchObject({ enable: true, ref: sha.toLowerCase() }))
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
    await waitFor(() => expect(installButton().disabled).toBe(false))
    fireEvent.click(installButton())
    await waitFor(() => expect(installCalls()).toHaveLength(1))
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

  it('holds both halves to the commit Electron inspected', async () => {
    const reviewed = '0123456789abcdef0123456789abcdef01234567'
    $connection.set({ mode: 'remote' } as NonNullable<ReturnType<typeof $connection.get>>)
    probePluginRepo.mockResolvedValue({
      ok: true,
      agent: true,
      agentName: 'pkg',
      desktop: true,
      sha: reviewed,
      warnings: []
    })
    requestGateway.mockImplementation(async (method, params) =>
      method === 'plugins.manage' && params?.action === 'install' ? { ok: true, plugin_name: 'pkg' } : { plugins: [] }
    )
    installDesktopPlugin.mockResolvedValue({ ok: true, pluginName: 'pkg' })
    vi.stubGlobal('hermesDesktop', { installDesktopPlugin, probePluginRepo, reconcileDesktopPlugins })
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/pkg' }))
    expect(await screen.findByText('This package includes')).toBeTruthy()
    await waitFor(() => expect(installButton().disabled).toBe(false))
    fireEvent.click(installButton())

    await waitFor(() => expect(installDesktopPlugin).toHaveBeenCalled())
    expect(installCalls()[0]?.[1]).toMatchObject({ action: 'install', inspected_sha: reviewed })
    expect(installDesktopPlugin).toHaveBeenCalledWith({
      identifier: 'https://github.com/example/pkg',
      force: false,
      sha: reviewed
    })
  })

  it('still clones the desktop half for a remote backend', async () => {
    // A remote backend's plugins/ folder is unreadable from this machine, so
    // the separate clone remains the only door for its desktop half.
    await installHybrid('remote')

    expect(installDesktopPlugin).toHaveBeenCalledWith({ identifier: 'https://github.com/example/pkg', force: false })
    expect(reconcileDesktopPlugins).not.toHaveBeenCalled()
  })

  it('turns the desktop half on with the agent half: install means ready for the whole package', async () => {
    // The half lands opt-in (marker => defaultEnabled false); the install turns it on with the agent half.
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, agentName: 'pkg', desktop: true, warnings: [] })
    requestGateway.mockImplementation(async (method, params) =>
      method === 'plugins.manage' && params?.action === 'install'
        ? { ok: true, plugin_name: 'pkg', enabled: true }
        : { plugins: [] }
    )
    reconcileDesktopPlugins.mockImplementation(async () => {
      publishPlugin(
        { id: 'pkg-ui', name: 'Pkg', kind: 'disk', status: 'disabled', packageName: 'pkg' },
        { activate: () => patchPlugin('pkg-ui', { status: 'loaded' }), deactivate: () => undefined }
      )

      return ['/app/desktop-plugins/pkg']
    })
    vi.stubGlobal('hermesDesktop', { installDesktopPlugin, probePluginRepo, reconcileDesktopPlugins })
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/pkg' }))
    expect(await screen.findByText('This package includes')).toBeTruthy()
    await waitFor(() => expect(installButton().disabled).toBe(false))
    fireEvent.click(installButton())

    await waitFor(() => expect($pluginRecords.get()['pkg-ui']?.status).toBe('loaded'))
    expect($pluginDecisions.get()['pkg-ui']).toBe(true)
    dropPlugin('pkg-ui')
    $pluginDecisions.set({})
  })

  it('never reports a desktop half installed when the reconcile only pruned the old copy', async () => {
    // A force reinstall whose new tree dropped its desktop entry: Electron's pass deletes
    // desktop-plugins/pkg and names it among the folders it touched.
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, agentName: 'pkg', desktop: true, warnings: [] })
    requestGateway.mockImplementation(async (method, params) =>
      method === 'plugins.manage' && params?.action === 'install'
        ? { ok: true, plugin_name: 'pkg', enabled: false }
        : { plugins: [] }
    )
    reconcileDesktopPlugins.mockImplementation(async () => ['/app/desktop-plugins/pkg'])
    const desktopPluginsRoot = vi.fn(async () => '/app/desktop-plugins')
    const readDir = vi.fn(async () => ({ entries: [] }))
    vi.stubGlobal('hermesDesktop', {
      desktopPluginsRoot,
      installDesktopPlugin,
      probePluginRepo,
      readDir,
      reconcileDesktopPlugins
    })
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/pkg' }))
    expect(await screen.findByText('This package includes')).toBeTruthy()
    await waitFor(() => expect(installButton().disabled).toBe(false))
    fireEvent.click(installButton())

    expect(await screen.findByText('Desktop plugin install failed')).toBeTruthy()
    expect(readDir).toHaveBeenCalledWith('/app/desktop-plugins')
    expect(installDesktopPlugin).not.toHaveBeenCalled()
  })

  it('does not start the desktop half or offer a retry when the agent install outcome is unknown', async () => {
    $connection.set({ mode: 'remote' } as NonNullable<ReturnType<typeof $connection.get>>)
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, agentName: 'pkg', desktop: true, warnings: [] })
    requestGateway.mockImplementation(async (method, params) => {
      if (method === 'plugins.manage' && params?.action === 'install') {
        throw new Error('request timed out after 120s: plugins.manage')
      }

      return { plugins: [] }
    })
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/pkg' }))
    expect(await screen.findByText('This package includes')).toBeTruthy()
    await waitFor(() => expect(installButton().disabled).toBe(false))
    fireEvent.click(installButton())

    const status = await screen.findByRole('status')
    expect(status.textContent).toContain('may still be installing')
    expect(installDesktopPlugin).not.toHaveBeenCalled()
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
    expect(installButton().disabled).toBe(true)
  })

  it('reports the desktop half as failed when the reconcile refused the package it just installed', async () => {
    // A combined package whose desktop/ folder holds a symlink: the agent half installs, the reconcile
    // refuses to publish the desktop half, and the dialog must say so instead of "installed".
    const reason = 'The desktop plugin contains a symbolic link (asset.txt); it is not installed.'
    const reconcileDesktopPluginsReport = vi.fn(async () => ({ published: [], refused: [{ name: 'pkg', reason }] }))
    $notifications.set([])
    probePluginRepo.mockResolvedValue({ ok: true, agent: true, agentName: 'pkg', desktop: true, warnings: [] })
    requestGateway.mockImplementation(async (method, params) =>
      method === 'plugins.manage' && params?.action === 'install'
        ? { ok: true, plugin_name: 'pkg', enabled: false }
        : { plugins: [] }
    )
    vi.stubGlobal('hermesDesktop', {
      installDesktopPlugin,
      probePluginRepo,
      reconcileDesktopPlugins,
      reconcileDesktopPluginsReport
    })
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/pkg' }))
    expect(await screen.findByText('This package includes')).toBeTruthy()
    await waitFor(() => expect(installButton().disabled).toBe(false))
    fireEvent.click(installButton())

    expect(await screen.findByText(reason)).toBeTruthy()
    const toasts = $notifications.get().map(n => n.message)
    expect(toasts).toContain('Agent plugin pkg installed')
    expect(toasts).not.toContain('Desktop plugin pkg installed')
    expect(installDesktopPlugin).not.toHaveBeenCalled()
  })
})
