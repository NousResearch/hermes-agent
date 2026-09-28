import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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

vi.mock('@/sdk/runtime', () => ({ installPluginSdk: vi.fn(), sdkImportMap: {} }))
vi.mock('@/store/confirm', () => ({ confirm: vi.fn() }))
import { queryClient } from '@/lib/query-client'
import { confirm } from '@/store/confirm'
import {
  $pluginInstallRequest,
  closePluginInstallRequest,
  openPluginInstallRequest
} from '@/store/plugin-install-request'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { PluginActions } from '../capabilities/plugins/plugins-tab'

import { PluginInstallModal } from './plugin-install-modal'

const probePluginRepo = vi.fn()
const installDesktopPlugin = vi.fn()

const renderFlow = () =>
  render(
    <MemoryRouter initialEntries={['/capabilities?tab=plugins']}>
      <QueryClientProvider client={queryClient}>
        <PluginActions profile={null} />
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
  it.each([false, true])('retries Python dependency review only with explicit acceptance=%s', async accepted => {
    $connection.set({ mode: 'remote' } as NonNullable<ReturnType<typeof $connection.get>>)
    vi.mocked(confirm).mockResolvedValue(accepted)
    const installs: Record<string, unknown>[] = []
    requestGateway.mockImplementation(async (_method, params) => {
      if (params?.action !== 'install') {return { plugins: [] }}
      installs.push(params)

      return params.dependency_consent
        ? { ok: true, plugin_name: 'example' }
        : {
            ok: false,
            consent_required: true,
            dependency_consent: 'candidate-A',
            python_dependencies: ['requests>=2,<3'],
            error: 'Review Python dependencies'
          }
    })
    installDesktopPlugin.mockResolvedValue({ ok: true, pluginName: 'example' })
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/plugin' }))
    await screen.findByText('This package includes')
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    await waitFor(() => expect(confirm).toHaveBeenCalledOnce())
    const ask = vi.mocked(confirm).mock.calls[0][0]

    // Consent copy is composed from the scanned list, never the backend's
    // composed error blob: intro + labelled requirements, no empty-list caveat.
    expect(ask.title).toBe('Install Python packages?')
    expect(ask.description).toBe(
      'This plugin installs the Python packages below into the shared Hermes environment.'
    )
    expect(ask.description).not.toContain('build backend')
    expect(ask.codeList).toEqual({ label: 'Python requirements', items: ['requests>=2,<3'] })
    await waitFor(() => expect(installs).toHaveLength(accepted ? 2 : 1))
    expect(installs[0]).not.toHaveProperty('dependency_consent')

    if (accepted) {
      expect(installs[1]).toEqual({ ...installs[0], dependency_consent: 'candidate-A' })
    } else {
      expect(installDesktopPlugin).not.toHaveBeenCalled()
      expect($pluginInstallRequest.get()).not.toBeNull()
    }
  })

  it('shows the empty-requirements caveat only when the scanned list is empty', async () => {
    requestGateway.mockImplementation(async (_method, params) => {
      if (params?.action !== 'install') {return { plugins: [] }}

      return params.dependency_consent
        ? { ok: true, plugin_name: 'example' }
        : {
            ok: false,
            consent_required: true,
            dependency_consent: 'candidate-empty',
            python_dependencies: [],
            error: 'Review Python dependencies'
          }
    })
    vi.mocked(confirm).mockResolvedValue(false)
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/plugin' }))
    await screen.findByText('This package includes')
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    await waitFor(() => expect(confirm).toHaveBeenCalledOnce())

    // Empty list: the build-backend caveat IS the copy, and there is no list.
    const ask = vi.mocked(confirm).mock.calls[0][0]
    expect(ask.description).toBe(
      'This plugin lists no Python requirements, but its build backend may still install some.'
    )
    expect(ask.codeList).toBeUndefined()
  })

  it('keeps the footer on its normal label while the consent answer is pending', async () => {
    // confirm() stays unresolved: the consent dialog is open, waiting.
    let answer!: (ok: boolean) => void
    vi.mocked(confirm).mockReturnValue(new Promise<boolean>(resolve => (answer = resolve)))
    requestGateway.mockImplementation(async (_method, params) => {
      if (params?.action !== 'install') {return { plugins: [] }}

      // The consented retry never settles: the footer must stay on the busy
      // label for as long as the install is actually in flight.
      if (params.dependency_consent) {return new Promise<never>(() => {})}

      return {
        ok: false,
        consent_required: true,
        dependency_consent: 'candidate-A',
        python_dependencies: ['requests>=2,<3'],
        error: 'Review Python dependencies'
      }
    })
    renderFlow()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/plugin' }))
    await screen.findByText('This package includes')
    const install = screen.getByRole('button', { name: 'Install' })
    fireEvent.click(install)
    await waitFor(() => expect(confirm).toHaveBeenCalledOnce())

    // Pending consent: the footer still reads Install, not Installing….
    expect(screen.getByRole('button', { name: 'Install' })).toBe(install)
    expect(screen.queryByRole('button', { name: 'Installing…' })).toBeNull()

    // Once answered, the install proceeds and the busy label returns.
    answer(true)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Installing…' })).toBeTruthy())
  })

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
