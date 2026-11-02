import { JsonRpcGatewayError } from '@hermes/shared'
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The browser-hosted app: no Electron preload, so no `window.hermesDesktop.probePluginRepo` and no
// `installDesktopPlugin`. Inspection and install both ride the gateway.
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
import { $notifications } from '@/store/notifications'
import {
  $pluginInstallRequest,
  closePluginInstallRequest,
  openPluginInstallRequest
} from '@/store/plugin-install-request'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { PluginInstallModal } from './plugin-install-modal'

const DELEGATION = {
  catalogName: 'advanced-delegation',
  repo: 'https://github.com/NousResearch/hermes-official-plugins#advanced-delegation',
  sha: 'e404b6816b20237a82f6fbe2749b5a576cf75dba',
  summary: 'Two extra delegation modes for delegate_task.',
  title: 'Advanced Delegation',
  tier: 'official'
}

const UNAVAILABLE = 'Plugin inspection is unavailable in this environment.'
const DESKTOP_UNAVAILABLE = 'Desktop plugin install is unavailable in this environment.'

const callsFor = (action: string) =>
  requestGateway.mock.calls.filter(([method, params]) => method === 'plugins.manage' && params?.action === action)

const installButton = () => screen.queryByRole('button', { name: /^Install/ }) as HTMLButtonElement | null

/** Answer `inspect` with *inspection* (or throw it), `install` with *install*, everything else an empty list. */
const backend = (inspection: unknown, install: unknown = { ok: true, plugin_name: 'advanced-delegation' }) =>
  requestGateway.mockImplementation(async (method, params) => {
    if (method !== 'plugins.manage') {
      return { plugins: [] }
    }

    if (params?.action === 'inspect') {
      if (inspection instanceof Error) {
        throw inspection
      }

      return inspection
    }

    return params?.action === 'install' ? install : { plugins: [] }
  })

const AGENT_ONLY = { ok: true, agent: true, agent_name: 'advanced-delegation', desktop: false, warnings: [] }

const renderModal = () =>
  render(
    <MemoryRouter initialEntries={['/']}>
      <QueryClientProvider client={queryClient}>
        <PluginInstallModal />
      </QueryClientProvider>
    </MemoryRouter>
  )

beforeEach(() => {
  vi.clearAllMocks()
  queryClient.clear()
  closePluginInstallRequest()
  $notifications.set([])
  $gatewayState.set('idle')
  // The browser bridge's connection is always `remote` (browser-connection.ts).
  $connection.set({ mode: 'remote' } as NonNullable<ReturnType<typeof $connection.get>>)
  $activeGatewayProfile.set('default')
  expect(window.hermesDesktop).toBeUndefined()
})
afterEach(() => {
  cleanup()
  closePluginInstallRequest()
})

describe('Plugin install in the browser-hosted app (no window.hermesDesktop)', () => {
  it('inspects the catalog pick through the gateway and installs the agent plugin', async () => {
    backend(AGENT_ONLY)
    renderModal()
    act(() => openPluginInstallRequest(DELEGATION))

    const dialog = await screen.findByRole('dialog', { name: 'Advanced Delegation' })
    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    expect(within(dialog).queryByText(UNAVAILABLE)).toBeNull()
    expect(callsFor('inspect')[0]?.[1]).toMatchObject({
      action: 'inspect',
      identifier: DELEGATION.repo,
      catalog_name: 'advanced-delegation',
      profile: 'default'
    })

    fireEvent.click(installButton()!)

    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
    expect(callsFor('install')).toHaveLength(1)
    expect(callsFor('install')[0]?.[1]).toMatchObject({ catalog_name: 'advanced-delegation', enable: true })
    expect($notifications.get().some(n => n.kind === 'warning' && n.message === DESKTOP_UNAVAILABLE)).toBe(false)
  })

  it('says inspection is unavailable, with no Retry, when the backend predates the inspect action', async () => {
    backend(new JsonRpcGatewayError('unknown plugins action: inspect', { code: 4017 }))
    renderModal()
    act(() => openPluginInstallRequest(DELEGATION))

    expect((await screen.findByRole('alert')).textContent).toBe(UNAVAILABLE)
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
    expect(installButton()).toBeNull()
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeTruthy()
    expect(callsFor('install')).toHaveLength(0)
  })

  it('shows a failed read and retries the inspection', async () => {
    backend({ ok: false, agent: false, desktop: false, warnings: [], error: 'Git clone failed: network down' })
    renderModal()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/plugin' }))

    expect((await screen.findByRole('alert')).textContent).toBe('Git clone failed: network down')
    backend({ ok: true, agent: true, agent_name: 'plugin', desktop: false, warnings: [] })
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    expect(callsFor('inspect')).toHaveLength(2)
    expect(callsFor('install')).toHaveLength(0)
  })

  it('offers Retry after a transient transport failure', async () => {
    backend(new Error('request timed out after 120s: plugins.manage'))
    renderModal()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/plugin' }))

    expect((await screen.findByRole('alert')).textContent).toBe('request timed out after 120s: plugins.manage')
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  })

  it('installs the agent half of a mixed package and reports the desktop half as skipped, not installed', async () => {
    backend(
      { ok: true, agent: true, agent_name: 'pkg', desktop: true, desktop_name: 'pkg', warnings: [] },
      { ok: true, plugin_name: 'pkg', enabled: true }
    )
    renderModal()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/pkg' }))

    expect(await screen.findByText(DESKTOP_UNAVAILABLE)).toBeTruthy()
    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    fireEvent.click(installButton()!)

    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
    expect(callsFor('install')).toHaveLength(1)
    const notes = $notifications.get()
    expect(notes.some(n => n.kind === 'warning' && n.message === DESKTOP_UNAVAILABLE)).toBe(true)
    expect(notes.some(n => n.kind === 'success' && /desktop/i.test(n.message))).toBe(false)
  })

  it('refuses a desktop-only package without a Retry loop', async () => {
    backend({ ok: true, agent: false, desktop: true, desktop_name: 'ui', warnings: [] })
    renderModal()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/ui' }))

    expect((await screen.findByRole('alert')).textContent).toBe('This package has nothing to install here.')
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
    expect(installButton()).toBeNull()
    expect(callsFor('install')).toHaveLength(0)
  })
})

// The browser-hosted app WITH a backend that can install a desktop half (`plugins.manage
// install_desktop`, new in this change): the host serving the tab clones the plugin into the folder
// the Webapp already loads, so a desktop package is installable from the dialog instead of refused.
describe('Plugin install in the browser-hosted app with a backend that installs desktop halves', () => {
  beforeEach(() => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
  })
  afterEach(() => {
    delete window.document.documentElement.dataset.hermesDesktopHost
  })

  const desktopBackend = (inspection: unknown, desktop: unknown, install: unknown = { ok: true, plugin_name: 'pkg' }) =>
    requestGateway.mockImplementation(async (method, params) => {
      if (method !== 'plugins.manage') {
        return { plugins: [] }
      }

      if (params?.action === 'inspect') {
        return inspection
      }

      if (params?.action === 'install_desktop') {
        return desktop
      }

      return params?.action === 'install' ? install : { plugins: [] }
    })

  it('installs a desktop-only package on the host through the backend, at the reviewed commit', async () => {
    const reviewed = 'e404b6816b20237a82f6fbe2749b5a576cf75dba'

    desktopBackend(
      { ok: true, agent: false, desktop: true, desktop_name: 'ui', sha: reviewed, warnings: [] },
      { ok: true, plugin_name: 'ui', paired_with_agent: false }
    )

    renderModal()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/ui' }))

    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    expect(screen.queryByText('This package has nothing to install here.')).toBeNull()
    fireEvent.click(installButton()!)

    await waitFor(() => expect(callsFor('install_desktop')).toHaveLength(1))
    expect(callsFor('install_desktop')[0][1]).toMatchObject({
      action: 'install_desktop',
      identifier: 'https://github.com/example/ui',
      force: false,
      ref: reviewed
    })
    // No agent half exists in this repository: nothing is installed into the plugins folder.
    expect(callsFor('install')).toHaveLength(0)
    await waitFor(() => expect($notifications.get().some(n => n.message === 'Desktop plugin ui installed')).toBe(true))
    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
  })

  it('holds the agent install to the commit the dialog inspected', async () => {
    const reviewed = 'e404b6816b20237a82f6fbe2749b5a576cf75dba'

    desktopBackend(
      { ok: true, agent: true, agent_name: 'pkg', desktop: false, sha: reviewed, warnings: [] },
      {},
      { ok: true, plugin_name: 'pkg', enabled: true }
    )

    renderModal()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/pkg' }))

    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    fireEvent.click(installButton()!)

    await waitFor(() => expect(callsFor('install')).toHaveLength(1))
    expect(callsFor('install')[0][1]).toMatchObject({ action: 'install', inspected_sha: reviewed })
    // Not a pin: the backend refuses a branch that moved, and updates still follow the branch.
    expect(callsFor('install')[0][1]).not.toHaveProperty('ref')
  })

  it('installs a mixed package once and has the host copy its desktop half out of that checkout', async () => {
    const reviewed = 'e404b6816b20237a82f6fbe2749b5a576cf75dba'
    // The browser bridge's door to the host's desktop-plugins folder (browser-desktop-plugins.ts): the
    // host's pass copies the package's half into desktop-plugins/pkg.
    const reconcileDesktopPlugins = vi.fn(async () => ['/srv/hermes/desktop-plugins/pkg'])
    window.hermesDesktop = { reconcileDesktopPlugins } as unknown as Window['hermesDesktop']

    try {
      desktopBackend(
        { ok: true, agent: true, agent_name: 'pkg', desktop: true, desktop_name: 'pkg', sha: reviewed, warnings: [] },
        { ok: true, plugin_name: 'pkg', paired_with_agent: true },
        { ok: true, plugin_name: 'pkg', enabled: true }
      )

      renderModal()
      act(() => openPluginInstallRequest({ repo: 'https://github.com/example/pkg' }))

      await waitFor(() => expect(installButton()?.disabled).toBe(false))
      fireEvent.click(installButton()!)

      await waitFor(() =>
        expect($notifications.get().some(n => n.message === 'Desktop plugin pkg installed')).toBe(true)
      )
      expect(callsFor('install')).toHaveLength(1)
      expect(callsFor('install')[0][1]).toMatchObject({
        action: 'install',
        identifier: 'https://github.com/example/pkg'
      })
      // One checkout: no second clone of the same repository racing the host's scan into the same folder.
      expect(callsFor('install_desktop')).toHaveLength(0)
      expect(reconcileDesktopPlugins).toHaveBeenCalled()
      expect($notifications.get().some(n => n.message === DESKTOP_UNAVAILABLE)).toBe(false)
    } finally {
      delete (window as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('never reports a desktop half installed when the host copied none out of the package', async () => {
    // The probe saw a desktop half, but the host's pass put nothing down for it (an entry it does not
    // read, a refused tree): the folder the loader scans has no pkg/, so the dialog must not say it does.
    const reconcileDesktopPlugins = vi.fn(async () => [] as string[])
    const desktopPluginsRoot = vi.fn(async () => '/srv/hermes/desktop-plugins')

    const readDir = vi.fn(async (path: string) => ({
      entries:
        path === '/srv/hermes/desktop-plugins' ? [{ name: 'other', path: `${path}/other`, isDirectory: true }] : []
    }))

    window.hermesDesktop = {
      desktopPluginsRoot,
      readDir,
      reconcileDesktopPlugins
    } as unknown as Window['hermesDesktop']

    try {
      desktopBackend(
        {
          ok: true,
          agent: true,
          agent_name: 'pkg',
          desktop: true,
          desktop_name: 'pkg',
          sha: 'e404b6816b20237a82f6fbe2749b5a576cf75dba',
          warnings: []
        },
        { ok: true, plugin_name: 'pkg', paired_with_agent: true },
        { ok: true, plugin_name: 'pkg', enabled: true }
      )

      renderModal()
      act(() => openPluginInstallRequest({ repo: 'https://github.com/example/pkg' }))

      await waitFor(() => expect(installButton()?.disabled).toBe(false))
      fireEvent.click(installButton()!)

      expect(await screen.findByText('Desktop plugin install failed')).toBeTruthy()
      expect(reconcileDesktopPlugins).toHaveBeenCalled()
      expect(readDir).toHaveBeenCalledWith('/srv/hermes/desktop-plugins')
      expect($notifications.get().some(n => n.message === 'Desktop plugin pkg installed')).toBe(false)
      expect($notifications.get().some(n => n.message.startsWith('Agent plugin pkg installed'))).toBe(true)
    } finally {
      delete (window as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('never reports a desktop half installed when the host pass only pruned the old copy', async () => {
    // A force reinstall whose new tree dropped its desktop entry: the host's pass deletes
    // desktop-plugins/pkg and names it among the folders it touched. Nothing is on disk to load.
    const reconcileDesktopPlugins = vi.fn(async () => ['/srv/hermes/desktop-plugins/pkg'])
    const desktopPluginsRoot = vi.fn(async () => '/srv/hermes/desktop-plugins')

    const readDir = vi.fn(async (path: string) => ({
      entries:
        path === '/srv/hermes/desktop-plugins' ? [{ name: 'other', path: `${path}/other`, isDirectory: true }] : []
    }))

    window.hermesDesktop = {
      desktopPluginsRoot,
      readDir,
      reconcileDesktopPlugins
    } as unknown as Window['hermesDesktop']

    try {
      desktopBackend(
        {
          ok: true,
          agent: true,
          agent_name: 'pkg',
          desktop: true,
          desktop_name: 'pkg',
          sha: 'e404b6816b20237a82f6fbe2749b5a576cf75dba',
          warnings: []
        },
        { ok: true, plugin_name: 'pkg', paired_with_agent: true },
        { ok: true, plugin_name: 'pkg', enabled: true }
      )

      renderModal()
      act(() => openPluginInstallRequest({ repo: 'https://github.com/example/pkg' }))

      await waitFor(() => expect(installButton()?.disabled).toBe(false))
      fireEvent.click(installButton()!)

      expect(await screen.findByText('Desktop plugin install failed')).toBeTruthy()
      expect(readDir).toHaveBeenCalledWith('/srv/hermes/desktop-plugins')
      expect($notifications.get().some(n => n.message === 'Desktop plugin pkg installed')).toBe(false)
    } finally {
      delete (window as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('says the install is unavailable when the backend predates the action', async () => {
    requestGateway.mockImplementation(async (method, params) => {
      if (method !== 'plugins.manage') {
        return { plugins: [] }
      }

      if (params?.action === 'inspect') {
        return { ok: true, agent: false, desktop: true, desktop_name: 'ui', warnings: [] }
      }

      if (params?.action === 'install_desktop') {
        throw new JsonRpcGatewayError('unknown plugins action: install_desktop', { code: 4017 })
      }

      return { plugins: [] }
    })

    renderModal()
    act(() => openPluginInstallRequest({ repo: 'https://github.com/example/ui' }))

    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    fireEvent.click(installButton()!)

    expect(await screen.findByText(DESKTOP_UNAVAILABLE)).toBeTruthy()
    expect(callsFor('install')).toHaveLength(0)
  })
})
