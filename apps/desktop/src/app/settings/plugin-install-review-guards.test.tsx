import { JsonRpcGatewayError } from '@hermes/shared'
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// What an install dialog may put down besides the commit it showed: only what the user explicitly chose.
// An older backend that cannot hold the install to that commit asks first; a catalog pick is held to the
// pin the backend read; a package whose desktop half could not follow its agent half never gets that half
// at another commit without a replacement of both.
const { requestGateway } = vi.hoisted(() => ({
  requestGateway: vi.fn(async (_method: string, _params?: Record<string, unknown>): Promise<unknown> => ({
    plugins: []
  }))
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))
const { lookupPluginCatalogEntry } = vi.hoisted(() => ({ lookupPluginCatalogEntry: vi.fn() }))

vi.mock('@/lib/plugin-catalog', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  lookupPluginCatalogEntry
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

const V1 = 'e404b6816b20237a82f6fbe2749b5a576cf75dba'
const V2 = '0a415bc5c1d2e3f405162738495a6b7c8d9e0f1a'
const REPO = 'https://github.com/example/pkg'

const ADMISSION_REFUSAL =
  'invalid params for plugins.manage: inspected_sha: Extra inputs are not permitted — the client and the ' +
  'Hermes backend are out of sync (different versions); run `hermes update` and restart both'

const callsFor = (action: string) =>
  requestGateway.mock.calls.filter(([method, params]) => method === 'plugins.manage' && params?.action === action)

const installButton = () => screen.queryByRole('button', { name: /^Install$|^Installing/ }) as HTMLButtonElement | null

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
  $connection.set({ mode: 'remote' } as NonNullable<ReturnType<typeof $connection.get>>)
  $activeGatewayProfile.set('default')
})
afterEach(() => {
  cleanup()
  closePluginInstallRequest()
  vi.unstubAllGlobals()
  delete window.document.documentElement.dataset.hermesDesktopHost
})

describe('a backend that predates inspected_sha', () => {
  const olderBackend = () =>
    requestGateway.mockImplementation(async (method, params) => {
      if (method !== 'plugins.manage') {
        return { plugins: [] }
      }

      if (params?.action === 'inspect') {
        return { ok: true, agent: true, agent_name: 'pkg', desktop: false, sha: V1, warnings: [] }
      }

      if (params?.action === 'install') {
        if ('inspected_sha' in params) {
          throw new JsonRpcGatewayError(ADMISSION_REFUSAL, { code: 4000 })
        }

        return { ok: true, plugin_name: 'pkg' }
      }

      return { plugins: [] }
    })

  it('explains the missing guarantee and installs nothing until the user confirms', async () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    olderBackend()

    renderModal()
    act(() => openPluginInstallRequest({ profile: 'research', repo: REPO }))
    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    fireEvent.click(installButton()!)

    expect(await screen.findByText(/cannot guarantee that the version you reviewed/)).toBeTruthy()
    expect(screen.getByText(/Update the backend to get that guarantee/)).toBeTruthy()
    expect(callsFor('install')).toHaveLength(1)
    expect($pluginInstallRequest.get()).not.toBeNull()
    expect($notifications.get().filter(n => n.kind === 'success')).toEqual([])

    fireEvent.click(screen.getByRole('button', { name: 'Install anyway' }))

    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
    const [, anyway] = callsFor('install')[1]
    expect(anyway).not.toHaveProperty('inspected_sha')
    expect(anyway).toMatchObject({ identifier: REPO, profile: 'research' })
  })

  it('Cancel leaves nothing installed', async () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    olderBackend()

    renderModal()
    act(() => openPluginInstallRequest({ repo: REPO }))
    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    fireEvent.click(installButton()!)
    await screen.findByRole('button', { name: 'Install anyway' })

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    expect($pluginInstallRequest.get()).toBeNull()
    expect(callsFor('install')).toHaveLength(1)
  })
})

describe('native Desktop on a remote backend: the desktop half cannot follow the agent half', () => {
  it('reports a partial install and replaces both halves only once Force is on', async () => {
    let tip = V1

    requestGateway.mockImplementation(async (method, params) => {
      if (method === 'plugins.manage' && params?.action === 'install') {
        return params.force || !callsFor('install').slice(0, -1).length
          ? { ok: true, plugin_name: 'pkg' }
          : { ok: false, error: "Plugin 'pkg' already exists." }
      }

      return { plugins: [] }
    })

    const probePluginRepo = vi.fn(async () => ({
      ok: true,
      agent: true,
      agentName: 'pkg',
      desktop: true,
      desktopName: 'pkg',
      sha: tip,
      warnings: []
    }))

    const installDesktopPlugin = vi
      .fn()
      .mockResolvedValueOnce({
        ok: false,
        sourceMoved: true,
        error: 'The commit you reviewed could not be checked out.'
      })
      .mockResolvedValue({ ok: true, pluginName: 'pkg' })

    vi.stubGlobal('hermesDesktop', { installDesktopPlugin, probePluginRepo })

    renderModal()
    act(() => openPluginInstallRequest({ repo: REPO }))
    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    tip = V2
    fireEvent.click(installButton()!)

    // The agent half is in at V1, the desktop half is not: said so, never a clean "review again".
    expect(await screen.findByText(new RegExp(`agent half is in at commit ${V1.slice(0, 12)}`))).toBeTruthy()
    await waitFor(() => expect(probePluginRepo).toHaveBeenCalledTimes(2))
    expect(installButton()?.disabled).toBe(true)
    fireEvent.click(installButton()!)
    expect(installDesktopPlugin).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole('switch'))
    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    fireEvent.click(installButton()!)

    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
    expect(callsFor('install').map(([, params]) => [params?.inspected_sha, params?.force])).toEqual([
      [V1, false],
      [V2, true]
    ])
    expect(installDesktopPlugin.mock.calls.map(([payload]) => [payload.sha, payload.force])).toEqual([
      [V1, false],
      [V2, true]
    ])
  })
})

describe('a catalog pick is held to the pin the dialog showed', () => {
  const catalogBackend = (refuseUnless?: () => string) =>
    requestGateway.mockImplementation(async (method, params) => {
      if (method === 'plugins.manage' && params?.action === 'inspect') {
        return { ok: true, agent: true, agent_name: 'pkg', desktop: false, sha: V1, warnings: [] }
      }

      if (method === 'plugins.manage' && params?.action === 'install') {
        if (refuseUnless && params.inspected_sha !== refuseUnless()) {
          throw new JsonRpcGatewayError('The source changed since you reviewed it.', {
            code: 5026,
            data: { reason: 'source_moved' }
          })
        }

        return { ok: true, plugin_name: 'pkg' }
      }

      return { plugins: [] }
    })

  const openCatalog = () => act(() => openPluginInstallRequest({ catalogName: 'pkg', repo: REPO, sha: V1 }))

  it('browser-hosted: the commit the backend read at the catalog pin', async () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    catalogBackend()

    renderModal()
    openCatalog()
    await waitFor(() => expect(callsFor('inspect')).toHaveLength(1))
    fireEvent.click(installButton()!)

    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
    expect(callsFor('install')[0][1]).toMatchObject({ catalog_name: 'pkg', inspected_sha: V1 })
  })

  it("native Desktop: the reviewed pin, never the branch tip Electron's probe read", async () => {
    catalogBackend()
    const probePluginRepo = vi.fn(async () => ({ ok: true, agent: true, desktop: false, sha: V2, warnings: [] }))
    vi.stubGlobal('hermesDesktop', { probePluginRepo })

    renderModal()
    openCatalog()
    await waitFor(() => expect(probePluginRepo).toHaveBeenCalledTimes(1))
    fireEvent.click(installButton()!)

    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
    expect(callsFor('install')[0][1]).toMatchObject({ catalog_name: 'pkg', inspected_sha: V1 })
  })

  it('native Desktop: a re-pin since the dialog opened is refused, and the next Install holds the new pin', async () => {
    const pin = { current: V1 }
    catalogBackend(() => pin.current)
    lookupPluginCatalogEntry.mockImplementation(async () => ({ ok: true, entry: { name: 'pkg', sha: pin.current } }))
    const probePluginRepo = vi.fn(async () => ({ ok: true, agent: true, desktop: false, sha: V2, warnings: [] }))
    vi.stubGlobal('hermesDesktop', { probePluginRepo })

    renderModal()
    openCatalog()
    await waitFor(() => expect(probePluginRepo).toHaveBeenCalledTimes(1))
    pin.current = V2 // the catalog re-pins while the dialog is open
    fireEvent.click(installButton()!)

    expect(await screen.findByText(new RegExp(`Review the new version \\(commit ${V2.slice(0, 12)}\\)`))).toBeTruthy()
    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    fireEvent.click(installButton()!)

    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
    expect(callsFor('install').map(([, params]) => params?.inspected_sha)).toEqual([V1, V2])
  })

  it('native Desktop: after a re-pin, Install stays off until the new pin is on screen', async () => {
    const pin = { current: V1 }
    catalogBackend(() => pin.current)
    let release: () => void = () => undefined
    lookupPluginCatalogEntry.mockImplementation(
      () =>
        new Promise(resolve => {
          release = () => resolve({ ok: true, entry: { name: 'pkg', sha: pin.current } })
        })
    )
    const probePluginRepo = vi.fn(async () => ({ ok: true, agent: true, desktop: false, sha: V2, warnings: [] }))
    vi.stubGlobal('hermesDesktop', { probePluginRepo })

    renderModal()
    openCatalog()
    await waitFor(() => expect(probePluginRepo).toHaveBeenCalledTimes(1))
    pin.current = V2
    fireEvent.click(installButton()!)

    await waitFor(() => expect(probePluginRepo).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(callsFor('install')).toHaveLength(1))
    // The re-read inspection is back but the catalog pin lookup is not: a click here must not install.
    expect(installButton()?.disabled).toBe(true)
    fireEvent.click(installButton()!)
    expect(callsFor('install')).toHaveLength(1)

    act(() => release())
    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    fireEvent.click(installButton()!)
    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
    expect(callsFor('install').map(([, params]) => params?.inspected_sha)).toEqual([V1, V2])
  })
})
