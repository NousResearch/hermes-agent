import { JsonRpcGatewayError } from '@hermes/shared'
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The source moves between the dialog's inspection and Install. The refusal says so with a structured
// marker; the dialog must read the source again and show the new commit, never resend the old one.
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
import { $pluginInstallRequest, closePluginInstallRequest, openPluginInstallRequest } from '@/store/plugin-install-request'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { PluginInstallModal } from './plugin-install-modal'

const V1 = 'e404b6816b20237a82f6fbe2749b5a576cf75dba'
const V2 = '0a415bc5c1d2e3f405162738495a6b7c8d9e0f1a'
const REPO = 'https://github.com/example/pkg'
const MOVED = `The source changed since you reviewed it: it is now at commit ${V2.slice(0, 12)}, not ${V1.slice(0, 12)}.`

const callsFor = (action: string) =>
  requestGateway.mock.calls.filter(([method, params]) => method === 'plugins.manage' && params?.action === action)

const installButton = () => screen.queryByRole('button', { name: /^Install/ }) as HTMLButtonElement | null

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

describe('Install after the source moved past the reviewed commit', () => {
  it('browser-hosted: re-inspects, shows the new commit, and installs that one instead of resending the old', async () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    let tip = V1

    requestGateway.mockImplementation(async (method, params) => {
      if (method !== 'plugins.manage') {
        return { plugins: [] }
      }

      if (params?.action === 'inspect') {
        return { ok: true, agent: true, agent_name: 'pkg', desktop: false, sha: tip, warnings: [] }
      }

      if (params?.action === 'install') {
        if (params.inspected_sha !== tip) {
          throw new JsonRpcGatewayError(MOVED, { code: 5026, data: { reason: 'source_moved' } })
        }

        return { ok: true, plugin_name: 'pkg' }
      }

      return { plugins: [] }
    })

    renderModal()
    act(() => openPluginInstallRequest({ repo: REPO }))
    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    tip = V2 // the branch moves while the dialog is open
    fireEvent.click(installButton()!)

    // The dialog read the source again and names the new commit; Install, not Retry, is offered.
    expect(await screen.findByText(new RegExp(`Review the new version \\(commit ${V2.slice(0, 12)}\\)`))).toBeTruthy()
    expect(callsFor('inspect')).toHaveLength(2)
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
    expect($pluginInstallRequest.get()).not.toBeNull()

    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    fireEvent.click(installButton()!)

    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
    expect(callsFor('install').map(([, params]) => params?.inspected_sha)).toEqual([V1, V2])
  })

  it('an ordinary install failure still offers Retry and never re-inspects', async () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    requestGateway.mockImplementation(async (method, params) => {
      if (method === 'plugins.manage' && params?.action === 'inspect') {
        return { ok: true, agent: true, agent_name: 'pkg', desktop: false, sha: V1, warnings: [] }
      }

      if (method === 'plugins.manage' && params?.action === 'install') {
        // The same words with no marker: text from a git server is not the backend's verdict.
        throw new JsonRpcGatewayError(MOVED, { code: 5026 })
      }

      return { plugins: [] }
    })

    renderModal()
    act(() => openPluginInstallRequest({ repo: REPO }))
    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    fireEvent.click(installButton()!)

    expect(await screen.findByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(callsFor('inspect')).toHaveLength(1)
  })

  it('native Desktop: Electron refusing the reviewed commit re-inspects and installs the new commit', async () => {
    const probePluginRepo = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, agent: false, desktop: true, desktopName: 'ui', sha: V1, warnings: [] })
      .mockResolvedValueOnce({ ok: true, agent: false, desktop: true, desktopName: 'ui', sha: V2, warnings: [] })

    const installDesktopPlugin = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, sourceMoved: true, error: 'The commit you reviewed could not be checked out.' })
      .mockResolvedValueOnce({ ok: true, pluginName: 'ui' })

    vi.stubGlobal('hermesDesktop', { installDesktopPlugin, probePluginRepo })

    renderModal()
    act(() => openPluginInstallRequest({ repo: REPO }))
    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    fireEvent.click(installButton()!)

    expect(await screen.findByText(new RegExp(`commit ${V2.slice(0, 12)}`))).toBeTruthy()
    expect(probePluginRepo).toHaveBeenCalledTimes(2)
    await waitFor(() => expect(installButton()?.disabled).toBe(false))
    fireEvent.click(installButton()!)

    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
    expect(installDesktopPlugin.mock.calls.map(([payload]) => payload.sha)).toEqual([V1, V2])
  })
})
