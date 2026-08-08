// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import type { McpCatalogEntry } from '@/types/hermes'

const activeProfile = atom('default')
const activeSessionId = atom<string | null>('session-1')
const gateway = atom<{ request: ReturnType<typeof vi.fn> } | null>(null)
const getMcpCatalog = vi.fn()
const installMcpCatalogEntry = vi.fn()
const setMcpServerEnabled = vi.fn()
const completeMcpDesktopOAuth = vi.fn()
const reloadMcp = vi.fn()

vi.mock('@/hermes', () => ({
  authMcpServer: vi.fn(),
  getActionStatus: vi.fn(),
  getMcpCatalog: () => getMcpCatalog(),
  getMcpOAuthFlow: vi.fn(),
  installMcpCatalogEntry: (name: string, env: Record<string, string>) => installMcpCatalogEntry(name, env),
  setMcpServerEnabled: (name: string, enabled: boolean) => setMcpServerEnabled(name, enabled)
}))

vi.mock('@/lib/mcp-dashboard-oauth', () => ({
  completeMcpDesktopOAuth: (options: unknown) => completeMcpDesktopOAuth(options)
}))

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: activeProfile,
  normalizeProfileKey: (profile: null | string | undefined) => profile?.trim() || 'default'
}))
vi.mock('@/store/gateway', () => ({ $gateway: gateway }))
vi.mock('@/store/session', () => ({ $activeSessionId: activeSessionId }))

function catalogEntry(patch: Partial<McpCatalogEntry> = {}): McpCatalogEntry {
  return {
    args: [],
    authenticated: null,
    auth_type: 'none',
    bootstrap: [],
    command: null,
    default_enabled: null,
    description: 'Work with Linear issues and projects.',
    enabled: false,
    install_ref: null,
    install_url: null,
    installed: false,
    name: 'linear',
    needs_install: false,
    post_install: '',
    required_env: [],
    source: 'https://linear.app',
    transport: 'http',
    url: 'https://mcp.linear.app',
    ...patch
  }
}

async function renderCatalog(props: { onlyInstalled?: boolean } = {}) {
  const { IntegrationsCatalog } = await import('./integrations-catalog')
  let result: ReturnType<typeof render>

  await act(async () => {
    result = render(
      <QueryClientProvider client={queryClient}>
        <IntegrationsCatalog {...props} />
      </QueryClientProvider>
    )
  })

  return result!
}

beforeEach(() => {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { openExternal: vi.fn().mockResolvedValue(undefined) }
  })
  activeProfile.set('default')
  activeSessionId.set('session-1')
  gateway.set({ request: reloadMcp })
  getMcpCatalog.mockResolvedValue({ diagnostics: [], entries: [catalogEntry()] })
  installMcpCatalogEntry.mockResolvedValue({ background: false, name: 'linear', ok: true })
  setMcpServerEnabled.mockResolvedValue({ ok: true })
  completeMcpDesktopOAuth.mockResolvedValue({ status: 'approved' })
  reloadMcp.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  Reflect.deleteProperty(window, 'hermesDesktop')
  vi.clearAllMocks()
})

describe('IntegrationsCatalog', () => {
  it('installs once, enables it, and loads the saved integration into the current chat', async () => {
    await renderCatalog()

    fireEvent.click(await screen.findByRole('button', { name: 'Install Linear' }))

    await waitFor(() => expect(installMcpCatalogEntry).toHaveBeenCalledWith('linear', {}))
    await waitFor(() =>
      expect(reloadMcp).toHaveBeenCalledWith('reload.mcp', { confirm: true, session_id: 'session-1' })
    )
  })

  it('treats an installed, enabled integration as ready without reinstalling it', async () => {
    getMcpCatalog.mockResolvedValue({ diagnostics: [], entries: [catalogEntry({ enabled: true, installed: true })] })

    await renderCatalog()

    const ready = await screen.findByRole('button', { name: 'Ready Linear' })
    expect((ready as HTMLButtonElement).disabled).toBe(true)
    expect(installMcpCatalogEntry).not.toHaveBeenCalled()
  })

  it('can show only MCP servers already installed in this profile', async () => {
    getMcpCatalog.mockResolvedValue({
      diagnostics: [],
      entries: [catalogEntry(), catalogEntry({ enabled: true, installed: true, name: 'notion' })]
    })

    await renderCatalog({ onlyInstalled: true })

    expect(await screen.findByRole('button', { name: 'Ready Notion' })).toBeTruthy()
    expect(screen.queryByText('Linear')).toBeNull()
  })

  it('only asks an OAuth integration to sign in when its saved token is missing', async () => {
    getMcpCatalog.mockResolvedValue({
      diagnostics: [],
      entries: [catalogEntry({ authenticated: false, auth_type: 'oauth', enabled: true, installed: true })]
    })

    await renderCatalog()

    const signIn = await screen.findByRole('button', { name: 'Sign in Linear' })
    expect(completeMcpDesktopOAuth).not.toHaveBeenCalled()
    fireEvent.click(signIn)

    await waitFor(() => expect(completeMcpDesktopOAuth).toHaveBeenCalledOnce())
    await waitFor(() => expect(reloadMcp).toHaveBeenCalledOnce())
  })

  it('activates a previously disabled OAuth integration and completes its one-time sign-in', async () => {
    getMcpCatalog.mockResolvedValue({
      diagnostics: [],
      entries: [catalogEntry({ authenticated: false, auth_type: 'oauth', enabled: false, installed: true })]
    })

    await renderCatalog()

    fireEvent.click(await screen.findByRole('button', { name: 'Activate Linear' }))

    await waitFor(() => expect(setMcpServerEnabled).toHaveBeenCalledWith('linear', true))
    await waitFor(() => expect(completeMcpDesktopOAuth).toHaveBeenCalledOnce())
    await waitFor(() => expect(reloadMcp).toHaveBeenCalledOnce())
  })
})
