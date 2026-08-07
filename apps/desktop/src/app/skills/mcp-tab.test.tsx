import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { McpCatalogEntry } from '@/types/hermes'

const installMcpCatalogEntry = vi.fn()
const getActionStatus = vi.fn()
const authMcpServer = vi.fn()
const getMcpOAuthFlow = vi.fn()
const completeMcpDesktopOAuth = vi.fn()
const notify = vi.fn()
const notifyError = vi.fn()

// Partial-mock @/hermes: only the four calls the install→OAuth chain exercises
// are stubbed. The rest stay real because sibling modules (e.g. @/store/profile)
// touch other exports at import time — a full mock would drop those and crash.
vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<typeof HermesApi>()

  return {
    ...actual,
    authMcpServer: (name: string) => authMcpServer(name),
    getActionStatus: (name: string, lines?: number) => getActionStatus(name, lines),
    getMcpOAuthFlow: (flowId: string) => getMcpOAuthFlow(flowId),
    installMcpCatalogEntry: (name: string, env?: Record<string, string>) => installMcpCatalogEntry(name, env)
  }
})

vi.mock('@/lib/mcp-dashboard-oauth', () => ({
  completeMcpDesktopOAuth: (opts: unknown) => completeMcpDesktopOAuth(opts)
}))

vi.mock('@/store/notifications', () => ({
  notify: (input: unknown) => notify(input),
  notifyError: (err: unknown, title?: string) => notifyError(err, title)
}))

function entry(overrides: Partial<McpCatalogEntry> = {}): McpCatalogEntry {
  return {
    name: 'linear',
    description: 'Linear issue tracker',
    source: 'nous',
    transport: 'http',
    auth_type: 'oauth',
    required_env: [],
    command: null,
    args: [],
    url: 'https://mcp.linear.app/mcp',
    install_url: null,
    install_ref: null,
    bootstrap: [],
    default_enabled: null,
    post_install: '',
    needs_install: false,
    installed: false,
    enabled: false,
    ...overrides
  }
}

beforeEach(() => {
  installMcpCatalogEntry.mockResolvedValue({ ok: true, background: false })
  completeMcpDesktopOAuth.mockResolvedValue({
    flow_id: 'f1',
    server_name: 'linear',
    status: 'approved',
    authorization_url: null,
    error: null,
    tools: [{ name: 'create_issue', description: 'Create an issue' }]
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('McpCatalog install → OAuth chaining', () => {
  it('refreshes immediately, then refreshes again after OAuth succeeds', async () => {
    const approvedFlow = {
      flow_id: 'f1',
      server_name: 'linear',
      status: 'approved',
      authorization_url: null,
      error: null,
      tools: [{ name: 'create_issue', description: 'Create an issue' }]
    }

    let approve: ((flow: typeof approvedFlow) => void) | undefined
    completeMcpDesktopOAuth.mockImplementation(
      () => new Promise<typeof approvedFlow>(resolve => (approve = resolve))
    )
    const { McpCatalog } = await import('./mcp-tab')
    const onInstalled = vi.fn()
    render(<McpCatalog entries={[entry()]} loading={false} onInstalled={onInstalled} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Install' }))

    await waitFor(() => expect(installMcpCatalogEntry).toHaveBeenCalledWith('linear', {}))
    // The auth flow is launched for the just-installed server, reusing the same
    // completeMcpDesktopOAuth machinery the Servers tab uses for re-auth.
    await waitFor(() =>
      expect(completeMcpDesktopOAuth).toHaveBeenCalledWith(expect.objectContaining({ serverName: 'linear' }))
    )
    await waitFor(() => expect(onInstalled).toHaveBeenCalledTimes(1))
    expect((screen.getByRole('button', { name: 'Install' }) as HTMLButtonElement).disabled).toBe(false)

    approve?.(approvedFlow)

    await waitFor(() => expect(onInstalled).toHaveBeenCalledTimes(2))
  })

  it('does not hold install completion behind persistent authorization_required polling', async () => {
    // Models completeMcpDesktopOAuth continuing to poll an authorization_required
    // flow after the user closes or abandons the browser.
    completeMcpDesktopOAuth.mockImplementation(() => new Promise(() => undefined))
    const { McpCatalog } = await import('./mcp-tab')
    const onInstalled = vi.fn()
    render(<McpCatalog entries={[entry()]} loading={false} onInstalled={onInstalled} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Install' }))

    await waitFor(() => expect(completeMcpDesktopOAuth).toHaveBeenCalled())
    await waitFor(() => expect(onInstalled).toHaveBeenCalledTimes(1))
    expect((screen.getByRole('button', { name: 'Install' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('does not launch OAuth for a non-oauth entry', async () => {
    const { McpCatalog } = await import('./mcp-tab')
    const onInstalled = vi.fn()
    render(
      <McpCatalog
        entries={[entry({ name: 'postgres', auth_type: 'api_key', transport: 'stdio', url: null })]}
        loading={false}
        onInstalled={onInstalled}
      />
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Install' }))

    await waitFor(() => expect(installMcpCatalogEntry).toHaveBeenCalledWith('postgres', {}))
    await waitFor(() => expect(onInstalled).toHaveBeenCalled())
    expect(completeMcpDesktopOAuth).not.toHaveBeenCalled()
  })

  it('does not launch OAuth for an oauth entry on a non-http transport', async () => {
    const { McpCatalog } = await import('./mcp-tab')
    const onInstalled = vi.fn()
    render(
      <McpCatalog
        entries={[entry({ name: 'stdio-oauth', transport: 'stdio' })]}
        loading={false}
        onInstalled={onInstalled}
      />
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Install' }))

    await waitFor(() => expect(installMcpCatalogEntry).toHaveBeenCalledWith('stdio-oauth', {}))
    await waitFor(() => expect(onInstalled).toHaveBeenCalled())
    expect(completeMcpDesktopOAuth).not.toHaveBeenCalled()
  })

  it('keeps the install successful when the chained OAuth fails', async () => {
    completeMcpDesktopOAuth.mockRejectedValue(new Error('user cancelled'))
    const { McpCatalog } = await import('./mcp-tab')
    const onInstalled = vi.fn()
    render(<McpCatalog entries={[entry()]} loading={false} onInstalled={onInstalled} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Install' }))

    // OAuth was attempted and threw...
    await waitFor(() => expect(completeMcpDesktopOAuth).toHaveBeenCalled())
    // ...yet the install still resolves: onInstalled fires and the error goes to
    // notify (connect-later), never to notifyError (which is the install-failed path).
    await waitFor(() => expect(onInstalled).toHaveBeenCalled())
    expect(notifyError).not.toHaveBeenCalled()
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'info' }))
  })
})

describe('McpCatalog post_install rendering', () => {
  it('renders setup notes when post_install is present', async () => {
    const { McpCatalog } = await import('./mcp-tab')
    render(
      <McpCatalog
        entries={[entry({ post_install: 'Grant the OAuth scopes in your Linear settings.' })]}
        loading={false}
        onInstalled={vi.fn()}
      />
    )

    expect(await screen.findByText('Setup notes')).toBeTruthy()
    expect(screen.getByText('Grant the OAuth scopes in your Linear settings.')).toBeTruthy()
  })

  it('omits the setup-notes block when post_install is empty', async () => {
    const { McpCatalog } = await import('./mcp-tab')
    render(<McpCatalog entries={[entry({ post_install: '' })]} loading={false} onInstalled={vi.fn()} />)

    await screen.findByRole('button', { name: 'Install' })
    expect(screen.queryByText('Setup notes')).toBeNull()
  })
})
