import type { ComposioAccountRow } from '@hermes/shared/gateway-events'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileScope } from '@/hermes'
import { I18nProvider } from '@/i18n'
import type * as GatewayModule from '@/store/gateway'

import { ComposioProviderRows, composioProvidersMatch, useComposioProviderCatalog } from './composio-providers'

const gatewayMocks = vi.hoisted(() => ({ requestGatewayForAgent: vi.fn() }))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayModule>()),
  requestGatewayForAgent: gatewayMocks.requestGatewayForAgent
}))

const mocks = vi.hoisted(() => {
  class McpOAuthCancelled extends Error {
    constructor() {
      super('OAuth cancelled by user')
      this.name = 'McpOAuthCancelled'
    }
  }

  return {
    completeMcpDesktopOAuth: vi.fn(),
    McpOAuthCancelled
  }
})

vi.mock('@/lib/mcp-dashboard-oauth', () => mocks)

const CATALOG = {
  providers: [
    { label: 'Gmail', provider: 'gmail', toolkit: 'gmail' },
    { label: 'Google Calendar', provider: 'calendar', toolkit: 'googlecalendar' },
    { label: 'Google Drive', provider: 'drive', toolkit: 'googledrive' }
  ]
}

const statusRow = (provider: string, status: string, accounts: ComposioAccountRow[] = []) => ({
  accounts,
  provider,
  status,
  toolkit: provider
})

let clients: QueryClient[] = []
let openExternal: ReturnType<typeof vi.fn>

function serveGateway({
  accounts = [
    { alias: 'work', id: 'gmail-work', status: 'active' },
    { alias: 'home', id: 'gmail-home', status: 'active' }
  ],
  connect = { provider: 'gmail', status: 'connected' },
  providers = [statusRow('gmail', 'connected', accounts)],
  afterRemove,
  removeStatus = 'removed'
}: {
  accounts?: ComposioAccountRow[]
  connect?: Record<string, unknown>
  providers?: Array<Record<string, unknown>>
  afterRemove?: (id: string) => void
  removeStatus?: 'not_found' | 'removed' | 'requires_mcp_auth'
} = {}) {
  const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === 'composio.catalog') {
      return CATALOG
    }

    if (method === 'composio.status') {
      return { providers }
    }

    if (method === 'composio.connect') {
      return connect
    }

    if (method === 'composio.remove') {
      afterRemove?.(String(params.account_id))

      return { account_id: params.account_id, provider: params.provider, status: removeStatus, toolkit: 'gmail' }
    }

    if (method === 'mcp.servers.add') {
      return { name: 'composio', ok: true, server: {} }
    }

    if (method === 'reload.mcp') {
      return { status: 'reloaded' }
    }

    throw new Error(`Unexpected gateway method ${method}`)
  })

  gatewayMocks.requestGatewayForAgent.mockImplementation(
    (_connectionId: unknown, _profile: unknown, method: string, params: Record<string, unknown>) => request(method, params)
  )

  return request
}

function RowsHarness({
  filterQuery,
  mcpServers,
  profile
}: {
  filterQuery: string
  mcpServers: Record<string, Record<string, unknown>>
  profile: ProfileScope
}) {
  const catalog = useComposioProviderCatalog(profile)

  return <ComposioProviderRows catalog={catalog} filterQuery={filterQuery} mcpServers={mcpServers} profile={profile} refreshMcpConfig={vi.fn()} />
}

function renderRows({ filterQuery = '', mcpServers = {} }: { filterQuery?: string; mcpServers?: Record<string, Record<string, unknown>> } = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } }
  })

  clients.push(queryClient)

  return render(
    <QueryClientProvider client={queryClient}>
      <I18nProvider>
        <RowsHarness
          filterQuery={filterQuery}
          mcpServers={mcpServers}
          profile={'default' as ProfileScope}
        />
      </I18nProvider>
    </QueryClientProvider>
  )
}

afterEach(() => {
  cleanup()
  clients.forEach(client => client.clear())
  clients = []
  gatewayMocks.requestGatewayForAgent.mockReset()
})

beforeEach(() => {
  mocks.completeMcpDesktopOAuth.mockReset()
  openExternal = vi.fn().mockResolvedValue(undefined)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { openExternal }
  })
})

describe('Composio provider rows', () => {
  it('matches provider labels, ids, and toolkit slugs for the directory search', () => {
    expect(composioProvidersMatch(CATALOG.providers, 'googlecalendar')).toBe(true)
    expect(composioProvidersMatch(CATALOG.providers, 'Google Drive')).toBe(true)
    expect(composioProvidersMatch(CATALOG.providers, 'zoho_inventory')).toBe(false)
  })

  it('shows every returned account and starts a replacement as a new auth without deleting the old account', async () => {
    let currentProviders = [
      statusRow('gmail', 'connected', [
        { alias: 'work', id: 'gmail-work', status: 'expired' },
        { alias: 'home', id: 'gmail-home', status: 'active' }
      ])
    ]

    const request = serveGateway({
      connect: { auth_url: 'https://accounts.google.com/authorize', provider: 'gmail', status: 'pending' },
      providers: currentProviders
    })

    request.mockImplementation(async (method, params) => {
      if (method === 'composio.catalog') {return CATALOG}

      if (method === 'composio.status') {return { providers: currentProviders }}

      if (method === 'mcp.servers.add') {return { name: 'composio', ok: true, server: {} }}

      if (method === 'reload.mcp') {return { status: 'reloaded' }}

      if (method === 'composio.connect') {
        currentProviders = [
          statusRow('gmail', 'pending', [
            { alias: 'work', id: 'gmail-work', status: 'expired' },
            { alias: 'home', id: 'gmail-home', status: 'active' },
            { alias: 'new', id: 'gmail-pending', status: 'pending' }
          ])
        ]

        return { auth_url: 'https://accounts.google.com/authorize', provider: 'gmail', status: 'pending' }
      }

      if (method === 'composio.remove') {return { account_id: params.account_id, provider: params.provider, status: 'removed' }}
      throw new Error(`Unexpected gateway method ${method}`)
    })

    renderRows()

    await screen.findByText('work')
    const row = screen.getByText('Gmail').closest('[data-provider="gmail"]') as HTMLElement
    expect(within(row).getByText('work')).toBeTruthy()
    expect(within(row).getByText('home')).toBeTruthy()
    expect(within(row).getAllByText('Connected')).toHaveLength(2)
    expect(within(row).getByText(/Reconnect starts a new sign-in/)).toBeTruthy()

    fireEvent.click(within(row).getByRole('button', { name: 'Reconnect' }))

    await waitFor(() => expect(openExternal).toHaveBeenCalledWith('https://accounts.google.com/authorize'))
    expect(request.mock.calls.filter(([method]) => method === 'composio.connect').map(([, params]) => params)).toEqual([
      { profile: 'default', provider: 'gmail' }
    ])
    expect(request.mock.calls.some(([method]) => method === 'composio.remove')).toBe(false)

    fireEvent.click(await within(row).findByRole('button', { name: 'Stop waiting' }))

    expect(await within(row).findByText('new')).toBeTruthy()
    expect(request.mock.calls.some(([method]) => method === 'composio.remove')).toBe(false)
  })

  it('uses the exact account id for confirmed removal', async () => {
    const removed: string[] = []
    const request = serveGateway({ afterRemove: id => removed.push(id) })

    renderRows()
    await screen.findByText('work')
    const row = screen.getByText('Gmail').closest('[data-provider="gmail"]') as HTMLElement

    fireEvent.click(within(row).getAllByRole('button', { name: 'Remove' })[0])
    fireEvent.click(within(row).getByRole('alertdialog').querySelector('button:last-child') as HTMLButtonElement)

    await waitFor(() => expect(removed).toEqual(['gmail-work']))
    expect(
      request.mock.calls.filter(([method]) => method === 'composio.remove').map(([, params]) => params)
    ).toEqual([{ account_id: 'gmail-work', profile: 'default', provider: 'gmail' }])
  })

  it('keeps the confirmation and account visible when removal needs native sign-in', async () => {
    const request = serveGateway({ removeStatus: 'requires_mcp_auth' })

    renderRows()
    await screen.findByText('work')
    const row = screen.getByText('Gmail').closest('[data-provider="gmail"]') as HTMLElement

    fireEvent.click(within(row).getAllByRole('button', { name: 'Remove' })[0])
    fireEvent.click(within(row).getByRole('alertdialog').querySelector('button:last-child') as HTMLButtonElement)

    expect(await within(row).findByRole('alert')).toBeTruthy()
    expect(within(row).getByRole('alertdialog')).toBeTruthy()
    expect(within(row).getByText('work')).toBeTruthy()
    expect(request.mock.calls.filter(([method]) => method === 'composio.remove')).toHaveLength(1)
  })

  it('reloads the profile MCP server after native OAuth before retrying provider consent', async () => {
    const order: string[] = []
    let connects = 0
    const request = serveGateway({ providers: [statusRow('gmail', 'requires_mcp_auth')] })

    request.mockImplementation(async (method, params) => {
      if (method === 'composio.catalog') {return CATALOG}

      if (method === 'composio.status') {return { providers: [statusRow('gmail', 'requires_mcp_auth')] }}

      if (method === 'mcp.servers.add') {
        order.push('add')

        return { name: 'composio', ok: true, server: {} }
      }

      if (method === 'composio.connect') {
        connects += 1
        order.push(`connect-${connects}`)

        return connects === 1
          ? { provider: 'gmail', status: 'requires_mcp_auth' }
          : { auth_url: 'https://accounts.google.com/authorize', provider: 'gmail', status: 'pending' }
      }

      if (method === 'reload.mcp') {
        order.push('reload')

        return { status: 'reloaded' }
      }

      if (method === 'composio.remove') {return { account_id: params.account_id, provider: params.provider, status: 'removed' }}
      throw new Error(`Unexpected gateway method ${method}`)
    })
    mocks.completeMcpDesktopOAuth.mockImplementation(async () => {
      order.push('oauth')

      return { status: 'approved' }
    })

    renderRows()
    await screen.findByText('Gmail')
    const row = screen.getByText('Gmail').closest('[data-provider="gmail"]') as HTMLElement

    fireEvent.click(within(row).getByRole('button', { name: 'Connect' }))

    await waitFor(() => expect(openExternal).toHaveBeenCalledWith('https://accounts.google.com/authorize'))
    expect(order).toEqual(['add', 'reload', 'connect-1', 'oauth', 'reload', 'connect-2'])
    expect(request.mock.calls.find(([method]) => method === 'reload.mcp')?.[1]).toEqual({ confirm: true })
    expect(request.mock.calls.filter(([method]) => method === 'composio.connect').map(([, params]) => params)).toEqual([
      { profile: 'default', provider: 'gmail' },
      { profile: 'default', provider: 'gmail' }
    ])
  })

  it('adds the untrusted MCP preset only after a click and leaves it in place when consent is cancelled', async () => {
    const request = serveGateway({
      providers: [statusRow('gmail', 'requires_mcp_auth')]
    })

    const order: string[] = []
    request.mockImplementation(async (method, params) => {
      if (method === 'composio.catalog') {return CATALOG}

      if (method === 'composio.status') {return { providers: [statusRow('gmail', 'requires_mcp_auth')] }}

      if (method === 'composio.connect') {return { provider: 'gmail', status: 'requires_mcp_auth' }}

      if (method === 'mcp.servers.add') {
        order.push('add')

        return { name: 'composio', ok: true, server: {} }
      }

      if (method === 'reload.mcp') {
        order.push('reload')

        return { status: 'reloaded' }
      }

      if (method === 'composio.remove') {return { account_id: params.account_id, provider: params.provider, status: 'removed' }}
      throw new Error(`Unexpected gateway method ${method}`)
    })
    mocks.completeMcpDesktopOAuth.mockImplementation(async () => {
      order.push('oauth')
      throw new mocks.McpOAuthCancelled()
    })

    renderRows()
    await screen.findByText('Gmail')
    const row = screen.getByText('Gmail').closest('[data-provider="gmail"]') as HTMLElement

    expect(request.mock.calls.some(([method]) => method === 'mcp.servers.add')).toBe(false)
    fireEvent.click(within(row).getByRole('button', { name: 'Connect' }))

    await waitFor(() => expect(order).toEqual(['add', 'reload', 'oauth']))
    expect(request.mock.calls.filter(([method]) => method === 'mcp.servers.add').map(([, params]) => params)).toEqual([
      {
        config: { trust: 'untrusted' },
        name: 'composio',
        preset: 'composio',
        profile: 'default'
      }
    ])
    expect(request.mock.calls.filter(([method]) => method === 'composio.connect')).toHaveLength(1)
    expect(request.mock.calls.some(([method]) => method === 'composio.remove')).toBe(false)
    expect(mocks.completeMcpDesktopOAuth).toHaveBeenCalledTimes(1)
  })
})
