// @vitest-environment jsdom
//
// The container, black-box: every RPC is stubbed at `data/rpc.ts` (the module
// the queries and mutations actually import) and the servers on this Mac come
// from a stubbed config record. Nothing below the container is mocked, so a
// break in the join, the derive or a component fails here too.

import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { queryClient } from '@/lib/query-client'

import type * as Rpc from './data/rpc'

const listConnectors = vi.fn()
const connectorCatalog = vi.fn()
const connectorAccounts = vi.fn()
const connectorPolicy = vi.fn()
const connectorTools = vi.fn()
const setConnectorPolicy = vi.fn()
const removeConnectorAccount = vi.fn()
const connectAccountConnectors = vi.fn()

// Partial mock: `ConnectorRpcError`, `asConnectorError` and `isConnectorReason`
// stay REAL, because the hooks branch on `instanceof` and on the typed reason.
vi.mock('./data/rpc', async importOriginal => ({
  ...(await importOriginal<typeof Rpc>()),
  connectAccountConnectors: (...args: unknown[]) => connectAccountConnectors(...args),
  connectorAccounts: (...args: unknown[]) => connectorAccounts(...args),
  connectorCatalog: (...args: unknown[]) => connectorCatalog(...args),
  connectorPolicy: (...args: unknown[]) => connectorPolicy(...args),
  connectorTools: (...args: unknown[]) => connectorTools(...args),
  listConnectors: (...args: unknown[]) => listConnectors(...args),
  removeConnectorAccount: (...args: unknown[]) => removeConnectorAccount(...args),
  setConnectorPolicy: (...args: unknown[]) => setConnectorPolicy(...args)
}))

const getHermesConfigRecord = vi.fn()
const getMcpCatalog = vi.fn()
const testMcpServer = vi.fn()
const getUsageAnalytics = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getHermesConfigRecord: () => getHermesConfigRecord(),
  getMcpCatalog: () => getMcpCatalog(),
  getUsageAnalytics: (days: number) => getUsageAnalytics(days),
  testMcpServer: (name: string) => testMcpServer(name)
}))

// `readableError` stays a real-shaped function: the Disconnect confirm feeds a
// refused write through it to get the sentence it shows inline.
vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn(),
  readableError: (error: unknown, fallback: string) => ({
    message: error instanceof Error ? error.message : fallback
  })
}))

// Module scope, after the hoisted mocks: the component tree is heavy and the
// import cost belongs to collection, not to the first test's budget.
const { ConnectorsTab } = await import('./connectors-tab')
const { ConnectorRpcError } = await import('./data/rpc')
const { $accountOperations, applyAccountConnectionUpdate } = await import('./data/account-operations')
const { notifyError } = await import('@/store/notifications')

const CATALOG = {
  connectors: [{ category: 'Docs', description: 'Pages and databases.', name: 'Notion', slug: 'notion' }]
}

const MEMBER_POLICY = {
  layers: [{ body: { disabled_connectors: [], mode: 'deny', tools: {} }, kind: 'member', revision: 'rev-1' }]
}

const ACCOUNT = {
  active: true,
  connection_id: 'conn-1',
  connector: 'notion',
  created_at: '2026-01-02T03:04:05Z',
  label: 'ada@example.com',
  status: 'active',
  updated_at: '2026-01-02T03:04:05Z'
}

const TOOLS = {
  connector: 'notion',
  etag: 'e1',
  fetched_at: Math.floor(Date.now() / 1000),
  source: 'cache',
  stale: false,
  tools: [
    {
      categories: [],
      deprecated: false,
      description: 'Read a page.',
      facet: 'read',
      hints: [],
      name: 'Get page',
      slug: 'get_page'
    },
    {
      categories: [],
      deprecated: false,
      description: 'Write a page.',
      facet: 'write',
      hints: [],
      name: 'Add page',
      slug: 'add_page'
    }
  ],
  toolkit_version: '1'
}

const OPERATION = {
  deadline_at: Math.floor(Date.now() / 1000) + 600,
  op_id: 'op-1',
  seq: 1,
  settled: false,
  targets: [
    {
      action: 'connect',
      connect_url: 'https://example.invalid/authorize',
      kind: 'connector',
      name: 'notion',
      state: 'initiated'
    }
  ]
}

async function renderTab() {
  await act(async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/capabilities?tab=connectors']}>
          <ConnectorsTab gateway={null} profile="default" />
        </MemoryRouter>
      </QueryClientProvider>
    )
  })
}

/** Open one app's dialog through the card's own control. */
async function openCard(name: string) {
  fireEvent.click(await screen.findByRole('button', { name: new RegExp(`${name} — Open ${name}`) }))
}

beforeEach(() => {
  vi.stubGlobal('hermesDesktop', undefined)
  listConnectors.mockResolvedValue({ available: true, connectors: [{ connector: 'notion', enabled: true }] })
  connectorCatalog.mockResolvedValue(CATALOG)
  connectorAccounts.mockResolvedValue({ accounts: [] })
  connectorPolicy.mockResolvedValue(MEMBER_POLICY)
  connectorTools.mockResolvedValue(TOOLS)
  setConnectorPolicy.mockResolvedValue({ revision: 'rev-2' })
  removeConnectorAccount.mockResolvedValue({ connection_id: 'conn-1', status: 'removed' })
  connectAccountConnectors.mockResolvedValue(OPERATION)

  getHermesConfigRecord.mockResolvedValue({ mcp_servers: {} })
  getMcpCatalog.mockResolvedValue({ entries: [] })
  testMcpServer.mockResolvedValue({ ok: true, tools: [] })
  getUsageAnalytics.mockResolvedValue({ tools: [] })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.unstubAllGlobals()
  queryClient.clear()
  // The operation store is module state: a connect one test started would still
  // be the operation the next test's card finds.
  $accountOperations.set({})
})

describe('ConnectorsTab', { timeout: 60_000 }, () => {
  it('builds each card from the catalog, the hosted list and the accounts read', async () => {
    connectorAccounts.mockResolvedValue({ accounts: [ACCOUNT] })
    await renderTab()

    // The name comes from the catalog, the group from the account row.
    expect(await screen.findByRole('button', { name: /Notion — Open Notion/ })).toBeTruthy()
    await waitFor(() => expect(screen.getAllByText('Connected').length).toBeGreaterThan(0))
    expect(screen.getByText('Pages and databases.')).toBeTruthy()
  })

  it('Connect shows the connect element, and an update frame reconnects the list', async () => {
    await renderTab()

    fireEvent.click(await screen.findByRole('button', { name: 'Connect' }))

    await waitFor(() => expect(connectAccountConnectors).toHaveBeenCalledWith('default', ['notion'], false))

    // The shipped connect element, not a redrawn one: it carries the chat
    // card's own title and its waiting cue.
    expect(await screen.findByText('Connect your apps')).toBeTruthy()
    expect(screen.getAllByText(/Waiting for your browser/).length).toBeGreaterThan(0)

    const before = listConnectors.mock.calls.length

    await act(async () => {
      applyAccountConnectionUpdate({
        ...OPERATION,
        owner: { type: 'account' },
        seq: 2,
        settled: true,
        settled_by: 'all_resolved',
        targets: [{ action: 'connect', kind: 'connector', name: 'notion', state: 'connected' }]
      })
    })

    await waitFor(() => expect(listConnectors.mock.calls.length).toBeGreaterThan(before))
  })

  it('Save writes against the revision the person saw, and a conflict shows the conflict view', async () => {
    connectorAccounts.mockResolvedValue({ accounts: [ACCOUNT] })
    setConnectorPolicy.mockRejectedValue(new ConnectorRpcError('conflict', 4009, 'POLICY_CONFLICT'))
    // The re-read after a losing write: their version turned the other tool off.
    await renderTab()
    await openCard('Notion')

    const toggle = await screen.findByRole('switch', { name: 'Turn Get page off' })

    // Only now does the other editor's version exist: the conflict re-read is a
    // direct RPC, so this never reaches the cached policy the editor is using.
    connectorPolicy.mockResolvedValue({
      layers: [
        {
          body: { disabled_connectors: [], mode: 'deny', tools: { notion: ['add_page'] } },
          kind: 'member',
          revision: 'rev-9'
        }
      ]
    })

    fireEvent.click(toggle)
    fireEvent.click(await screen.findByRole('button', { name: 'Save changes' }))

    await waitFor(() =>
      expect(setConnectorPolicy).toHaveBeenCalledWith(
        'default',
        { connector: 'notion', disabled_tools: ['get_page'], type: 'tools' },
        'rev-1'
      )
    )

    expect(await screen.findByText('Someone changed this rule while you were editing.')).toBeTruthy()
  })

  it('sends the revision the last save produced, not the one still on screen', async () => {
    connectorAccounts.mockResolvedValue({ accounts: [ACCOUNT] })
    await renderTab()
    await openCard('Notion')

    fireEvent.click(await screen.findByRole('switch', { name: 'Turn Get page off' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Save changes' }))

    await waitFor(() =>
      expect(setConnectorPolicy).toHaveBeenLastCalledWith(
        'default',
        { connector: 'notion', disabled_tools: ['get_page'], type: 'tools' },
        'rev-1'
      )
    )

    // The policy re-read is a slow call and the screen still says rev-1. The
    // second write has to use what the first one answered, or the backend
    // reports the person's own save as somebody else's change.
    fireEvent.click(await screen.findByRole('switch', { name: 'Turn Add page off' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Save changes' }))

    await waitFor(() =>
      expect(setConnectorPolicy).toHaveBeenLastCalledWith(
        'default',
        { connector: 'notion', disabled_tools: ['get_page', 'add_page'], type: 'tools' },
        'rev-2'
      )
    )
  })

  it('names a backend too old for these reads instead of offering a Retry that cannot win', async () => {
    listConnectors.mockRejectedValue(new ConnectorRpcError('Method not found', -32601, undefined))

    await renderTab()

    await waitFor(() => expect(notifyError).toHaveBeenCalledWith(expect.anything(), 'Could not reach your Nous apps.'))
  })

  it('opens the dialog when Stop waiting has no operation in this window to stop', async () => {
    // The account says "pending" — a connect another window (or another run of
    // the app) started, so nothing here holds it.
    connectorAccounts.mockResolvedValue({ accounts: [{ ...ACCOUNT, status: 'pending' }] })
    await renderTab()

    fireEvent.click(await screen.findByRole('button', { name: 'Stop waiting' }))

    expect(await screen.findByRole('dialog', { name: /Notion/ })).toBeTruthy()
  })

  it('keeps the Disconnect confirm open, and says why, when the write is refused', async () => {
    connectorAccounts.mockResolvedValue({ accounts: [ACCOUNT] })
    removeConnectorAccount.mockRejectedValue(new ConnectorRpcError('The portal refused it.', 4001, undefined))
    await renderTab()
    await openCard('Notion')

    fireEvent.click(await screen.findByRole('button', { name: 'Disconnect' }))

    const confirm = await screen.findByRole('dialog', { name: 'Disconnect Notion?' })
    fireEvent.click(within(confirm).getByRole('button', { name: 'Disconnect' }))

    expect(await within(confirm).findByText('The portal refused it.')).toBeTruthy()
  })

  it('Disconnect asks first, then removes the account', async () => {
    connectorAccounts.mockResolvedValue({ accounts: [ACCOUNT] })
    await renderTab()
    await openCard('Notion')

    fireEvent.click(await screen.findByRole('button', { name: 'Disconnect' }))

    const confirm = await screen.findByRole('dialog', { name: 'Disconnect Notion?' })
    expect(removeConnectorAccount).not.toHaveBeenCalled()

    fireEvent.click(within(confirm).getByRole('button', { name: 'Disconnect' }))

    await waitFor(() => expect(removeConnectorAccount).toHaveBeenCalledWith('default', 'conn-1'))
  })

  it('keeps the servers on this Mac when the hosted half fails', async () => {
    listConnectors.mockRejectedValue(new ConnectorRpcError('no route', undefined, undefined))
    connectorCatalog.mockRejectedValue(new ConnectorRpcError('no route', undefined, undefined))
    getHermesConfigRecord.mockResolvedValue({ mcp_servers: { ctx7: { url: 'https://ctx7.invalid/mcp' } } })

    await renderTab()

    expect(await screen.findByRole('button', { name: /ctx7 — Open ctx7/ })).toBeTruthy()
    expect(screen.getByText('Could not reach your Nous apps.')).toBeTruthy()
  })
})
