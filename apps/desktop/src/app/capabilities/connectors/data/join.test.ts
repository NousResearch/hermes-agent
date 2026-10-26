import type { ConnectorCatalogRow, McpServerRuntimeRow, McpServerSummary } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { localWay } from '../derive'
import type { AccountRow } from '../types'

import { EMPTY_POLICY, joinHostedConnectors, pluginServerRows } from './join'

const GMAIL: ConnectorCatalogRow = { category: 'mail', description: 'Mail', name: 'Gmail', slug: 'gmail' }

function account(overrides: Partial<AccountRow>): AccountRow {
  return {
    active: true,
    alias: null,
    connection_id: 'ca_1',
    connector: 'gmail',
    created_at: '2026-10-01T00:00:00Z',
    label: 'Gmail account',
    status: 'active',
    updated_at: '2026-10-01T00:00:00Z',
    ...overrides
  }
}

const join = (accounts: AccountRow[]) =>
  joinHostedConnectors({ accounts, catalog: [GMAIL], list: [], policy: EMPTY_POLICY })[0]

describe('joinHostedConnectors with several accounts', () => {
  it('keeps one row per account, the active one first, then the newest', () => {
    const row = join([
      account({ alias: 'home', connection_id: 'ca_home', created_at: '2026-10-01T00:00:00Z' }),
      account({
        active: false,
        alias: 'old',
        connection_id: 'ca_old',
        status: 'expired',
        created_at: '2026-10-09T00:00:00Z'
      }),
      account({ alias: 'work', connection_id: 'ca_work', created_at: '2026-10-05T00:00:00Z' })
    ])

    expect(row.accounts?.map(entry => entry.connection_id)).toEqual(['ca_work', 'ca_home', 'ca_old'])
  })

  it('names the summary by the account alias, falling back to the vendor label', () => {
    expect(join([account({ alias: 'work' })]).accountLabel).toBe('work')
    expect(join([account({ alias: null, label: 'me@example.com' })]).accountLabel).toBe('me@example.com')
  })

  it('folds a retired account out of the list and the summary, and still reports it', () => {
    const row = join([
      account({ alias: 'work', connection_id: 'ca_new', created_at: '2026-10-01T00:00:00Z' }),
      account({
        active: false,
        connection_id: 'ca_replaced',
        created_at: '2026-10-09T00:00:00Z',
        disabled: true,
        status: 'inactive'
      })
    ])

    expect(row.accounts?.map(entry => entry.connection_id)).toEqual(['ca_new'])
    expect(row.retiredAccounts?.map(entry => entry.connection_id)).toEqual(['ca_replaced'])
    expect(row.connectionStatus).toBe('active')
  })
})

function pluginServer(overrides: Partial<McpServerSummary>): McpServerSummary {
  return {
    args: [],
    enabled: true,
    env: [],
    name: 'sigrex',
    plugin: 'sigrex',
    source: 'plugin',
    transport: 'http',
    url: 'https://mcp.example',
    ...overrides
  }
}

const RUNTIME_FAILED: McpServerRuntimeRow = {
  connected: false,
  disabled: false,
  name: 'sigrex',
  plugin: 'sigrex',
  source: 'plugin',
  status: 'failed',
  tools: 0,
  transport: 'http'
}

describe('pluginServerRows', () => {
  it('offers Authenticate for a plugin server that declares OAuth and holds no token', () => {
    const [row] = pluginServerRows({
      runtime: [RUNTIME_FAILED],
      servers: [pluginServer({ auth: 'oauth', oauth_tokens_present: false })]
    })

    expect(row.status).toBe('needs-auth')
    expect(localWay(row).verb).toBe('authenticate')
  })

  it('reports a signed-in plugin server by its runtime state', () => {
    const [row] = pluginServerRows({
      runtime: [{ ...RUNTIME_FAILED, connected: true, status: 'connected', tools: 12 }],
      servers: [pluginServer({ auth: 'oauth', oauth_tokens_present: true })]
    })

    expect(row.status).toBe('ok')
    expect(localWay(row).verb).toBeUndefined()
  })

  it('never offers Authenticate for a plugin server without OAuth', () => {
    const [row] = pluginServerRows({ runtime: [RUNTIME_FAILED], servers: [pluginServer({})] })

    expect(row.status).toBe('error')
    expect(row.canAuthenticate).toBe(false)
  })
})
