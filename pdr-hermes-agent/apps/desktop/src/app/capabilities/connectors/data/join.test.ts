import type { ConnectorCatalogRow } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import type { AccountRow } from '../types'

import { EMPTY_POLICY, joinHostedConnectors } from './join'

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
