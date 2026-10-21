import type { ConnectorsConnectParams } from '@hermes/shared'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setPrimaryGateway } from '@/store/gateway'

import type { AccountRow } from '../types'

import { accountAddress } from './join'
import { connectAccountConnectors } from './rpc'

function account(overrides: Partial<AccountRow>): AccountRow {
  return {
    active: false,
    alias: null,
    connection_id: 'ca_unnamed',
    connector: 'gmail',
    created_at: '2026-10-01T00:00:00Z',
    label: 'Gmail account',
    status: 'expired',
    updated_at: '2026-10-01T00:00:00Z',
    ...overrides
  }
}

/** The primary socket records every `connectors.connect` and never answers. */
function serveConnect() {
  const request = vi.fn((_method: string, _params: ConnectorsConnectParams) => new Promise(() => undefined))

  // SAFETY: the connector RPC layer calls only `request` on the primary socket in these tests.
  setPrimaryGateway({ request } as never)

  return () => request.mock.calls.filter(([method]) => method === 'connectors.connect').map(([, params]) => params)
}

afterEach(() => setPrimaryGateway(null))

describe('reconnect addresses the account the user picked', () => {
  it('names a named account by its id, so an unconfirmed rename cannot point the repair at another account', () => {
    expect(accountAddress(account({ alias: 'office', connection_id: 'ca_work' }))).toEqual({ connectionId: 'ca_work' })
  })

  it('names an unnamed account by its id, the only address it has', () => {
    expect(accountAddress(account({}))).toEqual({ connectionId: 'ca_unnamed' })
  })

  it('sends the id of an unnamed account with the reconnect', async () => {
    const calls = serveConnect()

    void connectAccountConnectors('default', ['gmail'], true, accountAddress(account({})))

    await vi.waitFor(() => expect(calls()).toHaveLength(1))
    expect(calls()[0]).toEqual({
      connection_id: 'ca_unnamed',
      connectors: ['gmail'],
      owner: { type: 'account' },
      reconnect: true
    })
  })

  it('sends a named account by name and no id, and a new account never carries an id', async () => {
    const calls = serveConnect()

    void connectAccountConnectors('default', ['gmail'], true, { alias: 'work' })
    void connectAccountConnectors('default', ['gmail'], false, { connectionId: 'ca_unnamed' })

    await vi.waitFor(() => expect(calls()).toHaveLength(2))
    expect(calls()[0]).toEqual({ alias: 'work', connectors: ['gmail'], owner: { type: 'account' }, reconnect: true })
    expect(calls()[1]).toEqual({ connectors: ['gmail'], owner: { type: 'account' }, reconnect: false })
  })
})
