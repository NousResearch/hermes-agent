// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import { setPrimaryGateway } from '@/store/gateway'

import { renameLinkTarget } from '../hosted-dialog'
import type { AccountRow } from '../types'

import { accountRenameRoute, claimRouteLink, resetRouteLinksForTests } from './deep-link'
import { connectorsAccountsQueryKey } from './keys'
import { useConnectConnector } from './mutations'

const row = (connectionId: string, alias: string): AccountRow => ({
  active: true,
  alias,
  connection_id: connectionId,
  connector: 'gmail',
  created_at: '2026-10-01T00:00:00Z',
  label: 'Gmail account',
  status: 'active',
  updated_at: '2026-10-01T00:00:00Z'
})

afterEach(() => {
  resetRouteLinksForTests()
  setPrimaryGateway(null)
  vi.restoreAllMocks()
})

describe('the chat Rename link goes by account id', () => {
  it('carries the id, not the name the account had then', () => {
    const params = new URLSearchParams(
      accountRenameRoute('gmail', 'ca_work', { connectionId: null, profile: 'a' }).split('?')[1]
    )

    expect(params.get('rename')).toBe('ca_work')
  })

  it('opens the account with that id, never one that took its old name, and reports a missing one', () => {
    const pending = { connectionId: 'ca_work', slug: 'gmail' }
    const reused = [row('ca_new', 'work'), row('ca_work', 'office')]

    expect(renameLinkTarget(pending, 'gmail', reused, true)).toEqual(reused[1])
    expect(renameLinkTarget(pending, 'gmail', [row('ca_new', 'work')], true)).toBe('gone')
    expect(renameLinkTarget(pending, 'gmail', [], false)).toBeNull()
    expect(renameLinkTarget(pending, 'slack', reused, true)).toBeNull()
  })
})

describe('a link is acted on once, by the page on its scope', () => {
  it('leaves a link for another profile alone and lets only the first mounted page take it', () => {
    const params = new URLSearchParams({ connector: 'gmail', profile: 'b', rename: 'ca_b' })

    expect(claimRouteLink('k1', params, 'a')).toBe(false)
    expect(claimRouteLink('k1', params, 'b')).toBe(true)
    expect(claimRouteLink('k1', params, 'b')).toBe(false)
    expect(claimRouteLink('k2', params, 'b')).toBe(true)
  })
})

describe('starting an add or a reconnect', () => {
  it('refreshes the account list when the start returns, before the operation settles', async () => {
    const request = vi.fn(async () => ({ deadline_at: 0, op_id: 'op1', seq: 1, settled: false, targets: [] }))

    // SAFETY: the connector RPC layer calls only `request` on the primary socket here.
    setPrimaryGateway({ request } as never)
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries')
    const { result } = renderHook(() => useConnectConnector('default'))

    await act(async () => {
      await result.current.connect('gmail', { alias: 'work' })
    })

    expect(invalidate).toHaveBeenCalledWith({ queryKey: connectorsAccountsQueryKey('default') })
  })
})
