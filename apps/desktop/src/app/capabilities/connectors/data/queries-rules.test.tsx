import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import { $freeTierTurnCompleted, noteFreeTierTurnComplete } from '@/store/free-tier-sign-in'
import { setPrimaryGateway, setPrimaryGatewayConnectionId } from '@/store/gateway'

import { connectorsListQueryKey } from './keys'
import { useHostedConnectors } from './queries'

const row = { connected: false, connector: 'notion', enabled: true, gateway_disabled_tools: [] }

function gateway(listed: object[]) {
  const request = vi.fn(
    async (method: string) =>
      ({
        'connectors.accounts': { accounts: [] },
        'connectors.catalog': { connectors: [] },
        'connectors.list': { available: true, connectors: [...listed] },
        'connectors.policy.get': { effective: { mode: 'unrestricted' }, layers: [] }
      })[method]
  )

  // SAFETY: the page's queries call only `request` on the primary socket.
  setPrimaryGateway({ request } as never)
  setPrimaryGatewayConnectionId('connection-1')

  return request
}

// The page invalidates through the app's one query client, so the hook runs on it too.
function mount() {
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )

  return renderHook(() => useHostedConnectors('default'), { wrapper })
}

const asked = (request: ReturnType<typeof gateway>, method: string) =>
  request.mock.calls.some(([name]) => name === method)

afterEach(() => {
  cleanup()
  queryClient.clear()
})

describe('connector rules on the Connectors page', () => {
  it('are not asked for while the list is empty (no identity yet)', async () => {
    const request = gateway([])
    const { result } = mount()

    await waitFor(() => expect(result.current.phase).toBe('ready'))
    await waitFor(() => expect(asked(request, 'connectors.catalog')).toBe(true))

    expect(asked(request, 'connectors.policy.get')).toBe(false)
    expect(result.current.rulesFailed).toBe(false)
  })

  it('are asked for once the gateway lists apps', async () => {
    const request = gateway([row])

    mount()

    await waitFor(() => expect(asked(request, 'connectors.policy.get')).toBe(true))
  })

  it('are asked for once an identity appears outside the page (an agent connect in a chat)', async () => {
    const listed: object[] = []
    const request = gateway(listed)
    const { result } = mount()

    await waitFor(() => expect(result.current.phase).toBe('ready'))
    listed.push(row)
    noteFreeTierTurnComplete()

    await waitFor(() => expect(asked(request, 'connectors.policy.get')).toBe(true))
    // The accounts read cached while there was no identity is re-read too, or the new account stays hidden.
    expect(request.mock.calls.filter(([name]) => name === 'connectors.accounts')).toHaveLength(2)
  })
})

// The backend arms the sign-in offer on the first list read that shows a guest an app connected
// (an authorization past the watch, a connect by link). The page re-reads the offer timing then.
describe('a connected app first seen on a list read', () => {
  it('re-reads the sign-in offer timing', async () => {
    gateway([{ ...row, connected: true }])
    const before = $freeTierTurnCompleted.get()

    mount()

    await waitFor(() => expect($freeTierTurnCompleted.get()).toBe(before + 1))
  })

  it('waits for the backend read when the page starts from saved connected rows', async () => {
    const request = gateway([{ ...row, connected: true }])
    queryClient.setQueryData(connectorsListQueryKey('default'), { available: true, connectors: [{ ...row, connected: true }] }, { updatedAt: 1 })
    const before = $freeTierTurnCompleted.get()

    mount()
    expect($freeTierTurnCompleted.get()).toBe(before)

    await waitFor(() => expect(asked(request, 'connectors.list')).toBe(true))
    await waitFor(() => expect($freeTierTurnCompleted.get()).toBe(before + 1))
  })

  it('does not when nothing is connected', async () => {
    const request = gateway([row])
    const before = $freeTierTurnCompleted.get()
    const { result } = mount()

    await waitFor(() => expect(result.current.phase).toBe('ready'))
    await waitFor(() => expect(asked(request, 'connectors.policy.get')).toBe(true))

    expect($freeTierTurnCompleted.get()).toBe(before)
  })
})
