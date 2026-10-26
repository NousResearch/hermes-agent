import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import { noteFreeTierTurnComplete } from '@/store/free-tier-sign-in'
import { setPrimaryGateway, setPrimaryGatewayConnectionId } from '@/store/gateway'

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
