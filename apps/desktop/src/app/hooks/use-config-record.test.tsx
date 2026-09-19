import { QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import { queryClient } from '@/lib/query-client'

import { useHermesConfigRecord } from './use-config-record'

describe('useHermesConfigRecord writeScope ownership', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    queryClient.clear()
    api = vi.fn(async (request: { method?: string; connectionId?: string; profile?: string }) => {
      if (request.method === 'PUT') {
        return { ok: true }
      }

      const connectionId = String(request.connectionId ?? '').trim()
      const profile = String(request.profile ?? '').trim()

      return { model: `from-${connectionId || 'local'}` }
    })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
    setApiRequestConnection(null)
    setApiRequestProfile(null)
  })

  afterEach(() => {
    queryClient.clear()
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('two observers sharing a key retain the route paired with the cached record', async () => {
    setApiRequestConnection('connection-a')
    setApiRequestProfile('default')

    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    )

    const { result } = renderHook(() => useHermesConfigRecord(), { wrapper })

    await waitFor(() => expect(result.current.data?.model).toBe('from-connection-a'))
    expect(result.current.writeScope).toEqual({ connectionId: 'connection-a', profile: 'default' })

    setApiRequestConnection('connection-b')
    const second = renderHook(() => useHermesConfigRecord(), { wrapper })

    expect(second.result.current.data?.model).toBe('from-connection-a')
    expect(second.result.current.writeScope).toEqual({ connectionId: 'connection-a', profile: 'default' })

    await act(async () => {
      await result.current.refetch()
    })

    await waitFor(() => expect(second.result.current.data?.model).toBe('from-connection-b'))
    expect(second.result.current.writeScope).toEqual({ connectionId: 'connection-b', profile: 'default' })
  })
})
