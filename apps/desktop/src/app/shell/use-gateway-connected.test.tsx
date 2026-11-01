// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'

import { useGatewayConnected } from './use-gateway-connected'

describe('useGatewayConnected', () => {
  it('tracks closed and reopened gateway states', () => {
    let state: 'closed' | 'open' = 'closed'
    let listener: ((next: 'closed' | 'open') => void) | undefined
    const unsubscribe = vi.fn()

    const gateway = {
      get connectionState() {
        return state
      },
      onState(next: (value: 'closed' | 'open') => void) {
        listener = next
        next(state)

        return unsubscribe
      }
    } as unknown as HermesGateway

    const { result, unmount } = renderHook(() => useGatewayConnected(gateway))

    expect(result.current).toBe(false)

    act(() => {
      state = 'open'
      listener?.('open')
    })
    expect(result.current).toBe(true)

    act(() => {
      state = 'closed'
      listener?.('closed')
    })
    expect(result.current).toBe(false)

    unmount()
    expect(unsubscribe).toHaveBeenCalledOnce()
  })

  it('treats the legacy request path without an explicit gateway as connected', () => {
    const { result } = renderHook(() => useGatewayConnected(undefined))

    expect(result.current).toBe(true)
  })
})
