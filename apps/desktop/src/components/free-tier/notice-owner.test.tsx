import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: vi.fn() })
}))

describe('useFreeTierNoticeOwner', () => {
  it('hands the notice to a composer that is still mounted when the owner unmounts', async () => {
    const { useFreeTierNoticeOwner } = await import('./notice-strip')
    const first = renderHook(() => useFreeTierNoticeOwner())
    const second = renderHook(() => useFreeTierNoticeOwner())
    expect(first.result.current).toBe(true)
    expect(second.result.current).toBe(false)

    act(() => first.unmount())

    expect(second.result.current).toBe(true)
    second.unmount()
  })
})
