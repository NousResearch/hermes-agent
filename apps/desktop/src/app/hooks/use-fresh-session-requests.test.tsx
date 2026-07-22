import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $freshSessionRequest, requestFreshSession } from '@/store/profile'

import { useFreshSessionRequests } from './use-fresh-session-requests'

function Harness({ onRequest }: { onRequest: () => void }) {
  useFreshSessionRequests(onRequest)

  return null
}

describe('useFreshSessionRequests', () => {
  afterEach(() => {
    cleanup()
    $freshSessionRequest.set(0)
    vi.restoreAllMocks()
  })

  it('clears the foreground synchronously before profile activation can continue', () => {
    const order: string[] = []
    const onRequest = vi.fn(() => order.push('foreground-cleared'))

    render(<Harness onRequest={onRequest} />)

    expect(onRequest).not.toHaveBeenCalled()

    act(() => {
      requestFreshSession()
      order.push('gateway-activation')
    })

    expect(onRequest).toHaveBeenCalledTimes(1)
    expect(order).toEqual(['foreground-cleared', 'gateway-activation'])
  })
})
