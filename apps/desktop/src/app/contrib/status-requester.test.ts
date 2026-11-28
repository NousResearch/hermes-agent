import { renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { requestGatewayForAgent } from '@/store/gateway'
import { $shellChoice } from '@/store/shell-variant'

import { useStatusRequester } from './status-requester'
import type { GatewayRequester } from './types'

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn().mockResolvedValue({})
}))

afterEach(() => {
  $shellChoice.set(null)
  vi.clearAllMocks()
})

it.each([
  [false, 'the app gateway, as main does'],
  [true, 'the selected gateway and profile by name']
])('polls status (redesign=%s) through %s', async redesign => {
  $shellChoice.set(redesign)
  const requestGateway = vi.fn().mockResolvedValue({}) as unknown as GatewayRequester
  const { result } = renderHook(() => useStatusRequester(requestGateway))

  await result.current('status.get', {})

  expect(vi.mocked(requestGateway)).toHaveBeenCalledTimes(redesign ? 0 : 1)
  expect(requestGatewayForAgent).toHaveBeenCalledTimes(redesign ? 1 : 0)
})
