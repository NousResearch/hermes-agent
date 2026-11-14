import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $agentPlugins, $agentPluginsOwnerKey, $agentPluginsProfile, $agentPluginsStatus } from '@/store/agent-plugins'
import { $gatewayState } from '@/store/session'

import { useConnectorsCount, usePluginsCount } from './capability-counts'

const { request, connectors, config } = vi.hoisted(() => ({
  request: vi.fn(),
  connectors: { current: {} as Record<string, unknown> },
  config: { current: {} as Record<string, unknown> }
}))

vi.mock('../settings/settings-owner-request', () => ({
  useSettingsGatewayRequest: () => Object.assign(request, { ownerKey: '["local","ops"]' })
}))
vi.mock('./connectors/data/queries', () => ({ useConnectorsListQuery: () => connectors.current }))
vi.mock('../hooks/use-config-record', () => ({ useHermesConfigRecord: () => config.current }))

beforeEach(() => {
  $gatewayState.set('open')
})

afterEach(() => {
  // Unmount first: a mounted count would reload into the reset store.
  cleanup()
  request.mockReset()
  $agentPlugins.set([])
  $agentPluginsStatus.set('idle')
  $agentPluginsOwnerKey.set(undefined)
  $agentPluginsProfile.set(undefined)
})

const plugin = (key: string, source: string, status: 'enabled' | 'not enabled') => ({
  name: key,
  key,
  version: '1',
  description: '',
  source,
  status
})

it('an agent whose backend cannot answer settles as failed instead of loading forever, and Retry loads it', async () => {
  request.mockRejectedValueOnce(Object.assign(new Error('profile_parked'), { code: 'profile_parked' }))
  const { result } = renderHook(() => usePluginsCount('ops'))

  await waitFor(() => expect(result.current.failed).toBe(true))
  expect(result.current.value).toBeUndefined()
  expect(request).toHaveBeenCalledTimes(1)

  request.mockResolvedValueOnce({ plugins: [plugin('user/thing', 'user', 'enabled')] })
  act(() => result.current.retry())

  await waitFor(() => expect(result.current.value).toBe(1))
  expect(result.current.failed).toBe(false)
})

it('counts only the plugins the Plugins page lists that are on: never the bundled providers', async () => {
  request.mockResolvedValueOnce({
    plugins: [
      plugin('model-providers/openai', 'bundled', 'enabled'),
      plugin('platforms/telegram', 'bundled', 'enabled'),
      plugin('disk-cleanup', 'bundled', 'not enabled'),
      plugin('security-guidance', 'bundled', 'enabled'),
      plugin('user/linter', 'user', 'enabled')
    ]
  })
  const { result } = renderHook(() => usePluginsCount('ops'))

  await waitFor(() => expect(result.current.value).toBe(2))
})

it('connectors settle as failed when either read fails, not as an endless skeleton', () => {
  connectors.current = { data: undefined, isError: true, refetch: vi.fn() }
  config.current = { data: { mcp_servers: {} }, isError: false, refetch: vi.fn() }
  const { result, rerender } = renderHook(() => useConnectorsCount('ops'))

  expect(result.current.failed).toBe(true)
  expect(result.current.value).toBeUndefined()

  connectors.current = { data: { available: true, connectors: [{ connected: true }, { connected: false }] }, isError: false, refetch: vi.fn() }
  rerender()
  expect(result.current).toMatchObject({ failed: false, value: 1 })
})
