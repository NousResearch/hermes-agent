import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'
import { $sharedMetricsConsent } from '@/store/shared-metrics'

import { useDesktopMetrics } from './use-desktop-metrics'

const request = vi.hoisted(() => vi.fn())
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: request
}))
vi.mock('@/app/routes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  contributedRoutes: () => []
}))

beforeEach(() => {
  $connection.set({
    connectionId: 'local',
    mode: 'local',
    baseUrl: 'http://localhost:4242',
    wsUrl: 'ws://localhost:4242',
    token: 'test',
    logs: [],
    isFullscreen: false,
    nativeOverlayWidth: 0,
    windowButtonPosition: null
  })
  request.mockResolvedValue({ enabled: false, send: false, decided: true })
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  $sharedMetricsConsent.set(null)
  vi.clearAllMocks()
})

it.each(['default', 'work', 'custom'])('refreshes consent through the focused %s owning route', async profile => {
  renderHook(() => useDesktopMetrics({ enabled: true, gatewayOpen: true, pathname: '/', profile }))
  await waitFor(() =>
    expect(request).toHaveBeenCalledWith('local', profile, 'shared_metrics.status', {
      profile: profile === 'custom' ? undefined : profile
    })
  )
  const statusBefore = request.mock.calls.filter(call => call[2] === 'shared_metrics.status').length
  act(() => window.dispatchEvent(new Event('focus')))
  await waitFor(() =>
    expect(request.mock.calls.filter(call => call[2] === 'shared_metrics.status')).toHaveLength(statusBefore + 1)
  )
  expect(
    request.mock.calls.filter(call => call[2] === 'shared_metrics.status').every(call => call[1] === profile)
  ).toBe(true)
})

it('re-reads the new profile without reusing the old request scope', async () => {
  const view = renderHook(
    ({ profile }) => useDesktopMetrics({ enabled: true, gatewayOpen: true, pathname: '/', profile }),
    {
      initialProps: { profile: 'default' }
    }
  )

  await waitFor(() =>
    expect(request).toHaveBeenCalledWith('local', 'default', 'shared_metrics.status', { profile: 'default' })
  )
  view.rerender({ profile: 'work' })
  await waitFor(() =>
    expect(request).toHaveBeenCalledWith('local', 'work', 'shared_metrics.status', { profile: 'work' })
  )
})
