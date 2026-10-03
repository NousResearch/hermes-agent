import { cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { requestGatewayForAgent } from '@/store/gateway'
import { $sharedMetricsConsent, type SharedMetricsConsent } from '@/store/shared-metrics'

import { useDesktopMetrics } from './use-desktop-metrics'

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

const CONSENT: SharedMetricsConsent = { enabled: true, send: false, decided: true }

afterEach(() => {
  cleanup()
  vi.mocked(requestGatewayForAgent).mockReset()
  $sharedMetricsConsent.set(null)
})

describe('useDesktopMetrics', () => {
  it('reads the shared metrics consent from the focused profile', async () => {
    vi.mocked(requestGatewayForAgent).mockResolvedValue(CONSENT as never)

    renderHook(() =>
      useDesktopMetrics({
        enabled: true,
        gatewayOpen: true,
        pathname: '/settings',
        profile: 'havoc'
      })
    )

    await waitFor(() =>
      expect(requestGatewayForAgent).toHaveBeenCalledWith(null, 'havoc', 'shared_metrics.status', {
        profile: 'havoc'
      })
    )
  })
})
