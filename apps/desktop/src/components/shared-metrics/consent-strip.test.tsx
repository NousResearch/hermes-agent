import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $activeGatewayProfile } from '@/store/profile'
import { $sharedMetricsConsent, type SharedMetricsConsent } from '@/store/shared-metrics'

import { SharedMetricsConsentStrip } from './consent-strip'

const requestGateway = vi.fn()

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

const copy = en.sharedMetrics
const initialProfile = $activeGatewayProfile.get()

function installGatewayMock(initial: SharedMetricsConsent) {
  let stored = initial

  requestGateway.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'shared_metrics.set') {
      const enabled = params?.enabled === true
      stored = { enabled, send: enabled && params?.send === true, decided: true }
    }

    return stored
  })
}

beforeEach(() => {
  $activeGatewayProfile.set('havoc')
  $sharedMetricsConsent.set({ enabled: false, send: false, decided: false })
})

afterEach(() => {
  cleanup()
  requestGateway.mockReset()
  $activeGatewayProfile.set(initialProfile)
  $sharedMetricsConsent.set(null)
})

describe('SharedMetricsConsentStrip', () => {
  it('records the chosen answer on the focused profile', async () => {
    installGatewayMock({ enabled: false, send: false, decided: false })

    render(<SharedMetricsConsentStrip />)

    fireEvent.click(screen.getByRole('button', { name: copy.stripChoices.local }))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('shared_metrics.set', {
        enabled: true,
        send: false,
        first_run: true,
        profile: 'havoc'
      })
    )
  })
})