import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $settingsScopeOverride } from '@/store/settings-scope'
import type { SharedMetricsConsent } from '@/store/shared-metrics'

import { SharedMetricsSettings } from './shared-metrics-settings'

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

const initialProfile = $activeGatewayProfile.get()
const copy = en.sharedMetrics

function installGatewayMock(initial: SharedMetricsConsent) {
  let stored = initial

  vi.mocked(requestGatewayForAgent).mockImplementation(async (_connectionId, _profile, method, params) => {
    if (method === 'shared_metrics.set') {
      const enabled = params?.enabled === true
      stored = { enabled, send: enabled && params?.send === true, decided: true }
    }

    return stored as never
  })
}

beforeEach(() => {
  $activeGatewayProfile.set('havoc')
  $settingsScopeOverride.set(null)
})

afterEach(() => {
  cleanup()
  vi.mocked(requestGatewayForAgent).mockReset()
  $settingsScopeOverride.set(null)
  $activeGatewayProfile.set(initialProfile)
})

describe('SharedMetricsSettings', () => {
  it('reads and writes shared metrics consent for the settings profile', async () => {
    installGatewayMock({ enabled: false, send: false, decided: false })

    render(<SharedMetricsSettings />)

    await waitFor(() =>
      expect(requestGatewayForAgent).toHaveBeenCalledWith(
        null,
        'havoc',
        'shared_metrics.status',
        { profile: 'havoc' },
        undefined,
        undefined,
        { spawnPriority: 'foreground' }
      )
    )

    fireEvent.click(screen.getByRole('switch', { name: copy.collectLabel }))

    await waitFor(() =>
      expect(requestGatewayForAgent).toHaveBeenCalledWith(
        null,
        'havoc',
        'shared_metrics.set',
        { enabled: true, send: false, first_run: false, profile: 'havoc' },
        undefined,
        undefined,
        { spawnPriority: 'foreground' }
      )
    )
  })
})
