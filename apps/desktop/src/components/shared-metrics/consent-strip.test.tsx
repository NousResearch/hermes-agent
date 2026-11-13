import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { en } from '@/i18n/en'
import { bindDesktopMetrics, captureDesktopMetricsGate, resetDesktopMetricsForTests } from '@/store/desktop-metrics'
import type * as GatewayStore from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $sharedMetricsConsent, type SharedMetricsConsent, sharedMetricsOfferPending } from '@/store/shared-metrics'

import { SharedMetricsConsentStrip } from './consent-strip'

const requestGateway = vi.fn()

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  requestGatewayForAgent: (...args: unknown[]) => requestGateway(...args)
}))

const copy = en.sharedMetrics
const initialProfile = $activeGatewayProfile.get()
const initialScope = $apiRequestScope.get()

function installGatewayMock(initial: SharedMetricsConsent) {
  let stored = initial

  requestGateway.mockImplementation(async (_connection: unknown, _profile: unknown, method: string, params?: Record<string, unknown>) => {
    if (method === 'shared_metrics.set') {
      const enabled = params?.enabled === true
      stored = { enabled, send: enabled && params?.send === true, decided: true }
    }

    return stored
  })
}

beforeEach(() => {
  $apiRequestScope.set({ connectionId: null, profile: 'havoc' })
  resetDesktopMetricsForTests()
  bindDesktopMetrics(null, '|havoc')
  $activeGatewayProfile.set('havoc')
  $sharedMetricsConsent.set({ enabled: false, send: false, decided: false })
})

afterEach(() => {
  cleanup()
  resetDesktopMetricsForTests()
  vi.unstubAllGlobals()
  $apiRequestScope.set(initialScope)
  requestGateway.mockReset()
  $activeGatewayProfile.set(initialProfile)
  $sharedMetricsConsent.set(null)
})

describe('SharedMetricsConsentStrip', () => {
  it('retires the focused profile offer after saving both sharing opt-ins', async () => {
    installGatewayMock({ enabled: false, send: false, decided: false })

    render(<SharedMetricsConsentStrip />)

    fireEvent.click(screen.getByRole('button', { name: copy.stripChoices.share }))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(null, 'havoc', 'shared_metrics.set', {
        enabled: true,
        send: true,
        first_run: true,
        profile: 'havoc'
      })
    )
    await waitFor(() => expect(sharedMetricsOfferPending($sharedMetricsConsent.get())).toBe(false))
  })

  it('saves only the clicked connection when consent capture waits across a switch', async () => {
    const unanswered = { enabled: false, send: false, decided: false }

    const consentByConnection = new Map([
      ['source', { ...unanswered }],
      ['target', { ...unanswered }]
    ])

    requestGateway.mockImplementation(async (connection: string, _profile: string, method: string, params: Record<string, unknown>) => {
      if (method === 'shared_metrics.set') {
        consentByConnection.set(connection, {
          enabled: params.enabled === true,
          send: params.send === true,
          decided: true
        })
      }

      return consentByConnection.get(connection)
    })
    const entered = Promise.withResolvers<void>()
    const release = Promise.withResolvers<void>()
    vi.stubGlobal('navigator', {
      locks: {
        request: async (_name: string, run: () => unknown) => {
          entered.resolve()
          await release.promise

          return run()
        }
      }
    })
    $apiRequestScope.set({ connectionId: 'source', profile: 'havoc' })
    bindDesktopMetrics(null, 'source|havoc')
    render(<SharedMetricsConsentStrip />)
    fireEvent.click(screen.getByRole('button', { name: copy.stripChoices.share }))
    await entered.promise

    act(() => {
      $apiRequestScope.set({ connectionId: 'target', profile: 'havoc' })
      bindDesktopMetrics(null, 'target|havoc')
    })
    const targetWasInteractive = !screen.getByRole('button', { name: copy.stripChoices.share }).hasAttribute('disabled')
    release.resolve()
    expect(targetWasInteractive).toBe(true)

    await waitFor(() => expect(consentByConnection.get('source')?.decided).toBe(true))
    await captureDesktopMetricsGate()
    expect(consentByConnection.get('target')).toEqual(unanswered)
    expect(sharedMetricsOfferPending($sharedMetricsConsent.get())).toBe(true)
  })
})
