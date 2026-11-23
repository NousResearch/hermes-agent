import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { en } from '@/i18n/en'
import { bindDesktopMetrics, resetDesktopMetricsForTests } from '@/store/desktop-metrics'
import type * as GatewayStore from '@/store/gateway'
import { $desktopOnboarding } from '@/store/onboarding'
import { markOnboardingDecided } from '@/store/onboarding-presence'
import {
  $sharedMetricsConsent,
  $sharedMetricsDetailsOpen,
  type SharedMetricsConsent,
  sharedMetricsOfferPending
} from '@/store/shared-metrics'

import { SharedMetricsConsentDialog } from './consent-dialog'

// Onboarding's due check answered "not due": first-run surfaces are free to show.
markOnboardingDecided()

const copy = en.sharedMetrics
const initialScope = $apiRequestScope.get()
const requestGatewayForAgent = vi.fn()

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  requestGatewayForAgent: (...args: unknown[]) => requestGatewayForAgent(...args)
}))

function backend(initial: SharedMetricsConsent) {
  const calls: { method: string; params?: Record<string, unknown> }[] = []
  let stored = initial

  const requestGateway = async <T,>(method: string, params?: Record<string, unknown>): Promise<T> => {
    calls.push({ method, params })

    if (method === 'shared_metrics.set') {
      const enabled = params?.enabled === true
      stored = { enabled, send: enabled && params?.send === true, decided: true }
    }

    return stored as T
  }

  requestGatewayForAgent.mockImplementation(
    (_connection: unknown, _profile: unknown, method: string, params?: Record<string, unknown>) =>
      requestGateway(method, params)
  )

  return { calls }
}

const initialOnboarding = $desktopOnboarding.get()

beforeEach(() => {
  $apiRequestScope.set({ connectionId: null, profile: null })
  resetDesktopMetricsForTests()
  bindDesktopMetrics(null, '|havoc')
  $desktopOnboarding.set({ ...initialOnboarding, configured: true })
})

afterEach(() => {
  cleanup()
  resetDesktopMetricsForTests()
  requestGatewayForAgent.mockReset()
  $apiRequestScope.set(initialScope)
  $desktopOnboarding.set(initialOnboarding)
  $sharedMetricsConsent.set(null)
  $sharedMetricsDetailsOpen.set(false)
})

describe('SharedMetricsConsentDialog', () => {
  it('never blocks launch: an undecided profile becomes a composer offer, not a modal', async () => {
    backend({ enabled: false, send: false, decided: false })

    const { unmount } = render(<SharedMetricsConsentDialog enabled profile="default" />)

    await waitFor(() => expect(sharedMetricsOfferPending($sharedMetricsConsent.get())).toBe(true))
    expect(screen.queryByRole('dialog')).toBeNull()
    unmount()

    // An answer given in `hermes setup` is the same config keys: no offer at all.
    backend({ enabled: false, send: false, decided: true })
    render(<SharedMetricsConsentDialog enabled profile="work" />)
    await waitFor(() => expect($sharedMetricsConsent.get()?.decided).toBe(true))
    expect(sharedMetricsOfferPending($sharedMetricsConsent.get())).toBe(false)
  })

  it('details decide nothing when closed and record a chosen answer as both opt-ins', async () => {
    const { calls } = backend({ enabled: false, send: false, decided: false })

    render(<SharedMetricsConsentDialog enabled profile="havoc" />)
    await waitFor(() => expect(sharedMetricsOfferPending($sharedMetricsConsent.get())).toBe(true))
    expect(calls.find(c => c.method === 'shared_metrics.status')?.params).toEqual({ profile: 'havoc' })

    act(() => $sharedMetricsDetailsOpen.set(true))
    await screen.findByRole('dialog')
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(calls.some(c => c.method === 'shared_metrics.set')).toBe(false)
    expect(sharedMetricsOfferPending($sharedMetricsConsent.get())).toBe(true)

    act(() => $sharedMetricsDetailsOpen.set(true))
    const noThanks = await screen.findByRole('button', { name: copy.off })
    expect(screen.getByRole('button', { name: copy.share })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /computer|local/i })).toBeNull()
    // Nothing is preselected: no choice holds focus when the details open.
    expect(noThanks.ownerDocument.activeElement).not.toBe(noThanks)
    fireEvent.click(noThanks)

    await waitFor(() => expect(sharedMetricsOfferPending($sharedMetricsConsent.get())).toBe(false))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(calls.find(c => c.method === 'shared_metrics.set')?.params).toEqual({
      enabled: false,
      send: false,
      first_run: true,
      profile: 'havoc'
    })
  })
})
