import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'
import { $desktopOnboarding } from '@/store/onboarding'
import {
  $sharedMetricsConsent,
  $sharedMetricsDetailsOpen,
  type SharedMetricsConsent,
  sharedMetricsOfferPending
} from '@/store/shared-metrics'

import { SharedMetricsConsentDialog } from './consent-dialog'

const copy = en.sharedMetrics

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

  return { calls, requestGateway }
}

const initialOnboarding = $desktopOnboarding.get()

beforeEach(() => {
  $desktopOnboarding.set({ ...initialOnboarding, configured: true })
})

afterEach(() => {
  cleanup()
  $desktopOnboarding.set(initialOnboarding)
  $sharedMetricsConsent.set(null)
  $sharedMetricsDetailsOpen.set(false)
})

describe('SharedMetricsConsentDialog', () => {
  it('never blocks launch: an undecided profile becomes a composer offer, not a modal', async () => {
    const undecided = backend({ enabled: false, send: false, decided: false })

    const { unmount } = render(
      <SharedMetricsConsentDialog enabled profile="default" requestGateway={undecided.requestGateway} />
    )

    await waitFor(() => expect(sharedMetricsOfferPending($sharedMetricsConsent.get())).toBe(true))
    expect(screen.queryByRole('dialog')).toBeNull()
    unmount()

    // An answer given in `hermes setup` is the same config keys: no offer at all.
    const answered = backend({ enabled: false, send: false, decided: true })
    render(<SharedMetricsConsentDialog enabled profile="work" requestGateway={answered.requestGateway} />)
    await waitFor(() => expect($sharedMetricsConsent.get()?.decided).toBe(true))
    expect(sharedMetricsOfferPending($sharedMetricsConsent.get())).toBe(false)
  })

  it('details decide nothing when closed and record a chosen answer as both opt-ins', async () => {
    const { calls, requestGateway } = backend({ enabled: false, send: false, decided: false })

    render(<SharedMetricsConsentDialog enabled profile="default" requestGateway={requestGateway} />)
    await waitFor(() => expect(sharedMetricsOfferPending($sharedMetricsConsent.get())).toBe(true))

    $sharedMetricsDetailsOpen.set(true)
    await screen.findByRole('dialog')
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(calls.some(c => c.method === 'shared_metrics.set')).toBe(false)
    expect(sharedMetricsOfferPending($sharedMetricsConsent.get())).toBe(true)

    $sharedMetricsDetailsOpen.set(true)
    const keepLocal = await screen.findByRole('button', { name: copy.local })
    // Nothing is preselected: no choice holds focus when the details open.
    expect(keepLocal.ownerDocument.activeElement).not.toBe(keepLocal)
    fireEvent.click(keepLocal)

    await waitFor(() => expect(sharedMetricsOfferPending($sharedMetricsConsent.get())).toBe(false))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(calls.find(c => c.method === 'shared_metrics.set')?.params).toEqual({
      enabled: true,
      send: false,
      first_run: true
    })
  })
})
