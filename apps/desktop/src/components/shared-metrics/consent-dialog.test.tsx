import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'
import { $desktopOnboarding } from '@/store/onboarding'
import type { SharedMetricsConsent } from '@/store/shared-metrics'

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
})

describe('SharedMetricsConsentDialog', () => {
  it('stays hidden when the profile already answered (e.g. in hermes setup)', async () => {
    const { calls, requestGateway } = backend({ enabled: false, send: false, decided: true })

    render(<SharedMetricsConsentDialog enabled profile="default" requestGateway={requestGateway} />)

    await waitFor(() => expect(calls.map(c => c.method)).toContain('shared_metrics.status'))
    expect(screen.queryByText(copy.consentTitle)).toBeNull()
  })

  it('asks an undecided profile once and records the answer as both opt-ins', async () => {
    const { calls, requestGateway } = backend({ enabled: false, send: false, decided: false })

    render(<SharedMetricsConsentDialog enabled profile="default" requestGateway={requestGateway} />)

    const keepLocal = await screen.findByRole('button', { name: copy.local })
    // Nothing is preselected: no choice holds focus when the dialog opens.
    expect(keepLocal.ownerDocument.activeElement).not.toBe(keepLocal)

    fireEvent.click(keepLocal)

    await waitFor(() => expect(screen.queryByText(copy.consentTitle)).toBeNull())
    expect(calls.find(c => c.method === 'shared_metrics.set')?.params).toEqual({
      enabled: true,
      send: false,
      first_run: true
    })
  })
})
