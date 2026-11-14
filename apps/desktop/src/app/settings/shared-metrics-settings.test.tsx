// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as GatewayStore from '@/store/gateway'
import type { SharedMetricsConsent } from '@/store/shared-metrics'

const requestGatewayForAgent = vi.fn()

// The real gateway store, with only the per-agent request routed to the fake backend below.
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  requestGatewayForAgent: (...args: unknown[]) => requestGatewayForAgent(...args)
}))
vi.mock('@/store/desktop-metrics', () => ({ recordOnboarding: vi.fn(), setDesktopMetricsGate: vi.fn() }))

const { en } = await import('@/i18n/en')
const { SharedMetricsSettings } = await import('./shared-metrics-settings')

const copy = en.sharedMetrics

function backend(initial: SharedMetricsConsent) {
  let stored = initial

  requestGatewayForAgent.mockImplementation(
    async (_connection: unknown, _profile: unknown, method: string, params: Record<string, unknown> = {}) => {
      if (method === 'shared_metrics.set') {
        const enabled = params.enabled === true
        stored = { enabled, send: enabled && params.send === true, decided: true }
      }

      return stored
    }
  )
}

const setCalls = () =>
  requestGatewayForAgent.mock.calls
    .filter(call => call[2] === 'shared_metrics.set')
    .map(call => ({ enabled: call[3].enabled, send: call[3].send }))

afterEach(() => {
  cleanup()
  requestGatewayForAgent.mockReset()
})

describe('SharedMetricsSettings', () => {
  it('is one switch: on shares (both opt-ins), off clears both', async () => {
    backend({ enabled: false, send: false, decided: true })

    render(<SharedMetricsSettings />)

    const share = await screen.findByRole('switch', { name: copy.sendLabel })
    await waitFor(() => expect(share.hasAttribute('disabled')).toBe(false))
    expect(screen.getAllByRole('switch')).toHaveLength(1)
    expect(share.getAttribute('aria-checked')).toBe('false')

    fireEvent.click(share)
    await waitFor(() => expect(share.getAttribute('aria-checked')).toBe('true'))

    fireEvent.click(share)
    await waitFor(() => expect(share.getAttribute('aria-checked')).toBe('false'))

    expect(setCalls()).toEqual([
      { enabled: true, send: true },
      { enabled: false, send: false }
    ])
  })

  it('reads a retired collect-only answer as not sharing', async () => {
    backend({ enabled: true, send: false, decided: true })

    render(<SharedMetricsSettings />)

    const share = await screen.findByRole('switch', { name: copy.sendLabel })
    await waitFor(() => expect(share.hasAttribute('disabled')).toBe(false))
    expect(share.getAttribute('aria-checked')).toBe('false')
  })
})
