// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'

import { TelemetrySettings } from './telemetry-settings'

const mocks = vi.hoisted(() => ({
  consent: vi.fn(),
  get: vi.fn(),
  notify: vi.fn(),
  notifyError: vi.fn(),
  openDir: vi.fn(),
  status: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getSharedMetricsConsent: (profile?: string) => mocks.get(profile),
  getStatus: () => mocks.status(),
  setApiRequestProfile: () => undefined,
  setSharedMetricsConsent: (consent: unknown, profile?: unknown) => mocks.consent(consent, profile)
}))

vi.mock('@/store/notifications', () => ({
  notify: (...args: unknown[]) => mocks.notify(...args),
  notifyError: (...args: unknown[]) => mocks.notifyError(...args)
}))

vi.mock('@/store/session', async () => {
  const { atom } = await import('nanostores')

  return { $connection: atom({ mode: 'local' }) }
})

vi.mock('@/store/settings-scope', async () => {
  const { atom } = await import('nanostores')

  return { $settingsScopeProfile: atom(undefined) }
})

const SWITCH = 'Share anonymous usage metrics with Nous Research'

function renderPage() {
  return render(
    <QueryClientProvider client={queryClient}>
      <TelemetrySettings />
    </QueryClientProvider>
  )
}

describe('TelemetrySettings', () => {
  beforeEach(() => {
    queryClient.clear()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openDir: (dir: string) => mocks.openDir(dir) }
    })
    mocks.get.mockResolvedValue({ enabled: false, send: false, decided: false, source: 'default' })
    mocks.status.mockResolvedValue({ hermes_home: '/home/u/.hermes' })
    mocks.consent.mockImplementation(async (c: { enabled: boolean; send: boolean }) => ({
      ok: true,
      ...c,
      decided: true,
      source: 'profile'
    }))
    mocks.openDir.mockResolvedValue({ ok: true })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('shows the effective (backend-resolved) state and turns sharing on through the consent route', async () => {
    renderPage()
    const toggle = await screen.findByRole('switch', { name: SWITCH })
    expect(toggle).toHaveProperty('ariaChecked', 'false')

    await act(async () => {
      fireEvent.click(toggle)
    })

    expect(mocks.consent).toHaveBeenCalledWith({ enabled: true, send: true }, undefined)
    expect(screen.getByRole('switch', { name: SWITCH })).toHaveProperty('ariaChecked', 'true')
    expect(mocks.notify).toHaveBeenCalled()
  })

  it('an inherited (global) answer paints as on and explains where it came from', async () => {
    mocks.get.mockResolvedValue({ enabled: true, send: true, decided: true, source: 'global' })
    renderPage()

    expect(await screen.findByRole('switch', { name: SWITCH })).toHaveProperty('ariaChecked', 'true')
    expect(screen.getByText(/follows the answer you gave when first asked/)).toBeTruthy()
  })

  it('turning sharing off writes both keys off and rolls back on failure', async () => {
    mocks.get.mockResolvedValue({ enabled: true, send: true, decided: true, source: 'profile' })
    mocks.consent.mockRejectedValueOnce(new Error('backend down'))
    renderPage()
    const toggle = await screen.findByRole('switch', { name: SWITCH })
    expect(toggle).toHaveProperty('ariaChecked', 'true')

    await act(async () => {
      fireEvent.click(toggle)
    })

    expect(mocks.consent).toHaveBeenCalledWith({ enabled: false, send: false }, undefined)
    expect(screen.getByRole('switch', { name: SWITCH })).toHaveProperty('ariaChecked', 'true')
    expect(mocks.notifyError).toHaveBeenCalled()
  })

  it('opens the profile telemetry folder under the backend hermes_home', async () => {
    renderPage()
    const button = await screen.findByRole('button', { name: /Open folder/ })

    await act(async () => {
      fireEvent.click(button)
    })

    expect(mocks.openDir).toHaveBeenCalledWith('/home/u/.hermes/telemetry/shared_metrics')
  })

  it('collect-only (CLI-set) shows as off with an explanatory note', async () => {
    mocks.get.mockResolvedValue({ enabled: true, send: false, decided: true, source: 'profile' })
    renderPage()

    expect(await screen.findByRole('switch', { name: SWITCH })).toHaveProperty('ariaChecked', 'false')
    expect(screen.getByText(/collects metrics locally but does not send them/i)).toBeTruthy()
  })
})
