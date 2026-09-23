// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { TelemetryConsentPrompt } from './telemetry-consent-prompt'

interface OnboardingSlice {
  configured: boolean
  manual: boolean
  requested: boolean
}

const mocks = vi.hoisted(() => ({
  get: vi.fn(),
  invalidate: vi.fn(),
  notifyError: vi.fn(),
  set: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getSharedMetricsConsent: (profile?: string) => mocks.get(profile),
  getStatus: async () => ({}),
  setApiRequestProfile: () => undefined,
  setSharedMetricsConsent: (consent: unknown, profile?: string) => mocks.set(consent, profile)
}))

vi.mock('@/store/onboarding-gate', async () => {
  const { atom } = await import('nanostores')

  return { $onboardingGate: atom({ phase: 'idle', guideQueued: false }), guidedOnboardingActive: () => false }
})
vi.mock('@/store/notifications', () => ({ notifyError: (...a: unknown[]) => mocks.notifyError(...a) }))
vi.mock('../app/settings/telemetry-settings', () => ({
  invalidateSharedMetricsConsent: (p?: string) => mocks.invalidate(p)
}))

const { $onboarding } = vi.hoisted(() => {
  // Real nanostores atom, built inside the hoisted block so the mock factory can see it.
  let value: OnboardingSlice = { configured: true, manual: false, requested: false }
  const listeners = new Set<(v: OnboardingSlice) => void>()

  return {
    $onboarding: {
      get: () => value,
      listen: (fn: (v: OnboardingSlice) => void) => {
        listeners.add(fn)

        return () => listeners.delete(fn)
      },
      set: (next: OnboardingSlice) => {
        value = next
        listeners.forEach(fn => fn(next))
      },
      subscribe: (fn: (v: OnboardingSlice) => void) => {
        fn(value)
        listeners.add(fn)

        return () => listeners.delete(fn)
      }
    }
  }
})

vi.mock('@/store/onboarding', () => ({ $desktopOnboarding: $onboarding }))

const settled = { configured: true, manual: false, requested: false }
let profileCounter = 0
// Each test gets its own profile name: the component latches "asked" per profile per run.
const freshProfile = () => `p${++profileCounter}`

describe('TelemetryConsentPrompt', () => {
  beforeEach(() => {
    $onboarding.set(settled)
    mocks.get.mockResolvedValue({ enabled: false, send: false, decided: false, source: 'default' })
    mocks.set.mockImplementation(async (c: { enabled: boolean; send: boolean }) => ({
      ok: true,
      ...c,
      decided: true,
      source: 'profile'
    }))
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('asks when telemetry is off and undecided, and records Share', async () => {
    const profile = freshProfile()
    render(<TelemetryConsentPrompt enabled profile={profile} />)
    const share = await screen.findByRole('button', { name: 'Share' })

    await act(async () => {
      fireEvent.click(share)
    })

    expect(mocks.set).toHaveBeenCalledWith({ enabled: true, send: true }, profile)
    expect(mocks.invalidate).toHaveBeenCalledWith(profile)
  })

  it("records a decline for Don't share", async () => {
    const profile = freshProfile()
    render(<TelemetryConsentPrompt enabled profile={profile} />)
    const decline = await screen.findByRole('button', { name: "Don't share" })

    await act(async () => {
      fireEvent.click(decline)
    })

    expect(mocks.set).toHaveBeenCalledWith({ enabled: false, send: false }, profile)
  })

  it('stays silent once the backend holds the global answer', async () => {
    mocks.get.mockResolvedValue({ enabled: false, send: false, decided: true, source: 'profile' })
    render(<TelemetryConsentPrompt enabled profile={freshProfile()} />)
    await act(async () => {})

    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('does not ask while onboarding still owns the screen or the gateway is down', async () => {
    $onboarding.set({ configured: false, manual: false, requested: true })
    const { rerender } = render(<TelemetryConsentPrompt enabled profile={freshProfile()} />)
    await act(async () => {})
    expect(mocks.get).not.toHaveBeenCalled()

    $onboarding.set(settled)
    rerender(<TelemetryConsentPrompt enabled={false} profile={freshProfile()} />)
    await act(async () => {})
    expect(mocks.get).not.toHaveBeenCalled()
  })

  it('asks a profile at most once per app run even if the read fails', async () => {
    mocks.get.mockRejectedValueOnce(new Error('offline'))
    const profile = freshProfile()
    const { rerender } = render(<TelemetryConsentPrompt enabled profile={profile} />)
    await act(async () => {})
    rerender(<TelemetryConsentPrompt enabled={false} profile={profile} />)
    rerender(<TelemetryConsentPrompt enabled profile={profile} />)
    await act(async () => {})

    expect(mocks.get).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('dialog')).toBeNull()
  })
})
