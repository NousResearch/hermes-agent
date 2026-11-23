import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { writeKey } from '@/lib/storage'
import { $localModelOfferPending, LOCAL_MODEL_PENDING_KEY } from '@/store/local-model-offer'
import { $localSetupEligibility } from '@/store/local-setup-offer'
import { markOnboardingDecided } from '@/store/onboarding-presence'
import { $onboardingSettleFrom } from '@/store/onboarding-run'
import { $connection } from '@/store/session'
import type { LocalCatalogModel } from '@/types/hermes'

import { OnboardingEnd } from './onboarding-end'

const TITLE = 'This computer can run a model locally'
// Past the longest settle window plus one tick.
const SETTLED_MS = 10 * 60_000 + 30_000

const MODEL: LocalCatalogModel = {
  id: 'Qwen3.6-27B-UD-Q4_K_XL',
  display_name: 'Qwen3.6 27B',
  description: 'Best all-round agent model',
  size_bytes: 17.6 * 2 ** 30,
  size_label: '17.6 GB',
  native_context: 262144,
  native_context_label: '256K',
  recommended: true,
  downloaded: false,
  mtp: false,
  fits: true,
  fit_summary: 'runs at its full 256K context'
}

const LOCAL: HermesConnection = {
  baseUrl: 'http://127.0.0.1:1',
  isFullscreen: false,
  logs: [],
  mode: 'local',
  nativeOverlayWidth: 0,
  token: '',
  windowButtonPosition: null,
  wsUrl: 'ws://127.0.0.1:1'
}

beforeEach(() => {
  vi.useFakeTimers()
  // The quiet check: a focused, visible window with the boot onboarding check answered.
  vi.spyOn(window.document, 'hasFocus').mockReturnValue(true)
  vi.spyOn(window.document, 'visibilityState', 'get').mockReturnValue('visible')
  markOnboardingDecided()
  $connection.set(LOCAL)
  // Set after the connection, whose change drops the cached answer.
  $localSetupEligibility.set({ checkedAt: Date.now(), fit: { model: MODEL }, reason: 'test' })
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
  window.localStorage.clear()
  $onboardingSettleFrom.set(null)
  $connection.set(null)
})

describe('OnboardingEnd', () => {
  it('offers the local model once the settle window passes on a quiet app, once per install', async () => {
    render(<OnboardingEnd />)

    act(() => $onboardingSettleFrom.set(Date.now()))
    await act(() => vi.advanceTimersByTimeAsync(4 * 60_000))
    expect(screen.queryByText(TITLE)).toBeNull()

    await act(() => vi.advanceTimersByTimeAsync(SETTLED_MS))
    expect(screen.getByText(TITLE)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }))
    expect(screen.queryByText(TITLE)).toBeNull()

    act(() => $onboardingSettleFrom.set(Date.now()))
    await act(() => vi.advanceTimersByTimeAsync(SETTLED_MS))
    expect(screen.queryByText(TITLE)).toBeNull()
  })

  it('resumes the settle wait after a restart that came before the offer opened', async () => {
    const first = render(<OnboardingEnd />)

    act(() => $onboardingSettleFrom.set(Date.now()))
    first.unmount()
    act(() => $onboardingSettleFrom.set(null))

    render(<OnboardingEnd />)
    await act(() => vi.advanceTimersByTimeAsync(SETTLED_MS))
    expect(screen.getByText(TITLE)).toBeTruthy()
  })

  it('keeps sign-in waiting through a restart that mounts before the connection is up', async () => {
    writeKey(LOCAL_MODEL_PENDING_KEY, String(Date.now()))
    $connection.set(null)
    $localModelOfferPending.set(true)

    render(<OnboardingEnd />)
    await act(() => vi.advanceTimersByTimeAsync(0))
    expect($localModelOfferPending.get()).toBe(true)

    // The machine it connects to cannot run a local model: now the sign-in offer may go first.
    act(() => $connection.set({ ...LOCAL, mode: 'remote' }))
    await act(() => vi.advanceTimersByTimeAsync(0))
    expect($localModelOfferPending.get()).toBe(false)
  })
})
