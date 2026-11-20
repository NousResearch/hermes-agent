import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The Hermes Webapp is this same renderer served to a browser tab: the browser bridge installs
// `window.hermesDesktop` (openExternal -> window.open) and marks the document as a browser host.
// These tests mount the real dialog on that host to pin that the offer reaches Webapp users.

const requestGateway = vi.fn()

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))
vi.mock('@/store/windows', () => ({
  isSecondaryWindow: () => false,
  isBrowserWindow: () => false,
  isHudWindow: () => false
}))

import { isBrowserHostedDesktop } from '@/lib/platform'
import { $composerDraft } from '@/store/composer'
import { $freeTierStatus } from '@/store/free-tier'
import { $freeTierSignIn } from '@/store/free-tier-sign-in'
import { IDLE_SETTLE_MS, resetInAppOfferStateForTest } from '@/store/in-app-offer'
import { markOnboardingDecided, setOnboardingSurfaceActive } from '@/store/onboarding-presence'
import { $busy } from '@/store/session'
import type { FreeTierStatus } from '@/types/hermes'

import { InAppOfferDialog } from './dialog'

const OFFER = {
  id: 'offer_web',
  title: 'Welcome back',
  body: 'Line one.\n\nLine two.',
  cta_label: 'Claim the offer',
  cta_url: 'https://portal.nousresearch.com/billing'
}

const status = (offer: unknown = OFFER): FreeTierStatus => ({
  has_guest: false,
  enabled: true,
  available: true,
  notice_pending: false,
  model: 'nous/welcome',
  label: 'Nous',
  in_app_offer: offer as FreeTierStatus['in_app_offer']
})

let openSpy: ReturnType<typeof vi.fn>

beforeEach(() => {
  vi.useFakeTimers()
  localStorage.clear()
  requestGateway.mockReset().mockRejectedValue(new Error('not under test'))
  $busy.set(false)
  $composerDraft.set('')
  $freeTierSignIn.set({ status: 'closed' })
  markOnboardingDecided()
  setOnboardingSurfaceActive('onboarding', false)
  resetInAppOfferStateForTest()
  vi.spyOn(window.document, 'hasFocus').mockReturnValue(true)
  // The browser bridge's shape, as browser-desktop-bridge.ts installs it.
  openSpy = vi.fn()
  vi.stubGlobal('open', openSpy)
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
    openExternal: async (url: string) => {
      window.open(url, '_blank', 'noopener,noreferrer')
    }
  }
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
})

afterEach(() => {
  cleanup()
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
  delete window.document.documentElement.dataset.hermesDesktopHost
  $freeTierStatus.set(null)
  vi.unstubAllGlobals()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('InAppOfferDialog on the Webapp (browser host)', () => {
  it('opens on an idle moment and its CTA opens the offer in a new tab', async () => {
    expect(isBrowserHostedDesktop()).toBe(true)
    $freeTierStatus.set(status())
    render(<InAppOfferDialog />)

    await act(async () => vi.advanceTimersByTimeAsync(IDLE_SETTLE_MS))

    expect(screen.getByText('Welcome back')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Claim the offer' }))

    expect(openSpy).toHaveBeenCalledWith(OFFER.cta_url, '_blank', 'noopener,noreferrer')
    // The Portal's campaign counts: shown when it opened, clicked on the CTA.
    expect(requestGateway).toHaveBeenCalledWith('in_app_offer.report', { event: 'shown', offer_id: OFFER.id })
    expect(requestGateway).toHaveBeenCalledWith('in_app_offer.report', { event: 'clicked', offer_id: OFFER.id })
    expect(screen.queryByText('Welcome back')).toBeNull()
    expect(localStorage.getItem('hermes.in-app-offer.dismissed-v1')).toContain(OFFER.id)
  })

  it('stays closed while the user is typing in the composer', async () => {
    $composerDraft.set('half a sentence')
    $freeTierStatus.set(status())
    render(<InAppOfferDialog />)

    await act(async () => vi.advanceTimersByTimeAsync(IDLE_SETTLE_MS * 3))

    expect(screen.queryByText('Welcome back')).toBeNull()
  })

  it('does not put focus on the close button when it opens', async () => {
    $freeTierStatus.set(status())
    render(<InAppOfferDialog />)

    await act(async () => vi.advanceTimersByTimeAsync(IDLE_SETTLE_MS))

    const focused = window.document.activeElement as HTMLElement | null
    expect(focused?.getAttribute('aria-label') ?? '').not.toMatch(/close/i)
  })
})
