import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $freeTierStatus } from '@/store/free-tier'
import {
  $inAppOfferDialog,
  dismissInAppOffer,
  markInAppOfferShown,
  resetInAppOfferStateForTest,
  syncInAppOffer
} from '@/store/in-app-offer'
import {
  markOnboardingDecided,
  setOnboardingSurfaceActive
} from '@/store/onboarding-presence'

// The offer store rides the free-tier status the ambient snapshot already
// refreshes; these tests pin the presentation contract: show once, dismiss
// permanently, cap frequency, never during onboarding.

vi.mock('@/store/windows', () => ({
  isSecondaryWindow: () => false,
  isBrowserWindow: () => true,
  isHudWindow: () => false
}))

const OFFER = {
  id: 'offer_1',
  title: 'Welcome back',
  body: 'Here is an offer.',
  cta_label: 'See the offer',
  cta_url: 'https://portal.nousresearch.com/billing'
}

function statusWith(offer: unknown = OFFER) {
  return { has_guest: false, enabled: true, available: true, notice_pending: false, model: 'nous/welcome', label: 'Nous', in_app_offer: offer }
}

beforeEach(() => {
  localStorage.clear()
  $freeTierStatus.set(null)
  markOnboardingDecided()
  setOnboardingSurfaceActive('onboarding', false)
  resetInAppOfferStateForTest()
})

describe('syncInAppOffer', () => {
  it('opens the dialog for a valid offer on the status', () => {
    syncInAppOffer(statusWith())

    expect($inAppOfferDialog.get()).toEqual(OFFER)
  })

  it('ignores a status without an offer', () => {
    syncInAppOffer({ has_guest: false, enabled: false, available: false, notice_pending: false, model: 'nous/welcome', label: 'Nous' })

    expect($inAppOfferDialog.get()).toBeNull()
  })

  it('ignores a malformed offer', () => {
    syncInAppOffer(statusWith({ id: 'x', title: 'T', body: 'B', cta_label: 'C' }))

    expect($inAppOfferDialog.get()).toBeNull()
  })

  it('does not re-open an offer the dialog already shows', () => {
    syncInAppOffer(statusWith())
    const shown = $inAppOfferDialog.get()

    syncInAppOffer(statusWith())

    expect($inAppOfferDialog.get()).toBe(shown)
  })

  it('holds the offer back while onboarding owns the screen', () => {
    setOnboardingSurfaceActive('onboarding', true)

    syncInAppOffer(statusWith())

    expect($inAppOfferDialog.get()).toBeNull()
  })
})

describe('dismissInAppOffer', () => {
  it('closes the dialog and never re-opens that offer', () => {
    syncInAppOffer(statusWith())
    dismissInAppOffer(OFFER.id)

    expect($inAppOfferDialog.get()).toBeNull()

    syncInAppOffer(statusWith())

    expect($inAppOfferDialog.get()).toBeNull()
  })

  it('persists the dismissal across a renderer restart', () => {
    dismissInAppOffer(OFFER.id)

    expect(localStorage.getItem('hermes.in-app-offer.dismissed-v1')).toContain(OFFER.id)
  })

  it('lets a different offer through after the frequency cap clears', () => {
    syncInAppOffer(statusWith())
    markInAppOfferShown()
    dismissInAppOffer(OFFER.id)

    const next = { ...OFFER, id: 'offer_2' }
    syncInAppOffer(statusWith(next))

    // Inside the 24h cap: nothing shows.
    expect($inAppOfferDialog.get()).toBeNull()

    localStorage.removeItem('hermes.in-app-offer.last-shown-v1')
    syncInAppOffer(statusWith(next))

    expect($inAppOfferDialog.get()).toEqual(next)
  })
})

describe('the frequency cap', () => {
  it('blocks a second campaign inside a day of the last shown offer', () => {
    syncInAppOffer(statusWith())
    markInAppOfferShown()
    dismissInAppOffer(OFFER.id)

    const next = { ...OFFER, id: 'offer_2' }
    syncInAppOffer(statusWith(next))

    expect($inAppOfferDialog.get()).toBeNull()
  })
})
