import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $composerDraft } from '@/store/composer'
import { $freeTierStatus, type FreeTierRequester } from '@/store/free-tier'
import { $freeTierSignIn } from '@/store/free-tier-sign-in'
import {
  $inAppOfferDialog,
  appIsIdle,
  claimInAppOffer,
  dismissInAppOffer,
  eligibleOffer,
  FIRST_REFRESH_DELAY_MS,
  IDLE_SETTLE_MS,
  inAppOfferClaim,
  listenInAppOfferStatus,
  markInAppOfferShown,
  releaseInAppOffer,
  reportInAppOfferEvent,
  resetInAppOfferStateForTest,
  startInAppOfferRefresh,
  syncInAppOffer
} from '@/store/in-app-offer'
import { markOnboardingDecided, setOnboardingSurfaceActive } from '@/store/onboarding-presence'
import { $busy } from '@/store/session'
import type { FreeTierStatus } from '@/types/hermes'

// These tests pin the presentation contract: show once, only on an idle
// moment, never over another first-run or sign-in surface, one window,
// permanent dismissal, a daily cap, and withdrawal when the Portal stops
// serving the offer.

vi.mock('@/store/windows', () => ({
  isSecondaryWindow: () => false,
  isBrowserWindow: () => false,
  isHudWindow: () => false
}))

const OFFER = {
  id: 'offer_1',
  title: 'Welcome back',
  body: 'Here is an offer.',
  cta_label: 'See the offer',
  cta_url: 'https://portal.nousresearch.com/billing'
}

function statusWith(offer: unknown = OFFER): FreeTierStatus {
  return {
    has_guest: false,
    enabled: true,
    available: true,
    notice_pending: false,
    model: 'nous/welcome',
    label: 'Nous',
    in_app_offer: offer as FreeTierStatus['in_app_offer']
  }
}

const IDLE = { idle: true }

beforeEach(() => {
  localStorage.clear()
  $freeTierStatus.set(null)
  $busy.set(false)
  $composerDraft.set('')
  $freeTierSignIn.set({ status: 'closed' })
  markOnboardingDecided()
  setOnboardingSurfaceActive('onboarding', false)
  resetInAppOfferStateForTest()
  vi.spyOn(document, 'hasFocus').mockReturnValue(true)
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('syncInAppOffer', () => {
  it('opens the dialog for a valid offer on an idle moment', () => {
    syncInAppOffer(statusWith(), IDLE)

    expect($inAppOfferDialog.get()).toEqual(OFFER)
  })

  it('never opens while the app is not idle', () => {
    syncInAppOffer(statusWith(), { idle: false })

    expect($inAppOfferDialog.get()).toBeNull()
  })

  it('ignores a status without an offer', () => {
    syncInAppOffer(statusWith(null), IDLE)

    expect($inAppOfferDialog.get()).toBeNull()
  })

  it('ignores a malformed offer', () => {
    syncInAppOffer(statusWith({ id: 'x', title: 'T', body: 'B', cta_label: 'C' }), IDLE)

    expect($inAppOfferDialog.get()).toBeNull()
  })

  it('holds the offer back while onboarding owns the screen', () => {
    setOnboardingSurfaceActive('onboarding', true)

    syncInAppOffer(statusWith(), IDLE)

    expect($inAppOfferDialog.get()).toBeNull()
  })

  it('holds the offer back while the sign-in dialog is open', () => {
    $freeTierSignIn.set({ status: 'offer' } as ReturnType<typeof $freeTierSignIn.get>)

    syncInAppOffer(statusWith(), IDLE)

    expect($inAppOfferDialog.get()).toBeNull()
  })

  it('keeps an open dialog through later ticks, idle or not', () => {
    syncInAppOffer(statusWith(), IDLE)
    const shown = $inAppOfferDialog.get()

    syncInAppOffer(statusWith(), { idle: false })

    expect($inAppOfferDialog.get()).toBe(shown)
  })

  it('withdraws an open dialog when the Portal stops serving the offer', () => {
    syncInAppOffer(statusWith(), IDLE)

    syncInAppOffer(statusWith(null), { idle: false })

    expect($inAppOfferDialog.get()).toBeNull()
  })
})

describe('appIsIdle', () => {
  it('is false during a turn, with a draft, or with the window unfocused', () => {
    expect(appIsIdle()).toBe(true)

    $busy.set(true)
    expect(appIsIdle()).toBe(false)
    $busy.set(false)

    $composerDraft.set('half a sentence')
    expect(appIsIdle()).toBe(false)
    $composerDraft.set('   ')
    expect(appIsIdle()).toBe(true)

    vi.spyOn(document, 'hasFocus').mockReturnValue(false)
    expect(appIsIdle()).toBe(false)
  })
})

describe('listenInAppOfferStatus', () => {
  it('opens only after the app has stayed idle for the settle window', () => {
    vi.useFakeTimers()
    $freeTierStatus.set(statusWith())
    const stop = listenInAppOfferStatus()

    vi.advanceTimersByTime(IDLE_SETTLE_MS - 1)
    expect($inAppOfferDialog.get()).toBeNull()

    vi.advanceTimersByTime(1)
    expect($inAppOfferDialog.get()).toEqual(OFFER)

    stop()
  })

  it('restarts the settle clock when the user starts typing', () => {
    vi.useFakeTimers()
    $freeTierStatus.set(statusWith())
    const stop = listenInAppOfferStatus()

    vi.advanceTimersByTime(IDLE_SETTLE_MS - 1)
    $composerDraft.set('h')
    vi.advanceTimersByTime(IDLE_SETTLE_MS * 3)
    expect($inAppOfferDialog.get()).toBeNull()

    // Sent and cleared: idle again, and the clock starts over.
    $composerDraft.set('')
    vi.advanceTimersByTime(IDLE_SETTLE_MS)
    expect($inAppOfferDialog.get()).toEqual(OFFER)

    stop()
  })

  it('never opens during a running turn', () => {
    vi.useFakeTimers()
    $busy.set(true)
    $freeTierStatus.set(statusWith())
    const stop = listenInAppOfferStatus()

    vi.advanceTimersByTime(IDLE_SETTLE_MS * 5)
    expect($inAppOfferDialog.get()).toBeNull()

    stop()
  })
})

describe('offer validation', () => {
  it.each([
    ['http', 'http://portal.nousresearch.com/x'],
    ['a local path', '/Users/me/secret'],
    ['file', 'file:///etc/passwd'],
    ['javascript', 'javascript:alert(1)']
  ])('never shows a %s CTA (the external opener opens local paths)', (_label, url) => {
    expect(eligibleOffer(statusWith({ ...OFFER, cta_url: url }))).toBeNull()
  })

  it('never shows an offer with a blank title', () => {
    expect(eligibleOffer(statusWith({ ...OFFER, title: '  ' }))).toBeNull()
  })
})

describe('dismissInAppOffer', () => {
  it('closes the dialog and never re-opens that offer', () => {
    syncInAppOffer(statusWith(), IDLE)
    dismissInAppOffer(OFFER.id)

    expect($inAppOfferDialog.get()).toBeNull()

    localStorage.removeItem('hermes.in-app-offer.last-shown-v1')
    syncInAppOffer(statusWith(), IDLE)

    expect($inAppOfferDialog.get()).toBeNull()
  })

  it('persists the dismissal across a renderer restart', () => {
    dismissInAppOffer(OFFER.id)

    expect(localStorage.getItem('hermes.in-app-offer.dismissed-v1')).toContain(OFFER.id)
  })

  it('honours a dismissal another window wrote to shared storage', () => {
    localStorage.setItem('hermes.in-app-offer.dismissed-v1', JSON.stringify([OFFER.id]))

    syncInAppOffer(statusWith(), IDLE)

    expect($inAppOfferDialog.get()).toBeNull()
  })

  it("keeps another window's dismissals when recording a new one", () => {
    localStorage.setItem('hermes.in-app-offer.dismissed-v1', JSON.stringify(['offer_other']))

    dismissInAppOffer(OFFER.id)

    const stored = localStorage.getItem('hermes.in-app-offer.dismissed-v1') ?? ''
    expect(stored).toContain('offer_other')
    expect(stored).toContain(OFFER.id)
  })
})

describe('the frequency cap', () => {
  it('blocks a second campaign inside a day of the last shown offer', () => {
    syncInAppOffer(statusWith(), IDLE)
    markInAppOfferShown()
    dismissInAppOffer(OFFER.id)

    syncInAppOffer(statusWith({ ...OFFER, id: 'offer_2' }), IDLE)

    expect($inAppOfferDialog.get()).toBeNull()
  })

  it('lets a different offer through once the cap clears', () => {
    markInAppOfferShown()
    localStorage.removeItem('hermes.in-app-offer.last-shown-v1')

    const next = { ...OFFER, id: 'offer_2' }
    syncInAppOffer(statusWith(next), IDLE)

    expect($inAppOfferDialog.get()).toEqual(next)
  })
})

describe('the window claim', () => {
  it('gives the dialog to the first host only', () => {
    claimInAppOffer('window-a')
    claimInAppOffer('window-b')

    expect(inAppOfferClaim().get()).toBe('window-a')
  })

  it('hands it over when the owning host unmounts, and closes its dialog', () => {
    claimInAppOffer('window-a')
    syncInAppOffer(statusWith(), IDLE)

    releaseInAppOffer('window-a')

    expect(inAppOfferClaim().get()).toBeNull()
    expect($inAppOfferDialog.get()).toBeNull()
  })
})

describe('startInAppOfferRefresh', () => {
  it('fetches after the boot delay and folds the answer into the status', async () => {
    vi.useFakeTimers()
    $freeTierStatus.set(statusWith(null))
    const request = vi.fn().mockResolvedValue({ in_app_offer: OFFER })
    const stop = startInAppOfferRefresh(request)

    expect(request).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(FIRST_REFRESH_DELAY_MS)

    expect(request).toHaveBeenCalledWith('in_app_offer.refresh')
    expect($freeTierStatus.get()?.in_app_offer).toEqual(OFFER)

    stop()
  })

  it('clears the offer when the refresh says there is none', async () => {
    vi.useFakeTimers()
    $freeTierStatus.set(statusWith())
    const stop = startInAppOfferRefresh(vi.fn().mockResolvedValue({}))

    await vi.advanceTimersByTimeAsync(FIRST_REFRESH_DELAY_MS)

    expect($freeTierStatus.get()?.in_app_offer).toBeNull()

    stop()
  })

  it('drops an answer that arrives after the status changed (a profile switch)', async () => {
    vi.useFakeTimers()
    $freeTierStatus.set(statusWith(null))
    let answer: (value: unknown) => void = () => undefined
    const request = vi.fn(() => new Promise(resolve => (answer = resolve)))
    const stop = startInAppOfferRefresh(request as unknown as FreeTierRequester)

    await vi.advanceTimersByTimeAsync(FIRST_REFRESH_DELAY_MS)
    // Profile B's status lands while profile A's refresh is in flight.
    const profileB = statusWith(null)
    $freeTierStatus.set(profileB)
    answer({ in_app_offer: OFFER })
    await vi.advanceTimersByTimeAsync(0)

    expect($freeTierStatus.get()).toBe(profileB)

    stop()
  })

  it('leaves the status object alone when the offer did not change', async () => {
    vi.useFakeTimers()
    const status = statusWith()
    $freeTierStatus.set(status)
    const stop = startInAppOfferRefresh(vi.fn().mockResolvedValue({ in_app_offer: OFFER }))

    await vi.advanceTimersByTimeAsync(FIRST_REFRESH_DELAY_MS)

    // A fresh object would re-arm every status listener (the sign-in nudge).
    expect($freeTierStatus.get()).toBe(status)

    stop()
  })

  it('stays quiet when the backend lacks the method', async () => {
    vi.useFakeTimers()
    $freeTierStatus.set(statusWith())
    const stop = startInAppOfferRefresh(vi.fn().mockRejectedValue(new Error('method not found')))

    await vi.advanceTimersByTimeAsync(FIRST_REFRESH_DELAY_MS)

    // The last known answer is untouched.
    expect($freeTierStatus.get()?.in_app_offer).toEqual(OFFER)

    stop()
  })
})

describe('reportInAppOfferEvent', () => {
  it('reports each event once per offer on this install', async () => {
    const requestGateway = vi.fn().mockResolvedValue({ reported: true })

    reportInAppOfferEvent(requestGateway, 'offer_1', 'shown')
    // In flight: a re-render sends nothing more.
    reportInAppOfferEvent(requestGateway, 'offer_1', 'shown')
    await vi.waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(1))
    await Promise.resolve()
    reportInAppOfferEvent(requestGateway, 'offer_1', 'shown')
    reportInAppOfferEvent(requestGateway, 'offer_1', 'clicked')
    reportInAppOfferEvent(requestGateway, 'offer_2', 'shown')

    expect(requestGateway.mock.calls).toEqual([
      ['in_app_offer.report', { event: 'shown', offer_id: 'offer_1' }],
      ['in_app_offer.report', { event: 'clicked', offer_id: 'offer_1' }],
      ['in_app_offer.report', { event: 'shown', offer_id: 'offer_2' }]
    ])
  })

  it('does not re-report after a renderer restart (the record is in storage)', async () => {
    const first = vi.fn().mockResolvedValue({ reported: true })
    reportInAppOfferEvent(first, 'offer_1', 'shown')
    await vi.waitFor(() => expect(first).toHaveBeenCalled())
    await new Promise(resolve => setTimeout(resolve, 0))

    const afterRestart = vi.fn().mockResolvedValue({ reported: true })
    reportInAppOfferEvent(afterRestart, 'offer_1', 'shown')

    expect(afterRestart).not.toHaveBeenCalled()
  })

  it('sends a report again when the Portal did not take it (an expired token)', async () => {
    const lost = vi.fn().mockResolvedValue({ reported: false })
    reportInAppOfferEvent(lost, 'offer_1', 'shown')
    await new Promise(resolve => setTimeout(resolve, 0))

    const retry = vi.fn().mockResolvedValue({ reported: true })
    reportInAppOfferEvent(retry, 'offer_1', 'shown')

    expect(retry).toHaveBeenCalledWith('in_app_offer.report', { event: 'shown', offer_id: 'offer_1' })
  })

  it('stays silent when the backend lacks the method', async () => {
    const requestGateway = vi.fn().mockRejectedValue(new Error('method not found'))

    expect(() => reportInAppOfferEvent(requestGateway, 'offer_1', 'shown')).not.toThrow()
    await Promise.resolve()
  })
})
