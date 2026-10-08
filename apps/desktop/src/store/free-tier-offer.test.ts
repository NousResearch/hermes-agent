import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $freeTierStatus, type FreeTierRequester } from '@/store/free-tier'
import { $freeTierSignIn, closeFreeTierSignIn, stopFreeTierOffer, syncFreeTierOffer } from '@/store/free-tier-sign-in'
import { activeGatewayProfileKey, ensureGatewayForProfile } from '@/store/gateway'
import {
  $questionnaireDecided,
  markQuestionnaireDecided,
  setOnboardingSurfaceActive
} from '@/store/onboarding-presence'
import type { FreeTierStatus } from '@/types/hermes'

const status = (nudge_due_in: null | number, available = true): FreeTierStatus => ({
  available,
  enabled: true,
  has_guest: available,
  label: 'Nous · free tier',
  model: 'nous/welcome',
  notice_pending: false,
  nudge_due_in
})

// The backend's two answers: what `free_tier.status` reads now, and whether this caller won the claim.
function gateway({
  claim,
  claimed = true,
  statuses
}: {
  claim?: Promise<{ claimed: boolean }>
  claimed?: boolean
  statuses: FreeTierStatus[]
}) {
  const queue = [...statuses]

  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'free_tier.claim_nudge') {
      return claim ?? { claimed }
    }

    return queue.length > 1 ? queue.shift() : queue[0]
  })

  // SAFETY: the code under test asks only these two methods and reads exactly these shapes back.
  return requestGateway as typeof requestGateway & FreeTierRequester
}

const claims = (requestGateway: ReturnType<typeof gateway>) =>
  requestGateway.mock.calls.filter(([method]) => method === 'free_tier.claim_nudge').length

let stop: () => void = () => undefined

// What the dialog owner does: every status read goes through the offer sync.
function own(requestGateway: FreeTierRequester) {
  stop = $freeTierStatus.listen(next => syncFreeTierOffer(next, requestGateway))
}

beforeEach(() => {
  vi.useFakeTimers()
  // The questionnaire's due check answered "not due": nothing first-run owns the screen.
  markQuestionnaireDecided()
})

afterEach(() => {
  stop()
  stop = () => undefined
  stopFreeTierOffer()
  closeFreeTierSignIn()
  $freeTierStatus.set(null)
  setOnboardingSurfaceActive('questionnaire', false)
  vi.useRealTimers()
})

describe('sign-in offer after a finished task', () => {
  it('due now: claims once and opens the offer', async () => {
    const requestGateway = gateway({ statuses: [status(null)] })

    syncFreeTierOffer(status(0), requestGateway)
    await vi.advanceTimersByTimeAsync(0)

    expect(claims(requestGateway)).toBe(1)
    expect($freeTierSignIn.get()).toEqual({ status: 'offer' })
  })

  it('due later: waits out nudge_due_in, re-reads the status, then claims', async () => {
    const requestGateway = gateway({ statuses: [status(0), status(null)] })
    own(requestGateway)

    syncFreeTierOffer(status(180), requestGateway)
    await vi.advanceTimersByTimeAsync(179_000)

    expect(requestGateway).not.toHaveBeenCalled()
    expect($freeTierSignIn.get()).toEqual({ status: 'closed' })

    await vi.advanceTimersByTimeAsync(1_000)

    expect(requestGateway).toHaveBeenCalledWith('free_tier.status')
    expect(claims(requestGateway)).toBe(1)
    expect($freeTierSignIn.get()).toEqual({ status: 'offer' })
  })

  it('a claim another window won leaves the dialog closed', async () => {
    const requestGateway = gateway({ claimed: false, statuses: [status(null)] })

    syncFreeTierOffer(status(0), requestGateway)
    await vi.advanceTimersByTimeAsync(0)

    expect(claims(requestGateway)).toBe(1)
    expect($freeTierSignIn.get()).toEqual({ status: 'closed' })
  })

  it('does not claim before the due check answers', async () => {
    const requestGateway = gateway({ statuses: [status(0)] })

    $questionnaireDecided.set(false)
    syncFreeTierOffer(status(0), requestGateway)
    await vi.advanceTimersByTimeAsync(0)

    expect(claims(requestGateway)).toBe(0)
  })

  it('does not claim while the questionnaire is on screen, the dialog is open, or off the free tier', async () => {
    const requestGateway = gateway({ statuses: [status(0)] })

    setOnboardingSurfaceActive('questionnaire', true)
    syncFreeTierOffer(status(0), requestGateway)
    await vi.advanceTimersByTimeAsync(0)
    setOnboardingSurfaceActive('questionnaire', false)

    $freeTierSignIn.set({ status: 'requested' })
    syncFreeTierOffer(status(0), requestGateway)
    await vi.advanceTimersByTimeAsync(0)
    $freeTierSignIn.set({ status: 'closed' })

    syncFreeTierOffer(status(0, false), requestGateway)
    await vi.advanceTimersByTimeAsync(0)

    expect(claims(requestGateway)).toBe(0)
    expect($freeTierSignIn.get()).toEqual({ status: 'closed' })
  })

  it('a claim answered after the gateway route changed does not open the offer', async () => {
    let answer: (value: { claimed: boolean }) => void = () => undefined
    const claim = new Promise<{ claimed: boolean }>(resolve => (answer = resolve))
    const requestGateway = gateway({ claim, statuses: [status(null)] })

    syncFreeTierOffer(status(0), requestGateway)
    await vi.advanceTimersByTimeAsync(0)
    // Re-selecting the active route is a route change: it starts a new activation.
    await ensureGatewayForProfile(activeGatewayProfileKey())
    answer({ claimed: true })
    await vi.advanceTimersByTimeAsync(0)

    expect(claims(requestGateway)).toBe(1)
    expect($freeTierSignIn.get()).toEqual({ status: 'closed' })
  })
})
