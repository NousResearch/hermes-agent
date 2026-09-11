import { afterEach, describe, expect, it } from 'vitest'

import { $freeTierRoute, $freeTierStatus, freeTierStripPending } from '@/store/free-tier'
import { $desktopOnboarding, refreshOnboarding } from '@/store/onboarding'
import type { FreeTierStatus } from '@/types/hermes'

const READY: FreeTierStatus = {
  available: true,
  enabled: true,
  has_guest: true,
  label: 'Nous · free tier',
  model: 'nous/welcome',
  notice_pending: true
}

// A configured backend whose free-tier answer the test supplies. `setup.status`
// and `setup.runtime_check` both report ready so refreshOnboarding takes the
// "already configured" path — the one the intro hangs off.
function gatewayReturning(freeTier: FreeTierStatus, route = true) {
  return async <T>(method: string): Promise<T> => {
    if (method === 'free_tier.status') {
      return freeTier as T
    }

    if (method === 'setup.runtime_check') {
      return { free_tier: route, ok: true } as T
    }

    return { provider_configured: true } as T
  }
}

afterEach(() => {
  $freeTierStatus.set(null)
  $freeTierRoute.set(null)
})

describe('free-tier introduction branch table', () => {
  it('opens the ready screen when the free tier is the route inference runs on', async () => {
    await refreshOnboarding({ requestGateway: gatewayReturning(READY) })

    expect($desktopOnboarding.get().freeTierReady).toBe(true)
    // The overlay owns this case, so the composer strip must stay away.
    expect(freeTierStripPending($freeTierStatus.get(), $freeTierRoute.get())).toBe(false)
  })

  it('offers the composer strip instead when the user owns the provider carrying inference', async () => {
    await refreshOnboarding({ requestGateway: gatewayReturning(READY, false) })

    expect($desktopOnboarding.get().freeTierReady).toBe(false)
    expect(freeTierStripPending($freeTierStatus.get(), $freeTierRoute.get())).toBe(true)
  })

  it('shows nothing once the notice has been acknowledged', async () => {
    await refreshOnboarding({ requestGateway: gatewayReturning({ ...READY, notice_pending: false }) })

    expect($desktopOnboarding.get().freeTierReady).toBe(false)
    expect(freeTierStripPending($freeTierStatus.get(), $freeTierRoute.get())).toBe(false)
  })
})

describe('acknowledging the introduction', () => {
  it('reports a failed ack so the ready screen stays up', async () => {
    const { ackFreeTierIntro } = await import('@/store/onboarding')
    const failing = async <T>(method: string): Promise<T> => {
      if (method === 'free_tier.ack_notice') {
        throw new Error('gateway away')
      }

      return READY as T
    }

    expect(await ackFreeTierIntro({ requestGateway: failing })).toBe(false)
    expect(await ackFreeTierIntro({ requestGateway: gatewayReturning(READY) })).toBe(false) // status stub returns no {acked: true}
  })
})
