import { afterEach, describe, expect, it } from 'vitest'

import { $freeTierStatus, freeTierStripPending } from '@/store/free-tier'
import { $desktopOnboarding, refreshOnboarding } from '@/store/onboarding'
import type { FreeTierStatus } from '@/types/hermes'

const READY: FreeTierStatus = {
  carries_inference: true,
  enabled: true,
  has_guest: true,
  label: 'Nous · free tier',
  model: 'nous/welcome',
  notice_pending: true
}

// A configured backend whose free-tier answer the test supplies. `setup.status`
// and `setup.runtime_check` both report ready so refreshOnboarding takes the
// "already configured" path — the one the intro hangs off.
function gatewayReturning(freeTier: FreeTierStatus) {
  return async <T>(method: string): Promise<T> => {
    if (method === 'free_tier.status') {
      return freeTier as T
    }

    if (method === 'setup.runtime_check') {
      return { ok: true } as T
    }

    return { provider_configured: true } as T
  }
}

afterEach(() => {
  $freeTierStatus.set(null)
})

describe('free-tier introduction branch table', () => {
  it('opens the ready screen when the free tier carries inference', async () => {
    await refreshOnboarding({ requestGateway: gatewayReturning(READY) })

    expect($desktopOnboarding.get().freeTierReady).toBe(true)
    // The overlay owns this case, so the composer strip must stay away.
    expect(freeTierStripPending($freeTierStatus.get())).toBe(false)
  })

  it('offers the composer strip instead when the user owns the provider carrying inference', async () => {
    await refreshOnboarding({ requestGateway: gatewayReturning({ ...READY, carries_inference: false }) })

    expect($desktopOnboarding.get().freeTierReady).toBe(false)
    expect(freeTierStripPending($freeTierStatus.get())).toBe(true)
  })

  it('shows nothing once the notice has been acknowledged', async () => {
    await refreshOnboarding({ requestGateway: gatewayReturning({ ...READY, notice_pending: false }) })

    expect($desktopOnboarding.get().freeTierReady).toBe(false)
    expect(freeTierStripPending($freeTierStatus.get())).toBe(false)
  })
})
