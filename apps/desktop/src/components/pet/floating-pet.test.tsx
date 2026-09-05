import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { requestGateway } = vi.hoisted(() => ({
  requestGateway: vi.fn()
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'dark' })
}))

vi.mock('@/app/hooks/use-route-overlay-active', () => ({
  useRouteOverlayActive: () => false
}))

vi.mock('@/app/hooks/use-on-profile-switch', () => ({
  useOnProfileSwitch: () => {}
}))

import { $changeEventsAvailable } from '@/store/live-sync'
import { $petInfo, setPetInfo } from '@/store/pet'
import { $gatewayState } from '@/store/session'

import { FloatingPet } from './floating-pet'

describe('FloatingPet idle polling', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $gatewayState.set('open')
    $changeEventsAvailable.set(true)
    setPetInfo({ enabled: false })
    requestGateway.mockReset()
    requestGateway.mockResolvedValue({ enabled: false })
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('does not mutate $petInfo or notify listeners when polling an already-disabled pet', async () => {
    const initialPetInfo = $petInfo.get()
    const listener = vi.fn()
    const unsubscribe = $petInfo.listen(listener)

    render(<FloatingPet />)

    // Flush initial pull and startup retries
    await vi.runOnlyPendingTimersAsync()
    expect(requestGateway).toHaveBeenCalledWith('pet.info', expect.any(Object))

    // Advance through startup retry intervals (1s, 3s, 8s) and backstop (15s)
    await vi.advanceTimersByTimeAsync(20_000)

    // The atom should NOT have notified any listeners because enabled: false remained false
    expect(listener).not.toHaveBeenCalled()
    expect($petInfo.get()).toBe(initialPetInfo)

    unsubscribe()
  })
})
