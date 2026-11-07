import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as hapticsModule from '@/lib/haptics'
import { registerHapticTrigger } from '@/lib/haptics'
import { setHapticsMuted } from '@/store/haptics'

import { HapticsProvider } from './haptics-provider'

const { triggerMock } = vi.hoisted(() => ({ triggerMock: vi.fn() }))

vi.mock('web-haptics/react', () => ({
  useWebHaptics: () => ({ cancel: vi.fn(), isSupported: false, trigger: triggerMock })
}))

vi.mock('@/lib/haptics', async importOriginal => {
  const actual = await importOriginal<typeof hapticsModule>()

  return { ...actual, registerHapticTrigger: vi.fn() }
})

const registerMock = vi.mocked(registerHapticTrigger)

// The provider's registration contract (`muted ? null : trigger`) is not
// observable through triggerHaptic(): lib/haptics gates on $hapticsMuted
// itself, so a provider that ignored the muted flag would be masked. Spy on
// registerHapticTrigger directly to pin the registration behavior.
describe('HapticsProvider registration', () => {
  beforeEach(() => {
    setHapticsMuted(false)
    registerMock.mockClear()
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('registers the live trigger when unmuted and null when muted', async () => {
    render(<HapticsProvider>{null}</HapticsProvider>)

    expect(registerMock).toHaveBeenLastCalledWith(triggerMock)

    await act(async () => {
      setHapticsMuted(true)
    })
    expect(registerMock).toHaveBeenLastCalledWith(null)

    await act(async () => {
      setHapticsMuted(false)
    })
    expect(registerMock).toHaveBeenLastCalledWith(triggerMock)
  })
})
