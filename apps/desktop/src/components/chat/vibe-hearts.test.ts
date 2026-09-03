import { beforeEach, describe, expect, it, vi } from 'vitest'

const { flashPetActivity, forwardPetReaction, burst, overlay } = vi.hoisted(() => ({
  flashPetActivity: vi.fn(),
  forwardPetReaction: vi.fn(),
  burst: vi.fn(),
  overlay: { active: false }
}))

vi.mock('@/components/particles/particle-field', () => ({
  createParticleEmitter: () => ({ burst, subscribe: () => () => undefined }),
  ParticleField: () => null
}))

vi.mock('@/store/pet', () => ({
  $petActive: { get: () => false },
  flashPetActivity
}))

vi.mock('@/store/pet-overlay', () => ({
  $petOverlayActive: { get: () => overlay.active },
  forwardPetReaction
}))

import { burstVibeHearts } from '@/components/chat/vibe-hearts'
import { setVibeHeartsEnabled } from '@/store/vibe-hearts-enabled'

describe('burstVibeHearts', () => {
  beforeEach(() => {
    flashPetActivity.mockClear()
    forwardPetReaction.mockClear()
    burst.mockClear()
    overlay.active = false
    setVibeHeartsEnabled(true)
  })

  it('plays hearts when the preference is on', () => {
    burstVibeHearts()
    expect(burst).toHaveBeenCalledOnce()
  })

  it('no-ops when the preference is off', () => {
    setVibeHeartsEnabled(false)
    burstVibeHearts()
    expect(burst).not.toHaveBeenCalled()
    expect(flashPetActivity).not.toHaveBeenCalled()
    expect(forwardPetReaction).not.toHaveBeenCalled()
  })

  it('reads the live atom (toggle mid-session takes effect)', () => {
    setVibeHeartsEnabled(false)
    burstVibeHearts()
    expect(burst).not.toHaveBeenCalled()

    setVibeHeartsEnabled(true)
    burstVibeHearts()
    expect(burst).toHaveBeenCalledOnce()
  })

  it('forwards to the popped-out overlay when on', () => {
    overlay.active = true
    burstVibeHearts()
    expect(forwardPetReaction).toHaveBeenCalledWith('vibe')
    expect(flashPetActivity).toHaveBeenCalledOnce()
    expect(burst).not.toHaveBeenCalled()
  })

  it('off also silences the popped-out overlay path (vibe reactions only originate here)', () => {
    // The overlay window's playVibeHearts() only fires on a reaction forwarded
    // by this router, so gating the forward IS the overlay's off switch.
    overlay.active = true
    setVibeHeartsEnabled(false)
    burstVibeHearts()
    expect(forwardPetReaction).not.toHaveBeenCalled()
    expect(flashPetActivity).not.toHaveBeenCalled()
    expect(burst).not.toHaveBeenCalled()
  })
})
