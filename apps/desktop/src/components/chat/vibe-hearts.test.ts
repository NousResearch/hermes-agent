import { beforeEach, describe, expect, it, vi } from 'vitest'

const { flashPetActivity, forwardPetReaction, burst } = vi.hoisted(() => ({
  flashPetActivity: vi.fn(),
  forwardPetReaction: vi.fn(),
  burst: vi.fn()
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
  $petOverlayActive: { get: () => false },
  forwardPetReaction
}))

import { burstVibeHearts } from '@/components/chat/vibe-hearts'
import { setVibeHeartsEnabled } from '@/store/vibe-hearts-enabled'

describe('burstVibeHearts', () => {
  beforeEach(() => {
    flashPetActivity.mockClear()
    forwardPetReaction.mockClear()
    burst.mockClear()
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
})
