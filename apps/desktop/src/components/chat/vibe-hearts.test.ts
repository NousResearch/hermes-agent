import { beforeEach, describe, expect, it, vi } from 'vitest'

const { flashCompanionActivity, forwardCompanionReaction, burst, overlay } = vi.hoisted(() => ({
  flashCompanionActivity: vi.fn(),
  forwardCompanionReaction: vi.fn(),
  burst: vi.fn(),
  overlay: { active: false }
}))

vi.mock('@/components/particles/particle-field', () => ({
  createParticleEmitter: () => ({ burst, subscribe: () => () => undefined }),
  ParticleField: () => null
}))

vi.mock('@/store/companion', () => ({
  $companionActive: { get: () => false },
  flashCompanionActivity
}))

vi.mock('@/store/companion-overlay', () => ({
  $companionOverlayActive: { get: () => overlay.active },
  forwardCompanionReaction
}))

import { burstVibeHearts } from '@/components/chat/vibe-hearts'
import { setVibeHeartsEnabled } from '@/store/vibe-hearts-enabled'

describe('burstVibeHearts', () => {
  beforeEach(() => {
    flashCompanionActivity.mockClear()
    forwardCompanionReaction.mockClear()
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
    expect(flashCompanionActivity).not.toHaveBeenCalled()
    expect(forwardCompanionReaction).not.toHaveBeenCalled()
  })

  it('forwards to the popped-out overlay when on', () => {
    overlay.active = true
    burstVibeHearts()
    expect(forwardCompanionReaction).toHaveBeenCalledWith('vibe')
    expect(flashCompanionActivity).toHaveBeenCalledOnce()
    expect(burst).not.toHaveBeenCalled()
  })

  it('off also silences the popped-out overlay path (vibe reactions only originate here)', () => {
    // The overlay window's playVibeHearts() only fires on a reaction forwarded
    // by this router, so gating the forward IS the overlay's off switch.
    overlay.active = true
    setVibeHeartsEnabled(false)
    burstVibeHearts()
    expect(forwardCompanionReaction).not.toHaveBeenCalled()
    expect(flashCompanionActivity).not.toHaveBeenCalled()
    expect(burst).not.toHaveBeenCalled()
  })
})
