import { beforeEach, describe, expect, it, vi } from 'vitest'

import { consumePetReplyArm, resetPetLiveSessions } from './pet-live-session'
import { submitPetOverlayPrompt } from './pet-overlay-submit'

describe('pet overlay composer submit', () => {
  beforeEach(() => resetPetLiveSessions())

  it('keeps an exact-session reply arm only when the prompt was accepted', async () => {
    const accepted = vi.fn().mockResolvedValue(true)

    await submitPetOverlayPrompt({
      profile: 'work',
      runtimeSessionId: 'runtime',
      submitText: accepted,
      text: 'hello'
    })

    expect(accepted).toHaveBeenCalledWith('hello')
    expect(consumePetReplyArm('work', 'runtime')).toBe(true)
  })

  it('disarms exact and next-session captures after rejection or failure', async () => {
    await submitPetOverlayPrompt({
      profile: 'work',
      runtimeSessionId: 'runtime',
      submitText: vi.fn().mockResolvedValue(false),
      text: 'rejected'
    })
    await submitPetOverlayPrompt({
      profile: 'new-profile',
      runtimeSessionId: null,
      submitText: vi.fn().mockRejectedValue(new Error('submit failed')),
      text: 'failed'
    })

    expect(consumePetReplyArm('work', 'runtime')).toBe(false)
    expect(consumePetReplyArm('new-profile', 'new-runtime')).toBe(false)
  })
})
