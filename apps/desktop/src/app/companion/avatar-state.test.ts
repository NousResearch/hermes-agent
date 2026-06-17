import { describe, expect, it } from 'vitest'

import { nextCompanionAvatarState } from './avatar-state'

describe('companion avatar state', () => {
  it('switches to speaking when speech starts', () => {
    expect(nextCompanionAvatarState('idle', 'speech-start')).toBe('speaking')
  })

  it('returns to idle when reset fires', () => {
    expect(nextCompanionAvatarState('thinking', 'reset')).toBe('idle')
  })
})
