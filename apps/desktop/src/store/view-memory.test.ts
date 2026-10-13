import { beforeEach, describe, expect, it } from 'vitest'

import { rememberedViewSubState, rememberViewSubState, resetViewMemory } from './view-memory'

describe('view memory', () => {
  beforeEach(resetViewMemory)

  it('has nothing remembered for a view that has not been opened', () => {
    expect(rememberedViewSubState('capabilities')).toBeNull()
  })

  it('remembers the last sub-state of a view', () => {
    rememberViewSubState('capabilities', 'plugins')

    expect(rememberedViewSubState('capabilities')).toBe('plugins')
  })

  it('keys memory per view — one view never reads another', () => {
    rememberViewSubState('capabilities', 'plugins')
    rememberViewSubState('settings', 'keys')

    expect(rememberedViewSubState('capabilities')).toBe('plugins')
    expect(rememberedViewSubState('settings')).toBe('keys')
  })

  it('keeps only the latest sub-state', () => {
    rememberViewSubState('settings', 'keys')
    rememberViewSubState('settings', 'about')

    expect(rememberedViewSubState('settings')).toBe('about')
  })
})
