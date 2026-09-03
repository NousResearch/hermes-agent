import { describe, expect, it, vi } from 'vitest'

import { openPetAgentPicker, petAgentMotion } from './pet-agent-picker-hover'

describe('openPetAgentPicker', () => {
  it('opens exactly once from an explicit activation and includes the pet bounds', () => {
    const show = vi.fn()

    const pet = {
      getBoundingClientRect: () => ({ height: 80, width: 56, x: 24, y: 300 })
    } as unknown as Element

    openPetAgentPicker(show, pet)

    expect(show).toHaveBeenCalledOnce()
    expect(show).toHaveBeenCalledWith('agents', {
      height: 80,
      viewportHeight: window.innerHeight,
      viewportWidth: window.innerWidth,
      width: 56,
      x: 24,
      y: 300
    })
  })
})

describe('petAgentMotion', () => {
  it('greets on hover and gives working activity priority', () => {
    expect(petAgentMotion(false, {})).toBe('idle')
    expect(petAgentMotion(true, {})).toBe('greet')
    expect(petAgentMotion(true, { reasoning: true })).toBe('working')
    expect(petAgentMotion(false, { toolRunning: true })).toBe('working')
  })
})
