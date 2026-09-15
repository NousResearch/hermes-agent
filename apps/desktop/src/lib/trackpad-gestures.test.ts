import { describe, expect, it } from 'vitest'

import { createHorizontalSwipeDetector } from './trackpad-gestures'

const wheel = (deltaX: number, deltaY = 0, ctrlKey = false) => ({ ctrlKey, deltaX, deltaY })

describe('createHorizontalSwipeDetector', () => {
  it('accumulates one action per swipe and rearms on a fresh impulse inside the momentum tail', () => {
    const detect = createHorizontalSwipeDetector(36, 180)

    expect(detect(wheel(20), 1_000).direction).toBeNull()
    expect(detect(wheel(16), 1_020).direction).toBe(1)
    expect(detect(wheel(12), 1_100).direction).toBeNull()
    expect(detect(wheel(5), 1_200).direction).toBeNull()
    expect(detect(wheel(10), 1_240).direction).toBeNull()
    expect(detect(wheel(27), 1_260).direction).toBe(1)
    expect(detect(wheel(20), 1_280).direction).toBeNull()
  })

  it('does not claim vertical intent or pinch zoom', () => {
    const detect = createHorizontalSwipeDetector()

    expect(detect(wheel(0, 60), 1_000).claimed).toBe(false)
    expect(detect(wheel(30, 40), 1_010).claimed).toBe(false)
    expect(detect(wheel(60, 0, true), 1_020).claimed).toBe(false)
  })
})
