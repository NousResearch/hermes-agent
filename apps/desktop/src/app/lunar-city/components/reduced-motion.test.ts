import { describe, expect, it } from 'vitest'

import { reducedMotionPresentation } from './reduced-motion'

describe('reducedMotionPresentation', () => {
  it('snaps destinations and disables camera easing and looping clips when requested', () => {
    expect(reducedMotionPresentation(true)).toEqual({
      animateCamera: false,
      loopAnimations: false,
      snapToDestination: true
    })
  })

  it('preserves normal presentation when reduced motion is not requested', () => {
    expect(reducedMotionPresentation(false)).toEqual({
      animateCamera: true,
      loopAnimations: true,
      snapToDestination: false
    })
  })
})
