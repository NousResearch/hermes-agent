export interface ReducedMotionPresentation {
  animateCamera: boolean
  loopAnimations: boolean
  snapToDestination: boolean
}

/** Pure presentation policy shared by React and renderer adapters. */
export function reducedMotionPresentation(prefersReducedMotion: boolean): ReducedMotionPresentation {
  return prefersReducedMotion
    ? { animateCamera: false, loopAnimations: false, snapToDestination: true }
    : { animateCamera: true, loopAnimations: true, snapToDestination: false }
}
