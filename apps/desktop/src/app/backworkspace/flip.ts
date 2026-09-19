const FLIP_HALF_MS = 170
const EASE_OUT_OF_VIEW = 'cubic-bezier(0.55, 0, 0.85, 0.35)'
const EASE_INTO_VIEW = 'cubic-bezier(0.15, 0.65, 0.45, 1)'

export interface FlipOptions {
  /** 1 turns the right edge away (front → back), -1 turns it back. */
  direction: -1 | 1
  reducedMotion: boolean
}

function turn(degrees: number, scale: number): Keyframe {
  return { transform: `perspective(2400px) rotateY(${degrees}deg) scale(${scale})` }
}

function flipFrames({ direction, reducedMotion }: FlipOptions): { enter: Keyframe[]; exit: Keyframe[] } {
  if (reducedMotion) {
    return { enter: [{ opacity: 0 }, { opacity: 1 }], exit: [{ opacity: 1 }, { opacity: 0 }] }
  }

  return {
    enter: [turn(-90 * direction, 0.94), turn(0, 1)],
    exit: [turn(0, 1), turn(90 * direction, 0.94)]
  }
}

// A cancelled animation still hands over to the next state: motion never
// decides whether the side changes, only how it looks while it does.
function settled(animation: Animation): Promise<void> {
  return animation.finished.then(
    () => undefined,
    () => undefined
  )
}

/**
 * Turn `element` edge-on, run `swap` while nothing is visible, then turn it
 * back to face the user. The transform exists only while the flip runs, so at
 * rest fixed-position chrome and portals behave exactly as without it.
 */
export async function flipSurface(element: HTMLElement, swap: () => void, options: FlipOptions): Promise<void> {
  const { enter, exit } = flipFrames(options)
  const leaving = element.animate(exit, { duration: FLIP_HALF_MS, easing: EASE_OUT_OF_VIEW, fill: 'forwards' })

  await settled(leaving)
  swap()
  // Start the second half before dropping the first half's held frame, so
  // the surface never shows one unrotated frame at the midpoint.
  const arriving = element.animate(enter, { duration: FLIP_HALF_MS, easing: EASE_INTO_VIEW })

  leaving.cancel()
  await settled(arriving)
}
