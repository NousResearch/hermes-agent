// A card turned over by hand rather than a pane sliding out of the way: slow
// enough that the eye follows one object going over, with the surface losing
// focus as it swings edge-on the way anything does when it turns that fast.
const FLIP_HALF_MS = 420
// Near enough that the receding edge genuinely shortens while the near edge
// comes forward — the turn has a far side instead of being a flat rotation.
const PERSPECTIVE_PX = 1100
const EDGE_SCALE = 0.92
// Reached edge-on, where the surface is nearly invisible anyway. It rises late
// because that is where the angular speed is: an even ramp reads as the page
// going out of focus rather than as something moving.
const EDGE_BLUR_PX = 20
// Where the middle keyframe sits, as a fraction of the way over. Without it the
// browser interpolates the blur linearly against an eased rotation, and the
// softening arrives well before the movement that is supposed to cause it.
const MID_TURN_DEG = 52
const MID_SCALE = 1 - (1 - EDGE_SCALE) * 0.55
const MID_BLUR_PX = EDGE_BLUR_PX * 0.28
const EASE_OUT_OF_VIEW = 'cubic-bezier(0.55, 0, 0.85, 0.35)'
const EASE_INTO_VIEW = 'cubic-bezier(0.15, 0.65, 0.45, 1)'

export interface FlipOptions {
  /** 1 turns the right edge away (front → back), -1 turns it back. */
  direction: -1 | 1
  reducedMotion: boolean
}

function turn(degrees: number, scale: number, blur: number): Keyframe {
  return {
    filter: `blur(${blur}px)`,
    transform: `perspective(${PERSPECTIVE_PX}px) rotateY(${degrees}deg) scale(${scale})`
  }
}

function flipFrames({ direction, reducedMotion }: FlipOptions): { enter: Keyframe[]; exit: Keyframe[] } {
  if (reducedMotion) {
    return { enter: [{ opacity: 0 }, { opacity: 1 }], exit: [{ opacity: 1 }, { opacity: 0 }] }
  }

  return {
    enter: [
      turn(-90 * direction, EDGE_SCALE, EDGE_BLUR_PX),
      turn(-MID_TURN_DEG * direction, MID_SCALE, MID_BLUR_PX),
      turn(0, 1, 0)
    ],
    exit: [
      turn(0, 1, 0),
      turn(MID_TURN_DEG * direction, MID_SCALE, MID_BLUR_PX),
      turn(90 * direction, EDGE_SCALE, EDGE_BLUR_PX)
    ]
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
