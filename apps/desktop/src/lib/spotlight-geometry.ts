/** Geometry for pointing from a row at a place elsewhere in the window (the splash's nav
 *  spotlight). Boxes are viewport rectangles, as `getBoundingClientRect` returns them. */

export interface Box {
  left: number
  top: number
  right: number
  bottom: number
}

/** The overlap of two boxes, or null when they do not overlap by a positive area. */
export function intersect(a: Box, b: Box): Box | null {
  const box = {
    left: Math.max(a.left, b.left),
    top: Math.max(a.top, b.top),
    right: Math.min(a.right, b.right),
    bottom: Math.min(a.bottom, b.bottom)
  }

  return box.right > box.left && box.bottom > box.top ? box : null
}

/** What of `box` is on screen: clipped by every clipping box around it (the viewport, scrollers). */
export function visiblePart(box: Box, clips: readonly Box[]): Box | null {
  return clips.reduce<Box | null>((visible, clip) => (visible ? intersect(visible, clip) : null), box)
}

const middle = (box: Box) => (box.top + box.bottom) / 2

/**
 * An elbow from the side of `from` that faces `to` to the facing side of `to`: across, along, across,
 * so it never cuts through either. `to` must lie wholly to one side of `from` (the nav beside the
 * splash, either side, so right-to-left layouts work); otherwise there is no line to draw.
 */
export function connectorPath(from: Box, to: Box): null | string {
  const leftward = to.right <= from.left
  const rightward = to.left >= from.right

  if (!leftward && !rightward) {
    return null
  }

  const startX = leftward ? from.left : from.right
  const endX = leftward ? to.right : to.left
  const elbowX = Math.round((startX + endX) / 2)

  return `M ${startX} ${middle(from)} H ${elbowX} V ${middle(to)} H ${endX}`
}
