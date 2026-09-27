/**
 * Geometry for the main window as the guided chat grows it.
 *
 * Extracted from the `chat-onboarding:grow` handler so the resulting size can be asserted in a unit test
 * instead of checked by eye on a first run.
 */

import type { Rectangle } from 'electron'

export interface GrowRequest {
  bottom?: number
  left?: number
  /** Floor for the resulting viewport width in CSS pixels, used to clear a responsive breakpoint. */
  minWidth?: number
  right?: number
  top?: number
}

export interface GrowInputs {
  /** Current window bounds, frame included. */
  bounds: { height: number; width: number }
  /** `bounds.width` minus the content width, non-zero on platforms that draw a window frame. `minWidth` is a
   *  viewport floor, so the frame width is added to it. */
  frameWidth?: number
  /** Display work area the result is centred in and clamped to. */
  workArea: { height: number; width: number; x: number; y: number }
  /** Renderer zoom factor. Requests arrive in CSS pixels; window bounds are in DIP. */
  zoom?: number
}

/** Cap on each value converted from the request, so a malformed request cannot ask for an oversized window.
 *  The work area clamp below is usually the stricter limit. */
const MAX_DELTA_PX = 4000

/** Fraction of the display work area a grown window may fill. Below 1 so the result keeps a margin instead
 *  of looking maximized. */
const MAX_WORK_AREA = 0.92

export function growWindowBounds(
  request: GrowRequest | null | undefined,
  { bounds, frameWidth = 0, workArea, zoom = 1 }: GrowInputs
) {
  const dip = (value: number | undefined, round: (n: number) => number) =>
    Math.max(0, Math.min(MAX_DELTA_PX, round((Number(value) || 0) * zoom)))

  const toDip = (value?: number) => dip(value, Math.round)

  // The floor uses Math.ceil where the deltas round to nearest. At 118% zoom a 768px floor is 906.24 DIP:
  // rounding to nearest would give 906 DIP, a 767.8px viewport, and the media query the floor exists to
  // satisfy would stay false.
  const requestedMin = dip(request?.minWidth, Math.ceil)
  const grown = bounds.width + toDip(request?.left) + toDip(request?.right)

  // The floor applies before the work area clamp, so a floor wider than the display is dropped rather than
  // growing the window off-screen to satisfy the breakpoint.
  const width = Math.min(
    Math.max(grown, requestedMin ? requestedMin + frameWidth : 0),
    Math.round(workArea.width * MAX_WORK_AREA)
  )

  const height = Math.min(
    bounds.height + toDip(request?.top) + toDip(request?.bottom),
    Math.round(workArea.height * MAX_WORK_AREA)
  )

  return centeredBounds(workArea, width, height)
}

export function centeredBounds(workArea: Rectangle, width: number, height: number): Rectangle {
  return {
    height,
    width,
    x: Math.round(workArea.x + (workArea.width - width) / 2),
    y: Math.round(workArea.y + (workArea.height - height) / 2)
  }
}
