/**
 * Where the main window lands when the guided chat assembles the app around it.
 *
 * Pure geometry, extracted from the `chat-onboarding:grow` handler so the one
 * thing that has actually gone wrong here — ending up too small — can be
 * asserted rather than eyeballed on a first run.
 */

export interface GrowRequest {
  bottom?: number
  left?: number
  /** Floor for the resulting CSS-pixel viewport width, for a layout with a
   *  responsive breakpoint to clear. Optional: most growth is just deltas. */
  minWidth?: number
  right?: number
  top?: number
}

export interface GrowInputs {
  /** Current window bounds, frame included. */
  bounds: { height: number; width: number }
  /** Non-zero on framed platforms: `bounds.width` minus the content width. The
   *  floor is about the viewport, so the frame has to be added back on top. */
  frameWidth?: number
  /** Display work area the result is centred in and clamped to. */
  workArea: { height: number; width: number; x: number; y: number }
  /** Renderer zoom. Requests arrive in CSS pixels; windows live in DIP. */
  zoom?: number
}

/** Growth is bounded so a malformed request can't ask for a wall-sized window;
 *  the display clamp below is the real limit. */
const MAX_DELTA_PX = 4000

/** Never fill the whole display — a window pinned to every edge reads as broken
 *  rather than as an app that grew. */
const MAX_WORK_AREA = 0.92

export function growWindowBounds(
  request: GrowRequest | null | undefined,
  { bounds, frameWidth = 0, workArea, zoom = 1 }: GrowInputs
) {
  const dip = (value: number | undefined, round: (n: number) => number) =>
    Math.max(0, Math.min(MAX_DELTA_PX, round((Number(value) || 0) * zoom)))

  const toDip = (value?: number) => dip(value, Math.round)

  // The floor CEILS where the deltas round. Rounding a breakpoint down lands
  // fractionally under it — at 118% zoom a 768px floor becomes 906 DIP, a
  // 767.8px viewport, and the media query the floor exists to satisfy is still
  // false. Half a pixel, whole floating sidebar.
  const requestedMin = dip(request?.minWidth, Math.ceil)
  const grown = bounds.width + toDip(request?.left) + toDip(request?.right)

  // Order matters: the floor lifts, then the display clamps. A floor wider than
  // the screen loses — growing off-screen to satisfy a breakpoint would trade a
  // floating sidebar for an unusable window.
  const width = Math.min(
    Math.max(grown, requestedMin ? requestedMin + frameWidth : 0),
    Math.round(workArea.width * MAX_WORK_AREA)
  )

  const height = Math.min(
    bounds.height + toDip(request?.top) + toDip(request?.bottom),
    Math.round(workArea.height * MAX_WORK_AREA)
  )

  return {
    height,
    width,
    x: Math.round(workArea.x + (workArea.width - width) / 2),
    y: Math.round(workArea.y + (workArea.height - height) / 2)
  }
}
