// Pure luminance → RGB math for the sidebar portrait, extracted so the unit
// test exercises the real functions the canvas painter calls. Mirrors the
// gradient of gentle-pi's terminal portrait (lib/shell-sidebar-portrait.ts).

const WARM_RGB: readonly [number, number, number][] = [
  [23, 8, 6],
  [77, 21, 14],
  [143, 37, 22],
  [216, 58, 29],
  [255, 75, 32],
  [255, 122, 50],
  [255, 173, 102],
  [255, 215, 163]
]

export const PORTRAIT_GAMMA = 0.72
export const PORTRAIT_CYCLE_MS = 6_500
export const PORTRAIT_REVEAL_MS = 3_000
export const PORTRAIT_FRAME_MS = 100

/** Map a 0-255 intensity onto the warm gradient. Out-of-range input clamps. */
export function warmColor(value: number): [number, number, number] {
  const position = (Math.max(0, Math.min(255, value)) / 255) * (WARM_RGB.length - 1)
  const index = Math.min(WARM_RGB.length - 2, Math.floor(position))
  const mix = position - index
  const start = WARM_RGB[index]!
  const end = WARM_RGB[index + 1]!

  return [
    Math.round(start[0] + (end[0] - start[0]) * mix),
    Math.round(start[1] + (end[1] - start[1]) * mix),
    Math.round(start[2] + (end[2] - start[2]) * mix)
  ]
}

/** Gamma-shape one luminance sample and scale it by the reveal progress. */
export function shapeLuminance(luminance: number, elapsed: number): number {
  const reveal = Math.min(1, elapsed / PORTRAIT_REVEAL_MS)

  return 255 * Math.pow(Math.max(0, Math.min(255, luminance)) / 255, PORTRAIT_GAMMA) * reveal
}
