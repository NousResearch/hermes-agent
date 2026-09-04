export interface HudTranscriptHeightInput {
  /** Measured message rows, including the HUD sheet overhang. */
  contentHeight: number
  /** The composer's measured height. */
  barHeight: number
  /** The HUD window's current inner height. */
  viewportHeight: number
}

/**
 * The HUD transcript owns all available space after the composer once there is
 * something to show. A resizable HUD must expose more scrollback as it grows;
 * sizing the band to its message rows instead leaves a larger empty native
 * window around the same fixed-height transcript.
 */
export function hudTranscriptHeight({ barHeight, contentHeight, viewportHeight }: HudTranscriptHeightInput): number {
  if (contentHeight < 1) {
    return 0
  }

  return Math.max(0, Math.round(viewportHeight - barHeight))
}
