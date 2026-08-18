import { describe, expect, it } from 'vitest'

import { hudTranscriptHeight } from './layout'

describe('hudTranscriptHeight', () => {
  it('uses the resized window space for a non-empty transcript', () => {
    // The transcript is intentionally not constrained to its content height:
    // resizing HUD must reveal more scrollback instead of growing an empty
    // transparent window below a fixed-height chat band.
    expect(
      hudTranscriptHeight({
        barHeight: 58,
        contentHeight: 72,
        viewportHeight: 640
      })
    ).toBe(582)
  })

  it('keeps an empty HUD collapsed', () => {
    expect(hudTranscriptHeight({ barHeight: 58, contentHeight: 0, viewportHeight: 640 })).toBe(0)
  })
})
