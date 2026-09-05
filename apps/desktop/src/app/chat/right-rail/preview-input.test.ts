import { describe, expect, it } from 'vitest'

import { type PreviewInputEvent, scalePreviewInputForDpr } from './preview-input'

describe('scalePreviewInputForDpr', () => {
  // Issue #91130: on displays where the guest's devicePixelRatio is ≠ 1,
  // Chromium divides sendInputEvent coordinates by that scale before
  // dispatching into the page, so a CSS-space click lands at point/DPR. The
  // funnel multiplies by DPR to cancel the division. DPR 1 is the no-op
  // identity — the common 1x case must never change.

  it('is the identity at DPR 1', () => {
    const event: PreviewInputEvent = { type: 'mouseMove', x: 100, y: 100 }

    expect(scalePreviewInputForDpr(event, 1)).toBe(event)
  })

  it('scales mouse moves by a fractional DPR', () => {
    const event: PreviewInputEvent = { type: 'mouseMove', x: 100, y: 100 }

    expect(scalePreviewInputForDpr(event, 1.5)).toEqual({ type: 'mouseMove', x: 150, y: 150 })
  })

  it('scales mouse moves by integer DPR too', () => {
    const event: PreviewInputEvent = { type: 'mouseMove', x: 100, y: 100 }

    expect(scalePreviewInputForDpr(event, 2)).toEqual({ type: 'mouseMove', x: 200, y: 200 })
    expect(scalePreviewInputForDpr(event, 3)).toEqual({ type: 'mouseMove', x: 300, y: 300 })
  })

  it('rounds scaled coordinates to whole pixels', () => {
    const event: PreviewInputEvent = { type: 'mouseMove', x: 101, y: 99 }

    expect(scalePreviewInputForDpr(event, 1.25)).toEqual({ type: 'mouseMove', x: 126, y: 124 })
  })

  it('scales click coordinates and preserves button/clickCount', () => {
    const event: PreviewInputEvent = { button: 'left', clickCount: 1, type: 'mouseDown', x: 200, y: 150 }

    expect(scalePreviewInputForDpr(event, 1.5)).toEqual({
      button: 'left',
      clickCount: 1,
      type: 'mouseDown',
      x: 300,
      y: 225
    })
  })

  it('scales wheel position but keeps the wheel deltas untouched', () => {
    const event: PreviewInputEvent = { deltaX: 0, deltaY: -120, type: 'mouseWheel', x: 400, y: 300 }

    expect(scalePreviewInputForDpr(event, 1.25)).toEqual({
      deltaX: 0,
      deltaY: -120,
      type: 'mouseWheel',
      x: 500,
      y: 375
    })
  })

  it('never touches keyboard events (they carry no coordinates)', () => {
    const down: PreviewInputEvent = { keyCode: 'Enter', type: 'keyDown' }
    const char: PreviewInputEvent = { keyCode: 'a', modifiers: ['shift'], type: 'char' }

    expect(scalePreviewInputForDpr(down, 1.5)).toBe(down)
    expect(scalePreviewInputForDpr(char, 2)).toBe(char)
  })
})
