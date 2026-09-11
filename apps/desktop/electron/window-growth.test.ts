import { describe, expect, it } from 'vitest'

import { growWindowBounds } from './window-growth'

/** The solo guided chat: a small conversation card, centred. */
const SOLO = { height: 640, width: 600 }

const DISPLAY = { height: 1080, width: 1920, x: 0, y: 0 }

/** First-run zoom. The reason a request in CSS pixels can't be trusted as DIP. */
const ZOOM = 1.18

/** Where the sessions sidebar stops docking and becomes a floating Sheet. */
const DOCKED = 768

const viewport = (width: number, zoom = ZOOM) => width / zoom

describe('growWindowBounds', () => {
  it('grows outward by the deltas it was asked for', () => {
    const { height, width } = growWindowBounds(
      { bottom: 200, left: 220, right: 240 },
      { bounds: SOLO, workArea: DISPLAY, zoom: 1 }
    )

    expect(width).toBe(600 + 220 + 240)
    expect(height).toBe(640 + 200)
  })

  it('scales a CSS-pixel request into DIP at the renderer zoom', () => {
    const { width } = growWindowBounds({ left: 220 }, { bounds: SOLO, workArea: DISPLAY, zoom: ZOOM })

    expect(width).toBe(600 + Math.round(220 * ZOOM))
  })

  // The bug: Basic's sidebar delta is measured against the pane, not against
  // the viewport the sidebar needs to STAY docked. 600 + 220·1.18 is 860 DIP —
  // a 729px viewport, under the breakpoint — so the sidebar it just docked
  // arrived as a floating Sheet over the chat.
  it('clears the docked-sidebar breakpoint even when the deltas alone do not', () => {
    const bare = growWindowBounds({ left: 220 }, { bounds: SOLO, workArea: DISPLAY, zoom: ZOOM })

    expect(viewport(bare.width), 'no floor asked for, so nothing to prove').toBeLessThan(DOCKED)

    const floored = growWindowBounds({ left: 220, minWidth: DOCKED }, { bounds: SOLO, workArea: DISPLAY, zoom: ZOOM })

    expect(viewport(floored.width)).toBeGreaterThanOrEqual(DOCKED)
  })

  it('leaves a layout that already clears the floor at its own width', () => {
    const request = { bottom: 200, left: 220, minWidth: DOCKED, right: 240 }
    const { width } = growWindowBounds(request, { bounds: SOLO, workArea: DISPLAY, zoom: ZOOM })

    expect(width).toBe(600 + Math.round(220 * ZOOM) + Math.round(240 * ZOOM))
    expect(viewport(width)).toBeGreaterThan(DOCKED)
  })

  it('adds the window frame on top of the floor, so the viewport clears it', () => {
    const framed = growWindowBounds(
      { left: 220, minWidth: DOCKED },
      { bounds: SOLO, frameWidth: 16, workArea: DISPLAY, zoom: ZOOM }
    )

    expect(viewport(framed.width - 16)).toBeGreaterThanOrEqual(DOCKED)
  })

  it('lets the display win over the floor rather than growing off-screen', () => {
    const small = { height: 600, width: 800, x: 0, y: 0 }
    const { width } = growWindowBounds({ left: 220, minWidth: DOCKED }, { bounds: SOLO, workArea: small, zoom: ZOOM })

    expect(width).toBeLessThanOrEqual(small.width)
  })

  it('centres the result in the work area', () => {
    const { height, width, x, y } = growWindowBounds({ left: 220 }, { bounds: SOLO, workArea: DISPLAY, zoom: 1 })

    expect(x).toBe(Math.round((DISPLAY.width - width) / 2))
    expect(y).toBe(Math.round((DISPLAY.height - height) / 2))
  })

  it('treats a missing or malformed request as no growth at all', () => {
    for (const request of [null, undefined, {}, { left: Number.NaN }]) {
      expect(growWindowBounds(request, { bounds: SOLO, workArea: DISPLAY }).width).toBe(SOLO.width)
    }
  })
})
