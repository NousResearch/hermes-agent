import { describe, expect, it } from 'vitest'

import { measureOverlayInsets, type WindowControlsOverlayLike } from './use-window-controls-overlay-width'

const overlay = (rect: { left: number; right: number; width: number }, visible = true): WindowControlsOverlayLike => ({
  visible,
  getTitlebarAreaRect: () => rect as DOMRect,
  addEventListener: () => {},
  removeEventListener: () => {}
})

describe('measureOverlayInsets', () => {
  it('reads controls on the right', () => {
    expect(measureOverlayInsets(overlay({ left: 0, right: 928, width: 928 }), 1000)).toEqual({ left: 0, right: 72 })
  })

  it('reads controls on the left (Linux button order close,maximize,minimize:)', () => {
    expect(measureOverlayInsets(overlay({ left: 72, right: 1000, width: 928 }), 1000)).toEqual({ left: 72, right: 0 })
  })

  it('reads controls on both sides', () => {
    expect(measureOverlayInsets(overlay({ left: 28, right: 950, width: 922 }), 1000)).toEqual({ left: 28, right: 50 })
  })

  it('is null when there is no overlay or it is not laid out yet', () => {
    expect(measureOverlayInsets(null, 1000)).toBeNull()
    expect(measureOverlayInsets(overlay({ left: 0, right: 0, width: 0 }), 1000)).toBeNull()
    expect(measureOverlayInsets(overlay({ left: 72, right: 1000, width: 928 }, false), 1000)).toBeNull()
  })
})
