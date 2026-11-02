import { describe, expect, it } from 'vitest'

import { measureWindowControlsInsets } from './use-window-controls-overlay-width'

function overlay(rect: { left: number; right: number; width: number }, visible = true) {
  return {
    visible,
    getTitlebarAreaRect: () => rect as DOMRect,
    addEventListener: () => {},
    removeEventListener: () => {}
  }
}

describe('measureWindowControlsInsets', () => {
  it('reports right-placed controls (Windows default layout)', () => {
    expect(measureWindowControlsInsets(overlay({ left: 0, right: 1776, width: 1776 }), 1920)).toEqual({
      left: 0,
      right: 144
    })
  })

  it('reports left-placed controls (macOS-style KDE/GTK button layouts)', () => {
    expect(measureWindowControlsInsets(overlay({ left: 113, right: 1920, width: 1807 }), 1920)).toEqual({
      left: 113,
      right: 0
    })
  })

  it('reports no controls on a frameless/fullscreen window', () => {
    expect(measureWindowControlsInsets(overlay({ left: 0, right: 1920, width: 1920 }), 1920)).toEqual({
      left: 0,
      right: 0
    })
  })

  it('returns null when the overlay is unavailable or not laid out', () => {
    expect(measureWindowControlsInsets(null, 1920)).toBeNull()
    expect(measureWindowControlsInsets(overlay({ left: 0, right: 0, width: 0 }, false), 1920)).toBeNull()
  })
})
