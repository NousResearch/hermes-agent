import { describe, expect, it } from 'vitest'

import {
  showTitlebarCloseFallback,
  TITLEBAR_CONTROL_OFFSET_X,
  TITLEBAR_EDGE_INSET,
  TITLEBAR_FALLBACK_WINDOW_BUTTON_X,
  titlebarControlsPosition,
  titlebarSystemToolCount
} from './titlebar'

describe('titlebarControlsPosition', () => {
  it('offsets controls from visible traffic lights', () => {
    expect(titlebarControlsPosition({ x: 24, y: 10 }).left).toBe(24 + TITLEBAR_CONTROL_OFFSET_X)
  })

  it('pins to the edge when macOS fullscreen hides traffic lights', () => {
    expect(titlebarControlsPosition({ x: 24, y: 10 }, true).left).toBe(TITLEBAR_EDGE_INSET)
  })

  it('pins to the edge on Windows/Linux where native controls render on the right', () => {
    expect(titlebarControlsPosition(null).left).toBe(TITLEBAR_EDGE_INSET)
  })

  it('uses the macOS fallback while the initial window state is unknown', () => {
    expect(titlebarControlsPosition(undefined).left).toBe(TITLEBAR_FALLBACK_WINDOW_BUTTON_X + TITLEBAR_CONTROL_OFFSET_X)
  })

  it('shows the custom close button only on non-fullscreen fallback platforms', () => {
    expect(showTitlebarCloseFallback({ isFullscreen: false, showWindowControlsFallback: true })).toBe(true)
    expect(showTitlebarCloseFallback({ isFullscreen: true, showWindowControlsFallback: true })).toBe(false)
    expect(showTitlebarCloseFallback({ isFullscreen: false, showWindowControlsFallback: false })).toBe(false)
  })

  it('accounts for the optional close button in the fixed tool cluster width', () => {
    expect(titlebarSystemToolCount()).toBe(4)
    expect(titlebarSystemToolCount(true)).toBe(5)
  })
})
