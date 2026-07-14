import { describe, expect, it } from 'vitest'

import {
  TITLEBAR_CONTROL_OFFSET_X,
  TITLEBAR_EDGE_INSET,
  TITLEBAR_FALLBACK_WINDOW_BUTTON_X,
  resolveTitlebarFullscreen,
  titlebarControlsPosition
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
})

describe('resolveTitlebarFullscreen', () => {
  it('trusts the backend fullscreen state when it is true', () => {
    expect(resolveTitlebarFullscreen(true, false, { x: 24, y: 10 })).toBe(true)
  })

  it('does not treat macOS maximize as fullscreen when traffic lights are visible', () => {
    expect(resolveTitlebarFullscreen(false, true, { x: 24, y: 10 })).toBe(false)
  })

  it('uses the viewport fallback only when there are no left-side native controls', () => {
    expect(resolveTitlebarFullscreen(false, true, null)).toBe(true)
  })
})
