import { describe, expect, it } from 'vitest'

import {
  MACOS_TAHOE_DARWIN_MAJOR,
  TITLEBAR_CONTROL_OFFSET_X,
  TITLEBAR_EDGE_INSET,
  TITLEBAR_FALLBACK_WINDOW_BUTTON_X,
  TITLEBAR_MAC_TRAFFIC_LIGHTS_Y_NUDGE,
  titlebarControlsPosition,
  titlebarControlsYNudge,
  titlebarToolsRightCss
} from './titlebar'

describe('titlebarControlsPosition', () => {
  it('offsets controls from visible traffic lights', () => {
    expect(titlebarControlsPosition({ x: 20, y: 20 }).left).toBe(20 + TITLEBAR_CONTROL_OFFSET_X)
  })

  it('pins to the edge when macOS fullscreen hides traffic lights', () => {
    expect(titlebarControlsPosition({ x: 20, y: 20 }, true).left).toBe(TITLEBAR_EDGE_INSET)
  })

  it('pins to the edge on Windows/Linux where native controls render on the right', () => {
    expect(titlebarControlsPosition(null).left).toBe(TITLEBAR_EDGE_INSET)
  })

  it('uses the macOS fallback while the initial window state is unknown', () => {
    expect(titlebarControlsPosition(undefined).left).toBe(TITLEBAR_FALLBACK_WINDOW_BUTTON_X + TITLEBAR_CONTROL_OFFSET_X)
  })
})

describe('titlebarControlsYNudge', () => {
  it('nudges pre-Tahoe macOS when traffic lights are visible', () => {
    expect(titlebarControlsYNudge({ windowButtonPosition: { x: 20, y: 20 }, darwinMajor: 24 })).toBe(
      TITLEBAR_MAC_TRAFFIC_LIGHTS_Y_NUDGE
    )
  })

  it('stays flat on Tahoe, Windows/Linux, and macOS fullscreen', () => {
    expect(
      titlebarControlsYNudge({ windowButtonPosition: { x: 20, y: 20 }, darwinMajor: MACOS_TAHOE_DARWIN_MAJOR })
    ).toBe('0px')
    expect(titlebarControlsYNudge({ windowButtonPosition: null })).toBe('0px')
    expect(titlebarControlsYNudge({ windowButtonPosition: { x: 20, y: 20 }, isFullscreen: true })).toBe('0px')
  })

  it('nudges while macOS window-button position is still unknown on pre-Tahoe', () => {
    expect(titlebarControlsYNudge({ darwinMajor: 24 })).toBe(TITLEBAR_MAC_TRAFFIC_LIGHTS_Y_NUDGE)
  })
})

describe('titlebarToolsRightCss', () => {
  it('reserves the native overlay width when present', () => {
    expect(titlebarToolsRightCss(144)).toBe('144px')
  })

  it('matches the left edge inset on macOS fullscreen', () => {
    expect(titlebarToolsRightCss(0, { darwinMajor: 25, isFullscreen: true })).toBe(`${TITLEBAR_EDGE_INSET}px`)
  })
})
