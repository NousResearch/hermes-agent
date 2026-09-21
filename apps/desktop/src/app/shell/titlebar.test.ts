import { describe, expect, it } from 'vitest'

import {
  MACOS_TAHOE_DARWIN_MAJOR,
  TITLEBAR_CONTROL_OFFSET_X,
  TITLEBAR_CONTROL_SIZE,
  TITLEBAR_EDGE_INSET,
  TITLEBAR_EXTERNAL_BUTTON_FALLBACK_WIDTH,
  TITLEBAR_FALLBACK_WINDOW_BUTTON_X,
  TITLEBAR_ICON_SIZE,
  TITLEBAR_MAC_TRAFFIC_LIGHTS_Y_NUDGE,
  titlebarContentInsetCss,
  titlebarControlsPosition,
  titlebarControlsYNudge,
  titlebarExternalButtonsWidth,
  titlebarIconSizeCss,
  titlebarToolsRightCss,
  titlebarToolsWidthCss
} from './titlebar'

describe('titlebar sizing', () => {
  it('uses 24×24 hit targets and 13.9px glyphs', () => {
    expect(TITLEBAR_CONTROL_SIZE).toBe(24)
    expect(TITLEBAR_ICON_SIZE).toBe(13.9)
    expect(titlebarIconSizeCss()).toBe('13.9px')
  })

  it('reserves width from abutting hit targets only', () => {
    expect(titlebarToolsWidthCss(4)).toBe('calc(4 * var(--titlebar-control-size))')
  })
})

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

describe('titlebarControlsYNudge', () => {
  it('nudges pre-Tahoe macOS when traffic lights are visible', () => {
    expect(titlebarControlsYNudge({ windowButtonPosition: { x: 24, y: 10 }, darwinMajor: 24 })).toBe(
      TITLEBAR_MAC_TRAFFIC_LIGHTS_Y_NUDGE
    )
  })

  it('stays flat on Tahoe, Windows/Linux, and macOS fullscreen', () => {
    expect(
      titlebarControlsYNudge({ windowButtonPosition: { x: 24, y: 10 }, darwinMajor: MACOS_TAHOE_DARWIN_MAJOR })
    ).toBe('0px')
    expect(titlebarControlsYNudge({ windowButtonPosition: null })).toBe('0px')
    expect(titlebarControlsYNudge({ windowButtonPosition: { x: 24, y: 10 }, isFullscreen: true })).toBe('0px')
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

  it('keeps the default chrome inset otherwise', () => {
    expect(titlebarToolsRightCss(0)).toBe('0.75rem')
  })

  it('adds a third-party reservation on top of the measured overlay, and still moves the cluster while it is unmeasured', () => {
    expect(titlebarToolsRightCss(144, {}, 66)).toBe('210px')
    expect(titlebarToolsRightCss(0, {}, 66)).toBe('calc(0.75rem + 66px)')
  })
})

describe('titlebarExternalButtonsWidth', () => {
  it('scales a third-party button off the measured overlay, so DPI and UI zoom are followed', () => {
    // 138px of overlay = 3 native caption buttons; a third-party one is ~0.72 of that.
    expect(titlebarExternalButtonsWidth(138, 2)).toBe(66)
  })

  it('tracks the same window through a UI scale change', () => {
    // At 75% UI scale the measured overlay shrinks in CSS px by the same factor,
    // and so must the reservation — a fixed px constant would under-reserve.
    expect(titlebarExternalButtonsWidth(103.5, 2)).toBe(50)
  })

  it('falls back to a fixed width before the overlay is measured', () => {
    expect(titlebarExternalButtonsWidth(0, 2)).toBe(2 * TITLEBAR_EXTERNAL_BUTTON_FALLBACK_WIDTH)
  })

  it('reserves nothing when no external buttons are configured', () => {
    expect(titlebarExternalButtonsWidth(138, 0)).toBe(0)
  })
})

describe('titlebarContentInsetCss', () => {
  it('clears the left tool cluster', () => {
    expect(titlebarContentInsetCss(14, 4)).toBe('calc(14px + calc(4 * var(--titlebar-control-size)) + 0.75rem)')
  })
})
