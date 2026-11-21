import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  MACOS_TAHOE_DARWIN_MAJOR,
  TITLEBAR_CONTROL_OFFSET_X,
  TITLEBAR_EDGE_INSET,
  TITLEBAR_MAC_TRAFFIC_LIGHTS_Y_NUDGE,
  titlebarControlsPosition,
  titlebarControlsYNudge,
  titlebarToolsRightCss
} from './titlebar'

describe('titlebarControlsPosition', () => {
  afterEach(() => vi.restoreAllMocks())

  it('offsets controls from visible traffic lights', () => {
    expect(titlebarControlsPosition({ x: 24, y: 10 }).left).toBe(24 + TITLEBAR_CONTROL_OFFSET_X)
  })

  it('pins to the edge when macOS fullscreen hides traffic lights', () => {
    expect(titlebarControlsPosition({ x: 24, y: 10 }, true).left).toBe(TITLEBAR_EDGE_INSET)
  })

  it('pins to the edge on Windows/Linux where native controls render on the right', () => {
    expect(titlebarControlsPosition(null).left).toBe(TITLEBAR_EDGE_INSET)
  })

  it('offsets from left-side Linux window controls', () => {
    vi.spyOn(navigator, 'platform', 'get').mockReturnValue('Linux x86_64')
    expect(titlebarControlsPosition(null, false, 144).left).toBe(144 + TITLEBAR_EDGE_INSET)
  })

  it('ignores the left reservation in fullscreen', () => {
    vi.spyOn(navigator, 'platform', 'get').mockReturnValue('Linux x86_64')
    expect(titlebarControlsPosition(null, true, 144).left).toBe(TITLEBAR_EDGE_INSET)
  })

  it('keeps the previous left inset when the button count is zero', () => {
    vi.spyOn(navigator, 'platform', 'get').mockReturnValue('Linux x86_64')
    expect(titlebarControlsPosition(null, false, 0).left).toBe(TITLEBAR_EDGE_INSET)
  })

  it('keeps macOS traffic-light positioning even if a left reservation arrives', () => {
    vi.spyOn(navigator, 'platform', 'get').mockReturnValue('MacIntel')
    expect(titlebarControlsPosition({ x: 24, y: 10 }, false, 144).left).toBe(24 + TITLEBAR_CONTROL_OFFSET_X)
  })

  it('does not move macOS tools when the native traffic-light position arrives', () => {
    vi.spyOn(navigator, 'platform', 'get').mockReturnValue('MacIntel')
    expect(titlebarControlsPosition(undefined)).toEqual(titlebarControlsPosition({ x: 12, y: 10 }))
  })

  it.each(['Win32', 'Linux x86_64'])('preserves the previous startup spacing on %s', platform => {
    vi.spyOn(navigator, 'platform', 'get').mockReturnValue(platform)
    expect(titlebarControlsPosition(undefined).left).toBe(24 + TITLEBAR_CONTROL_OFFSET_X)
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
  it('reserves the native overlay reservation when present', () => {
    expect(titlebarToolsRightCss(144)).toBe('144px')
    expect(titlebarToolsRightCss(96)).toBe('96px')
  })

  it('uses the normal edge inset when no controls are on the right', () => {
    expect(titlebarToolsRightCss(0)).toBe('0.75rem')
  })

  it('matches the left edge inset on macOS fullscreen', () => {
    expect(titlebarToolsRightCss(0, { darwinMajor: 25, isFullscreen: true })).toBe(`${TITLEBAR_EDGE_INSET}px`)
  })
})
