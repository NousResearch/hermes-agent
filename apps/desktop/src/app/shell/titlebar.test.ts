import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import {
  MACOS_TAHOE_DARWIN_MAJOR,
  PORTAL_TITLEBAR_HEIGHT,
  TITLEBAR_CONTROL_OFFSET_X,
  TITLEBAR_EDGE_INSET,
  TITLEBAR_HEIGHT,
  TITLEBAR_MAC_TRAFFIC_LIGHTS_Y_NUDGE,
  titlebarBandHeight,
  titlebarControlsPosition,
  titlebarControlsYNudge,
  titlebarToolsRightCss,
  TOUCH_TITLEBAR_HEIGHT
} from './titlebar'
import { useTitlebarHeight } from './use-touch-titlebar'

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
  it('reserves the native overlay width when present', () => {
    expect(titlebarToolsRightCss(144)).toBe('144px')
  })

  it('matches the left edge inset on macOS fullscreen', () => {
    expect(titlebarToolsRightCss(0, { darwinMajor: 25, isFullscreen: true })).toBe(`${TITLEBAR_EDGE_INSET}px`)
  })
})

describe('titlebarBandHeight', () => {
  it('gives the redesign its 60px header and keeps classic compact (touch: 44px)', () => {
    expect(titlebarBandHeight({ portal: true, touch: false })).toBe(PORTAL_TITLEBAR_HEIGHT)
    expect(titlebarBandHeight({ portal: true, touch: true })).toBe(PORTAL_TITLEBAR_HEIGHT)
    expect(titlebarBandHeight({ portal: false, touch: false })).toBe(TITLEBAR_HEIGHT)
    expect(titlebarBandHeight({ portal: false, touch: true })).toBe(TOUCH_TITLEBAR_HEIGHT)
  })

  it('follows a live shell flip in the window band', () => {
    const { result } = renderHook(() => useTitlebarHeight())

    expect(result.current).toBe(PORTAL_TITLEBAR_HEIGHT)
    act(() => $shellChoice.set(false))
    expect(result.current).toBe(TITLEBAR_HEIGHT)
    act(() => $shellChoice.set(null))
    expect(result.current).toBe(PORTAL_TITLEBAR_HEIGHT)
  })
})
