import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { $touchPointer } from '@/lib/touch-interaction'
import { $connection } from '@/store/session'
import { $shellChoice } from '@/store/shell-variant'

import {
  MACOS_TAHOE_DARWIN_MAJOR,
  PORTAL_COMPACT_TITLEBAR_HEIGHT,
  PORTAL_RAIL_WIDTH,
  PORTAL_TITLEBAR_HEIGHT,
  portalBandLeadingInset,
  TITLEBAR_CONTROL_OFFSET_X,
  TITLEBAR_EDGE_INSET,
  TITLEBAR_HEIGHT,
  TITLEBAR_MAC_TRAFFIC_LIGHTS_Y_NUDGE,
  titlebarBandHeight,
  titlebarControlsPosition,
  titlebarControlsYNudge,
  titlebarToolsRightCss,
  TOUCH_TITLEBAR_HEIGHT,
  windowLightsInRail
} from './titlebar'
import { installTitlebarBandProperty, titlebarHeightNow, useTitlebarHeight } from './use-touch-titlebar'

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

  it('shrinks the redesign band only on a compact browser window (48, touch 44); native keeps 60', () => {
    const band = (portal: boolean, touch: boolean, compact: boolean, browserHost: boolean) =>
      titlebarBandHeight({ portal, touch, compact, browserHost })

    expect(band(true, false, true, true)).toBe(PORTAL_COMPACT_TITLEBAR_HEIGHT)
    expect(band(true, true, true, true)).toBe(TOUCH_TITLEBAR_HEIGHT)
    // Electron seats the traffic lights and caption buttons in the 60px band from the shell alone.
    expect(band(true, false, true, false)).toBe(PORTAL_TITLEBAR_HEIGHT)
    // Medium and wide windows keep Figma's header; touch only grows its buttons.
    expect(band(true, true, false, true)).toBe(PORTAL_TITLEBAR_HEIGHT)
    // Classic is untouched by width.
    expect(band(false, false, true, true)).toBe(TITLEBAR_HEIGHT)
    expect(band(false, true, true, true)).toBe(TOUCH_TITLEBAR_HEIGHT)
  })

  it('follows a live shell flip in the window band', () => {
    const { result } = renderHook(() => useTitlebarHeight())

    expect(result.current).toBe(PORTAL_TITLEBAR_HEIGHT)
    act(() => $shellChoice.set(false))
    expect(result.current).toBe(TITLEBAR_HEIGHT)
    act(() => $shellChoice.set(null))
    expect(result.current).toBe(PORTAL_TITLEBAR_HEIGHT)
  })

  it('publishes the band on the root for body-portalled surfaces, in both shells', () => {
    const root = document.createElement('html')
    const band = () => root.style.getPropertyValue('--titlebar-band-height')
    const stop = installTitlebarBandProperty(root)

    expect(band()).toBe(`${PORTAL_TITLEBAR_HEIGHT}px`)
    $shellChoice.set(false)
    expect(band()).toBe(`${TITLEBAR_HEIGHT}px`)
    $shellChoice.set(null)
    expect(band()).toBe(`${PORTAL_TITLEBAR_HEIGHT}px`)
    stop()
  })

  // Main placed the find bar, toast stack and floating HUDs (and pets, floating panes) against the
  // compact 34px band on every host; a touch browser host must not push them down in classic.
  it("keeps classic's body-portalled readers on the compact band under a touch pointer", () => {
    const root = document.createElement('html')
    const stop = installTitlebarBandProperty(root)

    $touchPointer.set(true)
    $shellChoice.set(false)
    expect(root.style.getPropertyValue('--titlebar-band-height')).toBe(`${TITLEBAR_HEIGHT}px`)
    expect(titlebarHeightNow()).toBe(TITLEBAR_HEIGHT)

    $shellChoice.set(null)
    expect(titlebarHeightNow()).toBe(PORTAL_TITLEBAR_HEIGHT)
    stop()
    $touchPointer.set(false)
  })
})

describe('portalBandLeadingInset', () => {
  it('starts at x=150 past macOS traffic lights', () => {
    expect(portalBandLeadingInset({ x: 23, y: 23 }, false, true)).toBe(150)
  })

  it('assumes lights on a mac before native state arrives', () => {
    expect(portalBandLeadingInset(undefined, false, true)).toBe(150)
  })

  it('aligns with the content inset (x=80) on Windows/Linux/web', () => {
    expect(portalBandLeadingInset(null, false, false)).toBe(80)
    expect(portalBandLeadingInset(undefined, false, false)).toBe(80)
  })

  it('aligns with the content inset in macOS fullscreen', () => {
    expect(portalBandLeadingInset({ x: 23, y: 23 }, true, true)).toBe(80)
    expect(portalBandLeadingInset(null, true, true)).toBe(80)
  })

  it('moves in by the rail on a compact window, where the rail has no column', () => {
    expect(portalBandLeadingInset(null, false, false, true)).toBe(80 - PORTAL_RAIL_WIDTH)
    expect(portalBandLeadingInset({ x: 23, y: 23 }, false, true, true)).toBe(150 - PORTAL_RAIL_WIDTH)
  })

  it('publishes the live start on the root for the band clusters and page toolbars', () => {
    const root = document.createElement('html')
    const start = () => root.style.getPropertyValue('--portal-band-start')
    const stop = installTitlebarBandProperty(root)

    $connection.set({ isFullscreen: false, windowButtonPosition: { x: 23, y: 23 } } as HermesConnection)
    expect(start()).toBe('150px')
    $connection.set({ isFullscreen: true, windowButtonPosition: { x: 23, y: 23 } } as HermesConnection)
    expect(start()).toBe('80px')
    $connection.set({ isFullscreen: false, windowButtonPosition: null } as HermesConnection)
    expect(start()).toBe('80px')
    stop()
    $connection.set(null)
  })
})

describe('windowLightsInRail', () => {
  it('holds for macOS traffic lights, reported or assumed before native state arrives', () => {
    expect(windowLightsInRail({ x: 23, y: 23 }, false, true)).toBe(true)
    expect(windowLightsInRail(undefined, false, true)).toBe(true)
  })

  it('is false in fullscreen, on Windows/Linux and in the browser', () => {
    expect(windowLightsInRail({ x: 23, y: 23 }, true, true)).toBe(false)
    expect(windowLightsInRail(null, false, true)).toBe(false)
    expect(windowLightsInRail(undefined, false, false)).toBe(false)
  })
})
