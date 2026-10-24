import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { windowControlsRect } from './geometry'

const linux = { windowButtonPosition: null, nativeOverlayWidth: 144, isFullscreen: false }

describe('windowControlsRect', () => {
  beforeEach(() => {
    vi.stubGlobal('hermesDesktop', {})
    vi.spyOn(navigator, 'platform', 'get').mockReturnValue('Linux x86_64')
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('uses the measured overlay on the right', () => {
    expect(windowControlsRect(linux, 1280, { left: 0, right: 120 })).toEqual({ x: 1160, y: 0, width: 120, height: 34 })
  })

  it('follows controls the desktop places on the left', () => {
    expect(windowControlsRect(linux, 1280, { left: 113, right: 0 })).toEqual({ x: 0, y: 0, width: 113, height: 34 })
  })

  it('falls back to the static right-side reservation before the overlay is measured', () => {
    expect(windowControlsRect(linux, 1280)).toEqual({ x: 1136, y: 0, width: 144, height: 34 })
  })

  it('keeps the traffic-light rect on macOS even when the overlay reports a left inset', () => {
    vi.spyOn(navigator, 'platform', 'get').mockReturnValue('MacIntel')

    expect(
      windowControlsRect({ windowButtonPosition: { x: 24, y: 10 }, nativeOverlayWidth: 0 }, 1280, { left: 90, right: 0 })
    ).toEqual({ x: 0, y: 0, width: 82, height: 34 })
  })
})
