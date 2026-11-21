// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'

import { windowControlsRects } from './geometry'

const VIEWPORT = 1024

afterEach(() => vi.unstubAllGlobals())

describe('windowControlsRects', () => {
  it('reserves a left-only Linux layout', () => {
    vi.stubGlobal('hermesDesktop', {})

    expect(windowControlsRects({ linuxWindowButtons: { left: 3, right: 0 } }, VIEWPORT)).toEqual([
      { x: 0, y: 0, width: 144, height: 34 }
    ])
  })

  it('reserves a right-only Linux layout', () => {
    vi.stubGlobal('hermesDesktop', {})

    expect(windowControlsRects({ linuxWindowButtons: { left: 0, right: 3 } }, VIEWPORT)).toEqual([
      { x: VIEWPORT - 144, y: 0, width: 144, height: 34 }
    ])
  })

  it('reserves a mixed Linux layout on both sides', () => {
    vi.stubGlobal('hermesDesktop', {})

    expect(windowControlsRects({ linuxWindowButtons: { left: 1, right: 2 } }, VIEWPORT)).toEqual([
      { x: 0, y: 0, width: 48, height: 34 },
      { x: VIEWPORT - 96, y: 0, width: 96, height: 34 }
    ])
  })

  it('keeps the right-side fallback for Windows/WSLg or an older Linux main', () => {
    vi.stubGlobal('hermesDesktop', {})

    expect(windowControlsRects({ nativeOverlayWidth: 144 }, VIEWPORT)).toEqual([
      { x: VIEWPORT - 144, y: 0, width: 144, height: 34 }
    ])
  })

  it('reserves nothing in fullscreen or outside Electron', () => {
    vi.stubGlobal('hermesDesktop', {})
    expect(windowControlsRects({ linuxWindowButtons: { left: 3, right: 0 }, isFullscreen: true }, VIEWPORT)).toEqual([])

    vi.unstubAllGlobals()
    expect(windowControlsRects({ linuxWindowButtons: { left: 3, right: 0 } }, VIEWPORT)).toEqual([])
  })
})
