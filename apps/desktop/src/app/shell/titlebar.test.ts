// @vitest-environment jsdom
import { describe, expect, it, vi, afterEach } from 'vitest'

import {
  TITLEBAR_CONTROL_OFFSET_X,
  TITLEBAR_EDGE_INSET,
  TITLEBAR_FALLBACK_WINDOW_BUTTON_X,
  titlebarControlsPosition
} from './titlebar'
import { viewportIsFullscreen } from './app-shell'

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

describe('viewportIsFullscreen', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  function mockScreen(overrides: Partial<Screen>) {
    const defaults = { width: 1920, height: 1080, availHeight: 1080, availWidth: 1920 }
    Object.defineProperty(window, 'screen', { value: { ...defaults, ...overrides }, configurable: true })
  }

  it('returns false when viewport is smaller than screen', () => {
    mockScreen({ width: 1920, height: 1080 })
    vi.stubGlobal('innerWidth', 1600)
    vi.stubGlobal('innerHeight', 900)
    expect(viewportIsFullscreen()).toBe(false)
  })

  it('returns true for true macOS fullscreen (menu bar hidden, availHeight equals height)', () => {
    mockScreen({ width: 1920, height: 1080, availHeight: 1080, availWidth: 1920 })
    vi.stubGlobal('innerWidth', 1920)
    vi.stubGlobal('innerHeight', 1080)
    expect(viewportIsFullscreen()).toBe(true)
  })

  it('returns false for macOS zoom (menu bar visible, availHeight < height) — #45264', () => {
    // macOS menu bar is 25px; availHeight is screen.height minus menu bar
    mockScreen({ width: 1920, height: 1080, availHeight: 1055, availWidth: 1920 })
    vi.stubGlobal('innerWidth', 1920)
    vi.stubGlobal('innerHeight', 1055)
    expect(viewportIsFullscreen()).toBe(false)
  })

  it('returns true on Windows/Linux where availHeight equals height (no menu bar)', () => {
    mockScreen({ width: 1920, height: 1080, availHeight: 1080, availWidth: 1920 })
    vi.stubGlobal('innerWidth', 1920)
    vi.stubGlobal('innerHeight', 1080)
    expect(viewportIsFullscreen()).toBe(true)
  })
})
