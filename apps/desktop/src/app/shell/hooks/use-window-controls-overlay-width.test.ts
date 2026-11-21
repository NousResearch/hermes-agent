// @vitest-environment jsdom
import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useWindowControlsOverlayWidth, windowControlsFallback } from './use-window-controls-overlay-width'

interface OverlayLike {
  visible: boolean
  getTitlebarAreaRect: () => DOMRect
  addEventListener: (type: 'geometrychange', cb: () => void) => void
  removeEventListener: (type: 'geometrychange', cb: () => void) => void
}

function stubOverlay(value: OverlayLike | undefined) {
  Object.defineProperty(navigator, 'windowControlsOverlay', {
    configurable: true,
    value
  })
}

function stubRect(x: number, width: number) {
  stubOverlay({
    addEventListener: vi.fn(),
    getTitlebarAreaRect: () => ({ right: x + width, width, x }) as DOMRect,
    removeEventListener: vi.fn(),
    visible: true
  })
}

afterEach(() => {
  stubOverlay(undefined)
  vi.restoreAllMocks()
})

describe('useWindowControlsOverlayWidth', () => {
  it('returns null when there is no native overlay (macOS/WSLg use the static fallback)', () => {
    stubOverlay(undefined)

    const { result } = renderHook(() => useWindowControlsOverlayWidth())

    expect(result.current).toBeNull()
  })

  it('returns null when the overlay exists but is not visible', () => {
    stubOverlay({
      addEventListener: vi.fn(),
      getTitlebarAreaRect: () => ({ right: 900, width: 124, x: 0 }) as DOMRect,
      removeEventListener: vi.fn(),
      visible: false
    })

    const { result } = renderHook(() => useWindowControlsOverlayWidth())

    expect(result.current).toBeNull()
  })

  it('measures a right-only layout', () => {
    vi.spyOn(window, 'innerWidth', 'get').mockReturnValue(1024)
    stubRect(0, 900)

    const { result } = renderHook(() => useWindowControlsOverlayWidth())

    expect(result.current).toEqual({ left: 0, right: 124 })
  })

  it('measures a left-only layout', () => {
    vi.spyOn(window, 'innerWidth', 'get').mockReturnValue(1024)
    stubRect(124, 900)

    const { result } = renderHook(() => useWindowControlsOverlayWidth())

    expect(result.current).toEqual({ left: 124, right: 0 })
  })

  it('measures a mixed layout on both sides', () => {
    vi.spyOn(window, 'innerWidth', 'get').mockReturnValue(1024)
    stubRect(48, 880)

    const { result } = renderHook(() => useWindowControlsOverlayWidth())

    expect(result.current).toEqual({ left: 48, right: 96 })
  })
})

describe('windowControlsFallback', () => {
  it('uses per-button width from the overlay fallback for left-only layouts', () => {
    expect(windowControlsFallback({ left: 3, right: 0 }, 144)).toEqual({ left: 144, right: 0 })
  })

  it('uses per-button width from the overlay fallback for right-only layouts', () => {
    expect(windowControlsFallback({ left: 0, right: 2 }, 144)).toEqual({ left: 0, right: 96 })
  })

  it('uses per-button width from the overlay fallback for mixed layouts', () => {
    expect(windowControlsFallback({ left: 1, right: 2 }, 144)).toEqual({ left: 48, right: 96 })
  })

  it('keeps the whole right-side fallback when no button-layout is available', () => {
    expect(windowControlsFallback(null, 144)).toEqual({ left: 0, right: 144 })
    expect(windowControlsFallback(undefined, 0)).toEqual({ left: 0, right: 0 })
  })
})
