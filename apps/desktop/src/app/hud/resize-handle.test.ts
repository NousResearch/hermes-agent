import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { HUD_RESIZE_DIRECTIONS, hudResizeBounds, hudResizeDirections, useHudResizeHandle } from './resize-handle'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
const setBounds = vi.fn()

function setWindowBounds(x: number, y: number, width: number, height: number): void {
  Object.defineProperty(window, 'screenX', { configurable: true, value: x })
  Object.defineProperty(window, 'screenY', { configurable: true, value: y })
  Object.defineProperty(window, 'outerWidth', { configurable: true, value: width })
  Object.defineProperty(window, 'outerHeight', { configurable: true, value: height })
}

function resizeTarget(): HTMLElement {
  const target = document.createElement('div')
  target.setPointerCapture = vi.fn()
  target.hasPointerCapture = vi.fn(() => false)
  target.releasePointerCapture = vi.fn()
  document.body.append(target)

  return target
}

beforeEach(() => {
  setBounds.mockClear()
  setWindowBounds(100, 200, 620, 320)
  desktopWindow.hermesDesktop = { hud: { setBounds } } as unknown as Window['hermesDesktop']
})

afterEach(() => {
  document.body.innerHTML = ''

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('hudResizeBounds', () => {
  const origin = { x: 100, y: 200, width: 620, height: 320 }

  it('resizes from every CanvasTTY edge and corner while preserving the opposite edges', () => {
    expect(hudResizeBounds(origin, 'n', 40, 30)).toEqual({ x: 100, y: 230, width: 620, height: 290 })
    expect(hudResizeBounds(origin, 'ne', 40, 30)).toEqual({ x: 100, y: 230, width: 660, height: 290 })
    expect(hudResizeBounds(origin, 'e', 40, 30)).toEqual({ x: 100, y: 200, width: 660, height: 320 })
    expect(hudResizeBounds(origin, 'se', 40, 30)).toEqual({ x: 100, y: 200, width: 660, height: 350 })
    expect(hudResizeBounds(origin, 's', 40, 30)).toEqual({ x: 100, y: 200, width: 620, height: 350 })
    expect(hudResizeBounds(origin, 'sw', 40, 30)).toEqual({ x: 140, y: 200, width: 580, height: 350 })
    expect(hudResizeBounds(origin, 'w', 40, 30)).toEqual({ x: 140, y: 200, width: 580, height: 320 })
    expect(hudResizeBounds(origin, 'nw', 40, 30)).toEqual({ x: 140, y: 230, width: 580, height: 290 })
  })

  it('clamps at the minimum without moving the opposite edge', () => {
    expect(hudResizeBounds(origin, 'nw', 1000, 1000)).toEqual({ x: 340, y: 360, width: 380, height: 160 })
    expect(hudResizeBounds(origin, 'se', -1000, -1000)).toEqual({ x: 100, y: 200, width: 380, height: 160 })
  })
})

describe('hudResizeDirections', () => {
  it('keeps every edge and corner on platforms with global window positioning', () => {
    expect(hudResizeDirections(true)).toBe(HUD_RESIZE_DIRECTIONS)
  })

  it('exposes only position-preserving handles when the client cannot place the window', () => {
    expect(hudResizeDirections(false)).toEqual(['e', 'se', 's'])
  })
})

describe('useHudResizeHandle', () => {
  it('sends the selected edge geometry through the HUD bridge', () => {
    const target = resizeTarget()
    const { result } = renderHook(() => useHudResizeHandle())

    act(() =>
      result.current.onPointerDown(
        {
          button: 0,
          currentTarget: target,
          pointerId: 4,
          preventDefault: vi.fn(),
          screenX: 300,
          screenY: 300,
          stopPropagation: vi.fn()
        } as never,
        'nw'
      )
    )
    act(() => void window.dispatchEvent(new PointerEvent('pointermove', { pointerId: 4, screenX: 250, screenY: 275 })))

    expect(setBounds).toHaveBeenCalledWith({ x: 50, y: 175, width: 670, height: 345 })
  })
})
