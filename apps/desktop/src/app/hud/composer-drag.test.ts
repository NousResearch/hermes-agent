import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useHudComposerDrag } from './composer-drag'

/** Matches LONG_PRESS_MS in composer-drag.ts. */
const LONG_PRESS_MS = 140

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const moveBy = vi.fn()

function setWindowSize(width: number, height: number) {
  Object.defineProperty(window, 'outerWidth', { configurable: true, value: width })
  Object.defineProperty(window, 'outerHeight', { configurable: true, value: height })
}

/** jsdom has no pointer capture. */
function pressTarget() {
  const target = document.createElement('div')
  target.setPointerCapture = vi.fn()
  target.hasPointerCapture = vi.fn(() => false)
  target.releasePointerCapture = vi.fn()
  document.body.append(target)

  return target
}

beforeEach(() => {
  vi.useFakeTimers()
  moveBy.mockClear()
  setWindowSize(620, 320)
  desktopWindow.hermesDesktop = { hud: { moveBy } } as unknown as Window['hermesDesktop']
})

afterEach(() => {
  vi.useRealTimers()
  document.body.innerHTML = ''

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('useHudComposerDrag', () => {
  it('sends every move with the size snapshotted at press, so main can pin it', () => {
    const target = pressTarget()
    const { result } = renderHook(() => useHudComposerDrag(true))

    act(() =>
      result.current.onPointerDown({
        button: 0,
        currentTarget: target,
        pointerId: 1,
        screenX: 100,
        screenY: 200
      } as never)
    )
    act(() => void vi.advanceTimersByTime(LONG_PRESS_MS))
    act(() => void window.dispatchEvent(new PointerEvent('pointermove', { pointerId: 1, screenX: 110, screenY: 210 })))

    expect(moveBy).toHaveBeenCalledWith({ x: 10, y: 10, width: 620, height: 320 })

    // A window that drifted wider mid-drag must not feed its new size back in —
    // that is exactly how the Windows growth compounded.
    setWindowSize(900, 500)
    act(() => void window.dispatchEvent(new PointerEvent('pointermove', { pointerId: 1, screenX: 115, screenY: 215 })))

    expect(moveBy).toHaveBeenLastCalledWith({ x: 5, y: 5, width: 620, height: 320 })
  })

  it('does not move the window until the hold arms', () => {
    const target = pressTarget()
    const { result } = renderHook(() => useHudComposerDrag(true))

    act(() =>
      result.current.onPointerDown({
        button: 0,
        currentTarget: target,
        pointerId: 1,
        screenX: 100,
        screenY: 200
      } as never)
    )
    act(() => void window.dispatchEvent(new PointerEvent('pointermove', { pointerId: 1, screenX: 102, screenY: 201 })))

    expect(moveBy).not.toHaveBeenCalled()
  })
})
