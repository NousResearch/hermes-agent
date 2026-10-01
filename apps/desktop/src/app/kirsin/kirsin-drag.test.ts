import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useKirsinHeaderDrag } from './kirsin-drag'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const beginMove = vi.fn()
const endMove = vi.fn()
const moveBy = vi.fn()

function setWindowSize(width: number, height: number) {
  Object.defineProperty(window, 'outerWidth', { configurable: true, value: width })
  Object.defineProperty(window, 'outerHeight', { configurable: true, value: height })
}

/** jsdom has no pointer capture. */
function pressTarget({ onControl = false }: { onControl?: boolean } = {}) {
  const target = document.createElement(onControl ? 'button' : 'div')

  if (onControl) {
    target.setAttribute('data-kirsin-ctl', '')
  }

  target.setPointerCapture = vi.fn()
  target.hasPointerCapture = vi.fn(() => false)
  target.releasePointerCapture = vi.fn()
  document.body.append(target)

  return target
}

function pressEvent(target: Element, pointerId: number, button = 0) {
  return { button, currentTarget: target, pointerId, target } as never
}

beforeEach(() => {
  beginMove.mockClear()
  endMove.mockClear()
  moveBy.mockClear()
  setWindowSize(576, 560)
  desktopWindow.hermesDesktop = { kirsin: { beginMove, endMove, moveBy } } as unknown as Window['hermesDesktop']
})

afterEach(() => {
  document.body.innerHTML = ''

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('useKirsinHeaderDrag', () => {
  it('drags immediately on a primary press, re-pinning the size snapshotted at press', () => {
    const target = pressTarget()
    const { result } = renderHook(() => useKirsinHeaderDrag())

    act(() => result.current.onPointerDown(pressEvent(target, 1)))

    expect(beginMove).toHaveBeenCalledTimes(1)
    expect(result.current.dragging).toBe(true)

    act(() => void window.dispatchEvent(new PointerEvent('pointermove', { pointerId: 1 })))
    expect(moveBy).toHaveBeenCalledWith({ width: 576, height: 560 })

    // A frameless window can drift wider mid-drag — main must keep pinning the
    // size snapshotted at press, never the drifted one (that is how the HUD
    // growth compounded on Windows).
    setWindowSize(900, 500)
    act(() => void window.dispatchEvent(new PointerEvent('pointermove', { pointerId: 1 })))
    expect(moveBy).toHaveBeenLastCalledWith({ width: 576, height: 560 })

    act(() => void window.dispatchEvent(new PointerEvent('pointerup', { pointerId: 1 })))
    expect(endMove).toHaveBeenCalledTimes(1)
  })

  it('ignores non-primary buttons', () => {
    const target = pressTarget()
    const { result } = renderHook(() => useKirsinHeaderDrag())

    act(() => result.current.onPointerDown(pressEvent(target, 2, 1)))

    expect(beginMove).not.toHaveBeenCalled()
    expect(result.current.dragging).toBe(false)
  })

  it('never starts a drag from a header control', () => {
    const target = pressTarget({ onControl: true })
    const { result } = renderHook(() => useKirsinHeaderDrag())

    act(() => result.current.onPointerDown(pressEvent(target, 1)))

    expect(beginMove).not.toHaveBeenCalled()
    expect(result.current.dragging).toBe(false)
  })

  it('ignores moves from a different pointer', () => {
    const target = pressTarget()
    const { result } = renderHook(() => useKirsinHeaderDrag())

    act(() => result.current.onPointerDown(pressEvent(target, 1)))

    act(() => void window.dispatchEvent(new PointerEvent('pointermove', { pointerId: 99 })))
    expect(moveBy).not.toHaveBeenCalled()
  })

  it('keeps the grab alive when crossing a display cancels the pointer', () => {
    const target = pressTarget()
    const { result } = renderHook(() => useKirsinHeaderDrag())

    act(() => result.current.onPointerDown(pressEvent(target, 3)))
    act(() => void window.dispatchEvent(new PointerEvent('pointermove', { pointerId: 3 })))
    expect(beginMove).toHaveBeenCalledTimes(1)

    act(() => void window.dispatchEvent(new PointerEvent('pointercancel', { cancelable: true, pointerId: 3 })))
    expect(endMove).not.toHaveBeenCalled()
    expect(moveBy).toHaveBeenLastCalledWith({ width: 576, height: 560 })
    expect(target.setPointerCapture).toHaveBeenCalled()

    act(() => void window.dispatchEvent(new MouseEvent('mouseup')))
    expect(endMove).toHaveBeenCalledTimes(1)
  })

  it('cleans up drag state on unmount without a pointless IPC round-trip', () => {
    const target = pressTarget()
    const { result, unmount } = renderHook(() => useKirsinHeaderDrag())

    act(() => result.current.onPointerDown(pressEvent(target, 1)))
    expect(beginMove).toHaveBeenCalledTimes(1)

    // Unmount mid-drag: the window is going away, so the move loop stops on its
    // own — sending end-move to a dying webContents is a pointless round-trip.
    // The cleanup only needs to reset the drag state.
    unmount()
    expect(endMove).not.toHaveBeenCalled()
  })
})
