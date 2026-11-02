import { act, render, renderHook } from '@testing-library/react'
import { createElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useHudComposerDrag } from './composer-drag'

/** Matches LONG_PRESS_MS in composer-drag.ts. */
const LONG_PRESS_MS = 140

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const beginMove = vi.fn()
const endMove = vi.fn()
const moveBy = vi.fn()
const setWorkspaceTransfer = vi.fn()

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
  beginMove.mockClear()
  endMove.mockClear()
  moveBy.mockClear()
  setWorkspaceTransfer.mockClear()
  setWindowSize(620, 320)
  desktopWindow.hermesDesktop = {
    hud: { beginMove, endMove, moveBy, setWorkspaceTransfer }
  } as unknown as Window['hermesDesktop']
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

    expect(beginMove).toHaveBeenCalledTimes(1)
    expect(moveBy).toHaveBeenCalledWith({ width: 620, height: 320 })
    expect(setWorkspaceTransfer).not.toHaveBeenCalled()

    // A window that drifted wider mid-drag must not feed its new size back in —
    // that is exactly how the Windows growth compounded.
    setWindowSize(900, 500)
    act(() => void window.dispatchEvent(new PointerEvent('pointermove', { pointerId: 1, screenX: 115, screenY: 215 })))

    expect(moveBy).toHaveBeenLastCalledWith({ width: 620, height: 320 })

    act(() => void window.dispatchEvent(new PointerEvent('pointerup', { pointerId: 1 })))

    expect(endMove).toHaveBeenCalledTimes(1)
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
    expect(beginMove).not.toHaveBeenCalled()
    expect(setWorkspaceTransfer).not.toHaveBeenCalled()
  })

  it('spans X11 workspaces only for an armed grab, then pins to the current desktop on release', () => {
    const target = pressTarget()

    const { result } = renderHook(() => useHudComposerDrag(true, { controlDrag: true, workspaceTransfer: true }))

    act(() =>
      result.current.onPointerDown({
        button: 0,
        ctrlKey: true,
        currentTarget: target,
        pointerId: 9,
        preventDefault: vi.fn(),
        screenX: 100,
        screenY: 200
      } as never)
    )

    expect(setWorkspaceTransfer).toHaveBeenLastCalledWith(true)
    expect(beginMove).toHaveBeenCalledTimes(1)

    act(() => void window.dispatchEvent(new PointerEvent('pointerup', { pointerId: 9 })))

    expect(setWorkspaceTransfer).toHaveBeenLastCalledWith(false)
    expect(endMove).toHaveBeenCalledTimes(1)
  })

  it('moves immediately with Ctrl over selected text without destroying the selection', () => {
    function Harness() {
      const { onPointerDown } = useHudComposerDrag(true, { controlDrag: true })

      return createElement(
        'form',
        { 'data-testid': 'composer', onPointerDownCapture: onPointerDown },
        createElement(
          'div',
          { contentEditable: true, 'data-testid': 'editor', suppressContentEditableWarning: true, tabIndex: 0 },
          'selected text stays selected'
        )
      )
    }

    const { getByTestId } = render(createElement(Harness))
    const editor = getByTestId('editor')
    const text = editor.firstChild!
    editor.focus()
    const range = document.createRange()
    range.setStart(text, 0)
    range.setEnd(text, 13)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(range)

    const down = new PointerEvent('pointerdown', {
      bubbles: true,
      button: 0,
      buttons: 1,
      cancelable: true,
      ctrlKey: true,
      pointerId: 7,
      screenX: 300,
      screenY: 180
    })

    act(() => void editor.dispatchEvent(down))

    expect(down.defaultPrevented).toBe(true)
    expect(document.activeElement).toBe(editor)
    expect(selection.toString()).toBe('selected text')

    const dragStart = new Event('dragstart', { bubbles: true, cancelable: true })
    const selectStart = new Event('selectstart', { bubbles: true, cancelable: true })
    act(() => void editor.dispatchEvent(dragStart))
    act(() => void editor.dispatchEvent(selectStart))

    expect(dragStart.defaultPrevented).toBe(true)
    expect(selectStart.defaultPrevented).toBe(true)

    act(
      () =>
        void window.dispatchEvent(
          new PointerEvent('pointermove', { buttons: 1, cancelable: true, pointerId: 7, screenX: 301, screenY: 182 })
        )
    )

    expect(moveBy).toHaveBeenCalledWith({ width: 620, height: 320 })
    expect(document.activeElement).toBe(editor)
    expect(selection.toString()).toBe('selected text')
  })

  it('keeps the grab alive when crossing a display cancels the pointer', () => {
    const target = pressTarget()
    const { result } = renderHook(() => useHudComposerDrag(true))

    act(() =>
      result.current.onPointerDown({
        button: 0,
        currentTarget: target,
        pointerId: 3,
        screenX: 1880,
        screenY: 400
      } as never)
    )
    act(() => void vi.advanceTimersByTime(LONG_PRESS_MS))
    act(() => void window.dispatchEvent(new PointerEvent('pointermove', { pointerId: 3, screenX: 1900, screenY: 400 })))

    expect(beginMove).toHaveBeenCalledTimes(1)
    expect(endMove).not.toHaveBeenCalled()

    act(
      () =>
        void window.dispatchEvent(
          new PointerEvent('pointercancel', { cancelable: true, pointerId: 3, screenX: 2100, screenY: 400 })
        )
    )

    expect(endMove).not.toHaveBeenCalled()
    expect(moveBy).toHaveBeenLastCalledWith({ width: 620, height: 320 })
    expect(target.setPointerCapture).toHaveBeenCalled()

    act(() => void window.dispatchEvent(new MouseEvent('mouseup')))

    expect(endMove).toHaveBeenCalledTimes(1)
  })
})
