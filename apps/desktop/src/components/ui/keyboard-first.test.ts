import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { onReleaseTypingFocus, releaseTypingFocus, usePointerQuiet } from './keyboard-first'

const move = () => act(() => void window.dispatchEvent(new MouseEvent('mousemove', { bubbles: true })))
const scroll = () => act(() => void window.dispatchEvent(new WheelEvent('wheel', { bubbles: true })))

afterEach(() => {
  vi.restoreAllMocks()
})

describe('usePointerQuiet', () => {
  it('starts quiet so a list opening under a parked cursor cannot hover-select', () => {
    const { result } = renderHook(() => usePointerQuiet())

    expect(result.current).toBe(true)
  })

  it('wakes on real pointer movement and stays awake', () => {
    const { result } = renderHook(() => usePointerQuiet())

    move()
    expect(result.current).toBe(false)

    // A later re-render must not re-arm the guard mid-interaction.
    move()
    expect(result.current).toBe(false)
  })

  it('treats a scroll as a hand on the mouse', () => {
    const { result } = renderHook(() => usePointerQuiet())

    scroll()
    expect(result.current).toBe(false)
  })

  it('re-arms for the next overlay (each open mounts a fresh guard)', () => {
    const first = renderHook(() => usePointerQuiet())

    move()
    expect(first.result.current).toBe(false)
    first.unmount()

    expect(renderHook(() => usePointerQuiet()).result.current).toBe(true)
  })

  it('drops its listeners on unmount', () => {
    const remove = vi.spyOn(window, 'removeEventListener')

    renderHook(() => usePointerQuiet()).unmount()

    expect(remove.mock.calls.map(([type]) => type)).toEqual(expect.arrayContaining(['mousemove', 'wheel']))
  })
})

describe('releaseTypingFocus', () => {
  it('notifies subscribers, and stops once unsubscribed', () => {
    const handler = vi.fn()
    const off = onReleaseTypingFocus(handler)

    releaseTypingFocus()
    expect(handler).toHaveBeenCalledTimes(1)

    off()
    releaseTypingFocus()
    expect(handler).toHaveBeenCalledTimes(1)
  })
})
