import { act, cleanup, fireEvent, renderHook } from '@testing-library/react'
import type { RefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { usePetRoam } from './use-pet-roam'

const rect = (left: number, top: number, width: number, height: number): DOMRect =>
  ({
    bottom: top + height,
    height,
    left,
    right: left + width,
    top,
    width,
    x: left,
    y: top,
    toJSON: () => ({})
  }) as DOMRect

describe('usePetRoam scheduling', () => {
  let visibilityDescriptor: PropertyDescriptor | undefined

  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(Math, 'random').mockReturnValue(0)
    visibilityDescriptor = Object.getOwnPropertyDescriptor(window.document, 'hidden')
    Object.defineProperty(window.document, 'hidden', { configurable: true, value: false })
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.useRealTimers()
    vi.restoreAllMocks()

    if (visibilityDescriptor) {
      Object.defineProperty(window.document, 'hidden', visibilityDescriptor)
    } else {
      Reflect.deleteProperty(window.document, 'hidden')
    }
  })

  it('uses a timeout while paused and requests frames only when movement resumes', () => {
    const frames: FrameRequestCallback[] = []
    let interacting = false

    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      frames.push(callback)

      return frames.length
    })
    vi.stubGlobal('cancelAnimationFrame', vi.fn())

    const element = window.document.createElement('div')

    vi.spyOn(element, 'getBoundingClientRect').mockReturnValue(rect(100, window.innerHeight - 40 + 4, 40, 40))

    const { unmount } = renderHook(() =>
      usePetRoam({
        commit: vi.fn(),
        containerRef: { current: element } as RefObject<HTMLDivElement>,
        enabled: true,
        isInteracting: () => interacting,
        loopMs: 800,
        overlayOpen: false,
        petH: 40,
        petW: 40
      })
    )

    expect(frames).toHaveLength(0)

    act(() => {
      vi.advanceTimersByTime(399)
    })
    expect(frames).toHaveLength(0)

    act(() => {
      vi.advanceTimersByTime(1)
    })
    expect(frames).toHaveLength(1)

    act(() => {
      frames.shift()!(performance.now())
    })

    // Math.random() = 0 selects another rest beat, which should return to a
    // timeout instead of starting a permanent frame loop.
    expect(frames).toHaveLength(0)
    expect(vi.getTimerCount()).toBe(1)

    interacting = true
    fireEvent.pointerDown(window)

    expect(frames).toHaveLength(1)
    expect(vi.getTimerCount()).toBe(0)

    act(() => {
      frames.shift()!(performance.now())
    })
    expect(frames).toHaveLength(1)

    interacting = false

    act(() => {
      frames.shift()!(performance.now())
    })

    expect(frames).toHaveLength(0)
    expect(vi.getTimerCount()).toBe(1)

    unmount()
    expect(vi.getTimerCount()).toBe(0)
  })
})
