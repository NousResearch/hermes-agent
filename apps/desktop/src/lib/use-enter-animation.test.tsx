import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useEnterAnimation } from './use-enter-animation'

const originalAnimate = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'animate')
const originalInnerWidth = Object.getOwnPropertyDescriptor(window, 'innerWidth')
const originalMatchMedia = Object.getOwnPropertyDescriptor(window, 'matchMedia')

const animate = vi.fn(() => ({}) as Animation)

function installMatchMedia(reducedMotion: boolean) {
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    value: vi.fn((query: string) => ({
      addEventListener: vi.fn(),
      addListener: vi.fn(),
      dispatchEvent: vi.fn(),
      matches: query === '(prefers-reduced-motion: reduce)' && reducedMotion,
      media: query,
      onchange: null,
      removeEventListener: vi.fn(),
      removeListener: vi.fn()
    }))
  })
}

function restoreProperty(target: object, key: PropertyKey, descriptor?: PropertyDescriptor) {
  if (descriptor) {
    Object.defineProperty(target, key, descriptor)
  } else {
    Reflect.deleteProperty(target, key)
  }
}

describe('useEnterAnimation', () => {
  beforeEach(() => {
    animate.mockClear()
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 390 })
    Object.defineProperty(HTMLElement.prototype, 'animate', { configurable: true, value: animate })
    installMatchMedia(false)
  })

  afterEach(() => {
    cleanup()
    restoreProperty(HTMLElement.prototype, 'animate', originalAnimate)
    restoreProperty(window, 'innerWidth', originalInnerWidth)
    restoreProperty(window, 'matchMedia', originalMatchMedia)
    vi.restoreAllMocks()
  })

  it('keeps the subtle message entry motion at a mobile viewport width', async () => {
    const { result } = renderHook(() => useEnterAnimation(true, 'mobile-entry'))
    const element = document.createElement('div')
    document.body.append(element)

    act(() => result.current(element))
    await Promise.resolve()

    expect(window.innerWidth).toBe(390)
    expect(animate).toHaveBeenCalledOnce()
    expect(animate).toHaveBeenCalledWith(
      [
        { opacity: 0, transform: 'translateY(0.375rem)' },
        { opacity: 1, transform: 'translateY(0)' }
      ],
      { duration: 180, easing: 'cubic-bezier(0.16, 1, 0.3, 1)', fill: 'both' }
    )

    act(() => result.current(element))
    expect(animate).toHaveBeenCalledOnce()
  })

  it('does not animate when the user requests reduced motion', () => {
    installMatchMedia(true)
    const { result } = renderHook(() => useEnterAnimation(true, 'reduced-motion-entry'))
    const element = document.createElement('div')
    document.body.append(element)

    act(() => result.current(element))

    expect(animate).not.toHaveBeenCalled()
  })
})
