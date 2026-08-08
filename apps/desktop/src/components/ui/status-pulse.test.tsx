import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { StatusPulse } from './status-pulse'

interface PlayedAnimation {
  cancel: ReturnType<typeof vi.fn>
  keyframes: Keyframe[]
  options: KeyframeAnimationOptions
}

interface WindowStatePayload {
  isMinimized?: boolean
  isVisible?: boolean
}

let windowStateCallback: ((payload: WindowStatePayload) => void) | undefined

function installMatchMedia(matches: boolean) {
  vi.stubGlobal(
    'matchMedia',
    vi.fn(() => ({
      addEventListener: vi.fn(),
      matches,
      media: '(prefers-reduced-motion: reduce)',
      onchange: null,
      removeEventListener: vi.fn()
    }))
  )
}

function installWindowStateBridge() {
  const off = vi.fn()

  window.hermesDesktop = {
    onWindowStateChanged: vi.fn(callback => {
      windowStateCallback = callback

      return off
    })
  } as unknown as typeof window.hermesDesktop

  return off
}

describe('StatusPulse', () => {
  const played: PlayedAnimation[] = []

  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(window.document, 'hasFocus').mockReturnValue(true)
    installMatchMedia(false)
    installWindowStateBridge()
    Object.defineProperty(HTMLElement.prototype, 'animate', {
      configurable: true,
      value: (keyframes: Keyframe[], options: KeyframeAnimationOptions) => {
        const animation = {
          cancel: vi.fn(),
          keyframes,
          options
        }

        played.push(animation)

        return animation as unknown as Animation
      },
      writable: true
    })
  })

  afterEach(() => {
    cleanup()
    played.length = 0
    windowStateCallback = undefined
    Reflect.deleteProperty(HTMLElement.prototype, 'animate')
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('plays a finite ping and sleeps between pulses', () => {
    render(<StatusPulse kind="ping" opacity={0.7} />)

    expect(played).toHaveLength(1)
    expect(played[0]?.keyframes).toEqual([
      { opacity: 0.7, transform: 'scale(1)' },
      { opacity: 0, transform: 'scale(2)' }
    ])
    expect(played[0]?.options).toMatchObject({ duration: 400, iterations: 1 })
    expect(vi.getTimerCount()).toBe(1)

    act(() => vi.advanceTimersByTime(4_999))
    expect(played).toHaveLength(1)

    act(() => vi.advanceTimersByTime(1))
    expect(played).toHaveLength(2)
  })

  it('cancels scheduled work while minimized and restarts once visible', () => {
    const offWindowState = installWindowStateBridge()
    const mounted = render(<StatusPulse kind="opacity" />)

    expect(played).toHaveLength(1)

    act(() => windowStateCallback?.({ isMinimized: true, isVisible: false }))

    expect(played[0]?.cancel).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(0)

    act(() => vi.advanceTimersByTime(5_000))
    expect(played).toHaveLength(1)

    act(() => windowStateCallback?.({ isMinimized: false, isVisible: true }))
    expect(played).toHaveLength(2)

    mounted.unmount()
    expect(played[1]?.cancel).toHaveBeenCalledTimes(1)
    expect(offWindowState).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('stays static when reduced motion is requested', () => {
    installMatchMedia(true)

    render(<StatusPulse kind="opacity" />)

    expect(played).toHaveLength(0)
    expect(vi.getTimerCount()).toBe(0)
  })
})
