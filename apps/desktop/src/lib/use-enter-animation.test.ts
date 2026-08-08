import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useEnterAnimation } from './use-enter-animation'

// The hook is a callback ref; we don't need React to exercise it — invoking
// the returned ref with a fake element is exactly what React does on mount.
// We render via a tiny manual harness to satisfy the rules-of-hooks (the hook
// calls useRef/useCallback). renderHook from @testing-library keeps it honest.
import { renderHook } from '@testing-library/react'

type FakeAnimation = {
  finish: ReturnType<typeof vi.fn>
  finished: Promise<void>
  _resolve: () => void
}

function makeFakeElement() {
  const animations: Array<{ keyframes: unknown; options: unknown; anim: FakeAnimation }> = []

  const el = {
    isConnected: true,
    animate(keyframes: unknown, options: unknown) {
      let resolve!: () => void
      const finished = new Promise<void>(r => {
        resolve = r
      })
      const anim: FakeAnimation = {
        finish: vi.fn(() => resolve()),
        finished,
        _resolve: resolve
      }
      animations.push({ keyframes, options, anim })

      return anim as unknown as Animation
    }
  } as unknown as HTMLElement & { animate: HTMLElement['animate'] }

  return { el, animations }
}

function setHidden(hidden: boolean) {
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden })
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    get: () => (hidden ? 'hidden' : 'visible')
  })
}

describe('useEnterAnimation', () => {
  beforeEach(() => {
    setHidden(false)
    // Default: motion allowed.
    window.matchMedia = vi.fn().mockReturnValue({ matches: false }) as unknown as typeof window.matchMedia
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('plays the enter animation when the document is visible', () => {
    const { result } = renderHook(() => useEnterAnimation(true))
    const { el, animations } = makeFakeElement()

    result.current(el)

    expect(animations).toHaveLength(1)
    // End keyframe must be fully opaque.
    expect((animations[0].keyframes as Array<{ opacity: number }>).at(-1)?.opacity).toBe(1)
  })

  it('does NOT create an opacity:0-pinned animation while the document is hidden', () => {
    setHidden(true)
    const { result } = renderHook(() => useEnterAnimation(true))
    const { el, animations } = makeFakeElement()

    result.current(el)

    // No animation at all → element keeps its natural (visible) styles, so it
    // can never be stranded transparent behind a paused timeline.
    expect(animations).toHaveLength(0)
  })

  it('force-finishes a mid-flight animation when the document becomes hidden', () => {
    const { result } = renderHook(() => useEnterAnimation(true))
    const { el, animations } = makeFakeElement()

    result.current(el)
    expect(animations).toHaveLength(1)

    // Simulate alt-tab mid-animation: window hides, visibilitychange fires.
    setHidden(true)
    document.dispatchEvent(new Event('visibilitychange'))

    // finish() jumps to the end keyframe (opacity:1) synchronously even though
    // the timeline is paused — the message can't be left invisible.
    expect(animations[0].anim.finish).toHaveBeenCalledTimes(1)
  })

  it('respects prefers-reduced-motion (no animation)', () => {
    window.matchMedia = vi.fn().mockReturnValue({ matches: true }) as unknown as typeof window.matchMedia
    const { result } = renderHook(() => useEnterAnimation(true))
    const { el, animations } = makeFakeElement()

    result.current(el)

    expect(animations).toHaveLength(0)
  })

  it('detaches the visibilitychange listener once the animation settles', async () => {
    const addSpy = vi.spyOn(document, 'addEventListener')
    const removeSpy = vi.spyOn(document, 'removeEventListener')

    const { result } = renderHook(() => useEnterAnimation(true))
    const { el, animations } = makeFakeElement()

    result.current(el)
    expect(addSpy).toHaveBeenCalledWith('visibilitychange', expect.any(Function))

    // Natural completion resolves `finished` → cleanup runs.
    animations[0].anim._resolve()
    await animations[0].anim.finished

    expect(removeSpy).toHaveBeenCalledWith('visibilitychange', expect.any(Function))
  })
})
