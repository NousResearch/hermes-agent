import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  createRendererLoopPauseController,
  installRendererAnimationPauseState,
  RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE
} from './renderer-loop-pause'

describe('installRendererAnimationPauseState', () => {
  afterEach(() => {
    document.documentElement.removeAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)
    vi.restoreAllMocks()
  })

  it('keeps visible animations running across blur and focus, and cleans up its root state', () => {
    let focused = true
    vi.spyOn(document, 'hasFocus').mockImplementation(() => focused)

    const dispose = installRendererAnimationPauseState()
    expect(document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)

    focused = false
    window.dispatchEvent(new Event('blur'))
    expect(document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)

    focused = true
    window.dispatchEvent(new Event('focus'))
    expect(document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)

    focused = false
    window.dispatchEvent(new Event('blur'))
    dispose()
    expect(document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)
  })
})

describe('createRendererLoopPauseController', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  function stubWindowStateBridge() {
    const listeners = new Set<(payload: { isMinimized?: boolean; isVisible?: boolean }) => void>()

    vi.stubGlobal('hermesDesktop', {
      onWindowStateChanged: (callback: (payload: { isMinimized?: boolean; isVisible?: boolean }) => void) => {
        listeners.add(callback)

        return () => listeners.delete(callback)
      }
    })

    return (payload: { isMinimized?: boolean; isVisible?: boolean }) => {
      for (const listener of listeners) {
        listener(payload)
      }
    }
  }

  it('starts paused when created after the window was hidden', () => {
    const emit = stubWindowStateBridge()
    const early = createRendererLoopPauseController(() => undefined)
    emit({ isMinimized: true })

    // The bridge reports changes only; a controller created now (a pulse
    // remounting in a hidden window) must not assume the window is shown.
    const late = createRendererLoopPauseController(() => undefined)

    expect(early.isPaused()).toBe(true)
    expect(late.isPaused()).toBe(true)

    emit({ isMinimized: false, isVisible: true })
    expect(late.isPaused()).toBe(false)

    early.dispose()
    late.dispose()
  })

  it('notifies each live controller once per change and none after dispose', () => {
    const emit = stubWindowStateBridge()
    const onA = vi.fn()
    const onB = vi.fn()
    const a = createRendererLoopPauseController(onA)
    const b = createRendererLoopPauseController(onB)

    emit({ isVisible: false })
    emit({ isVisible: false })
    expect(onA).toHaveBeenCalledTimes(1)
    expect(onB).toHaveBeenCalledTimes(1)

    b.dispose()
    emit({ isVisible: true })
    expect(onA).toHaveBeenCalledTimes(2)
    expect(onB).toHaveBeenCalledTimes(1)

    a.dispose()
  })
})
