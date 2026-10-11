import { afterEach, describe, expect, it, vi } from 'vitest'

import { installWindowStateBridge, setDocumentHidden } from '../test/window-state'

import {
  installRendererAnimationPauseState,
  RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE,
  setRendererBusyOverride
} from './renderer-loop-pause'

describe('installRendererAnimationPauseState', () => {
  afterEach(() => {
    setRendererBusyOverride(false)
    setDocumentHidden(false)
    document.documentElement.removeAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
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

  it('does not pause animations while busy even when hidden or minimized', () => {
    const windowState = installWindowStateBridge()
    const dispose = installRendererAnimationPauseState()

    setRendererBusyOverride(true)

    setDocumentHidden(true)
    document.dispatchEvent(new Event('visibilitychange'))
    expect(document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)

    windowState.emit({ isMinimized: true, isVisible: false })
    expect(document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)

    dispose()
  })

  it('clears a paused attribute once a session becomes busy', () => {
    const windowState = installWindowStateBridge()
    const dispose = installRendererAnimationPauseState()

    windowState.emit({ isMinimized: true, isVisible: false })
    expect(document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(true)

    setRendererBusyOverride(true)
    expect(document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)

    dispose()
  })

  it('pauses animations when idle and minimized (fail-open)', () => {
    const windowState = installWindowStateBridge()
    const dispose = installRendererAnimationPauseState()

    setRendererBusyOverride(false)
    windowState.emit({ isMinimized: true, isVisible: false })
    expect(document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(true)

    dispose()
  })
})
