import { afterEach, describe, expect, it, vi } from 'vitest'

import { installRendererAnimationPauseState, RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE } from './renderer-loop-pause'

const paused = () => document.documentElement.hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)

describe('installRendererAnimationPauseState', () => {
  afterEach(() => {
    document.documentElement.removeAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)
    vi.restoreAllMocks()
  })

  it('keeps animating while merely unfocused, since the window is still visible', () => {
    // Regression: pausing on blur froze every continuous animation whenever the
    // user clicked another window, which on a multi-monitor desktop leaves a
    // fully visible window looking hung. Only real invisibility may pause.
    let focused = true
    vi.spyOn(document, 'hasFocus').mockImplementation(() => focused)

    const dispose = installRendererAnimationPauseState()

    expect(paused()).toBe(false)

    focused = false
    window.dispatchEvent(new Event('blur'))
    expect(paused()).toBe(false)

    focused = true
    window.dispatchEvent(new Event('focus'))
    expect(paused()).toBe(false)

    dispose()
    expect(paused()).toBe(false)
  })

  it('pauses when the document is actually hidden, and cleans up its root state', () => {
    vi.spyOn(document, 'hasFocus').mockReturnValue(false)
    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')

    const dispose = installRendererAnimationPauseState()
    expect(paused()).toBe(true)

    visibility.mockReturnValue('visible')
    document.dispatchEvent(new Event('visibilitychange'))
    expect(paused()).toBe(false)

    visibility.mockReturnValue('hidden')
    document.dispatchEvent(new Event('visibilitychange'))
    expect(paused()).toBe(true)

    dispose()
    expect(paused()).toBe(false)
  })
})
