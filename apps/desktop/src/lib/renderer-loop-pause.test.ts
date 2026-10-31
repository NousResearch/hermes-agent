import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  installRendererAnimationPauseState,
  RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE,
  RENDERER_ANIMATIONS_UNFOCUSED_ATTRIBUTE
} from './renderer-loop-pause'

const root = () => document.documentElement

function clear() {
  root().removeAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)
  root().removeAttribute(RENDERER_ANIMATIONS_UNFOCUSED_ATTRIBUTE)
}

describe('installRendererAnimationPauseState', () => {
  afterEach(() => {
    clear()
    vi.restoreAllMocks()
  })

  it('leaves a visible window animating across blur and focus by default', () => {
    let focused = true
    vi.spyOn(document, 'hasFocus').mockImplementation(() => focused)

    const dispose = installRendererAnimationPauseState()
    expect(root().hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)
    expect(root().hasAttribute(RENDERER_ANIMATIONS_UNFOCUSED_ATTRIBUTE)).toBe(false)

    focused = false
    window.dispatchEvent(new Event('blur'))
    expect(root().hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)
    expect(root().hasAttribute(RENDERER_ANIMATIONS_UNFOCUSED_ATTRIBUTE)).toBe(false)

    dispose()
  })

  it('stops only decoration while the window is visible but unfocused when asked to', () => {
    let focused = true
    vi.spyOn(document, 'hasFocus').mockImplementation(() => focused)

    const dispose = installRendererAnimationPauseState({ pauseWhenUnfocused: true })
    expect(root().hasAttribute(RENDERER_ANIMATIONS_UNFOCUSED_ATTRIBUTE)).toBe(false)

    focused = false
    window.dispatchEvent(new Event('blur'))
    // Soft pause, not the hard one: status cues (spinners, cursor) keep moving.
    expect(root().hasAttribute(RENDERER_ANIMATIONS_UNFOCUSED_ATTRIBUTE)).toBe(true)
    expect(root().hasAttribute(RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE)).toBe(false)

    focused = true
    window.dispatchEvent(new Event('focus'))
    expect(root().hasAttribute(RENDERER_ANIMATIONS_UNFOCUSED_ATTRIBUTE)).toBe(false)

    dispose()
    expect(root().hasAttribute(RENDERER_ANIMATIONS_UNFOCUSED_ATTRIBUTE)).toBe(false)
  })

  it('heals a missed focus event rather than stranding the pause', () => {
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)

    const dispose = installRendererAnimationPauseState({ pauseWhenUnfocused: true })

    // The blur arrives but the matching focus event never does (system dialog,
    // IME handoff, focus-follows-mouse). The live check must win.
    window.dispatchEvent(new Event('blur'))
    expect(root().hasAttribute(RENDERER_ANIMATIONS_UNFOCUSED_ATTRIBUTE)).toBe(false)

    // And a stray focus event on a genuinely unfocused document must not clear
    // the pause either.
    vi.mocked(document.hasFocus).mockReturnValue(false)
    window.dispatchEvent(new Event('focus'))
    window.dispatchEvent(new Event('blur'))
    expect(root().hasAttribute(RENDERER_ANIMATIONS_UNFOCUSED_ATTRIBUTE)).toBe(true)

    dispose()
  })
})
