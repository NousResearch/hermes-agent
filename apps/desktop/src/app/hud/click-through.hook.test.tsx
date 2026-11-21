// @vitest-environment jsdom
import { renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useHudClickThrough } from './click-through'

function stubHud() {
  const setIgnoreMouse = vi.fn()
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    hud: { setIgnoreMouse }
  }

  return setIgnoreMouse
}

describe('useHudClickThrough', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
  })

  afterEach(() => {
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('stays solid until the pointer has been seen', () => {
    const setIgnoreMouse = stubHud()
    const shell = document.createElement('div')
    shell.setAttribute('data-hud-shell', '')
    document.body.append(shell)

    renderHook(() => useHudClickThrough({ current: shell }))

    // No pointer observation yet: the window must not arm click-through on a
    // guess, or a HUD opened under a stationary cursor swallows its first click.
    expect(setIgnoreMouse.mock.calls.every(([ignore]) => ignore === false)).toBe(true)
  })

  it('goes click-through once the pointer is known to be over dead space', () => {
    const setIgnoreMouse = stubHud()
    const shell = document.createElement('div')
    shell.setAttribute('data-hud-shell', '')
    document.body.append(shell)

    // jsdom has no layout, so the hit test is stubbed: outside the shell the
    // document body is what a real hit test reports — the "nothing painted
    // here" answer the click-through decision is built on.
    document.elementFromPoint = () => document.body

    renderHook(() => useHudClickThrough({ current: shell }))

    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 900, clientY: 900 }))

    expect(setIgnoreMouse).toHaveBeenLastCalledWith(true)
  })
})
