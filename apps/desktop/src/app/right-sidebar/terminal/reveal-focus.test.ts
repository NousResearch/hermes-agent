import { afterEach, describe, expect, it, vi } from 'vitest'

import { focusRevealedTerminal } from './reveal-focus'

afterEach(() => {
  vi.unstubAllGlobals()
  document.body.innerHTML = ''
})

function installRaf() {
  const frames: FrameRequestCallback[] = []
  let nextId = 1

  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    frames.push(callback)

    return nextId++
  })
  vi.stubGlobal('cancelAnimationFrame', () => {})

  // Run every frame scheduled so far, once — new rAF calls made while flushing
  // land in the NEXT drain, exactly like real frames.
  return () => frames.splice(0).forEach(callback => callback(0))
}

function mountTerminal(visible: boolean): HTMLTextAreaElement {
  const host = document.createElement('div')

  host.dataset.terminal = ''

  if (!visible) {
    host.className = 'invisible pointer-events-none'
  }

  const textarea = document.createElement('textarea')

  textarea.className = 'xterm-helper-textarea'
  host.appendChild(textarea)
  document.body.appendChild(host)

  return textarea
}

describe('focusRevealedTerminal', () => {
  it('focuses the visible terminal once the reveal settles', () => {
    const frame = installRaf()
    const hidden = mountTerminal(false)
    const visible = mountTerminal(true)

    focusRevealedTerminal()
    frame()

    expect(document.activeElement).toBe(visible)
    expect(document.activeElement).not.toBe(hidden)
  })

  it('re-asserts focus when another surface steals it back', () => {
    const frame = installRaf()
    const visible = mountTerminal(true)

    focusRevealedTerminal()
    frame()
    expect(document.activeElement).toBe(visible)

    // The composer (or any autofocus effect) wins focus after the reveal.
    visible.blur()
    expect(document.activeElement).not.toBe(visible)

    frame()
    expect(document.activeElement).toBe(visible)
  })

  it('stops once focus sits inside the terminal', () => {
    const frame = installRaf()
    const visible = mountTerminal(true)
    const focusSpy = vi.spyOn(visible, 'focus')

    focusRevealedTerminal()
    frame()
    frame()
    frame()

    expect(focusSpy).toHaveBeenCalledTimes(1)
  })

  it('gives up quietly when no terminal is on screen', () => {
    const frame = installRaf()

    focusRevealedTerminal()
    frame()
    frame()
    frame()

    expect(document.activeElement).toBe(document.body)
  })
})
