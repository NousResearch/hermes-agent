import { describe, expect, it } from 'vitest'

import { hudEscapeAction } from './escape'

function shell() {
  const root = document.createElement('div')
  const composer = document.createElement('div')
  composer.setAttribute('data-slot', 'composer-rich-input')
  root.appendChild(composer)
  document.body.appendChild(root)

  const overlay = document.createElement('div')
  overlay.setAttribute('role', 'dialog')
  document.body.appendChild(overlay)

  return { root, composer, overlay }
}

describe('hudEscapeAction', () => {
  it('closes the HUD when focus is inside the shell', () => {
    const { root, composer } = shell()

    expect(hudEscapeAction(root, composer, false)).toBe('close')
  })

  it('closes the HUD when nothing has focus', () => {
    const { root } = shell()

    expect(hudEscapeAction(root, null, false)).toBe('close')
    expect(hudEscapeAction(root, document.body, false)).toBe('close')
  })

  it('steps back when a portalled overlay holds focus — Escape belongs to it', () => {
    const { root, overlay } = shell()

    expect(hudEscapeAction(root, overlay, false)).toBe('ignore')
  })

  it('honours a press another handler already consumed', () => {
    const { root, composer } = shell()

    expect(hudEscapeAction(root, composer, true)).toBe('ignore')
  })
})
