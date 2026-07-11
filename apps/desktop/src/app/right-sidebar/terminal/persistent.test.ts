import { describe, expect, it } from 'vitest'

import { terminalOverlayZIndex } from './persistent'

describe('terminalOverlayZIndex', () => {
  it('tracks its own floating surface without punching through a different active window', () => {
    const slot = document.createElement('div')

    expect(terminalOverlayZIndex(slot)).toBe(4)

    const floating = document.createElement('div')
    floating.dataset.surfaceZIndex = '80'
    floating.append(slot)
    expect(terminalOverlayZIndex(slot)).toBe(81)

    floating.dataset.surfaceZIndex = '90'
    expect(terminalOverlayZIndex(slot)).toBe(91)

    floating.dataset.surfaceZIndex = 'invalid'
    expect(terminalOverlayZIndex(slot)).toBe(4)
  })
})
