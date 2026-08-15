import { afterEach, describe, expect, it, vi } from 'vitest'

import { markActiveComposer } from './focus'
import { composerTargetAtPoint } from './selection-target'

afterEach(() => {
  document.body.replaceChildren()
  markActiveComposer('main')
  vi.restoreAllMocks()
})

describe('composerTargetAtPoint', () => {
  it('prefers the chat surface under the native context-menu point', () => {
    const tile = document.createElement('div')
    tile.dataset.composerTarget = 'tile:stored-42'
    const selected = document.createElement('span')
    tile.append(selected)
    document.body.append(tile)
    markActiveComposer('main')
    Object.defineProperty(document, 'elementFromPoint', { configurable: true, value: () => selected })

    expect(composerTargetAtPoint(42, 17)).toBe('tile:stored-42')
  })

  it('falls back to the active composer outside a chat surface', () => {
    markActiveComposer('tile:active')
    Object.defineProperty(document, 'elementFromPoint', { configurable: true, value: () => document.body })

    expect(composerTargetAtPoint(0, 0)).toBe('tile:active')
  })
})
