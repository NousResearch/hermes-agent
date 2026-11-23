import { describe, expect, it } from 'vitest'

import { isComposerChord } from '@/lib/keybinds/chords'

import { isMacPlatform, shouldOwnAddSelectionShortcut, terminalSelectionAnchor } from './selection'

const key = (init: Partial<KeyboardEvent> & { key: string }) =>
  ({
    altKey: false,
    ctrlKey: false,
    metaKey: false,
    shiftKey: false,
    type: 'keydown',
    ...init
  }) as KeyboardEvent

/** Chord that the composer-focus ladder accepts on this host. */
const addSelectionKey = () => (isMacPlatform() ? key({ key: 'l', metaKey: true }) : key({ key: 'l', ctrlKey: true }))

describe('shouldOwnAddSelectionShortcut', () => {
  it('only the active tab claims the add-selection shortcut when text is selected (#76116)', () => {
    const chord = addSelectionKey()

    expect(shouldOwnAddSelectionShortcut(chord, { active: true, hasSelection: true })).toBe(true)

    // Inactive mounted tabs must not also fire — that was the N-pill bug.
    expect(shouldOwnAddSelectionShortcut(chord, { active: false, hasSelection: true })).toBe(false)
  })

  it('never claims the shortcut with no selection so clear-screen still reaches the shell', () => {
    const chord = addSelectionKey()

    expect(shouldOwnAddSelectionShortcut(chord, { active: true, hasSelection: false })).toBe(false)
    expect(isComposerChord(chord)).toBe(true)
  })
})

describe('terminalSelectionAnchor', () => {
  it('unscales selection rect deltas by currentCSSZoom so Add to chat aligns under browser zoom', () => {
    const host = document.createElement('div')
    const selectionLayer = document.createElement('div')
    selectionLayer.className = 'xterm-selection'
    const band = document.createElement('div')
    selectionLayer.appendChild(band)
    host.appendChild(selectionLayer)

    Object.defineProperty(host, 'clientWidth', { configurable: true, value: 400 })
    Object.defineProperty(host, 'clientHeight', { configurable: true, value: 200 })
    Object.defineProperty(host, 'currentCSSZoom', { configurable: true, value: 1.25 })
    host.getBoundingClientRect = () => ({ left: 100, top: 200, right: 600, bottom: 450, width: 500, height: 250 }) as DOMRect
    band.getBoundingClientRect = () => ({ left: 225, top: 250, right: 325, bottom: 275, width: 100, height: 25 }) as DOMRect

    expect(terminalSelectionAnchor(host)).toEqual({ left: 100, top: 64 })
  })
})
