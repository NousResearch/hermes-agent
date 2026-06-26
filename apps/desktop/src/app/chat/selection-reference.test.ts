import { describe, expect, it } from 'vitest'

import {
  currentThreadSelection,
  selectionIntersectsElement,
  THREAD_SELECTION_VIEWPORT_SELECTOR
} from './selection-reference'

describe('selection-reference', () => {
  it('detects a native selection inside the thread viewport', () => {
    document.body.innerHTML = `<div data-slot="aui_thread-viewport"><p id="msg">hello selected text</p></div>`
    const text = document.querySelector('#msg')?.firstChild

    expect(text).toBeTruthy()

    const range = document.createRange()
    range.setStart(text as Text, 6)
    range.setEnd(text as Text, 14)

    const selection = window.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)

    const match = currentThreadSelection()

    expect(match?.text).toBe('selected')
    expect(match?.viewport.matches(THREAD_SELECTION_VIEWPORT_SELECTOR)).toBe(true)
    expect(selectionIntersectsElement(selection, match?.viewport ?? null)).toBe(true)

    selection?.removeAllRanges()
  })

  it('ignores native selections outside the thread viewport', () => {
    document.body.innerHTML = `<main><p id="outside">outside text</p></main><div data-slot="aui_thread-viewport"></div>`
    const text = document.querySelector('#outside')?.firstChild

    expect(text).toBeTruthy()

    const range = document.createRange()
    range.selectNodeContents(text as Text)

    const selection = window.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)

    expect(currentThreadSelection()).toBeNull()

    selection?.removeAllRanges()
  })
})
