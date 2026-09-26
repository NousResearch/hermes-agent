import type { KeyboardEvent } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { applyListEdit, composerListKey } from './list-keys'
import { caretOffsetInEditor, RICH_INPUT_SLOT } from './rich-editor'

const key = (init: Partial<KeyboardEvent<HTMLElement>>) =>
  ({ altKey: false, ctrlKey: false, metaKey: false, shiftKey: false, ...init }) as KeyboardEvent<HTMLElement>

function editorWith(text: string) {
  const editor = document.createElement('div')
  editor.contentEditable = 'true'
  editor.dataset.slot = RICH_INPUT_SLOT
  editor.textContent = text
  document.body.append(editor)

  return editor
}

afterEach(() => document.body.replaceChildren())

describe('composerListKey', () => {
  it('never claims plain Enter, which sends', () => {
    expect(composerListKey(key({ key: 'Enter' }))).toBeNull()
    expect(composerListKey(key({ key: 'Enter', shiftKey: true }))).toBe('newline')
  })

  it('leaves shortcuts alone', () => {
    expect(composerListKey(key({ key: 'Enter', metaKey: true, shiftKey: true }))).toBeNull()
    expect(composerListKey(key({ ctrlKey: true, key: 'Tab' }))).toBeNull()
  })
})

describe('applyListEdit', () => {
  it('keeps the caret on a new empty last line', () => {
    const editor = editorWith('1. a\n2. ')

    // Ending the list at the bottom: the caret sits on the blank line after it.
    expect(applyListEdit(editor, { caret: 6, text: '1. a\n\n' })).toBe(true)
    expect(editor.lastChild?.nodeName).toBe('BR')
    expect(caretOffsetInEditor(editor)).toBe(6)
  })

  it('reports no change so the key is not recorded as an undo step', () => {
    expect(applyListEdit(editorWith('1. only'), { caret: 7, text: '1. only' })).toBe(false)
  })
})
