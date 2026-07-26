import { describe, expect, it } from 'vitest'

import { insertInlineRefsIntoEditor } from './inline-refs'
import {
  composerPlainText,
  deleteSelectionInEditor,
  insertComposerContentsAtCaret,
  normalizeComposerEditorDom,
  refChipElement,
  renderComposerContents,
  replaceBeforeCaret,
  RICH_INPUT_SLOT
} from './rich-editor'

const caretIn = (editor: HTMLElement) => {
  const range = document.createRange()
  const selection = window.getSelection()!

  range.selectNodeContents(editor)
  range.collapse(false)
  selection.removeAllRanges()
  selection.addRange(range)
}

describe('renderComposerContents', () => {
  it('renders refs and raw text without interpreting user text as HTML', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT

    renderComposerContents(editor, '@file:`<img src=x onerror=alert(1)>` <b>raw</b>')

    expect(editor.querySelector('img')).toBeNull()
    expect(editor.querySelector('b')).toBeNull()
    expect(editor.textContent).toContain('<img src=x onerror=alert(1)>')
    expect(editor.textContent).toContain('<b>raw</b>')
    expect(composerPlainText(editor)).toBe('@file:`<img src=x onerror=alert(1)>` <b>raw</b>')
  })
})

describe('normalizeComposerEditorDom', () => {
  it('unwraps a single insertHTML wrapper div so plain text stays one line', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.innerHTML = '<div><span data-ref-text="@file:`src/foo.ts`" contenteditable="false">foo.ts</span> </div>'

    normalizeComposerEditorDom(editor)

    expect(composerPlainText(editor)).toBe('@file:`src/foo.ts` ')
    expect(editor.querySelector(':scope > div')).toBeNull()
  })

  it('removes a trailing br after a ref chip', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.append(refChipElement('file', '`src/foo.ts`'), document.createElement('br'))

    normalizeComposerEditorDom(editor)

    expect(composerPlainText(editor)).toBe('@file:`src/foo.ts`')
    expect(editor.querySelector('br')).toBeNull()
  })
})

describe('insertInlineRefsIntoEditor', () => {
  it('inserts chips without wrapper divs or spurious newlines', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT

    insertInlineRefsIntoEditor(editor, ['@file:`src/foo.ts`'])

    expect(editor.querySelector(':scope > div')).toBeNull()
    expect(composerPlainText(editor)).toBe('@file:`src/foo.ts` ')
  })

  it('separates a chip from the word the caret sits after', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.append(document.createTextNode('review'))
    document.body.append(editor)
    caretIn(editor)

    expect(insertInlineRefsIntoEditor(editor, ['@file:`src/a.ts`'])).toBe('review @file:`src/a.ts` ')

    editor.remove()
  })

  it('does not double the space when one is already there', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.append(document.createTextNode('review '))
    document.body.append(editor)
    caretIn(editor)

    expect(insertInlineRefsIntoEditor(editor, ['@file:`src/a.ts`'])).toBe('review @file:`src/a.ts` ')

    editor.remove()
  })
})

describe('insertComposerContentsAtCaret', () => {
  it('inserts multiline text as text nodes + br', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
    caretIn(editor)

    insertComposerContentsAtCaret(editor, 'one\ntwo\nthree')

    expect(editor.querySelectorAll('br').length).toBe(2)
    expect(composerPlainText(editor)).toBe('one\ntwo\nthree')

    editor.remove()
  })

  it('replaces the selected span', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.textContent = 'abXYef'
    document.body.append(editor)

    const text = editor.firstChild!
    const selection = window.getSelection()!
    const range = document.createRange()

    range.setStart(text, 2)
    range.setEnd(text, 4)
    selection.removeAllRanges()
    selection.addRange(range)

    insertComposerContentsAtCaret(editor, 'cd')

    expect(composerPlainText(editor)).toBe('abcdef')

    editor.remove()
  })

  it('lands directives in the text as chips', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
    caretIn(editor)

    insertComposerContentsAtCaret(editor, 'read @url:`https://example.dev/a` now')

    expect(editor.querySelectorAll('[data-ref-kind="url"]').length).toBe(1)
    expect(composerPlainText(editor)).toBe('read @url:`https://example.dev/a` now')

    editor.remove()
  })
})

describe('replaceBeforeCaret', () => {
  it('swaps the token before the caret and leaves the caret after the insert', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.textContent = 'see foo'
    document.body.append(editor)

    const text = editor.firstChild!
    const selection = window.getSelection()!
    const range = document.createRange()

    range.setStart(text, 7)
    range.collapse(true)
    selection.removeAllRanges()
    selection.addRange(range)

    const fragment = document.createDocumentFragment()
    fragment.append(refChipElement('file', '`src/foo.ts`'), document.createTextNode(' '))

    expect(replaceBeforeCaret(editor, 3, fragment)).toBe(true)
    expect(composerPlainText(editor)).toBe('see @file:`src/foo.ts` ')
    expect(selection.getRangeAt(0).collapsed).toBe(true)

    editor.remove()
  })

  it('leaves the editor alone when the caret has no room for the token', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.textContent = 'hi'
    document.body.append(editor)

    const selection = window.getSelection()!
    const range = document.createRange()

    range.setStart(editor.firstChild!, 2)
    range.collapse(true)
    selection.removeAllRanges()
    selection.addRange(range)

    const fragment = document.createDocumentFragment()
    fragment.append(document.createTextNode('x'))

    expect(replaceBeforeCaret(editor, 20, fragment)).toBe(false)
    expect(composerPlainText(editor)).toBe('hi')

    editor.remove()
  })
})

describe('deleteSelectionInEditor', () => {
  it('clears a non-collapsed range and leaves a collapsed caret', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.textContent = 'hello world'
    document.body.append(editor)

    const selection = window.getSelection()!
    const range = document.createRange()

    range.selectNodeContents(editor)
    selection.removeAllRanges()
    selection.addRange(range)

    expect(deleteSelectionInEditor(editor)).toBe(true)
    expect(composerPlainText(editor)).toBe('')
    expect(selection.getRangeAt(0).collapsed).toBe(true)
    expect(deleteSelectionInEditor(editor)).toBe(false)

    editor.remove()
  })
})
