import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { composerPlainText, insertComposerLineBreak, RICH_INPUT_SLOT } from './rich-editor'

/** An editor element carrying `text` with a collapsed caret at the given offset. */
function mountedEditor(text: string, caret = text.length) {
  const editor = document.createElement('div')

  editor.setAttribute('data-slot', RICH_INPUT_SLOT)
  editor.contentEditable = 'true'
  editor.textContent = text
  document.body.append(editor)

  const range = document.createRange()
  range.setStart(editor.firstChild ?? editor, caret)
  range.collapse(true)

  const selection = window.getSelection()

  selection?.removeAllRanges()
  selection?.addRange(range)

  return editor
}

beforeEach(() => {
  // revealCaret defers to the next frame; jsdom has no layout to measure there.
  vi.stubGlobal('requestAnimationFrame', () => 0)
})

afterEach(() => {
  vi.unstubAllGlobals()
  document.body.replaceChildren()
  window.getSelection()?.removeAllRanges()
})

describe('insertComposerLineBreak', () => {
  it('leaves a break the draft can send', () => {
    const editor = mountedEditor('سطر')

    expect(insertComposerLineBreak(editor)).toBe(true)

    expect(composerPlainText(editor)).toBe('سطر\n')
    expect(editor.querySelectorAll('br')).toHaveLength(1)
  })

  it('never leaks the zero-width caret host into the serialized text', () => {
    const editor = mountedEditor('a')

    insertComposerLineBreak(editor)

    // The browser needs a node after the break to draw a caret line; that node
    // must not become part of the message.
    expect(editor.textContent).toContain('\u200B')
    expect(composerPlainText(editor)).toBe('a\n')
    expect(composerPlainText(editor)).not.toContain('\u200B')
  })

  it('breaks at the caret instead of always at the end', () => {
    const editor = mountedEditor('ab', 1)

    insertComposerLineBreak(editor)

    expect(composerPlainText(editor)).toBe('a\nb')
  })

  it('parks the caret after the break so the next keystroke lands on the new line', () => {
    const editor = mountedEditor('x')

    insertComposerLineBreak(editor)

    const range = window.getSelection()?.getRangeAt(0)

    expect(range?.collapsed).toBe(true)
    expect(composerPlainText(editor)).toBe('x\n')

    // Typing at that caret must extend the second line, not the first.
    range?.insertNode(document.createTextNode('y'))

    expect(composerPlainText(editor)).toBe('x\ny')
  })
})
