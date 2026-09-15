import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { rememberDesktopCommandsCatalog } from '@/lib/desktop-slash-commands'

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
import { placeCaretAtEnd } from './test-utils'

beforeEach(() => {
  rememberDesktopCommandsCatalog({
    commands: { '/goal': { argument_mode: 'mixed', desktop: null } }
  })
})

afterEach(() => {
  rememberDesktopCommandsCatalog(undefined)
})

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

  it('hydrates a committed leading slash command back to its pill', () => {
    // Text-hydration parity with @ refs: a re-render from serialized text
    // (draft restore, undo, the trigger commit fallback) must not demote a
    // committed no-arg command chip to plain text.
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT

    renderComposerContents(editor, '/some-skill @folder:`Desktop` ')

    const pill = editor.querySelector('[data-slash-kind]')

    expect(pill?.getAttribute('data-ref-text')).toBe('/some-skill')
    expect(editor.querySelector('[data-ref-kind="folder"]')).not.toBeNull()
    expect(composerPlainText(editor)).toBe('/some-skill @folder:`Desktop` ')
  })

  it('keeps a still-typed leading slash token as editable text', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT

    // No trailing whitespace — not committed yet.
    renderComposerContents(editor, '/some-skil')

    expect(editor.querySelector('[data-slash-kind]')).toBeNull()
    expect(composerPlainText(editor)).toBe('/some-skil')
  })

  it('keeps an arg-taking command as text — its tail may be uncommitted prose', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT

    renderComposerContents(editor, '/goal ship the redesign')

    expect(editor.querySelector('[data-slash-kind]')).toBeNull()
    expect(composerPlainText(editor)).toBe('/goal ship the redesign')
  })
})

describe('replaceBeforeCaret across split text nodes', () => {
  it('replaces a token that Chromium fragmented into multiple text nodes', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.contentEditable = 'true'
    document.body.append(editor)
    editor.append(document.createTextNode('see @Desk'), document.createTextNode('top/'))

    const caret = document.createRange()
    caret.setStart(editor.lastChild!, 4)
    caret.collapse(true)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(caret)

    const fragment = document.createDocumentFragment()
    fragment.append(refChipElement('folder', '`Desktop`'), document.createTextNode(' '))

    // Token `@Desktop/` (9 chars) spans both text nodes.
    expect(replaceBeforeCaret(editor, 9, fragment)).toBe(true)
    expect(composerPlainText(editor)).toBe('see @folder:`Desktop` ')

    editor.remove()
  })

  it('refuses when a chip interrupts the span — the token is not contiguous text', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.contentEditable = 'true'
    document.body.append(editor)
    editor.append(document.createTextNode('a'), refChipElement('file', '`x`'), document.createTextNode('bc'))

    const caret = document.createRange()
    caret.setStart(editor.lastChild!, 2)
    caret.collapse(true)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(caret)

    expect(replaceBeforeCaret(editor, 5, document.createDocumentFragment())).toBe(false)
    expect(composerPlainText(editor)).toBe('a@file:`x`bc')

    editor.remove()
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

  it('preserves a live caret anchored in an empty direct text node', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.contentEditable = 'true'
    document.body.append(editor)

    const leadingLitter = document.createTextNode('')
    const caretContainer = document.createTextNode('')
    const trailingLitter = document.createTextNode('')
    editor.append(leadingLitter, refChipElement('file', '`src/foo.ts`'), caretContainer, trailingLitter)

    const caret = document.createRange()
    caret.setStart(caretContainer, 0)
    caret.collapse(true)

    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(caret)

    normalizeComposerEditorDom(editor)

    expect(leadingLitter.isConnected).toBe(false)
    expect(trailingLitter.isConnected).toBe(false)
    expect(caretContainer.isConnected).toBe(true)
    expect(editor.contains(selection.getRangeAt(0).startContainer)).toBe(true)
    expect(selection.getRangeAt(0).startContainer).toBe(caretContainer)
    expect(composerPlainText(editor)).toBe('@file:`src/foo.ts`')

    selection.removeAllRanges()
    editor.remove()
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
    placeCaretAtEnd(editor)

    expect(insertInlineRefsIntoEditor(editor, ['@file:`src/a.ts`'])).toBe('review @file:`src/a.ts` ')

    editor.remove()
  })

  it('does not double the space when one is already there', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.append(document.createTextNode('review '))
    document.body.append(editor)
    placeCaretAtEnd(editor)

    expect(insertInlineRefsIntoEditor(editor, ['@file:`src/a.ts`'])).toBe('review @file:`src/a.ts` ')

    editor.remove()
  })
})

describe('insertComposerContentsAtCaret', () => {
  it('inserts multiline text as text nodes + br', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
    placeCaretAtEnd(editor)

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
    placeCaretAtEnd(editor)

    insertComposerContentsAtCaret(editor, 'read @url:`https://example.dev/a` now')

    expect(editor.querySelectorAll('[data-ref-kind="url"]').length).toBe(1)
    expect(composerPlainText(editor)).toBe('read @url:`https://example.dev/a` now')

    editor.remove()
  })

  // A directive typed by hand chips; the same directive pasted has to chip too,
  // or copy/pasting a prompt silently drops every command in it.
  it('chips a pasted slash command, including one that ends the paste', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
    placeCaretAtEnd(editor)

    insertComposerContentsAtCaret(editor, '/some-skill')

    expect(editor.querySelector('[data-slash-kind]')?.getAttribute('data-ref-text')).toBe('/some-skill')
    // Committed pills carry the trailing space the typed path appends, so a
    // later full re-render doesn't read the token as half-typed.
    expect(composerPlainText(editor)).toBe('/some-skill ')

    editor.remove()
  })

  it('chips a skill named mid-paste alongside a ref', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
    placeCaretAtEnd(editor)

    insertComposerContentsAtCaret(editor, 'clean @file:`a.ts` with /some-skill then ship')

    expect(editor.querySelectorAll('[data-slash-kind]').length).toBe(1)
    expect(editor.querySelectorAll('[data-ref-kind="file"]').length).toBe(1)
    expect(composerPlainText(editor)).toBe('clean @file:`a.ts` with /some-skill then ship')

    editor.remove()
  })

  it('leaves a pasted path alone — /usr/local is not a command', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
    placeCaretAtEnd(editor)

    insertComposerContentsAtCaret(editor, 'see /usr/local/bin and /goal ship it')

    expect(editor.querySelector('[data-slash-kind]')).toBeNull()
    expect(composerPlainText(editor)).toBe('see /usr/local/bin and /goal ship it')

    editor.remove()
  })

  it('does not chip a command pasted against a word — foo/clean is not a command', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.textContent = 'foo'
    document.body.append(editor)
    placeCaretAtEnd(editor)

    insertComposerContentsAtCaret(editor, '/some-skill')

    expect(editor.querySelector('[data-slash-kind]')).toBeNull()
    expect(composerPlainText(editor)).toBe('foo/some-skill')

    editor.remove()
  })

  it('chips a command pasted right after an existing chip', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.append(refChipElement('file', '`a.ts`'))
    document.body.append(editor)
    placeCaretAtEnd(editor)

    insertComposerContentsAtCaret(editor, '/some-skill')

    expect(editor.querySelector('[data-slash-kind]')).not.toBeNull()

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
