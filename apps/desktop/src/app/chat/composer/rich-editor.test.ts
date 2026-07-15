import { afterEach, describe, expect, it, vi } from 'vitest'

import { insertInlineRefsIntoEditor } from './inline-refs'
import {
  composerPlainText,
  deleteSelectionInEditor,
  insertPlainTextAtCaret,
  LARGE_PASTE_THRESHOLD_CHARS,
  normalizeComposerEditorDom,
  pastePlainTextIntoEditor,
  refChipElement,
  renderComposerContents,
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
})

describe('insertPlainTextAtCaret', () => {
  it('inserts multiline text as text nodes + br', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
    caretIn(editor)

    insertPlainTextAtCaret(editor, 'one\ntwo\nthree')

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

    insertPlainTextAtCaret(editor, 'cd')

    expect(composerPlainText(editor)).toBe('abcdef')

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

describe('pastePlainTextIntoEditor', () => {
  type ExecCommandFn = (command: string, showUi?: boolean, value?: string) => boolean

  const installExecStub = (impl?: ExecCommandFn): ExecCommandFn => {
    const fn = vi.fn(impl ?? (() => true)) as unknown as ExecCommandFn

    // jsdom doesn't define document.execCommand; install a stub so the
    // production code path is reachable from tests. vi.spyOn requires the
    // property to already exist on the target, so use direct assignment.
    ;(document as unknown as { execCommand: ExecCommandFn }).execCommand = fn

    return fn
  }

  const focusEditor = (editor: HTMLElement) => {
    editor.tabIndex = 0
    editor.focus()
  }

  afterEach(() => {
    delete (document as unknown as { execCommand?: unknown }).execCommand
  })

  it('routes a small paste through execCommand when the editor is focused (undo fidelity)', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
    focusEditor(editor)

    const execStub = installExecStub((command, _showUi, value) => {
      // Simulate Chromium writing the inserted text into the editor's DOM.
      if (command === 'insertText' && typeof value === 'string') {
        editor.append(value)
      }

      return true
    })

    pastePlainTextIntoEditor(editor, 'hello world')

    expect(execStub).toHaveBeenCalledWith('insertText', false, 'hello world')
    expect(composerPlainText(editor)).toBe('hello world')

    editor.remove()
  })

  it('falls back to insertPlainTextAtCaret when the editor is not focused', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
    // Intentionally no .focus() — execCommand would operate on whatever
    // selection lives elsewhere on the page, so we must use the direct path.

    const execStub = installExecStub()
    pastePlainTextIntoEditor(editor, 'orphan paste')

    expect(execStub).not.toHaveBeenCalled()
    expect(composerPlainText(editor)).toBe('orphan paste')

    editor.remove()
  })

  it('falls back to insertPlainTextAtCaret for pastes above the threshold (perf)', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
    focusEditor(editor)

    const bigBlob = 'x'.repeat(LARGE_PASTE_THRESHOLD_CHARS + 1)
    const execStub = installExecStub()

    pastePlainTextIntoEditor(editor, bigBlob)

    // Direct path bypasses execCommand — Chromium's O(n²) freeze is the reason
    // this branch exists; we accept the lost undo entry for pathological pastes.
    expect(execStub).not.toHaveBeenCalled()
    expect(composerPlainText(editor)).toBe(bigBlob)

    editor.remove()
  })

  it('routes through execCommand when a descendant element is focused (chip focus)', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)

    // Place a child element inside the editor and focus it — simulates focus on
    // a directive chip or inline ref. `document.activeElement === editor` is
    // false here, but `editor.contains(document.activeElement)` is true.
    const child = document.createElement('span')
    child.tabIndex = 0
    editor.append(child)
    child.focus()

    const execStub = installExecStub((command, _showUi, value) => {
      if (command === 'insertText' && typeof value === 'string') {
        editor.append(value)
      }

      return true
    })
    pastePlainTextIntoEditor(editor, 'chip paste')

    expect(execStub).toHaveBeenCalledWith('insertText', false, 'chip paste')
    expect(composerPlainText(editor)).toBe('chip paste')

    editor.remove()
  })

  it('falls back to insertPlainTextAtCaret when execCommand returns false', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
    focusEditor(editor)

    // execCommand returns false (e.g. unsupported command, disabled editing)
    const execStub = installExecStub(() => false)

    pastePlainTextIntoEditor(editor, 'fallback text')

    expect(execStub).toHaveBeenCalledWith('insertText', false, 'fallback text')
    expect(composerPlainText(editor)).toBe('fallback text')

    editor.remove()
  })

  it('falls back to insertPlainTextAtCaret when execCommand throws', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
    focusEditor(editor)

    // execCommand throws (e.g. jsdom without a stub, CSP-blocked API)
    const execStub = installExecStub(() => {
      throw new Error('Not supported')
    })

    pastePlainTextIntoEditor(editor, 'throwing paste')

    expect(execStub).toHaveBeenCalledWith('insertText', false, 'throwing paste')
    expect(composerPlainText(editor)).toBe('throwing paste')

    editor.remove()
  })
})
