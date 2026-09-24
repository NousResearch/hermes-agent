import { beforeEach, describe, expect, it } from 'vitest'

import {
  caretOffsetInEditor,
  composerPlainText,
  placeCaretAtOffset,
  refChipElement,
  renderComposerContents,
  RICH_INPUT_SLOT
} from './rich-editor'
import {
  createComposerUndoHistory,
  isCoalescableTypingInput,
  isRedoShortcut,
  isUndoShortcut
} from './undo-history'

const key = (over: Partial<KeyboardEvent> = {}) =>
  ({ altKey: false, ctrlKey: false, key: 'z', metaKey: false, shiftKey: false, ...over }) as KeyboardEvent

describe('undo/redo shortcut recognition', () => {
  it('claims Cmd+Z and Ctrl+Z as undo, but not with Shift or Alt', () => {
    expect(isUndoShortcut(key({ metaKey: true }))).toBe(true)
    expect(isUndoShortcut(key({ ctrlKey: true }))).toBe(true)
    expect(isUndoShortcut(key({ metaKey: true, shiftKey: true }))).toBe(false)
    expect(isUndoShortcut(key({ altKey: true, metaKey: true }))).toBe(false)
    expect(isUndoShortcut(key({ key: 'a', metaKey: true }))).toBe(false)
    expect(isUndoShortcut(key())).toBe(false)
  })

  it('claims Cmd+Shift+Z everywhere and Ctrl+Y as redo', () => {
    expect(isRedoShortcut(key({ metaKey: true, shiftKey: true }))).toBe(true)
    expect(isRedoShortcut(key({ ctrlKey: true, shiftKey: true }))).toBe(true)
    expect(isRedoShortcut(key({ ctrlKey: true, key: 'y' }))).toBe(true)
    expect(isRedoShortcut(key({ metaKey: true }))).toBe(false)
    expect(isRedoShortcut(key({ altKey: true, ctrlKey: true, key: 'y' }))).toBe(false)
  })

  it('never treats one keystroke as both undo and redo', () => {
    const chords = [
      key({ metaKey: true }),
      key({ ctrlKey: true }),
      key({ metaKey: true, shiftKey: true }),
      key({ ctrlKey: true, shiftKey: true }),
      key({ ctrlKey: true, key: 'y' })
    ]

    for (const chord of chords) {
      expect(isUndoShortcut(chord) && isRedoShortcut(chord)).toBe(false)
    }
  })
})

describe('isCoalescableTypingInput', () => {
  it('coalesces ordinary inserts and deletes, not paste/drop/history', () => {
    expect(isCoalescableTypingInput('insertText')).toBe(true)
    expect(isCoalescableTypingInput('insertReplacementText')).toBe(true)
    expect(isCoalescableTypingInput('deleteContentBackward')).toBe(true)
    expect(isCoalescableTypingInput('deleteContentForward')).toBe(true)
    expect(isCoalescableTypingInput('insertFromPaste')).toBe(false)
    expect(isCoalescableTypingInput('insertFromDrop')).toBe(false)
    expect(isCoalescableTypingInput('historyUndo')).toBe(false)
    expect(isCoalescableTypingInput('historyRedo')).toBe(false)
  })
})

describe('composer undo history', () => {
  const snap = (text: string, caret = text.length) => ({ caret, text })

  it('steps back through discrete edits, newest first', () => {
    const history = createComposerUndoHistory()

    history.record(snap(''))
    history.record(snap('a'))
    history.record(snap('ab'))

    expect(history.undo(snap('abc'))?.text).toBe('ab')
    expect(history.undo(snap('ab'))?.text).toBe('a')
    expect(history.undo(snap('a'))?.text).toBe('')
    expect(history.undo(snap(''))).toBeNull()
  })

  it('redoes back to the state undo left, and stops at the newest', () => {
    const history = createComposerUndoHistory()

    history.record(snap('before'))

    const undone = history.undo(snap('before + pasted'))

    expect(undone?.text).toBe('before')
    expect(history.redo(snap('before'))?.text).toBe('before + pasted')
    expect(history.redo(snap('before + pasted'))).toBeNull()
  })

  it('restores the caret along with the text', () => {
    const history = createComposerUndoHistory()

    history.record({ caret: 3, text: 'hello world' })

    expect(history.undo({ caret: 0, text: 'changed' })).toEqual({ caret: 3, text: 'hello world' })
  })

  it('collapses a run of typing into one entry, so undo steps back by a burst', () => {
    let now = 1_000
    const history = createComposerUndoHistory(200, () => now)

    history.record(snap(''), { coalesce: true })
    now += 50
    history.record(snap('h'), { coalesce: true })
    now += 50
    history.record(snap('he'), { coalesce: true })

    // One burst → one entry, back to the state before the burst started.
    expect(history.undo(snap('hel'))?.text).toBe('')
    expect(history.undo(snap(''))).toBeNull()
  })

  it('coalesces insertReplacementText into the same typing burst', () => {
    let now = 1_000
    const history = createComposerUndoHistory(200, () => now)

    history.record(snap(''), { coalesce: isCoalescableTypingInput('insertText') })
    now += 40
    history.record(snap('teh'), { coalesce: isCoalescableTypingInput('insertReplacementText') })

    expect(history.undo(snap('the'))?.text).toBe('')
  })

  it('starts a new entry once the typing pause exceeds the coalesce window', () => {
    let now = 1_000
    const history = createComposerUndoHistory(200, () => now)

    history.record(snap(''), { coalesce: true })
    now += 5_000
    history.record(snap('word one'), { coalesce: true })

    expect(history.undo(snap('word one two'))?.text).toBe('word one')
    expect(history.undo(snap('word one'))?.text).toBe('')
  })

  it('does not coalesce a paste into the typing burst that preceded it', () => {
    let now = 1_000
    const history = createComposerUndoHistory(200, () => now)

    history.record(snap(''), { coalesce: true })
    now += 20
    // A paste is a discrete edit — no coalesce flag.
    history.record(snap('typed '))

    expect(history.undo(snap('typed PASTED'))?.text).toBe('typed ')
    expect(history.undo(snap('typed '))?.text).toBe('')
  })

  it('drops the redo stack once a new edit lands', () => {
    const history = createComposerUndoHistory()

    history.record(snap('one'))
    history.undo(snap('one two'))
    history.record(snap('one'))

    expect(history.redo(snap('one three'))).toBeNull()
  })

  it('ignores a no-op edit so undo never looks stuck for a press', () => {
    const history = createComposerUndoHistory()

    history.record(snap('same'))
    history.record(snap('same'))

    expect(history.undo(snap('same'))?.text).toBe('same')
    expect(history.undo(snap('same'))).toBeNull()
  })

  it('bounds the stack, discarding the oldest entries', () => {
    const history = createComposerUndoHistory(3)

    for (const text of ['a', 'b', 'c', 'd', 'e']) {
      history.record(snap(text))
    }

    expect(history.undo(snap('f'))?.text).toBe('e')
    expect(history.undo(snap('e'))?.text).toBe('d')
    expect(history.undo(snap('d'))?.text).toBe('c')
    expect(history.undo(snap('c'))).toBeNull()
  })

  it('reset clears both directions', () => {
    const history = createComposerUndoHistory()

    history.record(snap('a'))
    history.undo(snap('ab'))
    history.reset()

    expect(history.undo(snap('ab'))).toBeNull()
    expect(history.redo(snap('ab'))).toBeNull()
  })
})

describe('caret offsets in composerPlainText coordinates', () => {
  let editor: HTMLDivElement

  beforeEach(() => {
    editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    document.body.append(editor)
  })

  const caretAfter = (node: Node, offset: number) => {
    const range = document.createRange()
    const selection = window.getSelection()!
    range.setStart(node, offset)
    range.collapse(true)
    selection.removeAllRanges()
    selection.addRange(range)
  }

  it('round-trips a caret in plain text', () => {
    renderComposerContents(editor, 'hello world')
    placeCaretAtOffset(editor, 5)

    expect(caretOffsetInEditor(editor)).toBe(5)
  })

  it('counts a chip as its whole @kind:value text', () => {
    editor.append(
      document.createTextNode('see '),
      refChipElement('file', '`src/a.ts`'),
      document.createTextNode(' now')
    )

    const chipText = '@file:`src/a.ts`'
    // Caret at the very end = everything before it.
    placeCaretAtOffset(editor, composerPlainText(editor).length)

    expect(caretOffsetInEditor(editor)).toBe(4 + chipText.length + 4)
  })

  it('lands the caret before a chip rather than splitting it', () => {
    editor.append(document.createTextNode('a '), refChipElement('file', '`x.ts`'))

    // An offset that falls midway through the chip's serialized text.
    placeCaretAtOffset(editor, 2 + 3)

    expect(caretOffsetInEditor(editor)).toBe(2)
  })

  it('counts a line break as one character', () => {
    renderComposerContents(editor, 'one\ntwo')
    placeCaretAtOffset(editor, 5)

    expect(caretOffsetInEditor(editor)).toBe(5)
    expect(composerPlainText(editor)).toBe('one\ntwo')
  })

  it('clamps past-the-end offsets to the end instead of throwing', () => {
    renderComposerContents(editor, 'short')
    placeCaretAtOffset(editor, 999)

    expect(caretOffsetInEditor(editor)).toBe(5)
  })

  it('reports the end when the selection is outside the editor', () => {
    renderComposerContents(editor, 'hello')

    const outside = document.createElement('div')
    outside.textContent = 'elsewhere'
    document.body.append(outside)
    caretAfter(outside.firstChild!, 3)

    expect(caretOffsetInEditor(editor)).toBe(5)

    outside.remove()
  })
})
