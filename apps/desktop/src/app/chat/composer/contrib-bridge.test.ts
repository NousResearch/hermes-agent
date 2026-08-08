import type { RefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type ComposerRenderContext, createComposerRenderContext } from './contrib'
import { composerPlainText, placeCaretEnd, renderComposerContents } from './rich-editor'
import { createComposerUndoHistory } from './undo-history'

/**
 * The render-area edit bridge is the acceptance surface of the composer
 * render seam: a composer.actions plugin inserting text must (1) land on the
 * app undo stack (recordUndoPoint fires BEFORE the DOM mutation), (2) be
 * selection-aware (replaces the live selection, caret lands after the insert),
 * (3) go through the app's own DOM pipeline (Range-based chip/text rendering,
 * never innerHTML). The IME guard exists because inserting into a live preedit
 * corrupts it — the exact bug class the execCommand path had.
 */

function makeEditor(text: string) {
  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.append(...text.split('\n').flatMap((line, i) => (i > 0 ? [document.createElement('br'), line] : [line])))
  document.body.append(editor)

  return editor
}

function selectRange(editor: HTMLElement, start: number, end: number) {
  const findTextAt = (offset: number): { node: Text; offset: number } => {
    const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT)
    let remaining = offset
    let node: Text | null

    while ((node = walker.nextNode() as Text | null)) {
      const length = node.textContent?.length ?? 0

      if (remaining <= length) {
        return { node, offset: remaining }
      }

      remaining -= length
    }

    throw new Error('offset past end of editor text')
  }

  const startPoint = findTextAt(start)
  const endPoint = findTextAt(end)
  const range = document.createRange()

  range.setStart(startPoint.node, startPoint.offset)
  range.setEnd(endPoint.node, endPoint.offset)

  const selection = window.getSelection()!

  selection.removeAllRanges()
  selection.addRange(range)

  return selection
}

function selectNode(node: Node) {
  const range = document.createRange()
  range.selectNode(node)

  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)
}

interface Harness {
  ctx: ComposerRenderContext
  editor: HTMLElement
  recordUndoPoint: ReturnType<typeof vi.fn>
  scheduleFlushEditorToDraft: ReturnType<typeof vi.fn>
  composing: { current: boolean }
}

function makeHarness(text: string): Harness {
  const editor = makeEditor(text)
  const composing = { current: false }
  const recordUndoPoint = vi.fn()
  const scheduleFlushEditorToDraft = vi.fn()

  const ctx = createComposerRenderContext({
    editorRef: { current: editor } as RefObject<HTMLDivElement | null>,
    composingRef: composing,
    recordUndoPoint,
    scheduleFlushEditorToDraft
  })

  return { ctx, editor, recordUndoPoint, scheduleFlushEditorToDraft, composing }
}

afterEach(() => {
  document.body.replaceChildren()
})

describe('createComposerRenderContext().insertText', () => {
  it('replaces the live selection and lands the caret after the insert', () => {
    const { ctx, editor } = makeHarness('hello world')
    selectRange(editor, 0, 5)

    ctx.insertText('**hello**')

    expect(editor.textContent).toBe('**hello** world')

    // Caret in composerPlainText coordinates sits right after the insert.
    // (jsdom's Range.insertNode leaves a zero-length split node before the
    // inserted text — same litter Chromium produces around chip edits — so
    // assert position semantically, not by node identity.)
    const caret = window.getSelection()!.getRangeAt(0)
    const before = document.createRange()

    before.selectNodeContents(editor)
    before.setEnd(caret.startContainer, caret.startOffset)

    const scratch = document.createElement('div')

    scratch.append(before.cloneContents())

    expect(scratch.textContent).toBe('**hello**')
  })

  it('inserts at the caret when the selection is collapsed', () => {
    const { ctx, editor } = makeHarness('hello world')
    selectRange(editor, 5, 5)

    ctx.insertText(' there')

    expect(editor.textContent).toBe('hello there world')
  })

  it('banks the pre-edit state BEFORE the DOM mutation (undo acceptance)', async () => {
    const { ctx, editor, recordUndoPoint } = makeHarness('hello world')
    selectRange(editor, 0, 5)

    const calls: string[] = []

    recordUndoPoint.mockImplementation(() => calls.push('record'))

    const probe = new MutationObserver(() => calls.push('mutate'))

    probe.observe(editor, { childList: true, subtree: true, characterData: true })
    ctx.insertText('**hello**')

    // The undo point is synchronous and lands before the first DOM mutation;
    // the observer's callback fires on the microtask queue.
    await new Promise(resolve => setTimeout(resolve, 0))
    probe.disconnect()

    expect(calls).toEqual(['record', 'mutate'])
  })

  it('no-ops without an editor', () => {
    const recordUndoPoint = vi.fn()
    const scheduleFlushEditorToDraft = vi.fn()

    const noop = createComposerRenderContext({
      editorRef: { current: null } as RefObject<HTMLDivElement | null>,
      composingRef: { current: false },
      recordUndoPoint,
      scheduleFlushEditorToDraft
    })

    noop.insertText('**x**')

    expect(recordUndoPoint).not.toHaveBeenCalled()
    expect(scheduleFlushEditorToDraft).not.toHaveBeenCalled()
  })

  it('no-ops during IME composition (never corrupt a preedit)', () => {
    const { ctx, editor, recordUndoPoint, scheduleFlushEditorToDraft, composing } = makeHarness('hello')
    composing.current = true

    ctx.insertText('**x**')

    expect(editor.textContent).toBe('hello')
    expect(recordUndoPoint).not.toHaveBeenCalled()
    expect(scheduleFlushEditorToDraft).not.toHaveBeenCalled()
  })

  it('flushes the draft state after the insert', () => {
    const { ctx, editor, scheduleFlushEditorToDraft } = makeHarness('hello')
    selectRange(editor, 5, 5)

    ctx.insertText('!')

    expect(scheduleFlushEditorToDraft).toHaveBeenCalledWith(editor)
  })

  it('keeps an empty insert a true no-op without touching the undo hook', () => {
    const { ctx, editor, recordUndoPoint, scheduleFlushEditorToDraft } = makeHarness('hello')
    selectRange(editor, 5, 5)

    ctx.insertText('')

    expect(editor.textContent).toBe('hello')
    expect(recordUndoPoint).not.toHaveBeenCalled()
    expect(scheduleFlushEditorToDraft).not.toHaveBeenCalled()
  })

  it('deletes a non-collapsed selection when the replacement is empty', () => {
    const { ctx, editor, recordUndoPoint, scheduleFlushEditorToDraft } = makeHarness('hello world')
    selectRange(editor, 0, 5)

    ctx.insertText('')

    expect(editor.textContent).toBe(' world')
    expect(recordUndoPoint).toHaveBeenCalledTimes(1)
    expect(scheduleFlushEditorToDraft).toHaveBeenCalledWith(editor)
  })

  it('preserves redo after an empty insert', () => {
    const editor = makeEditor('hello')
    const history = createComposerUndoHistory()

    const recordUndoPoint = vi.fn(() => {
      const text = composerPlainText(editor)

      history.record({ caret: text.length, text })
    })

    const ctx = createComposerRenderContext({
      editorRef: { current: editor } as RefObject<HTMLDivElement | null>,
      composingRef: { current: false },
      recordUndoPoint,
      scheduleFlushEditorToDraft: vi.fn()
    })

    selectRange(editor, 5, 5)
    ctx.insertText('!')

    const afterInsert = { caret: composerPlainText(editor).length, text: composerPlainText(editor) }
    const undone = history.undo(afterInsert)

    expect(undone?.text).toBe('hello')
    editor.replaceChildren(document.createTextNode(undone!.text))
    ctx.insertText('')

    const redone = history.redo({ caret: composerPlainText(editor).length, text: composerPlainText(editor) })

    expect(redone?.text).toBe('hello!')
    expect(recordUndoPoint).toHaveBeenCalledTimes(1)
  })

  it('keeps an identical selection replacement a true no-op', () => {
    const { ctx, editor, recordUndoPoint, scheduleFlushEditorToDraft } = makeHarness('hello world')
    selectRange(editor, 0, 5)

    ctx.insertText('hello')

    expect(editor.textContent).toBe('hello world')
    expect(recordUndoPoint).not.toHaveBeenCalled()
    expect(scheduleFlushEditorToDraft).not.toHaveBeenCalled()
  })

  it('preserves redo for an identical chip replacement', () => {
    const value = '@url:`https://example.dev`'
    const editor = makeEditor(`hello ${value}`)
    renderComposerContents(editor, `hello ${value}`)
    const history = createComposerUndoHistory()

    const recordUndoPoint = vi.fn(() => {
      history.record({ caret: composerPlainText(editor).length, text: composerPlainText(editor) })
    })

    const ctx = createComposerRenderContext({
      editorRef: { current: editor } as RefObject<HTMLDivElement | null>,
      composingRef: { current: false },
      recordUndoPoint,
      scheduleFlushEditorToDraft: vi.fn()
    })

    placeCaretEnd(editor)
    ctx.insertText('!')

    const afterInsert = { caret: composerPlainText(editor).length, text: composerPlainText(editor) }
    const undone = history.undo(afterInsert)

    expect(undone?.text).toBe(`hello ${value}`)
    renderComposerContents(editor, undone!.text)
    const chip = editor.querySelector('[data-ref-text]')
    expect(chip).not.toBeNull()
    selectNode(chip!)

    ctx.insertText(value)

    const redone = history.redo({ caret: composerPlainText(editor).length, text: composerPlainText(editor) })

    expect(redone?.text).toBe(afterInsert.text)
    expect(recordUndoPoint).toHaveBeenCalledTimes(1)
  })
})
