import { cleanup, fireEvent, render } from '@testing-library/react'
import type { FormEvent } from 'react'
import { useRef, useState } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

// Regression for the Korean/CJK IME truncation bug: in Chromium the final
// `input` event of an IME commit carries isComposing=true (so the input
// handler skips the state write), and no input event follows `compositionend`.
// If compositionend only flips the composing flag without committing the DOM
// text, the last composed cluster lives in the DOM but never reaches draft
// state — and is lost on submit. This harness mirrors index.tsx's exact
// wiring (composingRef guard + shared commitEditorState + compositionend
// commit) and drives the real Chromium IME event order.
function Harness({ onDraft }: { onDraft: (draft: string) => void }) {
  const editorRef = useRef<HTMLDivElement>(null)
  const composingRef = useRef(false)
  const draftRef = useRef('')
  const [, force] = useState(0)

  const commitEditorState = (editor: HTMLDivElement) => {
    const nextDraft = editor.textContent ?? ''

    if (nextDraft !== draftRef.current) {
      draftRef.current = nextDraft
      onDraft(nextDraft)
      force(n => n + 1)
    }
  }

  const handleEditorInput = (event: FormEvent<HTMLDivElement>) => {
    if (composingRef.current) {
      return
    }

    commitEditorState(event.currentTarget)
  }

  return (
    <div
      contentEditable
      data-testid="editor"
      onCompositionEnd={event => {
        composingRef.current = false
        commitEditorState(event.currentTarget)
      }}
      onCompositionStart={() => {
        composingRef.current = true
      }}
      onInput={handleEditorInput}
      ref={editorRef}
      suppressContentEditableWarning
    />
  )
}

describe('composer IME composition commit', () => {
  afterEach(cleanup)

  it('commits the composed text on compositionend even when the final input event is mid-composition', () => {
    let draft = ''
    const { getByTestId } = render(<Harness onDraft={d => (draft = d)} />)
    const editor = getByTestId('editor')

    // Real Chromium order for typing a Korean syllable via IME:
    fireEvent.compositionStart(editor)
    // Preedit appears in the DOM; the input event during composition is
    // isComposing=true and must be ignored by the input handler.
    editor.textContent = '안녕'
    fireEvent.input(editor, { nativeEvent: { isComposing: true } })

    // Before compositionend, the input handler has (correctly) written nothing.
    expect(draft).toBe('')

    // compositionend finalises the syllable — this is where the commit must
    // happen, since no further input event follows in Chromium.
    fireEvent.compositionEnd(editor)

    expect(draft).toBe('안녕')
  })

  it('still commits ordinary (non-IME) input via the input event', () => {
    let draft = ''
    const { getByTestId } = render(<Harness onDraft={d => (draft = d)} />)
    const editor = getByTestId('editor')

    editor.textContent = 'hello'
    fireEvent.input(editor, { nativeEvent: { isComposing: false } })

    expect(draft).toBe('hello')
  })
})
