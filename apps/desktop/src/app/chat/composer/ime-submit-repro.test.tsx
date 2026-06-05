import { cleanup, fireEvent, render } from '@testing-library/react'
import { type KeyboardEvent, useCallback, useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { RICH_INPUT_SLOT, syncComposerDraft } from './rich-editor'

// Faithful mirror of index.tsx's IME-relevant wiring (composingRef guard on
// input, the compositionend draft sync, and the Enter→submit keydown path),
// driven through REAL DOM events on a contentEditable. Same approach as
// slash-nav-dom-repro.test.tsx: rendering the full ChatBar needs the
// assistant-ui runtime context + a large prop surface, so we exercise the
// shared real helper (syncComposerDraft) through the same handler shape.
//
// Reproduces #39025: on Windows/Electron the trailing `input` event after
// `compositionend` is sometimes dropped, so without the compositionend sync
// the draft stays stale and Enter no-ops despite visible text.
function Harness({ onSubmit }: { onSubmit: (text: string) => void }) {
  const editorRef = useRef<HTMLDivElement>(null)
  const composingRef = useRef(false)
  const draftRef = useRef('')

  const sync = useCallback((editor: HTMLElement | null = editorRef.current) => {
    // The real component routes setText into the assistant-ui composer state;
    // this submit-path test only cares that draftRef tracks the live DOM.
    draftRef.current = syncComposerDraft(editor, draftRef.current, () => {})

    return draftRef.current
  }, [])

  return (
    <div
      contentEditable
      data-slot={RICH_INPUT_SLOT}
      data-testid="editor"
      onCompositionEnd={() => {
        composingRef.current = false
        sync()
      }}
      onCompositionStart={() => {
        composingRef.current = true
      }}
      onInput={() => {
        if (composingRef.current) {
          return
        }

        sync()
      }}
      onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => {
        if (composingRef.current || event.nativeEvent.isComposing) {
          return
        }

        if (event.key === 'Enter' && !event.shiftKey) {
          event.preventDefault()

          // Mirrors the hardened submitDraft(): commit the live DOM and decide
          // from the synchronously-fresh value, never the async draft state.
          const live = sync()

          if (live.trim()) {
            onSubmit(live)
          }
        }
      }}
      ref={editorRef}
      role="textbox"
      suppressContentEditableWarning
    />
  )
}

describe('composer IME submit (#39025)', () => {
  afterEach(() => cleanup())

  it('submits IME-finalised text on Enter even when the post-composition input event is dropped', () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    // IME composition: the trailing `input` event is deliberately NOT fired to
    // model the Windows/Electron drop; the DOM still ends up with the text.
    fireEvent.compositionStart(editor)
    editor.textContent = '你好世界'
    fireEvent.compositionEnd(editor)

    // Enter (composition already ended) must send the visible text.
    fireEvent.keyDown(editor, { key: 'Enter' })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('你好世界')
  })

  it('does not submit while composition is still active (Enter confirms the candidate)', () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    fireEvent.compositionStart(editor)
    editor.textContent = '你好'
    // Enter during composition (isComposing) confirms the candidate, never submits.
    fireEvent.keyDown(editor, { key: 'Enter', isComposing: true })

    expect(onSubmit).not.toHaveBeenCalled()
  })
})
