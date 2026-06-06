import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef, useState } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

// No global setupFiles registers auto-cleanup, so unmount between tests —
// otherwise a second render() leaks the first editor and getByTestId('editor')
// matches multiple nodes.
afterEach(cleanup)

// Faithful mirror of index.tsx's composer text wiring for IME input, driven
// through REAL DOM composition + input events on a contentEditable.
//
// Regression repro for #39614: typing committed multi-character IME text (e.g.
// Chinese "你好") used to leave the send button hidden. The input events fired
// during composition carry uncommitted preedit text and are intentionally
// skipped; Chromium then does NOT reliably emit a trailing input event after
// compositionend on Windows IMEs, so the finalized text never reached composer
// state and `hasPayload` stayed false until an unrelated edit forced a sync.
// The fix flushes the live DOM text in onCompositionEnd.
function Harness({ onPayload }: { onPayload: (hasPayload: boolean) => void }) {
  const editorRef = useRef<HTMLDivElement>(null)
  const composingRef = useRef(false)
  const draftRef = useRef('')
  const [draft, setDraft] = useState('')

  const flushEditorToDraft = (editor: HTMLDivElement) => {
    const next = editor.textContent ?? ''

    if (next !== draftRef.current) {
      draftRef.current = next
      setDraft(next)
    }
  }

  onPayload(draft.trim().length > 0)

  return (
    <div
      contentEditable
      data-testid="editor"
      onCompositionEnd={event => {
        composingRef.current = false
        flushEditorToDraft(event.currentTarget)
      }}
      onCompositionStart={() => {
        composingRef.current = true
      }}
      onInput={event => {
        if (composingRef.current) {
          return
        }

        flushEditorToDraft(event.currentTarget)
      }}
      ref={editorRef}
      suppressContentEditableWarning
    />
  )
}

describe('composer IME composition — send button visibility (#39614)', () => {
  it('shows the send button after committing CJK text without a trailing edit', async () => {
    let hasPayload = false
    const { getByTestId } = render(<Harness onPayload={p => (hasPayload = p)} />)
    const editor = getByTestId('editor')

    // Compose "你好" the way a Windows Chinese IME does: compositionstart, then
    // input events carrying uncommitted preedit text, then compositionend with
    // the committed text already in the DOM — and crucially NO input event
    // afterwards.
    await act(async () => {
      fireEvent.compositionStart(editor)
      editor.textContent = '你'
      fireEvent.input(editor)
      editor.textContent = '你好'
      fireEvent.input(editor)
      fireEvent.compositionEnd(editor)
    })

    // Before the fix this was false (button hidden) until a further edit.
    expect(hasPayload).toBe(true)
    expect(editor.textContent).toBe('你好')
  })

  it('also covers Japanese/Korean and any IME-composed script', async () => {
    let hasPayload = false
    const { getByTestId } = render(<Harness onPayload={p => (hasPayload = p)} />)
    const editor = getByTestId('editor')

    for (const committed of ['こんにちは', '안녕하세요']) {
      await act(async () => {
        fireEvent.compositionStart(editor)
        editor.textContent = committed
        fireEvent.input(editor)
        fireEvent.compositionEnd(editor)
      })

      expect(hasPayload).toBe(true)

      // Clear for the next script.
      await act(async () => {
        editor.textContent = ''
        fireEvent.input(editor)
      })
      expect(hasPayload).toBe(false)
    }
  })
})
