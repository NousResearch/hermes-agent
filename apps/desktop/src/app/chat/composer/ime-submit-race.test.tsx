import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useCallback, useRef, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

afterEach(cleanup)

// Minimal harness that mirrors the IME composition + submitDraft flow from
// index.tsx. The key difference from ime-composition-dom-repro.test.tsx is
// that this harness exercises a `submitDraft`-equivalent function and
// verifies the *submitted* text (not just `hasPayload`).
//
// Regression repro for #40633: pressing Enter immediately after IME
// compositionend used to submit stale React state because
// `aui.composer().setText()` is async — the re-render hadn't committed yet.
function Harness({ onSubmit }: { onSubmit: (text: string) => void }) {
  const editorRef = useRef<HTMLDivElement>(null)
  const composingRef = useRef(false)
  const draftRef = useRef('')
  const [draft, setDraft] = useState('')

  const flushEditorToDraft = useCallback((editor: HTMLDivElement) => {
    const next = editor.textContent ?? ''
    if (next !== draftRef.current) {
      draftRef.current = next
      setDraft(next)
    }
  }, [])

  // Mirrors the fixed submitDraft: flush DOM → read draftRef, NOT React state.
  const submitDraft = useCallback(() => {
    if (editorRef.current) {
      flushEditorToDraft(editorRef.current)
    }
    const currentDraft = draftRef.current
    if (currentDraft.trim()) {
      onSubmit(currentDraft)
    }
  }, [flushEditorToDraft, onSubmit])

  return (
    <div>
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
          if (composingRef.current) return
          flushEditorToDraft(event.currentTarget)
        }}
        onKeyDown={event => {
          if (composingRef.current || event.nativeEvent.isComposing) return
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            submitDraft()
          }
        }}
        ref={editorRef}
        suppressContentEditableWarning
      />
      <span data-testid="draft">{draft}</span>
    </div>
  )
}

describe('IME composition — submitDraft race condition (#40633)', () => {
  it('submits the full composed text when Enter is pressed right after compositionend', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    // Simulate Korean IME: compositionstart → intermediate input →
    // compositionend → immediate Enter (no trailing input event).
    await act(async () => {
      fireEvent.compositionStart(editor)
      editor.textContent = '안녕'
      fireEvent.input(editor)
      fireEvent.compositionEnd(editor)
      // Press Enter immediately — before React re-renders with the new draft.
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    // Before the fix, submitted text was stale ("안" or "") because
    // React state hadn't updated. After the fix, the full text is submitted.
    expect(onSubmit).toHaveBeenCalledWith('안녕')
  })

  it('submits full Chinese text on immediate Enter after composition', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    await act(async () => {
      fireEvent.compositionStart(editor)
      editor.textContent = '你好世界'
      fireEvent.input(editor)
      fireEvent.compositionEnd(editor)
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onSubmit).toHaveBeenCalledWith('你好世界')
  })

  it('submits full Japanese text on immediate Enter after composition', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    await act(async () => {
      fireEvent.compositionStart(editor)
      editor.textContent = 'こんにちは'
      fireEvent.input(editor)
      fireEvent.compositionEnd(editor)
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onSubmit).toHaveBeenCalledWith('こんにちは')
  })

  it('does not submit when composer is empty after composition', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    await act(async () => {
      fireEvent.compositionStart(editor)
      fireEvent.compositionEnd(editor)
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })
})
