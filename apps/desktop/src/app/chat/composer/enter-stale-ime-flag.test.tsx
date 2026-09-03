import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

afterEach(cleanup)

// Faithful mirror of index.tsx's IME wiring: the composition guard at the top
// of handleEditorKeyDown (self-heal + swallow), the compositionstart/end
// handlers, and the blur reset.
//
// Regression repro for #44135: compositionend can be missed (focus jumps,
// input-source switches, programmatic DOM swaps mid-preedit), leaving
// composingRef wedged true. Before the fix, a wedged flag silently swallowed
// every Enter — and, via the form onSubmit guard, the Send button — until the
// composer remounted, which read as "Enter has no effect, no error, nothing
// reaches the gateway". The fix trusts Chromium's per-keydown isComposing flag
// to clear a stale ref, and clears it on blur (a composition never survives
// focus loss).
function Harness({ onSubmit, wedgeComposing }: { onSubmit: (text: string) => void; wedgeComposing?: boolean }) {
  const editorRef = useRef<HTMLDivElement>(null)
  const composingRef = useRef(Boolean(wedgeComposing))

  const submitDraft = () => {
    onSubmit(editorRef.current?.textContent ?? '')
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (composingRef.current && !event.nativeEvent.isComposing) {
      composingRef.current = false
    }

    if (composingRef.current || event.nativeEvent.isComposing) {
      return
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submitDraft()
    }
  }

  return (
    <div>
      <div
        contentEditable
        data-testid="editor"
        onBlur={() => {
          composingRef.current = false
        }}
        onCompositionEnd={() => {
          composingRef.current = false
        }}
        onCompositionStart={() => {
          composingRef.current = true
        }}
        onKeyDown={handleKeyDown}
        ref={editorRef}
        suppressContentEditableWarning
      />
      <button
        data-testid="send"
        onClick={() => {
          // Mirrors the form onSubmit guard in index.tsx.
          if (composingRef.current) {
            return
          }

          submitDraft()
        }}
        type="button"
      />
    </div>
  )
}

describe('composer Enter — stale IME composition flag recovery (#44135)', () => {
  it('sends on Enter despite a wedged composing flag when the native event says not composing', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} wedgeComposing />)
    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'hello after wedge'
      fireEvent.keyDown(editor, { key: 'Enter', isComposing: false })
    })

    expect(onSubmit).toHaveBeenCalledWith('hello after wedge')
  })

  it('still swallows Enter during a genuine composition (isComposing keydown)', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    await act(async () => {
      fireEvent.compositionStart(editor)
      editor.textContent = '你好'
      // The Enter that confirms the preedit: Chromium stamps isComposing=true.
      fireEvent.keyDown(editor, { key: 'Enter', isComposing: true })
    })

    expect(onSubmit).not.toHaveBeenCalled()

    // After compositionend, the next Enter sends normally.
    await act(async () => {
      fireEvent.compositionEnd(editor)
      fireEvent.keyDown(editor, { key: 'Enter', isComposing: false })
    })

    expect(onSubmit).toHaveBeenCalledWith('你好')
  })

  it('unblocks the Send button after blur even when compositionend was missed', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    await act(async () => {
      fireEvent.compositionStart(editor)
      editor.textContent = '发送'
      // compositionend never fires (the wedge) — the user mouses to Send,
      // blurring the editor.
      fireEvent.blur(editor)
      fireEvent.click(getByTestId('send'))
    })

    expect(onSubmit).toHaveBeenCalledWith('发送')
  })
})
