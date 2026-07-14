import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef, useState } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

// No global setupFiles registers auto-cleanup, so unmount between tests —
// otherwise a second render() leaks the first editor and getByTestId('editor')
// matches multiple nodes.
afterEach(cleanup)

// Faithful mirror of index.tsx's IME-guard wiring after the #39649 fix:
// - keydown and input both resync composingRef from the current event's
//   native isComposing before deciding whether to block, so a stuck `true`
//   can't outlive the (possibly fake) composition that set it.
// - blur clears composingRef synchronously, covering the Send-button submit
//   path, which has no composition event of its own to resync from.
//
// Regression repro for #39649: on Windows, Chromium/Electron can fire
// `compositionstart` for perfectly ordinary English keystrokes and never
// fire the matching `compositionend`, leaving composingRef.current stuck
// `true` forever. Before the fix that silently swallowed every later Enter
// (and blocked the Send button too) as if IME composition were still active.
function Harness({ onSubmit }: { onSubmit: () => void }) {
  const editorRef = useRef<HTMLDivElement>(null)
  const composingRef = useRef(false)
  const draftRef = useRef('')
  const [draft, setDraft] = useState('')

  const flush = (editor: HTMLDivElement) => {
    const next = editor.textContent ?? ''

    if (next !== draftRef.current) {
      draftRef.current = next
      setDraft(next)
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
        onCompositionEnd={event => {
          composingRef.current = false
          flush(event.currentTarget)
        }}
        onCompositionStart={() => {
          composingRef.current = true
        }}
        onInput={event => {
          if (composingRef.current && (event.nativeEvent as InputEvent).isComposing === false) {
            composingRef.current = false
          }

          if (composingRef.current) {
            return
          }

          flush(event.currentTarget)
        }}
        onKeyDown={event => {
          if (composingRef.current && event.nativeEvent.isComposing === false) {
            composingRef.current = false
          }

          if (composingRef.current || event.nativeEvent.isComposing) {
            return
          }

          if (event.key === 'Enter') {
            onSubmit()
          }
        }}
        ref={editorRef}
        suppressContentEditableWarning
      />
      <button
        data-testid="send-button"
        onClick={() => {
          if (composingRef.current) {
            return
          }

          onSubmit()
        }}
        type="button"
      >
        Send
      </button>
      <span data-testid="draft">{draft}</span>
    </div>
  )
}

describe('composer IME guard — Windows stale composingRef (#39649)', () => {
  it('Enter still submits after a false compositionstart with no compositionend', async () => {
    let submitCount = 0
    const { getByTestId } = render(<Harness onSubmit={() => submitCount++} />)
    const editor = getByTestId('editor')

    // The Windows quirk: compositionstart fires for plain input, but no
    // compositionend ever follows, so composingRef is left stuck `true`.
    await act(async () => {
      fireEvent.compositionStart(editor)
      editor.textContent = 'hello'
      fireEvent.input(editor, { isComposing: false })
    })

    expect(getByTestId('draft').textContent).toBe('hello')

    await act(async () => {
      fireEvent.keyDown(editor, { isComposing: false, key: 'Enter' })
    })

    expect(submitCount).toBe(1)
  })

  it('genuine IME composition still blocks Enter from submitting', async () => {
    let submitCount = 0
    const { getByTestId } = render(<Harness onSubmit={() => submitCount++} />)
    const editor = getByTestId('editor')

    await act(async () => {
      fireEvent.compositionStart(editor)
      editor.textContent = 'ㅎ'
      fireEvent.input(editor, { isComposing: true })
    })

    // Enter while genuinely composing must confirm the preedit, not submit.
    await act(async () => {
      fireEvent.keyDown(editor, { isComposing: true, key: 'Enter' })
    })

    expect(submitCount).toBe(0)

    await act(async () => {
      editor.textContent = '한'
      fireEvent.compositionEnd(editor)
    })

    await act(async () => {
      fireEvent.keyDown(editor, { isComposing: false, key: 'Enter' })
    })

    expect(submitCount).toBe(1)
  })

  it('Send button still works after a false compositionstart with no compositionend', async () => {
    let submitCount = 0
    const { getByTestId } = render(<Harness onSubmit={() => submitCount++} />)
    const editor = getByTestId('editor')

    await act(async () => {
      fireEvent.compositionStart(editor)
      editor.textContent = 'hello'
      fireEvent.input(editor, { isComposing: false })
    })

    // Clicking the Send button blurs the editor first, same as a real click.
    await act(async () => {
      fireEvent.blur(editor)
      fireEvent.click(getByTestId('send-button'))
    })

    expect(submitCount).toBe(1)
  })
})
