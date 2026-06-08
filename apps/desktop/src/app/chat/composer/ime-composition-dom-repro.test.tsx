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

function SubmitHarness({ onSubmit }: { onSubmit: (text: string) => void }) {
  const editorRef = useRef<HTMLDivElement>(null)
  const composingRef = useRef(false)
  const compositionJustEndedRef = useRef(false)
  const submitAfterCompositionRef = useRef(false)
  const compositionSubmitTimerRef = useRef<number | null>(null)
  const draftRef = useRef('')

  const flushEditorToDraft = (editor: HTMLDivElement) => {
    draftRef.current = editor.textContent ?? ''

    return draftRef.current
  }

  const submitDraft = () => {
    const editor = editorRef.current

    if (editor) {
      flushEditorToDraft(editor)
    }

    if (draftRef.current.trim()) {
      onSubmit(draftRef.current)
    }
  }

  const scheduleSubmitAfterComposition = () => {
    if (compositionSubmitTimerRef.current !== null) {
      window.clearTimeout(compositionSubmitTimerRef.current)
    }

    compositionSubmitTimerRef.current = window.setTimeout(() => {
      compositionSubmitTimerRef.current = null
      compositionJustEndedRef.current = false
      submitAfterCompositionRef.current = false
      submitDraft()
    }, 0)
  }

  return (
    <div
      contentEditable
      data-testid="submit-editor"
      onCompositionEnd={event => {
        composingRef.current = false
        compositionJustEndedRef.current = true
        flushEditorToDraft(event.currentTarget)

        if (submitAfterCompositionRef.current) {
          scheduleSubmitAfterComposition()
        } else {
          window.setTimeout(() => {
            compositionJustEndedRef.current = false
          }, 0)
        }
      }}
      onCompositionStart={() => {
        composingRef.current = true
        compositionJustEndedRef.current = false
        submitAfterCompositionRef.current = false
      }}
      onInput={event => {
        if (composingRef.current) {
          return
        }

        flushEditorToDraft(event.currentTarget)
      }}
      onKeyDown={event => {
        const nativeEvent = event.nativeEvent as globalThis.KeyboardEvent & { keyCode?: number }
        const plainEnter =
          event.key === 'Enter' && !event.shiftKey && !event.metaKey && !event.ctrlKey && !event.altKey

        if (composingRef.current || event.nativeEvent.isComposing || nativeEvent.keyCode === 229) {
          if (plainEnter) {
            submitAfterCompositionRef.current = true
          }

          return
        }

        if (plainEnter && compositionJustEndedRef.current) {
          event.preventDefault()
          submitAfterCompositionRef.current = true
          scheduleSubmitAfterComposition()

          return
        }

        if (plainEnter) {
          event.preventDefault()
          submitDraft()
        }
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


describe('composer IME composition — Enter submit preserves Korean text', () => {
  it('submits the committed Korean syllable after Enter finalizes active composition', async () => {
    const submitted: string[] = []
    const { getByTestId } = render(<SubmitHarness onSubmit={text => submitted.push(text)} />)
    const editor = getByTestId('submit-editor')

    await act(async () => {
      editor.textContent = '안'
      fireEvent.input(editor)
      fireEvent.compositionStart(editor)
      editor.textContent = '안녀'
      fireEvent.input(editor)
      fireEvent.keyDown(editor, { key: 'Enter', keyCode: 229 })
      editor.textContent = '안녕'
      fireEvent.compositionEnd(editor)
      await new Promise(resolve => window.setTimeout(resolve, 0))
    })

    expect(submitted).toEqual(['안녕'])
  })

  it('waits one tick when compositionend precedes the Enter keydown but the final DOM commit trails it', async () => {
    const submitted: string[] = []
    const { getByTestId } = render(<SubmitHarness onSubmit={text => submitted.push(text)} />)
    const editor = getByTestId('submit-editor')

    await act(async () => {
      editor.textContent = '안'
      fireEvent.input(editor)
      fireEvent.compositionStart(editor)
      editor.textContent = '안녀'
      fireEvent.input(editor)
      editor.textContent = '안'
      fireEvent.compositionEnd(editor)
      fireEvent.keyDown(editor, { key: 'Enter', keyCode: 13 })
      editor.textContent = '안녕'
      await new Promise(resolve => window.setTimeout(resolve, 0))
    })

    expect(submitted).toEqual(['안녕'])
  })
})
