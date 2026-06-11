import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef, useState, KeyboardEvent } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

// No global setupFiles registers auto-cleanup, so unmount between tests —
// otherwise a second render() leaks the first editor and getByTestId('editor')
// matches multiple nodes.
afterEach(cleanup)

// Faithful mirror of index.tsx's composer text wiring for IME input, driven
// through REAL DOM composition + input events on a contentEditable.
function Harness({ onPayload, onSubmit }: { onPayload: (hasPayload: boolean) => void; onSubmit?: (text: string) => void }) {
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

  const handleEditorKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    // #44278: Prevent submission if composition is ongoing or Windows IME firmed code 229
    if (composingRef.current || event.nativeEvent.isComposing || event.keyCode === 229) {
      return
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      // Read directly from live DOM like index.tsx's submitDraft fix
      const editorNode = editorRef.current
      const submitted = editorNode ? editorNode.textContent?.trim() || draft : draft
      if (onSubmit) {
        onSubmit(submitted)
      }
    }
  }

  onPayload(draft.trim().length > 0)

  return (
    <div
      contentEditable
      data-testid="editor"
      onKeyDown={handleEditorKeyDown}
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

    await act(async () => {
      fireEvent.compositionStart(editor)
      editor.textContent = '你'
      fireEvent.input(editor)
      editor.textContent = '你好'
      fireEvent.input(editor)
      fireEvent.compositionEnd(editor)
    })

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

      await act(async () => {
        editor.textContent = ''
        fireEvent.input(editor)
      })
      expect(hasPayload).toBe(false)
    }
  })
})

describe('composer Korean IME Enter submission (#44278)', () => {
  it('does not drop the final syllable when Enter is pressed on Windows Korean IME', async () => {
    let submittedText = ''
    let hasPayload = false
    const { getByTestId } = render(
      <Harness 
        onPayload={p => (hasPayload = p)} 
        onSubmit={text => { submittedText = text }} 
      />
    )
    const editor = getByTestId('editor')

    // Simulate Windows Korean IME sequence for "테스트" (Test)
    // where Enter fires keydown 229 mid-composition before compositionend flushes to React state
    await act(async () => {
      fireEvent.compositionStart(editor)
      editor.textContent = '테스트'
      
      // First keydown with 229 while composing
      fireEvent.keyDown(editor, { keyCode: 229, key: 'Process' })
      
      // Then IME commits and compositionend triggers, followed by the clean Enter keydown
      fireEvent.compositionEnd(editor)
      fireEvent.keyDown(editor, { keyCode: 13, key: 'Enter' })
    })

    // The full word should be preserved and submitted successfully
    expect(submittedText).toBe('테스트')
  })
})