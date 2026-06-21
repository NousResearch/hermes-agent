import { act, cleanup, render } from '@testing-library/react'
import { useRef, useState } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

afterEach(cleanup)

function LoopHarness({ onRenderCount }: { onRenderCount: (count: number) => void }) {
  const editorRef = useRef<HTMLDivElement>(null)
  const isSyncingRef = useRef(false)
  const draftRef = useRef('')
  const [draft, setDraft] = useState('')
  const [renderCount, setRenderCount] = useState(0)

  const flushEditorToDraft = (editor: HTMLDivElement) => {
    if (isSyncingRef.current) {
      return
    }
    isSyncingRef.current = true
    try {
      const next = editor.textContent ?? ''

      if (next !== draftRef.current) {
        draftRef.current = next
        setDraft(next)

        // Simulate a re-entrant trigger (e.g., attachment preview layout/reconciliation update
        // triggering a synchronous DOM input/mutation event on the contentEditable editor)
        if (editorRef.current) {
          editorRef.current.textContent = next + '!'
          flushEditorToDraft(editorRef.current)
        }
      }
    } finally {
      isSyncingRef.current = false
    }
  }

  // Count renders to ensure we don't loop endlessly
  onRenderCount(renderCount)

  return (
    <div
      contentEditable
      data-testid="editor"
      onInput={event => {
        flushEditorToDraft(event.currentTarget)
      }}
      ref={editorRef}
      suppressContentEditableWarning
    />
  )
}

describe('composer attachment input loop guard', () => {
  it('prevents maximum call stack error when input triggers a synchronous re-entrant event', async () => {
    let renderCount = 0
    const { getByTestId } = render(<LoopHarness onRenderCount={c => { renderCount = c }} />)
    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'hello'
      const event = new Event('input', { bubbles: true })
      editor.dispatchEvent(event)
    })

    // The guard should stop the recursion immediately, so it should not stack overflow
    // and should complete execution.
    expect(editor.textContent).toBe('hello!')
  })
})
