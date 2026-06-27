// @vitest-environment jsdom

import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef, useState } from 'react'
import { afterEach, describe, expect, it } from 'vitest'
import { vi } from 'vitest'

afterEach(cleanup)

// Faithful mirror of UserEditComposer's inline-edit textbox wiring in
// thread.tsx. This path is distinct from the main composer and needs the same
// IME composition guards.
function Harness({ onSubmit }: { onSubmit: (draft: string) => void }) {
  const draftRef = useRef('already sent text')
  const [draft, setDraft] = useState('already sent text')
  const composingRef = useRef(false)

  const syncDraftFromEditor = (editor: HTMLDivElement) => {
    const nextDraft = editor.textContent ?? ''

    if (nextDraft !== draftRef.current) {
      draftRef.current = nextDraft
      setDraft(nextDraft)
    }

    return nextDraft
  }

  const flushEditorToDraft = (editor: HTMLDivElement) => {
    const nextDraft = syncDraftFromEditor(editor)

    return nextDraft
  }

  const submitEdit = (editor: HTMLDivElement) => {
    if (composingRef.current) {
      return
    }

    const nextDraft = flushEditorToDraft(editor)

    if (!nextDraft.trim()) {
      return
    }

    onSubmit(nextDraft)
  }

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
      onKeyDown={event => {
        if (composingRef.current || event.nativeEvent.isComposing) {
          return
        }

        if (event.key === 'Enter' && !event.shiftKey) {
          event.preventDefault()
          submitEdit(event.currentTarget)
        }
      }}
      role="textbox"
      suppressContentEditableWarning
    >
      {draft}
    </div>
  )
}

describe('UserEditComposer IME reproduction', () => {
  it('does not submit an inline edit while IME composition is still active', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    // Edit a previously-sent message, start a CJK IME composition, then press
    // Enter to commit the candidate. Inline edit should treat that Enter as IME
    // confirmation, not "submit edited message".
    await act(async () => {
      fireEvent.compositionStart(editor)
      editor.textContent = 'already sent text你'
      fireEvent.input(editor)
      fireEvent.keyDown(editor, { isComposing: true, key: 'Enter' })
    })

    expect(onSubmit).not.toHaveBeenCalled()

    await act(async () => {
      fireEvent.compositionEnd(editor)
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('already sent text你')
  })
})
