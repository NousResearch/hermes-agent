import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { DOUBLE_ENTER_WINDOW_MS, isDoubleEnter } from './composer-utils'

// No global setupFiles registers auto-cleanup, so unmount between tests —
// otherwise a second render() leaks the first editor and getByTestId('editor')
// matches multiple nodes.
afterEach(cleanup)

// Faithful mirror of index.tsx's Enter branch AFTER the double-Enter change:
// the live-DOM payload read, the queue drain / busy send-now / empty-Enter
// gates, and the new double-Enter tracker that converts a second empty Enter
// into the "Continue" filler via loadIntoComposer + submitDraft.
//
// The gesture arms on a real send (payload Enter), so the filler flows through
// the same routing ladder as a typed message: steer while busy (the mirror's
// onQueue), send fresh when idle (onSubmit).
function Harness({
  busy = false,
  disabled = false,
  queued = [],
  continueEnabled = true,
  queueEdit = false,
  repeat = false,
  now = () => 0,
  onSubmit,
  onQueue,
  onDrain,
  onSendNow,
  onSaveQueueEdit,
  onCancel
}: {
  busy?: boolean
  disabled?: boolean
  queued?: readonly string[]
  continueEnabled?: boolean
  queueEdit?: boolean
  repeat?: boolean
  now?: () => number
  onSubmit: (text: string) => void
  onQueue: (text: string) => void
  onDrain: () => void
  onSendNow?: (id: string) => void
  onSaveQueueEdit?: () => void
  onCancel: () => void
}) {
  const editorRef = useRef<HTMLDivElement>(null)
  const draftRef = useRef('')
  const [draft, setDraft] = useState('')
  const lastEmptyEnterAtRef = useRef<number | null>(null)
  const attachments: unknown[] = []

  const composerPlainText = (el: HTMLElement) => el.textContent ?? ''

  const setText = (next: string) => {
    draftRef.current = next
    setDraft(next)
  }

  // Mirror of loadIntoComposer's paint: synchronous DOM write + draftRef sync.
  const loadIntoComposer = (text: string) => {
    if (editorRef.current) {
      editorRef.current.textContent = text
    }

    setText(text)
  }

  const submitDraft = () => {
    if (disabled) {
      return
    }

    const editor = editorRef.current

    if (editor) {
      const domText = composerPlainText(editor)

      if (domText !== draftRef.current) {
        draftRef.current = domText
        setDraft(domText)
      }
    }

    const text = draftRef.current
    const payloadPresent = text.trim().length > 0 || attachments.length > 0

    if (busy) {
      if (payloadPresent) {
        onQueue(text)
      } else {
        onCancel()
      }
    } else if (!payloadPresent && queued.length > 0) {
      onDrain()
    } else if (payloadPresent) {
      onSubmit(text)
    }
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()

      if (queueEdit) {
        onSaveQueueEdit?.()

        return
      }

      const editorText = editorRef.current ? composerPlainText(editorRef.current) : draftRef.current
      const hasLivePayload = editorText.trim().length > 0 || attachments.length > 0

      if (disabled) {
        return
      }

      if (!busy && !hasLivePayload && queued.length > 0) {
        onDrain()

        return
      }

      if (!hasLivePayload) {
        const head = busy ? queued[0] : undefined

        if (head) {
          onSendNow?.(head)

          return
        }

        const continueFire =
          continueEnabled && !queueEdit && !event.repeat && isDoubleEnter(lastEmptyEnterAtRef.current, now())

        if (continueFire) {
          lastEmptyEnterAtRef.current = null
          loadIntoComposer('Continue')
        } else {
          lastEmptyEnterAtRef.current = now()

          if (busy) {
            return
          }
        }

        submitDraft()

        return
      }

      lastEmptyEnterAtRef.current = now()

      submitDraft()
    }
  }

  // `draft` is read so the lint/compiler treats the stale-state mirror as live;
  // the tracker holds its own ref, mirroring index.tsx.
  void draft

  return (
    <div
      contentEditable
      data-testid="editor"
      onInput={event => setText(composerPlainText(event.currentTarget))}
      onKeyDown={handleKeyDown}
      ref={editorRef}
      suppressContentEditableWarning
    />
  )
}

describe('composer double-Enter Continue', () => {
  it('steers the live turn when the second empty Enter lands within the window', async () => {
    const onQueue = vi.fn()
    const onCancel = vi.fn()
    let clock = 0
    const now = () => clock

    const { getByTestId } = render(
      <Harness busy now={now} onCancel={onCancel} onDrain={vi.fn()} onQueue={onQueue} onSubmit={vi.fn()} />
    )

    const editor = getByTestId('editor')

    // First Enter sends a message and empties the composer.
    await act(async () => {
      editor.textContent = 'fix the failing test'
      fireEvent.keyDown(editor, { key: 'Enter' })
      editor.textContent = ''
      clock += 100
    })

    expect(onQueue).toHaveBeenCalledWith('fix the failing test')

    // Second empty Enter, quickly: converts to the Continue nudge.
    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Enter', repeat: false })
    })

    expect(onQueue).toHaveBeenCalledWith('Continue')
    expect(onCancel).not.toHaveBeenCalled()
    // The filler was loaded into the composer, not silently sent from nothing.
    expect(editor.textContent).toBe('Continue')
  })

  it('sends Continue as a fresh message when the turn already settled', async () => {
    const onSubmit = vi.fn()
    let clock = 0
    const now = () => clock

    const { getByTestId } = render(
      <Harness now={now} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'go on'
      fireEvent.keyDown(editor, { key: 'Enter' })
      editor.textContent = ''
      clock += 50
    })

    expect(onSubmit).toHaveBeenCalledWith('go on')

    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Enter', repeat: false })
    })

    expect(onSubmit).toHaveBeenLastCalledWith('Continue')
  })

  it('never fires from a held-down Enter (key repeat)', async () => {
    const onQueue = vi.fn()
    const onCancel = vi.fn()
    let clock = 0
    const now = () => clock

    const { getByTestId } = render(
      <Harness busy now={now} onCancel={onCancel} onDrain={vi.fn()} onQueue={onQueue} onSubmit={vi.fn()} />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'hello'
      fireEvent.keyDown(editor, { key: 'Enter' })
      editor.textContent = ''
      clock += 10
    })

    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Enter', repeat: true })
    })

    expect(onQueue).not.toHaveBeenCalledWith('Continue')
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('stays a no-op when the second empty Enter is outside the window', async () => {
    const onQueue = vi.fn()
    const onCancel = vi.fn()
    let clock = 0
    const now = () => clock

    const { getByTestId } = render(
      <Harness busy now={now} onCancel={onCancel} onDrain={vi.fn()} onQueue={onQueue} onSubmit={vi.fn()} />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'hello'
      fireEvent.keyDown(editor, { key: 'Enter' })
      editor.textContent = ''
      clock += DOUBLE_ENTER_WINDOW_MS + 500
    })

    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Enter', repeat: false })
    })

    expect(onQueue).not.toHaveBeenCalledWith('Continue')
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('never fires when the setting is off (default behavior preserved)', async () => {
    const onQueue = vi.fn()
    const onCancel = vi.fn()
    let clock = 0
    const now = () => clock

    const { getByTestId } = render(
      <Harness
        busy
        continueEnabled={false}
        now={now}
        onCancel={onCancel}
        onDrain={vi.fn()}
        onQueue={onQueue}
        onSubmit={vi.fn()}
      />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'hello'
      fireEvent.keyDown(editor, { key: 'Enter' })
      editor.textContent = ''
      clock += 50
    })

    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Enter', repeat: false })
    })

    expect(onQueue).not.toHaveBeenCalledWith('Continue')
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('does not fire while editing a queued turn (Enter saves the edit)', async () => {
    const onSaveQueueEdit = vi.fn()
    let clock = 0
    const now = () => clock

    const { getByTestId } = render(
      <Harness
        busy
        now={now}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSaveQueueEdit={onSaveQueueEdit}
        onSubmit={vi.fn()}
        queueEdit
      />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'hello'
      fireEvent.keyDown(editor, { key: 'Enter' })
      editor.textContent = ''
      clock += 50
    })

    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Enter', repeat: false })
    })

    expect(onSaveQueueEdit).toHaveBeenCalled()
  })
})
