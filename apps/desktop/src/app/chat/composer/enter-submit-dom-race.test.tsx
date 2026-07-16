import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { composerPlainText, insertPlainTextAtCaret, RICH_INPUT_SLOT } from './rich-editor'

// No global setupFiles registers auto-cleanup, so unmount between tests —
// otherwise a second render() leaks the first editor and getByTestId('editor')
// matches multiple nodes.
afterEach(cleanup)

const caretIn = (editor: HTMLElement) => {
  const range = globalThis.document.createRange()
  const selection = globalThis.window.getSelection()!

  range.selectNodeContents(editor)
  range.collapse(false)
  selection.removeAllRanges()
  selection.addRange(range)
}

// Faithful mirror of index.tsx's Enter wiring (handleEditorKeyDown's Enter
// branch + submitDraft), driven through REAL DOM keydown events on a
// contentEditable.
//
// Regression repro for #39630: pressing Enter right after typing (fast typing /
// IME) did nothing. The composer state (`draft` from useAuiState) and its
// derived `hasComposerPayload` lag the DOM by a render, so the keydown handler
// read empty state and either dropped the message, drained a queued prompt
// instead of sending, or (while busy) refused to queue. The fix reads the live
// editor text — `hasLivePayload` in the handler and a DOM re-sync at the top of
// submitDraft — so the just-typed text always wins.
//
// We model the race deterministically the way the IME repro does: mutate the
// editor's textContent WITHOUT firing an input event, so the React `draft`
// state stays stale while the DOM already holds the text.
function Harness({
  busy = false,
  disabled = false,
  queued = [],
  onSubmit,
  onQueue,
  onCancel,
  onDrain
}: {
  busy?: boolean
  disabled?: boolean
  queued?: readonly string[]
  onSubmit: (text: string) => void
  onQueue: (text: string) => void
  onCancel: () => void
  onDrain: () => void
}) {
  const editorRef = useRef<HTMLDivElement>(null)
  const draftRef = useRef('')
  // Mirrors `useAuiState(s => s.composer.text)` — updated only via setText, so
  // it lags the DOM until React re-renders (the source of the bug).
  const [draft, setDraft] = useState('')
  const attachments: unknown[] = []

  const setText = (next: string) => {
    draftRef.current = next
    setDraft(next)
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
    if (event.key === 'Enter' && event.shiftKey && !event.metaKey && !event.ctrlKey && !event.altKey) {
      event.preventDefault()
      insertPlainTextAtCaret(event.currentTarget, '\n')
      setText(composerPlainText(event.currentTarget))

      return
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()

      const editorText = editorRef.current ? composerPlainText(editorRef.current) : draftRef.current
      const hasLivePayload = editorText.trim().length > 0 || attachments.length > 0

      if (disabled) {
        return
      }

      if (!busy && !hasLivePayload && queued.length > 0) {
        onDrain()

        return
      }

      if (busy && !hasLivePayload) {
        return
      }

      submitDraft()
    }
  }

  // `draft` is read so the lint/compiler treats the stale-state mirror as live;
  // the assertions prove the handler never relies on it.
  void draft

  return (
    <div
      contentEditable
      data-slot={RICH_INPUT_SLOT}
      data-testid="editor"
      onInput={event => setText(composerPlainText(event.currentTarget))}
      onKeyDown={handleKeyDown}
      ref={editorRef}
      suppressContentEditableWarning
    />
  )
}

describe('composer Enter submit — live DOM vs stale composer state (#39630)', () => {
  it('inserts a Shift+Enter newline without submitting and preserves it on send', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'first line'
      caretIn(editor)
      fireEvent.keyDown(editor, { key: 'Enter', shiftKey: true })
    })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(composerPlainText(editor)).toBe('first line\n')

    await act(async () => {
      editor.append('second line')
      caretIn(editor)
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onSubmit).toHaveBeenCalledWith('first line\nsecond line')
  })

  it('sends the just-typed text on Enter even when composer state has not synced', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />
    )

    const editor = getByTestId('editor')

    // Fast typing: the DOM has the text but NO input event fired, so `draft`
    // state is still empty (the exact stale-state race).
    await act(async () => {
      editor.textContent = 'hello world'
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onSubmit).toHaveBeenCalledWith('hello world')
  })

  it('queues a fast-typed message while busy instead of draining the queue or cancelling', async () => {
    const onQueue = vi.fn()
    const onDrain = vi.fn()
    const onCancel = vi.fn()

    const { getByTestId } = render(
      <Harness busy onCancel={onCancel} onDrain={onDrain} onQueue={onQueue} onSubmit={vi.fn()} queued={['queued-1']} />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'urgent follow-up'
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onQueue).toHaveBeenCalledWith('urgent follow-up')
    expect(onDrain).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('treats an empty Enter while busy as a no-op (never an accidental Stop)', async () => {
    const onCancel = vi.fn()
    const onSubmit = vi.fn()
    const onQueue = vi.fn()

    const { getByTestId } = render(
      <Harness busy onCancel={onCancel} onDrain={vi.fn()} onQueue={onQueue} onSubmit={onSubmit} />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = ''
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onCancel).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(onQueue).not.toHaveBeenCalled()
  })

  it('drains the next queued prompt on Enter when idle with a truly empty editor', async () => {
    const onDrain = vi.fn()
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness onCancel={vi.fn()} onDrain={onDrain} onQueue={vi.fn()} onSubmit={onSubmit} queued={['queued-1']} />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = ''
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onDrain).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('keeps reconnect drafts editable but blocks Enter submit until the gateway returns', async () => {
    const onSubmit = vi.fn()
    const onDrain = vi.fn()

    const { getByTestId } = render(
      <Harness
        disabled
        onCancel={vi.fn()}
        onDrain={onDrain}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        queued={['queued-1']}
      />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'draft while reconnecting'
      fireEvent.input(editor)
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(editor.textContent).toBe('draft while reconnecting')
    expect(onDrain).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })
})
