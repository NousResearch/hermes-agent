import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

afterEach(cleanup)

// Faithful mirror of index.tsx's Send-button wiring: the button is
// type="submit" inside the composer form, gated by canSubmit = busy ||
// hasComposerPayload, where hasComposerPayload derives from AUI composer
// state. That state syncs from the contentEditable only via the rAF-coalesced
// input flush (or compositionend), so it lags the DOM by a frame or more under
// main-thread pressure. A disabled button swallows the whole click (Chromium
// drops mousedown/click on disabled controls and Tailwind's
// disabled:pointer-events-none removes the hit target), with no feedback —
// the #52950 symptom: clicks on Send do nothing while Enter still sends,
// because the Enter path reads the live DOM (#39630).
//
// The fix mirrors the #39630 seam for the click path: the form's
// pointerdown-capture syncs the live editor text into composer state, so the
// gate is open before the press's hit-test resolves on the button. Chromium
// anchors a click's target at the press's hit-test, so syncing any later than
// pointerdown is too late.
function Harness({
  busy = false,
  disabled = false,
  onSync,
  onSubmit
}: {
  busy?: boolean
  disabled?: boolean
  onSync?: (text: string) => void
  onSubmit: (text: string) => void
}) {
  const editorRef = useRef<HTMLDivElement>(null)
  const draftRef = useRef('')
  // Mirrors `useAuiState(s => s.composer.text)` — only updated via setText
  // (the rAF-coalesced flush), so it lags the DOM.
  const [draft, setDraft] = useState('')

  const composerPlainText = (el: HTMLElement) => el.textContent ?? ''

  const setText = (next: string) => {
    draftRef.current = next
    setDraft(next)
  }

  const syncDraftFromEditor = () => {
    const editor = editorRef.current

    if (!editor) {
      return draftRef.current
    }

    const text = composerPlainText(editor)

    if (text !== draftRef.current) {
      setText(text)
      onSync?.(text)
    }

    return text
  }

  const hasComposerPayload = draft.trim().length > 0
  const canSubmit = busy || hasComposerPayload

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

    if (text.trim().length > 0) {
      onSubmit(text)
    }
  }

  return (
    <form
      onPointerDownCapture={() => {
        // The fix: sync the live editor BEFORE the press hit-tests the button.
        syncDraftFromEditor()
      }}
      onSubmit={event => {
        event.preventDefault()
        submitDraft()
      }}
    >
      <div
        contentEditable
        data-testid="editor"
        onInput={event => setText(composerPlainText(event.currentTarget))}
        ref={editorRef}
        suppressContentEditableWarning
      />
      <button data-testid="send" disabled={disabled || !canSubmit} type="submit" />
    </form>
  )
}

// jsdom does not implement disabled-button event suppression (a disabled
// <button> still fires click in jsdom), so the race is asserted at the two
// seams that produce it in Chromium: (1) the gate opens synchronously inside
// pointerdown-capture — before any mousedown/click dispatch — and (2) the
// submit that then fires carries the live editor text, not the stale state.
describe('composer Send button — pointerdown syncs the live draft (#52950)', () => {
  it('opens the send gate synchronously on pointerdown when the DOM has text state has not synced', async () => {
    const onSync = vi.fn()
    const onSubmit = vi.fn()

    const { getByTestId } = render(<Harness onSubmit={onSubmit} onSync={onSync} />)

    const editor = getByTestId('editor')
    const send = getByTestId('send') as HTMLButtonElement

    // Fast typing / stalled flush: the DOM holds text, composer state does
    // not (the exact stale-state race from #39630, on the click path).
    await act(async () => {
      editor.textContent = 'hello world'
    })

    expect(send.disabled).toBe(true)

    await act(async () => {
      fireEvent.pointerDown(send)
    })

    expect(send.disabled).toBe(false)
    expect(onSync).toHaveBeenCalledWith('hello world')

    await act(async () => {
      fireEvent.click(send)
      // jsdom does not implicitly submit forms from a submit-button click.
      fireEvent.submit(send.closest('form')!)
    })

    expect(onSubmit).toHaveBeenCalledWith('hello world')
  })

  it('keeps the gate closed when the editor is genuinely empty', async () => {
    const onSync = vi.fn()
    const onSubmit = vi.fn()

    const { getByTestId } = render(<Harness onSubmit={onSubmit} onSync={onSync} />)

    const send = getByTestId('send') as HTMLButtonElement

    await act(async () => {
      fireEvent.pointerDown(send)
    })

    expect(send.disabled).toBe(true)
    expect(onSync).not.toHaveBeenCalled()

    await act(async () => {
      fireEvent.click(send)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('does not flip the gate for a whitespace-only editor', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)

    const editor = getByTestId('editor')
    const send = getByTestId('send') as HTMLButtonElement

    await act(async () => {
      editor.textContent = '   '
      fireEvent.pointerDown(send)
    })

    expect(send.disabled).toBe(true)
  })
})
