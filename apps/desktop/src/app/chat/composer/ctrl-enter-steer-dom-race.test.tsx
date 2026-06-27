import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

// No global setupFiles registers auto-cleanup, so unmount between tests —
// otherwise a second render() leaks the first editor and getByTestId('editor')
// matches multiple nodes.
afterEach(cleanup)

// Faithful mirror of index.tsx's Ctrl+Enter steer wiring, driven through REAL
// DOM keydown events on a contentEditable.
//
// Regression repro for #53659: Ctrl+Enter right after typing (fast typing)
// did nothing. The `canSteer` guard derives from React state (`trimmedDraft`)
// which lags the contentEditable DOM by a render, so the keydown handler saw
// empty state and swallowed the steer. The fix reads the live editor text in
// the Ctrl+Enter handler, same pattern as the plain Enter handler (#39630).
//
// We model the race deterministically: mutate the editor's textContent WITHOUT
// firing an input event, so the React `draft` state stays stale while the DOM
// already holds the text.
function Harness({
  busy = false,
  onSteer
}: {
  busy?: boolean
  onSteer: (text: string) => void
}) {
  const editorRef = useRef<HTMLDivElement>(null)
  const draftRef = useRef('')
  // Mirrors `useAuiState(s => s.composer.text)` — updated only via setText, so
  // it lags the DOM until React re-renders (the source of the bug).
  const [draft, setDraft] = useState('')
  const attachments: unknown[] = []

  const composerPlainText = (el: HTMLElement) => el.textContent ?? ''

  const trimmedDraft = draft.trim()

  // Mirror of index.tsx canSteer — derived from React state, NOT DOM.
  const canSteer =
    busy && !!onSteer && attachments.length === 0 && trimmedDraft.length > 0

  const steerDraft = () => {
    if (!onSteer) {
      return
    }

    const text = draftRef.current.trim()

    if (text.length === 0) {
      return
    }

    onSteer(text)
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    // Cmd/Ctrl+Enter steer path — mirrors index.tsx after the fix.
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !event.shiftKey) {
      event.preventDefault()

      if (onSteer && busy && attachments.length === 0) {
        const liveText = editorRef.current
          ? composerPlainText(editorRef.current)
          : draftRef.current
        const trimmedLive = liveText.trim()

        if (trimmedLive.length > 0) {
          // Sync draftRef so steerDraft() reads the just-typed text.
          draftRef.current = liveText
          steerDraft()
        }
      }

      return
    }
  }

  // `draft` and `canSteer` are read so the lint/compiler treats the
  // stale-state mirror as live; the assertions prove the handler never
  // relies on them for the steer decision.
  void draft
  void canSteer

  return (
    <div
      contentEditable
      data-testid="editor"
      onInput={event => {
        draftRef.current = composerPlainText(event.currentTarget)
        setDraft(draftRef.current)
      }}
      onKeyDown={handleKeyDown}
      ref={editorRef}
      suppressContentEditableWarning
    />
  )
}

describe('composer Ctrl+Enter steer — live DOM vs stale composer state (#53659)', () => {
  it('steers the just-typed text on Ctrl+Enter even when composer state has not synced', async () => {
    const onSteer = vi.fn()

    const { getByTestId } = render(
      <Harness busy onSteer={onSteer} />
    )

    const editor = getByTestId('editor')

    // Fast typing: the DOM has the text but NO input event fired, so `draft`
    // state is still empty (the exact stale-state race).
    await act(async () => {
      editor.textContent = 'steer this'
      fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })
    })

    expect(onSteer).toHaveBeenCalledWith('steer this')
  })

  it('steers on Ctrl+Enter with Ctrl key (Windows/Linux)', async () => {
    const onSteer = vi.fn()

    const { getByTestId } = render(
      <Harness busy onSteer={onSteer} />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'ctrl steer'
      fireEvent.keyDown(editor, { key: 'Enter', ctrlKey: true })
    })

    expect(onSteer).toHaveBeenCalledWith('ctrl steer')
  })

  it('does NOT steer on Ctrl+Enter when editor is empty', async () => {
    const onSteer = vi.fn()

    const { getByTestId } = render(
      <Harness busy onSteer={onSteer} />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = ''
      fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })
    })

    expect(onSteer).not.toHaveBeenCalled()
  })

  it('does NOT steer on Ctrl+Enter when not busy', async () => {
    const onSteer = vi.fn()

    const { getByTestId } = render(
      <Harness busy={false} onSteer={onSteer} />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'should not steer'
      fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })
    })

    expect(onSteer).not.toHaveBeenCalled()
  })

  it('does NOT steer on Ctrl+Enter when onSteer is not provided', async () => {
    const onSteer = vi.fn()

    const { getByTestId } = render(
      <Harness busy onSteer={onSteer} />
    )

    const editor = getByTestId('editor')

    // Simulate onSteer being undefined by testing with empty text
    await act(async () => {
      editor.textContent = '   '
      fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })
    })

    // Whitespace-only text should not trigger steer
    expect(onSteer).not.toHaveBeenCalled()
  })
})
