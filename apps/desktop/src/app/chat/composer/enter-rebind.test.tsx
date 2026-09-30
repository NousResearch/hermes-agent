import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { eventMatchesCombos } from '@/lib/keybinds/combo'
import { bindingsFor, resetBinding, setBinding } from '@/store/keybinds'

// No global setupFiles registers auto-cleanup, so unmount between tests —
// otherwise a second render() leaks the first editor and getByTestId('editor')
// matches multiple nodes.
afterEach(cleanup)

// Faithful mirror of index.tsx's Enter wiring (handleEditorKeyDown's Enter
// branches + submitDraft), driven through REAL DOM keydown events on a
// contentEditable — the same harness contract as enter-submit-dom-race.test.tsx.
//
// #46525 / #49422: send and newline are rebindable (`composer.send` /
// `composer.newline`). The editor resolves both chords from the same
// $bindings store the Keyboard Shortcuts panel writes, via comboFromEvent —
// never a hardcoded key check. IME composition still owns Enter at all times.
function Harness({
  busy = false,
  disabled = false,
  queued = [],
  onDrain,
  onQueue,
  onSubmit,
  onSendNow
}: {
  busy?: boolean
  disabled?: boolean
  queued?: readonly string[]
  onDrain: () => void
  onQueue: (text: string) => void
  onSubmit: (text: string) => void
  onSendNow?: (id: string) => void
}) {
  const editorRef = useRef<HTMLDivElement>(null)
  const draftRef = useRef('')

  const [draft, setDraft] = useState('')
  const attachments: unknown[] = []

  const composerPlainText = (el: HTMLElement) => el.textContent ?? ''

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
      }
    } else if (!payloadPresent && queued.length > 0) {
      onDrain()
    } else if (payloadPresent) {
      onSubmit(text)
    }
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    // IME composition owns the keyboard: composingRef || isComposing || the
    // legacy VK_PROCESSKEY (keyCode 229) Enter that follows a late
    // compositionend — none of these may ever send.
    if (event.nativeEvent.isComposing || (event.key === 'Enter' && event.keyCode === 229)) {
      return
    }

    if (event.key === 'PageUp' || event.key === 'PageDown') {
      event.preventDefault()

      return
    }

    // The rebindable pair (#46525, #49422) — an explicit rebind outranks the
    // fixed Cmd/Ctrl+Enter queue chord.
    const sendChord = event.key === 'Enter' && eventMatchesCombos(event.nativeEvent, bindingsFor('composer.send'))

    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !event.shiftKey && !sendChord) {
      return
    }

    if (sendChord) {
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
        const head = queued[0]

        if (head) {
          onSendNow?.(head)
        }

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
      data-testid="editor"
      onInput={event => setText(composerPlainText(event.currentTarget))}
      onKeyDown={handleKeyDown}
      ref={editorRef}
      suppressContentEditableWarning
    />
  )
}

describe('composer Enter wiring follows the rebindable send/newline chords (#46525, #49422)', () => {
  afterEach(() => {
    resetBinding('composer.send')
    resetBinding('composer.newline')
  })

  it('sends on the default Enter and never on the default Shift+Enter (native newline)', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(<Harness onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />)

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'hello world'
      fireEvent.keyDown(editor, { code: 'Enter', key: 'Enter' })
    })

    expect(onSubmit).toHaveBeenCalledWith('hello world')

    await act(async () => {
      fireEvent.keyDown(editor, { code: 'Enter', key: 'Enter', shiftKey: true })
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('sends on the rebound chord and turns plain Enter into a newline', async () => {
    const onSubmit = vi.fn()

    setBinding('composer.send', ['mod+enter'])
    setBinding('composer.newline', ['enter'])

    const { getByTestId } = render(<Harness onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />)

    const editor = getByTestId('editor')

    // The inverse mapping: plain Enter falls through to the browser's own
    // newline insert (asserted by absence of a submit), and the mod chord sends.
    await act(async () => {
      editor.textContent = 'typed line'
      fireEvent.keyDown(editor, { code: 'Enter', key: 'Enter' })
    })

    expect(onSubmit).not.toHaveBeenCalled()

    await act(async () => {
      fireEvent.keyDown(editor, { code: 'Enter', key: 'Enter', metaKey: true })
    })

    expect(onSubmit).toHaveBeenCalledWith('typed line')
  })

  it('sends on a Ctrl+Enter rebind on non-mac modifiers (canonical ctrl folding)', async () => {
    const onSubmit = vi.fn()

    // What a Win/Linux user picks in the panel: `ctrl+enter`.
    setBinding('composer.send', ['ctrl+enter'])
    setBinding('composer.newline', ['enter'])

    const { getByTestId } = render(<Harness onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />)

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'win path'
      fireEvent.keyDown(editor, { code: 'Enter', key: 'Enter', ctrlKey: true })
    })

    expect(onSubmit).toHaveBeenCalledWith('win path')
  })

  it('never sends during IME composition, on any binding', async () => {
    const onSubmit = vi.fn()

    setBinding('composer.send', ['mod+enter'])

    const { getByTestId } = render(<Harness onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />)

    const editor = getByTestId('editor')

    // isComposing keydown — the IME's own commit key must not reach send.
    await act(async () => {
      editor.textContent = '打字中'
      fireEvent.keyDown(editor, { code: 'Enter', isComposing: true, key: 'Enter', metaKey: true })
    })

    expect(onSubmit).not.toHaveBeenCalled()

    // The late-commit Enter some macOS Chinese IMEs emit: isComposing already
    // false, keyCode 229 (VK_PROCESSKEY) — still not a send.
    await act(async () => {
      fireEvent.keyDown(editor, { code: 'Enter', keyCode: 229, key: 'Enter' })
    })

    expect(onSubmit).not.toHaveBeenCalled()

    // Once composition is truly over, the chord sends the committed text.
    await act(async () => {
      fireEvent.keyDown(editor, { code: 'Enter', key: 'Enter', metaKey: true })
    })

    expect(onSubmit).toHaveBeenCalledWith('打字中')
  })

  it('keeps the queue behaviors on the send chord: drain when idle, send-now when busy', async () => {
    const onDrain = vi.fn()
    const onQueue = vi.fn()
    const onSendNow = vi.fn()
    const onSubmit = vi.fn()

    setBinding('composer.send', ['mod+enter'])

    const { getByTestId } = render(
      <Harness busy onDrain={onDrain} onQueue={onQueue} onSendNow={onSendNow} onSubmit={onSubmit} queued={['queued-1']} />
    )

    const editor = getByTestId('editor')

    // Busy + empty: the head of the queue goes now, never an accidental Stop.
    await act(async () => {
      editor.textContent = ''
      fireEvent.keyDown(editor, { code: 'Enter', key: 'Enter', metaKey: true })
    })

    expect(onSendNow).toHaveBeenCalledWith('queued-1')

    // Busy + typed: the words queue as the next turn.
    await act(async () => {
      editor.textContent = 'follow-up'
      fireEvent.keyDown(editor, { code: 'Enter', key: 'Enter', metaKey: true })
    })

    expect(onQueue).toHaveBeenCalledWith('follow-up')
  })
})
