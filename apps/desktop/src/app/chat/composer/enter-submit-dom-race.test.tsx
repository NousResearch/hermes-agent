import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { isPostCompositionCommitEnter } from '@/lib/ime'

// No global setupFiles registers auto-cleanup, so unmount between tests —
// otherwise a second render() leaks the first editor and getByTestId('editor')
// matches multiple nodes.
afterEach(cleanup)

// Faithful mirror of index.tsx's Enter wiring (handleEditorKeyDown's Enter
// branches + submitDraft), driven through REAL DOM keydown events on a
// contentEditable, across all three send modes.
//
// Contract under test (Settings → Keyboards → Send with):
//   · `enter` (default) — Enter commits the draft; Shift+Enter breaks the line;
//   · `double-enter` — Enter breaks the line, a second Enter inside the window
//     commits (and the trailing break the first press added is stripped);
//   · `mod-enter` — a bare Enter only breaks the line; ⌘/Ctrl+Enter commits;
//   · every mode — empty Enter keeps its single-press queue gestures (drain
//     when idle, promote the queue head while busy), and Shift+Enter never
//     sends.
//
// The stale-composer-state race from #39630 is covered here too: pressing Enter
// right after typing (fast typing / IME) must not read empty React state and
// drop the message. We model the race deterministically the way the IME repro
// does: mutate the editor's textContent WITHOUT firing an input event, so the
// React `draft` state stays stale while the DOM already holds the text.
const DOUBLE_ENTER_MS = 400

type SendMode = 'double-enter' | 'enter' | 'mod-enter'

function Harness({
  busy = false,
  compositionEndedMsAgo,
  disabled = false,
  doubleEnterMs = DOUBLE_ENTER_MS,
  mode = 'enter',
  queued = [],
  onSubmit,
  onQueue,
  onCancel,
  onDrain,
  onSendNow
}: {
  busy?: boolean
  /** Mirrors the composer's `compositionEndedAtRef`: an IME composition ended
   *  this many ms ago (undefined = never). */
  compositionEndedMsAgo?: number
  disabled?: boolean
  doubleEnterMs?: number
  mode?: SendMode
  queued?: readonly string[]
  onSubmit: (text: string) => void
  onQueue: (text: string) => void
  onCancel: () => void
  onDrain: () => void
  onSendNow?: (id: string) => void
}) {
  const editorRef = useRef<HTMLDivElement>(null)
  const draftRef = useRef('')
  // Mirrors `useAuiState(s => s.composer.text)` — updated only via setText, so
  // it lags the DOM until React re-renders (the source of the bug).
  const [draft, setDraft] = useState('')
  const lastEnterAtRef = useRef(0)
  const compositionEndedAtRef = useRef(compositionEndedMsAgo === undefined ? 0 : Date.now() - compositionEndedMsAgo)
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
    // PageUp/PageDown: no text-editing purpose in the single-line editor —
    // swallow the default so the browser cannot scroll the nearest ancestor
    // (which breaks the desktop pane layout, #49978). The routed transcript
    // page-scroll lives in the global keybind, not here.
    if (event.key === 'PageUp' || event.key === 'PageDown') {
      event.preventDefault()

      return
    }

    // IME commit guard (lib/ime.ts): a bare Enter on the back of a composition
    // end is that composition's commit, not a send.
    if (
      event.key === 'Enter' &&
      !event.shiftKey &&
      !event.metaKey &&
      !event.ctrlKey &&
      isPostCompositionCommitEnter(compositionEndedAtRef.current, Date.now())
    ) {
      return
    }

    // ⌘/Ctrl+Enter commits in every mode (queues while busy).
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !event.shiftKey) {
      event.preventDefault()
      lastEnterAtRef.current = 0

      if (disabled) {
        return
      }

      if (busy) {
        const editorText = editorRef.current ? composerPlainText(editorRef.current) : draftRef.current

        if (editorText !== draftRef.current) {
          draftRef.current = editorText
          setDraft(editorText)
        }

        onQueue(editorText)

        return
      }

      submitDraft()

      return
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      const editorText = editorRef.current ? composerPlainText(editorRef.current) : draftRef.current
      const hasLivePayload = editorText.trim().length > 0 || attachments.length > 0

      if (disabled) {
        return
      }

      if (!hasLivePayload) {
        event.preventDefault()
        lastEnterAtRef.current = 0

        if (!busy && queued.length > 0) {
          onDrain()

          return
        }

        if (busy) {
          const head = queued[0]

          if (head) {
            onSendNow?.(head)
          }
        }

        return
      }

      if (mode === 'enter') {
        event.preventDefault()
        submitDraft()

        return
      }

      if (mode === 'mod-enter') {
        return
      }

      const now = Date.now()
      const doubleTap = now - lastEnterAtRef.current <= doubleEnterMs

      lastEnterAtRef.current = now

      if (!doubleTap) {
        // Production falls through UNPREVENTED here so the editor inserts the
        // break; jsdom does not, so the tests append it themselves.
        return
      }

      event.preventDefault()

      const editor = editorRef.current
      const live = editor ? composerPlainText(editor) : ''

      if (editor && live.endsWith('\n')) {
        editor.textContent = live.replace(/\n+$/, '')
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

/** Two presses of Enter, `gapMs` apart, with the editor's line break inserted
 *  in between the way a real contenteditable does it. */
function typeThenTap(editor: HTMLElement, text: string, gapMs = 0) {
  editor.textContent = text
  fireEvent.keyDown(editor, { key: 'Enter' })

  if (gapMs === 0) {
    fireEvent.keyDown(editor, { key: 'Enter' })

    return
  }

  // Slow second press: a real editor would have broken the line by now.
  editor.textContent = `${text}\n`
  vi.setSystemTime(new Date(Date.now() + gapMs))
  fireEvent.keyDown(editor, { key: 'Enter' })
}

describe('composer Enter — send modes', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-14T12:00:00Z'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('mode: enter (default)', () => {
    it('sends the just-typed text on a single Enter even when composer state has not synced', async () => {
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
  })

  describe('mode: double-enter', () => {
    it('does not send on a single Enter — the draft survives as a line break', async () => {
      const onSubmit = vi.fn()

      const { getByTestId } = render(
        <Harness mode="double-enter" onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        editor.textContent = 'first line'
        fireEvent.keyDown(editor, { key: 'Enter' })
      })

      expect(onSubmit).not.toHaveBeenCalled()
      expect(editor.textContent).toBe('first line')

      // And the SECOND press only sends once it lands inside the window.
      await act(async () => {
        fireEvent.keyDown(editor, { key: 'Enter' })
      })

      expect(onSubmit).toHaveBeenCalledTimes(1)
    })

    it('strips the trailing break the first tap inserted before sending', async () => {
      const onSubmit = vi.fn()

      const { getByTestId } = render(
        <Harness mode="double-enter" onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        editor.textContent = 'no trailing newline'
        fireEvent.keyDown(editor, { key: 'Enter' })
        editor.textContent = 'no trailing newline\n'
        fireEvent.keyDown(editor, { key: 'Enter' })
      })

      expect(onSubmit).toHaveBeenCalledWith('no trailing newline')
    })

    it('treats two slow presses as two newlines, not a send', async () => {
      const onSubmit = vi.fn()

      const { getByTestId } = render(
        <Harness mode="double-enter" onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        typeThenTap(editor, 'line one', DOUBLE_ENTER_MS + 200)
      })

      expect(onSubmit).not.toHaveBeenCalled()
      expect(editor.textContent).toBe('line one\n')
    })

    it('honours a custom window', async () => {
      const onSubmit = vi.fn()

      const { getByTestId } = render(
        <Harness
          doubleEnterMs={800}
          mode="double-enter"
          onCancel={vi.fn()}
          onDrain={vi.fn()}
          onQueue={vi.fn()}
          onSubmit={onSubmit}
        />
      )

      const editor = getByTestId('editor')

      // 600ms apart: a send on the shipped 400ms window, two newlines on 800ms.
      await act(async () => {
        typeThenTap(editor, 'slow double tap', 600)
      })

      expect(onSubmit).toHaveBeenCalledWith('slow double tap')
      expect(editor.textContent).toBe('slow double tap')
    })

    it('keeps a multi-line message intact across the double-tap', async () => {
      const onSubmit = vi.fn()

      const { getByTestId } = render(
        <Harness mode="double-enter" onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        // Break a line the slow way (two presses > window apart), type on, then
        // commit with a fast double-tap.
        typeThenTap(editor, 'paragraph one', DOUBLE_ENTER_MS + 200)
        editor.textContent = 'paragraph one\nparagraph two'
        fireEvent.keyDown(editor, { key: 'Enter' })
        editor.textContent = 'paragraph one\nparagraph two\n'
        fireEvent.keyDown(editor, { key: 'Enter' })
      })

      expect(onSubmit).toHaveBeenCalledWith('paragraph one\nparagraph two')
    })

    it('queues a fast-typed message on a double-tap while busy instead of draining the queue', async () => {
      const onQueue = vi.fn()
      const onDrain = vi.fn()
      const onCancel = vi.fn()

      const { getByTestId } = render(
        <Harness
          busy
          mode="double-enter"
          onCancel={onCancel}
          onDrain={onDrain}
          onQueue={onQueue}
          onSubmit={vi.fn()}
          queued={['queued-1']}
        />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        typeThenTap(editor, 'urgent follow-up')
      })

      expect(onQueue).toHaveBeenCalledWith('urgent follow-up')
      expect(onDrain).not.toHaveBeenCalled()
      expect(onCancel).not.toHaveBeenCalled()
    })
  })

  describe('mode: mod-enter', () => {
    it('leaves a bare Enter to the editor and sends on the chord', async () => {
      const onSubmit = vi.fn()

      const { getByTestId } = render(
        <Harness mode="mod-enter" onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        editor.textContent = 'chord only'
        fireEvent.keyDown(editor, { key: 'Enter' })
        fireEvent.keyDown(editor, { key: 'Enter' })
      })

      expect(onSubmit).not.toHaveBeenCalled()

      await act(async () => {
        fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })
      })

      expect(onSubmit).toHaveBeenCalledWith('chord only')
    })

    it('queues on the chord while busy', async () => {
      const onQueue = vi.fn()

      const { getByTestId } = render(
        <Harness busy mode="mod-enter" onCancel={vi.fn()} onDrain={vi.fn()} onQueue={onQueue} onSubmit={vi.fn()} />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        editor.textContent = 'queued by chord'
        fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })
      })

      expect(onQueue).toHaveBeenCalledWith('queued by chord')
    })
  })

  describe('every mode', () => {
    const modes: SendMode[] = ['enter', 'double-enter', 'mod-enter']

    it('swallows the IME commit Enter that follows compositionend (#49422)', async () => {
      const onSubmit = vi.fn()

      // Default mode: a bare Enter sends. This one is the pinyin candidate
      // confirmation, arriving 40ms after the composition ended with no
      // isComposing flag and no keyCode 229 left on it — the case the flag
      // guards cannot see.
      const { getByTestId } = render(
        <Harness
          compositionEndedMsAgo={40}
          onCancel={vi.fn()}
          onDrain={vi.fn()}
          onQueue={vi.fn()}
          onSubmit={onSubmit}
        />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        editor.textContent = '你好'
        fireEvent.keyDown(editor, { key: 'Enter' })
      })

      expect(onSubmit).not.toHaveBeenCalled()
    })

    it('lets a send through once the composition is in the past', async () => {
      const onSubmit = vi.fn()

      const { getByTestId } = render(
        <Harness
          compositionEndedMsAgo={5_000}
          onCancel={vi.fn()}
          onDrain={vi.fn()}
          onQueue={vi.fn()}
          onSubmit={onSubmit}
        />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        editor.textContent = '你好，这是一条消息'
        fireEvent.keyDown(editor, { key: 'Enter' })
      })

      expect(onSubmit).toHaveBeenCalledWith('你好，这是一条消息')
    })

    it('does not swallow the explicit chords behind the guard', async () => {
      const onSubmit = vi.fn()
      const onQueue = vi.fn()

      const { getByTestId } = render(
        <Harness
          busy
          compositionEndedMsAgo={40}
          onCancel={vi.fn()}
          onDrain={vi.fn()}
          onQueue={onQueue}
          onSubmit={onSubmit}
        />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        editor.textContent = 'committed then queued'
        fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })
      })

      // ⌘Enter is a deliberate act even mid-composition-commit; it must still
      // reach the queue rather than being eaten as an IME commit.
      expect(onQueue).toHaveBeenCalledWith('committed then queued')
      expect(onSubmit).not.toHaveBeenCalled()
    })

    it('never sends on Shift+Enter', async () => {
      const onSubmit = vi.fn()

      const { getByTestId } = render(
        <Harness mode="double-enter" onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        editor.textContent = 'held shift'
        fireEvent.keyDown(editor, { key: 'Enter', shiftKey: true })
        fireEvent.keyDown(editor, { key: 'Enter', shiftKey: true })
      })

      expect(onSubmit).not.toHaveBeenCalled()
    })

    it('treats an empty Enter while busy with nothing queued as a no-op (never an accidental Stop)', async () => {
      for (const mode of modes) {
        cleanup()

        const onCancel = vi.fn()
        const onSubmit = vi.fn()
        const onQueue = vi.fn()
        const onSendNow = vi.fn()

        const { getByTestId } = render(
          <Harness
            busy
            mode={mode}
            onCancel={onCancel}
            onDrain={vi.fn()}
            onQueue={onQueue}
            onSendNow={onSendNow}
            onSubmit={onSubmit}
          />
        )

        const editor = getByTestId('editor')

        await act(async () => {
          editor.textContent = ''
          fireEvent.keyDown(editor, { key: 'Enter' })
        })

        expect(onCancel).not.toHaveBeenCalled()
        expect(onSubmit).not.toHaveBeenCalled()
        expect(onQueue).not.toHaveBeenCalled()
        expect(onSendNow).not.toHaveBeenCalled()
      }
    })

    it('double-send: an empty Enter while busy with a queued turn sends that turn now', async () => {
      const onCancel = vi.fn()
      const onSendNow = vi.fn()

      const { getByTestId } = render(
        <Harness
          busy
          mode="double-enter"
          onCancel={onCancel}
          onDrain={vi.fn()}
          onQueue={vi.fn()}
          onSendNow={onSendNow}
          onSubmit={vi.fn()}
          queued={['queued-1', 'queued-2']}
        />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        editor.textContent = ''
        fireEvent.keyDown(editor, { key: 'Enter' })
      })

      // Head of the queue, and NOT a bare cancel — send-now promotes + interrupts.
      expect(onSendNow).toHaveBeenCalledWith('queued-1')
      expect(onCancel).not.toHaveBeenCalled()
    })

    it('drains the next queued prompt on Enter when idle with a truly empty editor', async () => {
      const onDrain = vi.fn()
      const onSubmit = vi.fn()

      const { getByTestId } = render(
        <Harness
          mode="double-enter"
          onCancel={vi.fn()}
          onDrain={onDrain}
          onQueue={vi.fn()}
          onSubmit={onSubmit}
          queued={['queued-1']}
        />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        editor.textContent = ''
        fireEvent.keyDown(editor, { key: 'Enter' })
      })

      expect(onDrain).toHaveBeenCalledTimes(1)
      expect(onSubmit).not.toHaveBeenCalled()
    })

    it('keeps reconnect drafts editable but blocks send until the gateway returns', async () => {
      const onSubmit = vi.fn()
      const onDrain = vi.fn()

      const { getByTestId } = render(
        <Harness
          disabled
          mode="double-enter"
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
        fireEvent.keyDown(editor, { key: 'Enter' })
      })

      expect(editor.textContent).toBe('draft while reconnecting')
      expect(onDrain).not.toHaveBeenCalled()
      expect(onSubmit).not.toHaveBeenCalled()
    })
  })

  // #49978 — the browser's default for PageUp/PageDown in a focused
  // contentEditable scrolls the nearest scrollable ancestor, which breaks the
  // desktop pane layout (sidebar squeezed out, content shifted left). The
  // editor must swallow the default for both keys.
  it('prevents the browser default for PageUp and PageDown keydowns in the editor', async () => {
    const { getByTestId } = render(
      <Harness onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={vi.fn()} />
    )

    const editor = getByTestId('editor')

    for (const key of ['PageUp', 'PageDown']) {
      // React's synthetic handlers run on the root, so a native listener on
      // the editor sees the event BEFORE React's — capture the native event
      // and read its defaultPrevented flag after the dispatch (the flag is
      // mutable on the same native event React handled).
      let event: KeyboardEvent | undefined

      await act(async () => {
        editor.addEventListener(
          'keydown',
          e => {
            event = e
          },
          { once: true, capture: true }
        )
        fireEvent.keyDown(editor, { key })
      })

      expect(event?.defaultPrevented, `keydown ${key} must be default-prevented`).toBe(true)
    }
  })
})
