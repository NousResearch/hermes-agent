import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { isPostCompositionCommitEnter } from '@/lib/ime'
import type { SendGraceReason } from '@/store/composer-prefs'

import { composerEnterPressOwner, isComposerDoubleTap } from './enter-gesture'

// No global setupFiles registers auto-cleanup, so unmount between tests —
// otherwise a second render() leaks the first editor and getByTestId('editor')
// matches multiple nodes.
afterEach(cleanup)

// Faithful mirror of index.tsx's Enter wiring (handleEditorKeyDown's Enter
// branches + submitDraft), driven through REAL DOM keydown events on a
// contentEditable. The branch ORDER is deliberately NOT mirrored: it comes from
// ./enter-gesture, the same module the composer calls, so a precedence change
// cannot pass here while failing in the app.
//
// Contract under test (Settings → Keyboards → Enter and sending):
//   · the gate — `enterSends: true` (the default) commits the draft on a bare
//     Enter; false means a lone press never sends, whatever else is armed;
//   · `enterNewline` — what that press does instead: break the line (the
//     default), or nothing at all;
//   · the gestures — double tap, the pause rule and the hold, each independent
//     and each with its own window. The double-tap strips the break the first
//     press inserted, and has nothing to strip when the break is off;
//   · every configuration — empty Enter keeps its single-press queue gestures
//     (drain when idle, promote the queue head while busy), and Shift+Enter
//     never sends.
//
// The stale-composer-state race from #39630 is covered here too: pressing Enter
// right after typing (fast typing / IME) must not read empty React state and
// drop the message. We model the race deterministically the way the IME repro
// does: mutate the editor's textContent WITHOUT firing an input event, so the
// React `draft` state stays stale while the DOM already holds the text.
const DOUBLE_ENTER_MS = 400
const HOLD_MS = 350
const GRACE_MS = 900

/** Every prop the harness takes. The send settings arrive as a CONFIGURATION
 *  rather than a mode, because the gestures are independent: a test can arm one,
 *  or all of them, which is the point of the model. */
interface HarnessProps {
  busy?: boolean
  /** Mirrors the composer's `compositionEndedAtRef`: an IME composition ended
   *  this many ms ago (undefined = never). */
  compositionEndedMsAgo?: number
  disabled?: boolean
  doubleEnterMs?: number
  /** Mirrors `enterNewline`: what the press does when it is not sending. */
  enterNewline?: boolean
  enterSends?: boolean
  holdMs?: number
  queued?: readonly string[]
  /** Mirrors `sendGraceFor`: the situations whose sends wait out a grace
   *  window. Empty means every send commits on the keystroke, which is what the
   *  other tests in this file assume. */
  sendGraceFor?: readonly SendGraceReason[]
  /** Mirrors `commitOnPress`: a press during a wait commits it on the spot. */
  commitOnPress?: boolean
  /** Mirrors `canSteer`: a turn is running and the draft can redirect it. */
  canSteer?: boolean
  sendOnDoubleTap?: boolean
  sendOnHold?: boolean
  sendOnIdle?: boolean
  sendOnPause?: boolean
  /** Mirrors the composer's typing-recency: ms since the last keystroke. */
  typedIdleMsAgo?: number
  typingIdleMs?: number
  onSubmit: (text: string) => void
  onQueue: (text: string) => void
  onCancel: () => void
  onDrain: () => void
  onSendNow?: (id: string) => void
  onSteer?: () => void
}

const DEFAULT_TYPING_IDLE_MS = 1000

function Harness({
  busy = false,
  compositionEndedMsAgo,
  disabled = false,
  doubleEnterMs = DOUBLE_ENTER_MS,
  enterNewline = true,
  enterSends = true,
  holdMs = HOLD_MS,
  queued = [],
  sendGraceFor = [],
  commitOnPress = true,
  canSteer = false,
  sendOnDoubleTap = false,
  sendOnHold = false,
  sendOnPause = false,
  typedIdleMsAgo = 0,
  typingIdleMs = DEFAULT_TYPING_IDLE_MS,
  onSubmit,
  onQueue,
  onCancel,
  onDrain,
  onSendNow,
  onSteer
}: HarnessProps) {
  const editorRef = useRef<HTMLDivElement>(null)
  const draftRef = useRef('')
  // Mirrors `useAuiState(s => s.composer.text)` — updated only via setText, so
  // it lags the DOM until React re-renders (the source of the bug).
  const [draft, setDraft] = useState('')
  const lastEnterAtRef = useRef(0)
  // The steer chord's own double-press stamp (the composer's `lastSteerAtRef`).
  const lastSteerAtRef = useRef(0)
  const enterHoldTimerRef = useRef<number | undefined>(undefined)
  // What the release should do when the press did not become a gesture (the
  // composer's `pendingPressRef`).
  const pendingPressRef = useRef<'pause' | 'commit' | null>(null)
  // The grace window's timer (the composer's `useComposerSendGrace`).
  const graceTimerRef = useRef<number | undefined>(undefined)
  const typedAtRef = useRef(Date.now() - typedIdleMsAgo)
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

  const cancelHold = () => {
    window.clearTimeout(enterHoldTimerRef.current)
    enterHoldTimerRef.current = undefined
  }

  /** Mirrors `commitHeldEnter`: drop the break the press inserted, then send. */
  const commitHeld = () => {
    const editor = editorRef.current
    const live = editor ? composerPlainText(editor) : ''

    if (editor && live.endsWith('\n')) {
      editor.textContent = live.replace(/\n+$/, '')
    }

    holdOrSubmit('hold', submitDraft)
  }

  /** The composer's grace window. A send configured to wait is held here; the
   *  draft deliberately stays in the editor for the whole hold, so a later
   *  press can still commit it. */
  const cancelGrace = () => {
    window.clearTimeout(graceTimerRef.current)
    graceTimerRef.current = undefined
  }

  const holdOrSubmit = (reason: SendGraceReason, commit: () => void) => {
    cancelGrace()

    if (!sendGraceFor.includes(reason)) {
      commit()

      return
    }

    graceTimerRef.current = window.setTimeout(() => {
      graceTimerRef.current = undefined
      commit()
    }, GRACE_MS)
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

    // A held key repeats, and a repeat is never a press: it must not complete a
    // double tap, drain a queue, or start a hold.
    if (event.key === 'Enter' && event.repeat) {
      if (enterSends || sendOnDoubleTap || sendOnHold || sendOnPause) {
        event.preventDefault()
      }

      return
    }

    // ⌘/Ctrl+Enter commits in every configuration (queues while busy).
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

    // Shift+Enter steers a running turn: the multiline-first binding the
    // shortcuts panel, the settings description and the docs all promise. With
    // the double tap armed it takes the same guard as every other send.
    if (!enterSends && event.key === 'Enter' && event.shiftKey && canSteer) {
      if (sendOnDoubleTap) {
        const steeredAt = Date.now()

        if (steeredAt - lastSteerAtRef.current > doubleEnterMs) {
          lastSteerAtRef.current = steeredAt

          if (!enterNewline) {
            event.preventDefault()
          }

          return
        }

        lastSteerAtRef.current = 0
      }

      event.preventDefault()
      onSteer?.()

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

      if (enterSends) {
        event.preventDefault()
        submitDraft()

        return
      }

      const now = Date.now()
      const doubleTap = isComposerDoubleTap(now, lastEnterAtRef.current, doubleEnterMs)

      lastEnterAtRef.current = now

      const pausedEnough = sendOnPause && now - typedAtRef.current > typingIdleMs

      const owner = composerEnterPressOwner({
        doubleTap,
        pausedEnough,
        sendOnDoubleTap,
        sendOnHold,
        sendOnPause
      })

      cancelHold()

      if (sendOnHold) {
        enterHoldTimerRef.current = window.setTimeout(() => {
          enterHoldTimerRef.current = undefined
          commitHeld()
        }, holdMs)
      }

      if (owner === 'doubleTap') {
        event.preventDefault()

        pendingPressRef.current = null

        const editor = editorRef.current
        const live = editor ? composerPlainText(editor) : ''

        if (enterNewline && editor && live.endsWith('\n')) {
          editor.textContent = live.replace(/\n+$/, '')
        }

        cancelHold()

        // The previous press may already have sent the draft, and submitting an
        // empty composer drains or steers.
        if (live.trim().length === 0 && attachments.length === 0) {
          return
        }

        if (sendGraceFor.includes('doubleTap')) {
          holdOrSubmit('doubleTap', submitDraft)

          return
        }

        // Not configured to wait, so this press is the user saying "now": commit
        // on the spot and take back any window the previous press opened, or it
        // would fire a second send into an empty composer later.
        cancelGrace()
        submitDraft()

        return
      }

      // A press during a wait commits it, unless this press can still become a
      // hold (the composer's `commitOnPress`).
      if (commitOnPress && graceTimerRef.current !== undefined) {
        event.preventDefault()

        if (sendOnHold) {
          pendingPressRef.current = 'commit'

          return
        }

        cancelGrace()

        const editor = editorRef.current
        const live = editor ? composerPlainText(editor) : ''

        if (live.trim().length > 0 || attachments.length > 0) {
          submitDraft()
        }

        return
      }

      if (owner === 'pauseOnRelease') {
        event.preventDefault()
        pendingPressRef.current = 'pause'

        return
      }

      if (owner === 'pause') {
        event.preventDefault()
        holdOrSubmit('pause', submitDraft)

        return
      }

      if (!sendOnDoubleTap) {
        // Falls through UNPREVENTED when the press may break the line: the
        // editor inserts the break, which jsdom does not do, so the tests append
        // it themselves.
        if (!enterNewline) {
          event.preventDefault()
        }

        return
      }

      // The first press of a possible pair: jsdom does not insert the break, so
      // the tests append it themselves.
      if (!enterNewline) {
        event.preventDefault()
      }
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
      onKeyUp={event => {
        if (event.key === 'Enter') {
          cancelHold()

          // The release of a press that was deferred because the hold was armed
          // and it could still have become one: the press was a tap after all,
          // so what it deferred runs now.
          const deferred = pendingPressRef.current

          pendingPressRef.current = null

          if (deferred === 'pause') {
            holdOrSubmit('pause', submitDraft)
          } else if (deferred === 'commit' && graceTimerRef.current !== undefined) {
            cancelGrace()

            const editor = editorRef.current
            const live = editor ? composerPlainText(editor) : ''

            if (live.trim().length > 0 || attachments.length > 0) {
              submitDraft()
            }
          }
        }
      }}
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
        <Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} sendOnDoubleTap />
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
        <Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} sendOnDoubleTap />
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
        <Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} sendOnDoubleTap />
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
          enterSends={false}
          onCancel={vi.fn()}
          onDrain={vi.fn()}
          onQueue={vi.fn()}
          onSubmit={onSubmit}
          sendOnDoubleTap
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
        <Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} sendOnDoubleTap />
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
          enterSends={false}
          onCancel={onCancel}
          onDrain={onDrain}
          onQueue={onQueue}
          onSubmit={vi.fn()}
          queued={['queued-1']}
          sendOnDoubleTap
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
        <Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />
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
        <Harness busy enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={onQueue} onSubmit={vi.fn()} />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        editor.textContent = 'queued by chord'
        fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })
      })

      expect(onQueue).toHaveBeenCalledWith('queued by chord')
    })
  })

  describe('every non-sending configuration', () => {
    const configurations: Partial<HarnessProps>[] = [
      { enterSends: false },
      { enterSends: false, sendOnDoubleTap: true },
      { enterSends: false, sendOnHold: true, sendOnPause: true }
    ]

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
        <Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} sendOnDoubleTap />
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
      for (const configuration of configurations) {
        cleanup()

        const onCancel = vi.fn()
        const onSubmit = vi.fn()
        const onQueue = vi.fn()
        const onSendNow = vi.fn()

        const { getByTestId } = render(
          <Harness
            busy
            {...configuration}
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
          enterSends={false}
          onCancel={onCancel}
          onDrain={vi.fn()}
          onQueue={vi.fn()}
          onSendNow={onSendNow}
          onSubmit={vi.fn()}
          queued={['queued-1', 'queued-2']}
          sendOnDoubleTap
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
          enterSends={false}
          onCancel={vi.fn()}
          onDrain={onDrain}
          onQueue={vi.fn()}
          onSubmit={onSubmit}
          queued={['queued-1']}
          sendOnDoubleTap
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
          enterSends={false}
          onCancel={vi.fn()}
          onDrain={onDrain}
          onQueue={vi.fn()}
          onSubmit={onSubmit}
          queued={['queued-1']}
          sendOnDoubleTap
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

describe('composer Enter — key repeat is not a press', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-14T12:00:00Z'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  /** A held key: the OS delivers one real press, then repeats. */
  function holdThenRepeat(editor: HTMLElement, text: string, repeats = 3) {
    editor.textContent = text
    fireEvent.keyDown(editor, { key: 'Enter' })

    for (let i = 0; i < repeats; i += 1) {
      editor.textContent = `${editor.textContent}\n`
      fireEvent.keyDown(editor, { key: 'Enter', repeat: true })
    }
  }

  it('does not let a held Enter fake a double tap', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} sendOnDoubleTap />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      holdThenRepeat(editor, 'holding the key')
    })

    // Before this guard, repeat #2 landed inside the double-tap window and sent.
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('swallows the repeat in `enter` mode instead of leaving a stray break', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(<Harness  onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />)

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'sent on the press'
      fireEvent.keyDown(editor, { key: 'Enter' })
      fireEvent.keyDown(editor, { key: 'Enter', repeat: true })
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    // The one press sends; the repeats must not append a break to the emptied box.
    expect(editor.textContent).toBe('sent on the press')
  })

  it('re-sends nothing on repeated ⌘Enter', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(<Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />)

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'once'
      fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })
      fireEvent.keyDown(editor, { key: 'Enter', metaKey: true, repeat: true })
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
  })
})

describe('composer Enter — the press-and-hold flag', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-14T12:00:00Z'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('sends once the key has been down for holdMs, without the break the tap inserted', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendOnHold
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'held message'
    act(() => {
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onSubmit).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(HOLD_MS)
    })

    expect(onSubmit).toHaveBeenCalledWith('held message')
  })

  it('never sends on a tap, however long the pause between taps', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} sendOnHold />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'tap tap'
    act(() => {
      fireEvent.keyDown(editor, { key: 'Enter' })
      fireEvent.keyUp(editor, { key: 'Enter' })
      vi.advanceTimersByTime(200)
      fireEvent.keyDown(editor, { key: 'Enter' })
      fireEvent.keyUp(editor, { key: 'Enter' })
      vi.advanceTimersByTime(HOLD_MS)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('cancels the hold when the key comes up early, even at the last moment', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} sendOnHold />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'nearly held'
    act(() => {
      fireEvent.keyDown(editor, { key: 'Enter' })
      vi.advanceTimersByTime(HOLD_MS - 1)
      fireEvent.keyUp(editor, { key: 'Enter' })
      vi.advanceTimersByTime(HOLD_MS * 4)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('holds off while the OS is repeating, because the timer decides', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} sendOnHold />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'held'
    act(() => {
      fireEvent.keyDown(editor, { key: 'Enter' })
      // A repeat arriving BEFORE holdMs must not send early — a user with a fast
      // repeat rate would otherwise get a shorter gesture than everyone else.
      fireEvent.keyDown(editor, { key: 'Enter', repeat: true })
    })

    expect(onSubmit).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(HOLD_MS)
    })

    expect(onSubmit).toHaveBeenCalledWith('held')
  })

  it('composes with the mode rather than replacing it: double tap AND hold both send', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} sendOnDoubleTap sendOnHold />
    )

    const editor = getByTestId('editor')

    // The mode's own gesture, with no hold in play.
    act(() => {
      editor.textContent = 'by double tap'
      fireEvent.keyDown(editor, { key: 'Enter' })
      fireEvent.keyUp(editor, { key: 'Enter' })
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onSubmit).toHaveBeenCalledWith('by double tap')

    // ...and the flag's gesture, from the same composer.
    act(() => {
      editor.textContent = 'by hold'
      fireEvent.keyDown(editor, { key: 'Enter' })
      vi.advanceTimersByTime(HOLD_MS)
    })

    expect(onSubmit).toHaveBeenLastCalledWith('by hold')
    expect(onSubmit).toHaveBeenCalledTimes(2)
  })

  it('still lets the ⌘Enter chord send while a bare press only breaks the line', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} sendOnHold />
    )

    const editor = getByTestId('editor')

    await act(async () => {
      editor.textContent = 'chord send'
      fireEvent.keyDown(editor, { key: 'Enter', metaKey: true })
    })

    expect(onSubmit).toHaveBeenCalledWith('chord send')
  })
})

describe('composer Enter — gestures composing', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-14T12:00:00Z'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('sends on a press once typing has stopped, and not before', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendOnPause
        typedIdleMsAgo={50}
        typingIdleMs={1000}
      />
    )

    const editor = getByTestId('editor')

    // Mid-flow: a press only breaks the line.
    await act(async () => {
      editor.textContent = 'still typing'
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('lets the pause gesture send the press it was waiting for', () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendOnPause
        typedIdleMsAgo={5000}
        typingIdleMs={1000}
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'thought about it'
    fireEvent.keyDown(editor, { key: 'Enter' })

    expect(onSubmit).toHaveBeenCalledWith('thought about it')
  })

  it('carries two gestures at once, which is the whole point of the model', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendOnDoubleTap
        sendOnHold
      />
    )

    const editor = getByTestId('editor')

    // The double tap.
    act(() => {
      editor.textContent = 'by double tap'
      fireEvent.keyDown(editor, { key: 'Enter' })
      fireEvent.keyUp(editor, { key: 'Enter' })
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onSubmit).toHaveBeenCalledWith('by double tap')

    // And the long press, from the same composer, without a settings change.
    act(() => {
      editor.textContent = 'by holding'
      fireEvent.keyDown(editor, { key: 'Enter' })
      vi.advanceTimersByTime(HOLD_MS)
    })

    expect(onSubmit).toHaveBeenLastCalledWith('by holding')
  })

  it('never sends on a stray tap, which is why the settings exist', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendOnHold
      />
    )

    const editor = getByTestId('editor')

    // Someone reaching for the apostrophe: a tap, released immediately.
    await act(async () => {
      editor.textContent = "don't"
      fireEvent.keyDown(editor, { key: 'Enter' })
      fireEvent.keyUp(editor, { key: 'Enter' })
      vi.advanceTimersByTime(HOLD_MS * 4)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  describe('a bare Enter that does nothing at all', () => {
    it('swallows the press when the line break is switched off', async () => {
      const onSubmit = vi.fn()

      const { getByTestId } = render(
        <Harness enterNewline={false} enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} />
      )

      const editor = getByTestId('editor')

      let reachedTheEditor = true

      await act(async () => {
        editor.textContent = 'stays put'
        // Cancelled means the editor never sees it: no break, no send.
        reachedTheEditor = fireEvent.keyDown(editor, { key: 'Enter' }) !== false
      })

      expect(reachedTheEditor).toBe(false)
      expect(onSubmit).not.toHaveBeenCalled()
      expect(editor.textContent).toBe('stays put')
    })

    it('breaks the line instead when the break is switched on, which is the default', async () => {
      const { getByTestId } = render(
        <Harness enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={vi.fn()} />
      )

      const editor = getByTestId('editor')

      let reachedTheEditor = false

      await act(async () => {
        editor.textContent = 'breaks instead'
        reachedTheEditor = fireEvent.keyDown(editor, { key: 'Enter' }) !== false
      })

      expect(reachedTheEditor).toBe(true)
    })

    it('still sends on a double tap, with no trailing break to strip', async () => {
      const onSubmit = vi.fn()

      const { getByTestId } = render(
        <Harness enterNewline={false} enterSends={false} onCancel={vi.fn()} onDrain={vi.fn()} onQueue={vi.fn()} onSubmit={onSubmit} sendOnDoubleTap />
      )

      const editor = getByTestId('editor')

      await act(async () => {
        editor.textContent = 'no break to strip'
        fireEvent.keyDown(editor, { key: 'Enter' })
        fireEvent.keyDown(editor, { key: 'Enter' })
      })

      expect(onSubmit).toHaveBeenCalledWith('no break to strip')
    })
  })
})

describe('composer Enter — a deliberate gesture outranks the pause rule', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-14T12:00:00Z'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('reads a second press after a pause as the gesture, and commits the held send at once', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendGraceFor={['pause']}
        sendOnDoubleTap
        sendOnPause
        typedIdleMsAgo={5000}
        typingIdleMs={1000}
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'one press'

    // The first press is the pause send, held for its grace window. The draft
    // deliberately stays in the editor for the whole hold.
    fireEvent.keyDown(editor, { key: 'Enter' })
    fireEvent.keyUp(editor, { key: 'Enter' })

    expect(onSubmit).not.toHaveBeenCalled()

    // The second press lands inside the double-tap window. It is the gesture,
    // and the double tap is not configured to wait, so the held send commits on
    // the spot instead of starting another window.
    fireEvent.keyDown(editor, { key: 'Enter' })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('one press')
  })

  it('gives the double tap its own window when the user asked that gesture to wait', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendGraceFor={['pause', 'doubleTap']}
        sendOnDoubleTap
        sendOnPause
        typedIdleMsAgo={5000}
        typingIdleMs={1000}
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'waits its turn'

    fireEvent.keyDown(editor, { key: 'Enter' })
    fireEvent.keyUp(editor, { key: 'Enter' })
    fireEvent.keyDown(editor, { key: 'Enter' })

    expect(onSubmit).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(GRACE_MS)
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('waits its turn')
  })

  it('lets the hold claim a press that arrived after a pause', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendOnHold
        sendOnPause
        typedIdleMsAgo={5000}
        typingIdleMs={1000}
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'held after a pause'
    fireEvent.keyDown(editor, { key: 'Enter' })

    // With the hold armed the pause send waits for the release, so the press is
    // still free to become the gesture.
    expect(onSubmit).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(HOLD_MS)
    })

    expect(onSubmit).toHaveBeenCalledWith('held after a pause')
  })

  it('still runs the pause send when the press turns out to be a tap', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendOnHold
        sendOnPause
        typedIdleMsAgo={5000}
        typingIdleMs={1000}
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'a tap, not a hold'
    fireEvent.keyDown(editor, { key: 'Enter' })
    fireEvent.keyUp(editor, { key: 'Enter' })

    expect(onSubmit).toHaveBeenCalledWith('a tap, not a hold')
  })

  it('gives the hold its own window when the user asked that gesture to wait', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendGraceFor={['hold']}
        sendOnHold
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'held, then waits'

    act(() => {
      fireEvent.keyDown(editor, { key: 'Enter' })
      vi.advanceTimersByTime(HOLD_MS)
    })

    expect(onSubmit).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(GRACE_MS)
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('held, then waits')
  })

  it('waits the gesture own window even after a pause handed it the press, and commits once', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendGraceFor={['pause', 'hold']}
        sendOnHold
        sendOnPause
        typedIdleMsAgo={5000}
        typingIdleMs={1000}
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'pause, then hold'

    // The pause hands the press to the hold, which fires at its threshold and
    // then waits out its own window.
    act(() => {
      fireEvent.keyDown(editor, { key: 'Enter' })
      vi.advanceTimersByTime(HOLD_MS)
    })

    expect(onSubmit).not.toHaveBeenCalled()

    // The release must not also run the pause rule the hold superseded.
    fireEvent.keyUp(editor, { key: 'Enter' })

    expect(onSubmit).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(GRACE_MS)
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('pause, then hold')
  })
})

describe('composer Enter — a press during the grace window', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-14T12:00:00Z'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('commits the waiting send when a press lands during the wait', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        commitOnPress
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendGraceFor={['pause']}
        sendOnPause
        typedIdleMsAgo={5000}
        typingIdleMs={1000}
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'still here'

    fireEvent.keyDown(editor, { key: 'Enter' })
    fireEvent.keyUp(editor, { key: 'Enter' })

    expect(onSubmit).not.toHaveBeenCalled()

    // A second press, slow enough that it is not a double tap: it is the answer
    // to the question the window asked.
    act(() => {
      vi.advanceTimersByTime(DOUBLE_ENTER_MS + 1)
    })

    fireEvent.keyDown(editor, { key: 'Enter' })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('still here')
  })

  it('restarts the window instead when the option is off, which is what it replaces', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        commitOnPress={false}
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendGraceFor={['pause']}
        sendOnPause
        typedIdleMsAgo={5000}
        typingIdleMs={1000}
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'slides instead'

    fireEvent.keyDown(editor, { key: 'Enter' })
    fireEvent.keyUp(editor, { key: 'Enter' })

    act(() => {
      vi.advanceTimersByTime(DOUBLE_ENTER_MS + 1)
    })

    fireEvent.keyDown(editor, { key: 'Enter' })

    // The press moved the deadline rather than meeting it: the original window
    // has elapsed and nothing has gone.
    act(() => {
      vi.advanceTimersByTime(GRACE_MS - DOUBLE_ENTER_MS)
    })

    expect(onSubmit).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(DOUBLE_ENTER_MS + 1)
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('waits for the release when the press could still become a hold', async () => {
    const onSubmit = vi.fn()

    const { getByTestId } = render(
      <Harness
        commitOnPress
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSubmit={onSubmit}
        sendGraceFor={['pause']}
        sendOnHold
        sendOnPause
        typedIdleMsAgo={5000}
        typingIdleMs={1000}
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'a tap after all'

    fireEvent.keyDown(editor, { key: 'Enter' })
    fireEvent.keyUp(editor, { key: 'Enter' })

    act(() => {
      vi.advanceTimersByTime(DOUBLE_ENTER_MS + 1)
    })

    fireEvent.keyDown(editor, { key: 'Enter' })

    expect(onSubmit).not.toHaveBeenCalled()

    fireEvent.keyUp(editor, { key: 'Enter' })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('a tap after all')
  })
})

describe('composer Enter — the multiline-first steer chord', () => {
  it('steers the running turn on Shift+Enter', async () => {
    const onSteer = vi.fn()

    const { getByTestId } = render(
      <Harness
        canSteer
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSteer={onSteer}
        onSubmit={vi.fn()}
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'not that, this'
    fireEvent.keyDown(editor, { key: 'Enter', shiftKey: true })

    expect(onSteer).toHaveBeenCalledTimes(1)
  })

  it('leaves Shift+Enter to the editor when there is nothing to steer', async () => {
    const onSteer = vi.fn()

    const { getByTestId } = render(
      <Harness
        canSteer={false}
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSteer={onSteer}
        onSubmit={vi.fn()}
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'just a newline'
    // Unprevented: in multiline-first a Shift+Enter that cannot steer is the
    // line break a bare Enter already is.
    const reachedTheEditor = fireEvent.keyDown(editor, { key: 'Enter', shiftKey: true }) !== false

    expect(onSteer).not.toHaveBeenCalled()
    expect(reachedTheEditor).toBe(true)
  })

  it('never steers on Shift+Enter while a bare Enter still sends', async () => {
    const onSteer = vi.fn()

    const { getByTestId } = render(
      <Harness
        canSteer
        enterSends
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSteer={onSteer}
        onSubmit={vi.fn()}
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'the default mode keeps this a newline'
    fireEvent.keyDown(editor, { key: 'Enter', shiftKey: true })

    expect(onSteer).not.toHaveBeenCalled()
  })

  it('needs the second press when the double tap is armed, so a lone Shift+Enter cannot steer', async () => {
    const onSteer = vi.fn()

    const { getByTestId } = render(
      <Harness
        canSteer
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSteer={onSteer}
        onSubmit={vi.fn()}
        sendOnDoubleTap
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'a stray shift, then the real one'
    fireEvent.keyDown(editor, { key: 'Enter', shiftKey: true })

    expect(onSteer).not.toHaveBeenCalled()

    // The pair completes on the second press, inside the window.
    fireEvent.keyDown(editor, { key: 'Enter', shiftKey: true })

    expect(onSteer).toHaveBeenCalledTimes(1)
  })

  it('gives a lone Shift+Enter nothing at all when the line break is off', async () => {
    const onSteer = vi.fn()

    const { getByTestId } = render(
      <Harness
        canSteer
        enterNewline={false}
        enterSends={false}
        onCancel={vi.fn()}
        onDrain={vi.fn()}
        onQueue={vi.fn()}
        onSteer={onSteer}
        onSubmit={vi.fn()}
        sendOnDoubleTap
      />
    )

    const editor = getByTestId('editor')

    editor.textContent = 'nothing lands'
    const reachedTheEditor = fireEvent.keyDown(editor, { key: 'Enter', shiftKey: true }) !== false

    expect(onSteer).not.toHaveBeenCalled()
    expect(reachedTheEditor).toBe(false)
  })
})
