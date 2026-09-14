import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { type FormEvent, useEffect, useRef, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { ComposerControls } from './controls'
import { useAutoSendIdle } from './hooks/use-auto-send-idle'
import type { ChatBarState } from './types'

interface HarnessProps {
  attachmentUploading?: boolean
  awaitingInput?: boolean
  blockingPrompt?: boolean
  busy?: boolean
  compacting?: boolean
  delayMs?: number
  disabled?: boolean
  enabled?: boolean
  inputDisabled?: boolean
  minimal?: boolean
  onSubmit: (text: string) => void
  queueEdit?: boolean
  resetKey?: string
  trigger?: null | { kind: string }
  voiceLive?: boolean
}

function Harness({
  attachmentUploading = false,
  awaitingInput = false,
  blockingPrompt = false,
  busy = false,
  compacting = false,
  delayMs = 2000,
  disabled = false,
  enabled = true,
  inputDisabled = false,
  minimal = false,
  onSubmit,
  queueEdit = false,
  resetKey = 'session-1',
  trigger = null,
  voiceLive = false
}: HarnessProps) {
  const editorRef = useRef<HTMLDivElement>(null)
  const draftRef = useRef('')
  const [draft, setDraft] = useState('')
  const composingRef = useRef(false)

  const composerPlainText = (el: HTMLElement) => el.textContent ?? ''

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
    const payloadPresent = text.trim().length > 0

    if (payloadPresent) {
      onSubmit(text)

      if (editorRef.current) {
        editorRef.current.textContent = ''
      }

      draftRef.current = ''
      setDraft('')
    }
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      cancelAutoSend()
      submitDraft()
    }
  }

  const {
    armedInSeconds,
    cancel: cancelAutoSend,
    noteCommittedComposition: noteAutoSendComposition,
    noteEdit: noteAutoSendEdit
  } = useAutoSendIdle({
    canAutoSend: () =>
      !busy &&
      !disabled &&
      !inputDisabled &&
      !compacting &&
      !composingRef.current &&
      !queueEdit &&
      !awaitingInput &&
      !blockingPrompt &&
      trigger === null &&
      !minimal &&
      !voiceLive &&
      !attachmentUploading &&
      (editorRef.current ? editorRef.current.ownerDocument.hasFocus() : true) &&
      !!editorRef.current &&
      editorRef.current.contains(editorRef.current.ownerDocument.activeElement),
    delayMs,
    enabled,
    onFire: submitDraft,
    readText: () => (editorRef.current ? composerPlainText(editorRef.current) : draftRef.current),
    resetKey
  })

  useEffect(() => {
    if (
      attachmentUploading ||
      awaitingInput ||
      blockingPrompt ||
      busy ||
      compacting ||
      disabled ||
      inputDisabled ||
      minimal ||
      queueEdit ||
      trigger ||
      voiceLive
    ) {
      cancelAutoSend()
    }
  }, [
    attachmentUploading,
    awaitingInput,
    blockingPrompt,
    busy,
    cancelAutoSend,
    compacting,
    disabled,
    inputDisabled,
    minimal,
    queueEdit,
    trigger,
    voiceLive
  ])

  useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined
    }

    window.addEventListener('blur', cancelAutoSend)

    return () => window.removeEventListener('blur', cancelAutoSend)
  }, [cancelAutoSend])

  const flushEditorToDraft = (editor: HTMLDivElement) => {
    const nextDraft = composerPlainText(editor)

    if (nextDraft !== draftRef.current) {
      draftRef.current = nextDraft
      setDraft(nextDraft)
    }
  }

  const handleEditorInput = (event: FormEvent<HTMLDivElement>) => {
    // Hands-free send: only a trusted insert arms the idle timer. A paste, a
    // delete, or a programmatic DOM write (restored draft, undo restore, queue
    // load) disarms it instead — none of those may ever auto-send.
    const nativeInput = event.nativeEvent as InputEvent

    noteAutoSendEdit(nativeInput.isTrusted, nativeInput.inputType)

    if (composingRef.current) {
      return
    }

    flushEditorToDraft(event.currentTarget)
  }

  void draft

  return (
    <div>
      <div
        contentEditable={!inputDisabled}
        data-testid="editor"
        onBlur={() => {
          composingRef.current = false
          cancelAutoSend()
        }}
        onCompositionEnd={event => {
          composingRef.current = false
          flushEditorToDraft(event.currentTarget)
          noteAutoSendComposition()
        }}
        onCompositionStart={() => {
          composingRef.current = true
          cancelAutoSend()
        }}
        onInput={handleEditorInput}
        onKeyDown={handleKeyDown}
        ref={editorRef}
        suppressContentEditableWarning
      />
      <input data-testid="other-input" />
      <span data-testid="countdown">{armedInSeconds ?? 'idle'}</span>
    </div>
  )
}

function dispatchTrustedInput(editor: HTMLElement, text: string, inputType = 'insertText') {
  editor.textContent = text

  const event = new InputEvent('input', {
    bubbles: true,
    cancelable: true,
    inputType
  })

  // In jsdom, EventTarget.dispatchEvent resets eventImpl.isTrusted to false.
  // Intercept the internal symbol so isTrusted remains true when read by React's nativeEvent.
  for (const symbol of Object.getOwnPropertySymbols(event)) {
    const candidate = (event as unknown as Record<symbol, unknown>)[symbol]

    if (candidate && typeof candidate === 'object' && 'isTrusted' in candidate) {
      Object.defineProperty(candidate, 'isTrusted', {
        get() {
          return true
        },
        set() {}
      })
    }
  }

  fireEvent(editor, event)
}

describe('composer hands-free auto-send DOM behaviour', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('1. trusted insert advancing 2000 ms calls onSubmit exactly once and clears editor', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    act(() => {
      editor.focus()
    })

    act(() => {
      dispatchTrustedInput(editor, 'hello world')
    })

    expect(onSubmit).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('hello world')
    expect(editor.textContent).toBe('')

    // Further advancement does not call onSubmit again
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('2. advancing 1999 ms does not call; second keystroke at 1000 ms restarts window', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    act(() => {
      editor.focus()
    })

    act(() => {
      dispatchTrustedInput(editor, 'first')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    expect(onSubmit).not.toHaveBeenCalled()

    // Second keystroke at 1000 ms restarts the 2000 ms window
    act(() => {
      dispatchTrustedInput(editor, 'first second')
    })

    // Advance to 1999 ms from start (999 ms after second keystroke)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(999)
    })
    expect(onSubmit).not.toHaveBeenCalled()

    // Advance to 2000 ms from start (original deadline) -> no fire
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })
    expect(onSubmit).not.toHaveBeenCalled()

    // Advance remaining 1000 ms (2000 ms after second keystroke) -> fires once
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('first second')
    expect(editor.textContent).toBe('')
  })

  it('3. programmatic write without trusted input event never auto-sends', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    act(() => {
      editor.focus()
      editor.textContent = 'restored draft'
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000)
    })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(editor.textContent).toBe('restored draft')
  })

  it('4. untrusted new Event("input") dispatched on editor never auto-sends', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    act(() => {
      editor.focus()
      editor.textContent = 'untrusted text'
      fireEvent(editor, new Event('input', { bubbles: true }))
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('5. inputType of insertFromPaste never auto-sends', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    act(() => {
      editor.focus()
      dispatchTrustedInput(editor, 'pasted text', 'insertFromPaste')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('6. composition sequence auto-sends only after compositionend, never mid-composition', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    act(() => {
      editor.focus()
      fireEvent.compositionStart(editor)
      editor.textContent = 'nihongo'
      fireEvent(editor, new InputEvent('input', { bubbles: true, inputType: 'insertCompositionText' }))
    })

    // Advance mid-composition: must not fire
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })
    expect(onSubmit).not.toHaveBeenCalled()

    // Commit composition
    act(() => {
      editor.textContent = '日本語'
      fireEvent.compositionEnd(editor)
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1999)
    })
    expect(onSubmit).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('日本語')
  })

  it('7. enabled: false never auto-sends', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness enabled={false} onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    act(() => {
      editor.focus()
      dispatchTrustedInput(editor, 'hello')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('8. popover-open gate (canAutoSend false) never auto-sends', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} trigger={{ kind: 'slash' }} />)
    const editor = getByTestId('editor')

    act(() => {
      editor.focus()
      dispatchTrustedInput(editor, 'some command')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('focus-gate: moving focus away cancels/blocks pending auto-send', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')
    const otherInput = getByTestId('other-input')

    act(() => {
      editor.focus()
    })

    act(() => {
      dispatchTrustedInput(editor, 'unfocused message')
    })

    // Advance 1000 ms while focused
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    expect(onSubmit).not.toHaveBeenCalled()

    // Move focus away to another element
    act(() => {
      otherInput.focus()
      fireEvent.blur(editor)
    })

    // Advance past the remaining delay
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('multi-composer: only the focused composer auto-sends', async () => {
    const firstSubmit = vi.fn()
    const secondSubmit = vi.fn()

    const view = render(
      <>
        <Harness onSubmit={firstSubmit} resetKey="session-1" />
        <Harness onSubmit={secondSubmit} resetKey="session-2" />
      </>
    )

    const editors = view.getAllByTestId('editor')

    // Both composers receive a trusted insert (a dictation stream can land in
    // either); only the one the user is actually in may send.
    act(() => {
      editors[1].focus()
      dispatchTrustedInput(editors[0], 'from the unfocused composer')
      dispatchTrustedInput(editors[1], 'from the focused composer')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })

    expect(firstSubmit).not.toHaveBeenCalled()
    expect(secondSubmit).toHaveBeenCalledTimes(1)
    expect(secondSubmit).toHaveBeenCalledWith('from the focused composer')
  })

  it('mid-composition: a trusted insert cannot auto-send until the composition commits', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    act(() => {
      editor.focus()
      fireEvent.compositionStart(editor)
    })

    // Some IMEs emit a TRUSTED insert for preedit text rather than a
    // composition-typed one. That arms the timer, so the fire-time composing
    // gate is the only thing standing between a mid-thought pause and a
    // half-composed message being sent.
    act(() => {
      dispatchTrustedInput(editor, '你好', 'insertText')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })
    expect(onSubmit).not.toHaveBeenCalled()

    // The commit is the point where the text becomes sendable.
    act(() => {
      fireEvent.compositionEnd(editor)
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('你好')
  })

  it('window blur cancels a pending auto-send', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    act(() => {
      editor.focus()
    })

    act(() => {
      dispatchTrustedInput(editor, 'hello world')
    })

    act(() => {
      fireEvent(window, new Event('blur'))
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('loss of window focus blocks the fire', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')
    // jsdom's hasFocus defaults to false until something in the document is
    // focused, so stub the gate's own read rather than depending on that.
    const hasFocusSpy = vi.spyOn(editor.ownerDocument, 'hasFocus').mockReturnValue(false)

    try {
      act(() => {
        editor.focus()
      })

      act(() => {
        dispatchTrustedInput(editor, 'hello world')
      })

      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000)
      })

      expect(onSubmit).not.toHaveBeenCalled()
    } finally {
      hasFocusSpy.mockRestore()
    }
  })

  it('minimal layout never auto-sends', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness minimal onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    act(() => {
      editor.focus()
    })

    act(() => {
      dispatchTrustedInput(editor, 'hello world')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('a live voice conversation never auto-sends', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} voiceLive />)
    const editor = getByTestId('editor')

    act(() => {
      editor.focus()
    })

    act(() => {
      dispatchTrustedInput(editor, 'hello world')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('an attachment mid-upload never auto-sends', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness attachmentUploading onSubmit={onSubmit} />)
    const editor = getByTestId('editor')

    act(() => {
      editor.focus()
    })

    act(() => {
      dispatchTrustedInput(editor, 'hello world')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('manual Enter cancels the pending countdown', async () => {
    const onSubmit = vi.fn()
    const { getByTestId } = render(<Harness onSubmit={onSubmit} />)
    const editor = getByTestId('editor')
    const countdown = getByTestId('countdown')

    act(() => {
      editor.focus()
    })

    act(() => {
      dispatchTrustedInput(editor, 'hello world')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })

    expect(onSubmit).not.toHaveBeenCalled()

    act(() => {
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('hello world')
    expect(countdown.textContent).toBe('idle')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('a voice conversation starting mid-countdown disarms it', async () => {
    const onSubmit = vi.fn()
    const view = render(<Harness onSubmit={onSubmit} />)
    const editor = view.getByTestId('editor')

    act(() => {
      editor.focus()
      dispatchTrustedInput(editor, 'hello world')
    })
    expect(view.getByTestId('countdown').textContent).toBe('2')

    // The voice loop submits its own turns — a pending send must give way.
    view.rerender(<Harness onSubmit={onSubmit} voiceLive />)
    expect(view.getByTestId('countdown').textContent).toBe('idle')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('an upload starting mid-countdown disarms it', async () => {
    const onSubmit = vi.fn()
    const view = render(<Harness onSubmit={onSubmit} />)
    const editor = view.getByTestId('editor')

    act(() => {
      editor.focus()
      dispatchTrustedInput(editor, 'hello world')
    })
    expect(view.getByTestId('countdown').textContent).toBe('2')

    view.rerender(<Harness attachmentUploading onSubmit={onSubmit} />)
    expect(view.getByTestId('countdown').textContent).toBe('idle')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })
    expect(onSubmit).not.toHaveBeenCalled()
  })

  describe('ComposerControls countdown affordance', () => {
    const dummyState: ChatBarState = {
      model: { canSwitch: false, model: '', provider: '' },
      tools: { enabled: false, label: '' },
      voice: { active: false, enabled: false }
    }

    it('renders countdown before send button when armed, and hides when minimal', () => {
      const { queryByText, rerender } = render(
        <I18nProvider configClient={null} initialLocale="en">
          <ComposerControls
            autoSendArmedInSeconds={2}
            autoSpeak={false}
            busy={false}
            busyAction="stop"
            canSubmit={true}
            conversation={{
              active: false,
              level: 0,
              muted: false,
              onEnd: vi.fn(),
              onStart: vi.fn(),
              onStopTurn: vi.fn(),
              onToggleMute: vi.fn(),
              status: 'idle'
            }}
            disabled={false}
            hasComposerPayload={true}
            onDictate={vi.fn()}
            onQueue={vi.fn()}
            onToggleAutoSpeak={vi.fn()}
            state={dummyState}
            voiceStatus="idle"
          />
        </I18nProvider>
      )

      expect(queryByText('Sending in 2s')).toBeTruthy()

      // When minimal is true, the label is hidden
      rerender(
        <I18nProvider configClient={null} initialLocale="en">
          <ComposerControls
            autoSendArmedInSeconds={2}
            autoSpeak={false}
            busy={false}
            busyAction="stop"
            canSubmit={true}
            conversation={{
              active: false,
              level: 0,
              muted: false,
              onEnd: vi.fn(),
              onStart: vi.fn(),
              onStopTurn: vi.fn(),
              onToggleMute: vi.fn(),
              status: 'idle'
            }}
            disabled={false}
            hasComposerPayload={true}
            minimal={true}
            onDictate={vi.fn()}
            onQueue={vi.fn()}
            onToggleAutoSpeak={vi.fn()}
            state={dummyState}
            voiceStatus="idle"
          />
        </I18nProvider>
      )

      expect(queryByText('Sending in 2s')).toBeNull()
    })
  })
})
