import type { ThreadMessageLike } from '@assistant-ui/react'
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import { composerPrefsFromConfig } from '@hermes/shared'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ChatBar } from '@/app/chat/composer'
import { RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'
import type { ChatBarState } from '@/app/chat/composer/types'
import { stubThreadEnvironment, stubThreadViewportSize } from '@/components/assistant-ui/test-utils'
import { I18nProvider } from '@/i18n'
import { en } from '@/i18n/en'
import { $composerSendPrefs } from '@/store/composer-prefs'

stubThreadEnvironment()
stubThreadViewportSize()

afterEach(cleanup)

// Real-ChatBar regressions for the press-owned send lifecycle:
//
//  · blur ends the press (bug 3): the hold timer started by the keydown is
//    cancelled and the deferred pause/commit is dropped, so nothing fires after
//    focus moves — and a keyup delivered elsewhere cannot revive it;
//  · a session change takes the window back (bug 2): the composer stays mounted
//    across a swap, so a window armed for session A must not survive to submit
//    session B (the visible cue is the SendHoldChip).
//
// `onSubmit` is reached through the composer's async middleware chain, so every
// step that can send is awaited inside an async act() before asserting.
const chatBarState: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function ComposerHarness({
  onSubmit,
  sessionId
}: {
  onSubmit: (text: string) => void
  sessionId?: string
}) {
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: false,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <ChatBar
            busy={false}
            disabled={false}
            gateway={null}
            onCancel={vi.fn()}
            onSubmit={async text => {
              onSubmit(text)

              return true
            }}
            sessionId={sessionId}
            state={chatBarState}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

function editorOf(container: HTMLElement): HTMLElement {
  const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)

  expect(editor).not.toBeNull()

  return editor!
}

describe('composer Enter — blur ends the press-owned state', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    $composerSendPrefs.set(composerPrefsFromConfig({}))
  })

  it('cancels the hold timer when the editor blurs, so focus moving cannot send', async () => {
    const onSubmit = vi.fn()

    $composerSendPrefs.set(composerPrefsFromConfig({ enter_sends: false, send_on_hold: true, hold_ms: 350 }))

    const { container } = render(<ComposerHarness onSubmit={onSubmit} />)
    const editor = editorOf(container)

    await act(async () => {
      editor.textContent = 'held message'
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    expect(onSubmit).not.toHaveBeenCalled()

    // The user clicks away before the threshold: the press is over, so the
    // hold timer must not still fire the send after focus has moved.
    await act(async () => {
      fireEvent.blur(editor)
      vi.advanceTimersByTime(350)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('drops a press deferred to the release, so a later keyup cannot run it', async () => {
    const onSubmit = vi.fn()

    // pause + hold: the press is deferred to the release while the hold is
    // armed (composerEnterPressOwner → pauseOnRelease).
    $composerSendPrefs.set(composerPrefsFromConfig({ enter_sends: false, send_on_hold: true, send_on_pause: true }))

    const { container } = render(<ComposerHarness onSubmit={onSubmit} />)
    const editor = editorOf(container)

    await act(async () => {
      editor.textContent = 'deferred press'
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    await act(async () => {
      fireEvent.blur(editor)
    })

    // The keyup arrives after focus moved — it must not run the pause rule the
    // blur cancelled (whose own grace window would then commit the draft).
    await act(async () => {
      fireEvent.keyUp(editor, { key: 'Enter' })
      vi.advanceTimersByTime(900)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })
})

describe('composer Enter — a session change takes back a waiting window', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    $composerSendPrefs.set(composerPrefsFromConfig({}))
  })

  it('cancels the grace window when the active session changes', async () => {
    const onSubmit = vi.fn()
    const label = en.composer.sendHold

    $composerSendPrefs.set(
      composerPrefsFromConfig({
        enter_sends: false,
        send_on_hold: false,
        send_on_pause: true,
        send_grace_for: ['pause'],
        send_grace_ms: 900
      })
    )

    const { container, rerender } = render(<ComposerHarness onSubmit={onSubmit} sessionId="session-a" />)
    const editor = editorOf(container)

    await act(async () => {
      editor.textContent = 'session a draft'
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    // The pause send is waiting, so the chip is up.
    expect(screen.queryByText(label)).not.toBeNull()

    // The user switches sessions. ChatBar stays mounted; the window belongs to
    // session A and must be taken back rather than repointed at session B.
    rerender(<ComposerHarness onSubmit={onSubmit} sessionId="session-b" />)

    expect(screen.queryByText(label)).toBeNull()

    await act(async () => {
      vi.advanceTimersByTime(900)
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })
})
