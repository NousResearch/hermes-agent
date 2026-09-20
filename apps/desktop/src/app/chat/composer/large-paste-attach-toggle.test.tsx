// @vitest-environment jsdom
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { mainComposerScope } from '@/store/composer'
import { $largePasteAttachEnabled, setLargePasteAttachEnabled } from '@/store/large-paste-attach'

import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  $largePasteAttachEnabled.set(true)
  window.localStorage.clear()
})

// THE INVARIANT: the large-paste-to-attachment conversion is a preference,
// not a hard-wired policy. A paste over the threshold attaches ONLY while the
// device-local toggle is on (and by default); turned off, the very same paste
// lands inline in the composer. The user's stated case is a 20-30k character
// prompt — the paste IS the message, and an attachment chip is not an
// equivalent surface for it.
//
// The handler is driven through the REAL ChatBar with a real paste event on
// the contentEditable (same harness as paste-url-is-text.test.tsx). The
// attachment hook resolves true — if the gate is open the paste is consumed
// and the editor stays empty; if the gate is closed the text must survive.

const LONG_PASTE = 'x'.repeat(3_500)

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness({ onAttachPastedText }: { onAttachPastedText: (text: string) => boolean }) {
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
            onAttachPastedText={onAttachPastedText}
            onCancel={vi.fn()}
            onSubmit={vi.fn(async () => true)}
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

/** Focus `editor` and fire the paste event a ⌘V into the composer produces. */
function pasteInto(editor: HTMLElement, text: string) {
  Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })
  editor.focus()

  const event = new Event('paste', { bubbles: true, cancelable: true }) as ClipboardEvent

  Object.defineProperty(event, 'clipboardData', {
    value: {
      getData: (type: string) => (type === 'text' || type === 'text/plain' ? text : ''),
      files: [],
      items: []
    }
  })

  act(() => {
    fireEvent(editor, event)
  })

  return event
}

function editorOf(container: HTMLElement): HTMLElement {
  return container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
}

describe('large paste conversion follows the device-local toggle', () => {
  it('attaches an over-threshold paste while the toggle is on (the default)', () => {
    const onAttachPastedText = vi.fn(() => true)

    const { container } = render(<Harness onAttachPastedText={onAttachPastedText} />)

    pasteInto(editorOf(container), LONG_PASTE)

    expect(onAttachPastedText).toHaveBeenCalledWith(LONG_PASTE)
    expect(composerPlainText(editorOf(container))).not.toContain('x'.repeat(100))
  })

  it('keeps the very same paste inline when the toggle is off', () => {
    setLargePasteAttachEnabled(false)

    const onAttachPastedText = vi.fn(() => true)

    const { container } = render(<Harness onAttachPastedText={onAttachPastedText} />)

    pasteInto(editorOf(container), LONG_PASTE)

    expect(onAttachPastedText).not.toHaveBeenCalled()
    expect(composerPlainText(editorOf(container))).toContain(LONG_PASTE)
  })

  it('a toggle flip is picked up live — no remount needed', () => {
    const onAttachPastedText = vi.fn(() => true)

    const { container } = render(<Harness onAttachPastedText={onAttachPastedText} />)

    // On: consumed as an attachment.
    pasteInto(editorOf(container), LONG_PASTE)
    expect(onAttachPastedText).toHaveBeenCalledTimes(1)

    // Off: identical paste lands inline.
    setLargePasteAttachEnabled(false)
    pasteInto(editorOf(container), LONG_PASTE)
    expect(onAttachPastedText).toHaveBeenCalledTimes(1)
    expect(composerPlainText(editorOf(container))).toContain(LONG_PASTE)
  })
})
