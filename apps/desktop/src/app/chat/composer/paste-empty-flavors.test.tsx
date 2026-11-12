// @vitest-environment jsdom
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { mainComposerScope } from '@/store/composer'

import { RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

afterEach(cleanup)

// THE INVARIANT: an unreadable paste is not cancelled.
//
// The ChatGPT iOS app's "Copy" puts RTFD/HTML on the pasteboard with no
// synchronously-readable text flavor: inside the paste event, `getData` comes
// back empty for EVERY type (plain, uri-list, html). The flavor rescue in
// `extractClipboardText` has nothing to work with, and the handler used to
// `preventDefault()` any text-less paste anyway — deleting a paste WebKit's
// NATIVE insertion (the "Paste and Match Style" machinery) converts just fine.
// On iPhone Safari those rich pastes were a silent no-op: typed text in the
// composer survived the send, the pasted body vanished with it.
//
// So when no flavor reads as text and there is no image, the handler must
// leave the event UNcancelled (a bridge-less surface — the webapp — has no
// fallback left; let the browser convert the paste itself). The WSL image
// bridge keeps its preventDefault: that path has a real fallback to run.
// Sibling of #132591 and #150719.
const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness({ onPasteClipboardImage }: { onPasteClipboardImage?: (opts?: { silent?: boolean }) => void | Promise<boolean> }) {
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
            onPasteClipboardImage={onPasteClipboardImage}
            onSubmit={vi.fn(async () => true)}
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

/** Focus `editor` and fire a paste whose clipboard reads empty for every type. */
function pasteEmptyClipboard(editor: HTMLElement) {
  // jsdom does not implement isContentEditable, so the app's window-level paste
  // router would treat this target as page chrome; pin it (see the sibling
  // paste-url-is-text.test.tsx for why).
  Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })
  editor.focus()

  const event = new Event('paste', { bubbles: true, cancelable: true }) as ClipboardEvent

  Object.defineProperty(event, 'clipboardData', {
    value: {
      getData: () => '',
      files: [],
      items: []
    }
  })

  act(() => {
    fireEvent(editor, event)
  })

  return event
}

describe('a paste no clipboard flavor can describe', () => {
  afterEach(() => {
    mainComposerScope.clear()
  })

  it('is left uncancelled so WebKit native insertion converts it', () => {
    const { container } = render(<Harness />)

    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    const event = pasteEmptyClipboard(editor)

    expect(event.defaultPrevented).toBe(false)
  })

  it('still hands the WSL image bridge a cancelled event (it has a fallback to run)', () => {
    const onPasteClipboardImage = vi.fn(() => {})

    const { container } = render(<Harness onPasteClipboardImage={onPasteClipboardImage} />)

    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    const event = pasteEmptyClipboard(editor)

    expect(event.defaultPrevented).toBe(true)
    expect(onPasteClipboardImage).toHaveBeenCalledWith({ silent: true })
  })
})
