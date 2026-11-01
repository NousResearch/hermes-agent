// @vitest-environment jsdom
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { clearSessionDraft, mainComposerScope, NEW_SESSION_DRAFT_KEY } from '@/store/composer'

import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

afterEach(cleanup)

// THE INVARIANT: when the clipboard carries meaningful text alongside an image
// flavor, the text is inserted and the image is NOT attached.
//
// Rich-text sources put both flavors on the clipboard for one copy operation:
// Word attaches a rendered image of the selected prose (#42280), and macOS
// adds a PDF/image ghost flavor beside every plain-text copy (#42827). The
// image is a rendering artifact of the same content, so attaching it produced
// a duplicate `composer_….jpg` attachment next to the inserted text — for
// macOS plain-text copies, a ghost attachment on EVERY paste.
//
// The handler is driven through the REAL ChatBar with a real paste event on
// the contentEditable, focused first (an unfocused ⌘V routes through
// paste-to-focus, which shares the attachment path). jsdom has no clipboard,
// so the event carries the fake DataTransfer shape the sibling
// paste-url-is-text tests use.
//
// A bare `data:image` URL is an image in TEXT form: extractClipboardImageBlobs
// turns it into a blob, so it must NOT count as meaningful text — otherwise
// the attachment path would starve and the URI would land in the editor as
// literal text.

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

const CLIP_IMAGE = () =>
  new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])], 'clip.png', {
    type: 'image/png',
    lastModified: 1_700_000_000_002
  })

// 1×1 transparent PNG — a valid base64 payload so dataUrlToBlob's atob path
// produces a non-empty blob.
const PNG_DATA_URL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='

function Harness({ onAttachImageBlob, sessionId }: { onAttachImageBlob: () => void; sessionId: string }) {
  // The adapter's message generic infers to `never` from an empty array, so the
  // empty store is typed explicitly. The runtime itself is only here to satisfy
  // ChatBar's provider; nothing in this test reads it.
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
            onAttachImageBlob={onAttachImageBlob}
            onCancel={vi.fn()}
            onSubmit={vi.fn(async () => true)}
            sessionId={sessionId}
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

interface PastePayload {
  text?: string
  imageFile?: File
}

/** Focus `editor` and fire the paste event a ⌘V into the composer produces. */
function pasteInto(editor: HTMLElement, { text = '', imageFile }: PastePayload) {
  // jsdom does not implement isContentEditable, so the app's window-level paste
  // router would treat this target as page chrome and insert through the
  // composer bus instead of the editor's own handler. Pin it.
  Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })
  editor.focus()

  const event = new Event('paste', { bubbles: true, cancelable: true }) as ClipboardEvent

  Object.defineProperty(event, 'clipboardData', {
    value: {
      getData: (type: string) => (type === 'text' || type === 'text/plain' ? text : ''),
      files: { length: 0, item: () => null },
      items: imageFile
        ? [{ kind: 'file', type: imageFile.type, getAsFile: () => imageFile }]
        : []
    }
  })

  act(() => {
    fireEvent(editor, event)
  })

  return event
}

describe('paste text+image flavor priority', () => {
  // Each test gives its ChatBar a unique sessionId so the draft stashes land in
  // that test's own session bucket — a session-less composer shares the
  // fresh-draft bucket, and a pending persist from one test can re-stash into
  // it after the sweep below and leak into the next test's restored composer.
  afterEach(async () => {
    mainComposerScope.clear()
    // Let any pending rAF/debounced draft persist land BEFORE the sweep.
    await new Promise(resolve => setTimeout(resolve, 20))
    clearSessionDraft(undefined)
    clearSessionDraft(NEW_SESSION_DRAFT_KEY)
    clearSessionDraft('paste-word')
    clearSessionDraft('paste-image-only')
    clearSessionDraft('paste-data-url')
  })

  it('inserts the text and attaches nothing when Word ships text with a rendered image', () => {
    const onAttachImageBlob = vi.fn()

    const { container } = render(<Harness onAttachImageBlob={onAttachImageBlob} sessionId="paste-word" />)

    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    const event = pasteInto(editor, { text: 'hello from Word', imageFile: CLIP_IMAGE() })

    // The prose reached the editor…
    expect(composerPlainText(editor)).toContain('hello from Word')
    // …and the rendered copy of that same prose was not attached.
    expect(onAttachImageBlob).not.toHaveBeenCalled()
    expect(event.defaultPrevented).toBe(true)
  })

  it('still attaches an image-only paste (a real screenshot has no text flavor)', () => {
    const onAttachImageBlob = vi.fn()

    const { container } = render(<Harness onAttachImageBlob={onAttachImageBlob} sessionId="paste-image-only" />)

    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    pasteInto(editor, { imageFile: CLIP_IMAGE() })

    expect(onAttachImageBlob).toHaveBeenCalledTimes(1)
    expect(composerPlainText(editor)).toBe('')
  })

  it('treats a pasted data:image URL as an image, not literal text', () => {
    const onAttachImageBlob = vi.fn()

    const { container } = render(<Harness onAttachImageBlob={onAttachImageBlob} sessionId="paste-data-url" />)

    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    pasteInto(editor, { text: PNG_DATA_URL })

    // The URI is extracted as an image blob and attached…
    expect(onAttachImageBlob).toHaveBeenCalledTimes(1)
    // …never inserted into the editor as literal text.
    expect(composerPlainText(editor)).toBe('')
  })
})
