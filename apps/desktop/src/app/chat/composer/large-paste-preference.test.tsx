// @vitest-environment jsdom
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { clearSessionDraft, mainComposerScope } from '@/store/composer'
import { setLargePasteAttachmentThreshold } from '@/store/large-paste-threshold'

import { LARGE_PASTE_ATTACHMENT_THRESHOLD } from './large-paste'
import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

afterEach(cleanup)

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness({ onAttachPastedText }: { onAttachPastedText?: (text: string) => Promise<boolean> }) {
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
  // jsdom does not implement isContentEditable, so the app's window-level paste
  // router would treat this target as page chrome and insert through the
  // composer bus instead of the editor's own handler. Pin it.
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

describe('large-paste preference in the real composer', () => {
  beforeEach(() => {
    setLargePasteAttachmentThreshold(LARGE_PASTE_ATTACHMENT_THRESHOLD)
    mainComposerScope.clear()
    clearSessionDraft(null)
  })

  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
    clearSessionDraft(null)
    setLargePasteAttachmentThreshold(LARGE_PASTE_ATTACHMENT_THRESHOLD)
  })

  const goal = '/goal ' + 'a'.repeat(11_694)

  it.each([
    [LARGE_PASTE_ATTACHMENT_THRESHOLD - 1, false],
    [LARGE_PASTE_ATTACHMENT_THRESHOLD, false],
    [LARGE_PASTE_ATTACHMENT_THRESHOLD + 1, true]
  ] as const)('routes a default paste of %s characters (attachment: %s)', async (length, attached) => {
    const attach = vi.fn(async () => true)
    const { container } = render(<Harness onAttachPastedText={attach} />)
    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
    const text = 'a'.repeat(length)

    await act(async () => {
      pasteInto(editor, text)
    })

    if (attached) {
      expect(attach).toHaveBeenCalledExactlyOnceWith(text)
      expect(composerPlainText(editor)).toBe('')
    } else {
      expect(attach).not.toHaveBeenCalled()
      expect(composerPlainText(editor)).toBe(text)
    }
  })

  it.each([0, 50_000, 100_000])('keeps a long goal inline with threshold %s', async threshold => {
    setLargePasteAttachmentThreshold(threshold)
    const attach = vi.fn(async () => true)
    const { container } = render(<Harness onAttachPastedText={attach} />)
    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    await act(async () => {
      pasteInto(editor, goal)
    })

    expect(attach).not.toHaveBeenCalled()
    expect(composerPlainText(editor)).toBe(goal)
  })

  it('reads preference changes on the next paste without remounting', async () => {
    const attach = vi.fn(async () => true)
    const { container } = render(<Harness onAttachPastedText={attach} />)
    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    await act(async () => {
      pasteInto(editor, goal)
    })
    expect(attach).toHaveBeenCalledExactlyOnceWith(goal)
    expect(composerPlainText(editor)).toBe('')

    setLargePasteAttachmentThreshold(0)
    await act(async () => {
      pasteInto(editor, goal)
    })
    expect(attach).toHaveBeenCalledTimes(1)
    expect(composerPlainText(editor)).toBe(goal)
  })

  it('falls back to inline insertion when attachment creation fails', async () => {
    const attach = vi.fn(async () => false)
    const { container } = render(<Harness onAttachPastedText={attach} />)
    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    await act(async () => {
      pasteInto(editor, goal)
    })

    expect(attach).toHaveBeenCalledExactlyOnceWith(goal)
    expect(composerPlainText(editor)).toBe(goal)
  })

  it('keeps the paste inline when attachment creation is unavailable', async () => {
    const { container } = render(<Harness />)
    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    await act(async () => {
      pasteInto(editor, goal)
    })

    expect(composerPlainText(editor)).toBe(goal)
  })
})
