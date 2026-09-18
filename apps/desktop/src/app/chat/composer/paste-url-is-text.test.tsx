// @vitest-environment jsdom
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { mainComposerScope } from '@/store/composer'

import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

afterEach(cleanup)

// THE INVARIANT: a pasted URL is never swallowed.
//
// A GitHub PR-comment deep link (`…/pull/1#issuecomment-2`) used to be
// special-cased inside ChatBar's paste handler: it called
// `onAttachPrCommentUrl`, ran `event.preventDefault()`, and returned — so the
// clipboard payload never reached the editor (proven: with this harness the
// editor ends up EMPTY and the hook is called once) and the user saw only an
// attachment pill above the composer. Removing that interception is what makes
// the two URLs behave identically again.
//
// The handler is driven through the REAL ChatBar with a real paste event on the
// contentEditable, focused first (an unfocused ⌘V routes through paste-to-focus,
// which shares this insertion path). jsdom has no clipboard, so the event
// carries the fake DataTransfer shape the sibling paste-to-focus tests use.
//
// `onAttachPrCommentUrl` is spread in under a cast deliberately: the prop was
// deleted along with the interception, and passing it anyway is what proves the
// handler no longer consults it. On the pre-fix tree it is called once and the
// payload is dropped; here it must never be called and the URL must survive.
//
// `composerPlainText` round-trips a `@url:` chip to its directive text, so the
// assertions hold whether the link lands chipped or raw — the chip form is
// url-refs.test.ts's contract, this file's is only that the paste survives.
const PR_COMMENT_URL = 'https://github.com/o/r/pull/1#issuecomment-2'

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness({ onAttachPrCommentUrl }: { onAttachPrCommentUrl: (url: string) => boolean }) {
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
            onCancel={vi.fn()}
            onSubmit={vi.fn(async () => true)}
            state={state}
            {...({ onAttachPrCommentUrl } as Record<string, unknown>)}
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

describe('a pasted URL survives the paste', () => {
  afterEach(() => {
    mainComposerScope.clear()
  })

  it('keeps a GitHub PR-comment deep link in the composer and attaches nothing', () => {
    const onAttachPrCommentUrl = vi.fn(() => true)

    const { container } = render(<Harness onAttachPrCommentUrl={onAttachPrCommentUrl} />)

    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    const event = pasteInto(editor, PR_COMMENT_URL)

    // The payload reached the editor instead of being consumed by the handler…
    expect(composerPlainText(editor)).toContain(PR_COMMENT_URL)
    // …nothing was attached above the composer…
    expect(mainComposerScope.$attachments.get()).toEqual([])
    // …and the interception hook was never consulted.
    expect(onAttachPrCommentUrl).not.toHaveBeenCalled()
    expect(event.defaultPrevented).toBe(true)
  })

  it('treats it exactly like a plain URL paste — same insertion, no attachment', () => {
    const onAttachPrCommentUrl = vi.fn(() => true)
    const plain = render(<Harness onAttachPrCommentUrl={onAttachPrCommentUrl} />)
    const deep = render(<Harness onAttachPrCommentUrl={onAttachPrCommentUrl} />)

    const plainEditor = plain.container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
    const deepEditor = deep.container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    pasteInto(plainEditor, 'https://github.com/o/r/pull/1')
    pasteInto(deepEditor, PR_COMMENT_URL)

    // Same shape of result for both URLs: the text is present, nothing is attached.
    expect(composerPlainText(plainEditor)).toContain('https://github.com/o/r/pull/1')
    expect(composerPlainText(deepEditor)).toContain(PR_COMMENT_URL)
    expect(mainComposerScope.$attachments.get()).toEqual([])
    expect(onAttachPrCommentUrl).not.toHaveBeenCalled()
  })
})
