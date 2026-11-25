// @vitest-environment jsdom
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { mainComposerScope } from '@/store/composer'

import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

afterEach(cleanup)

// THE INVARIANT: Ctrl/Cmd+Shift+V is the plain-text escape hatch — what the
// web chat's Ctrl+Shift+V does. The regular paste pipeline chips links
// (linkifyUrls) and promotes paths (pathifyRefs) and attaches images; the
// shortcut must insert the raw clipboard text verbatim and trigger none of
// those conversions. The paste EVENT carries no modifier state, so the
// intercept lives on the editor's keydown — which is also why the harness
// fires keydown, not paste.
//
// jsdom has no async clipboard, so `navigator.clipboard.readText` is stubbed;
// the Electron bridge (`window.hermesDesktop.readClipboard`) is the production
// first choice and is exercised by the second test.

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness() {
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
          <ChatBar busy={false} disabled={false} gateway={null} onCancel={vi.fn()} onSubmit={vi.fn(async () => true)} state={state} />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

function focusEditor(container: HTMLElement): HTMLElement {
  const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

  // jsdom does not implement isContentEditable; pin it like the sibling paste tests.
  Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })
  editor.focus()

  return editor
}

describe('Ctrl/Cmd+Shift+V pastes plain text', () => {
  afterEach(() => {
    mainComposerScope.clear()

    if ('clipboard' in navigator) {
      // @ts-expect-error test stub teardown
      delete navigator.clipboard
    }

    delete (window as unknown as Record<string, unknown>).hermesDesktop
  })

  it('keeps a pasted URL literal — no @url: chip, no conversion', async () => {
    const url = 'https://github.com/o/r/pull/1#issuecomment-2'

    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { readText: async () => url }
    })

    const { container } = render(<Harness />)
    const editor = focusEditor(container)

    let prevented = false

    await act(async () => {
      const event = fireEvent.keyDown(editor, { altKey: false, ctrlKey: true, key: 'v', shiftKey: true })

      prevented = !event
    })

    await waitFor(() => {
      expect(composerPlainText(editor)).toContain(url)
    })

    // The regular paste pipeline would chip this into `@url:` — the shortcut
    // must keep the characters the user copied.
    expect(composerPlainText(editor)).not.toContain('@url:')
    expect(prevented).toBe(true)
  })

  it('reads the Electron bridge first and only falls back to navigator.clipboard', async () => {
    const readClipboard = vi.fn(async () => 'veio pela ponte')

    ;(window as unknown as Record<string, unknown>).hermesDesktop = { readClipboard }

    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { readText: async () => 'veio do navigator' }
    })

    const { container } = render(<Harness />)

    const editor = focusEditor(container)

    await act(async () => {
      fireEvent.keyDown(editor, { altKey: false, metaKey: true, key: 'V', shiftKey: true })
    })

    await waitFor(() => {
      expect(composerPlainText(editor)).toContain('veio pela ponte')
    })

    expect(composerPlainText(editor)).not.toContain('veio do navigator')
    expect(readClipboard).toHaveBeenCalledTimes(1)
  })
})
