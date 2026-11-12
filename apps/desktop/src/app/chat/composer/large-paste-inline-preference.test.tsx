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

// A user who turned "Convert large pastes to attachments" off keeps that
// choice across app restarts: the preference is already in storage when the
// app's modules load (vi.hoisted runs before the imports above), and the very
// first over-threshold paste lands inline. Nothing here touches the toggle's
// store directly; the stored value alone must decide.
vi.hoisted(() => {
  window.localStorage.setItem('hermes.desktop.composer.largePasteAttach', 'false')
})

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
})

const LONG_PASTE = 'y'.repeat(3_500)

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
}

describe('a stored "keep large pastes inline" preference', () => {
  it('keeps the first over-threshold paste after startup inline', () => {
    const onAttachPastedText = vi.fn(() => true)
    const { container } = render(<Harness onAttachPastedText={onAttachPastedText} />)
    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    pasteInto(editor, LONG_PASTE)

    expect(onAttachPastedText).not.toHaveBeenCalled()
    expect(composerPlainText(editor)).toContain(LONG_PASTE)
  })
})
