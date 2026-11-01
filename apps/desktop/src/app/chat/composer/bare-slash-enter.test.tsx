// @vitest-environment jsdom
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { mainComposerScope } from '@/store/composer'

import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarProps, ChatBarState } from './types'

import { ChatBar } from './index'

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
})

// THE INVARIANT: Enter on a bare `/` with no arrowed pick sends nothing (#139774).
//
// `implicitSlashAcceptIndex` deliberately returns null for a bare `/` the user
// has not arrowed through (#98590), so Enter accepts no row. The keypress then
// fell through to the submit branch, where a lone `/` passes the non-empty
// payload check, and the trigger token itself went out as a message.
//
// Driven through the REAL ChatBar: a gateway stub answers `commands.catalog` so
// the popover has rows, the editor holds `/` with the caret after it, and Enter
// is a real keydown on the contentEditable.
const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

const gateway = {
  request: vi.fn(async (method: string) =>
    method === 'commands.catalog'
      ? {
          pairs: [
            ['/new', 'Start a new session'],
            ['/help', 'Show help']
          ]
        }
      : {}
  )
} as unknown as HermesGateway

function Harness({ onSubmit }: { onSubmit: ChatBarProps['onSubmit'] }) {
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
            gateway={gateway}
            onCancel={vi.fn()}
            onSubmit={onSubmit}
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

/** Put `text` in the editor with the caret at its end, as typing would. */
function typeInto(editor: HTMLElement, text: string) {
  Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })
  editor.focus()
  editor.textContent = text

  const range = editor.ownerDocument.createRange()
  range.selectNodeContents(editor)
  range.collapse(false)
  const selection = window.getSelection()
  selection?.removeAllRanges()
  selection?.addRange(range)

  act(() => {
    fireEvent.input(editor)
  })
}

describe('Enter on a bare `/`', () => {
  it('sends nothing when no row was arrowed to', async () => {
    const onSubmit = vi.fn<ChatBarProps['onSubmit']>(async () => true)
    const { container } = render(<Harness onSubmit={onSubmit} />)
    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    typeInto(editor, '/')

    // The picker is open with rows, so Enter goes through the accept branch.
    await waitFor(() =>
      expect(editor.ownerDocument.querySelector('[data-slot="composer-completion-drawer"]')?.textContent).toContain('/new')
    )

    act(() => {
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(composerPlainText(editor)).toBe('/')
  })

  // Shift+Enter never sends, and the picker must not swallow it either: it
  // falls through to the browser's default, a line break after the `/`.
  it('leaves Shift+Enter to insert a line break', async () => {
    const onSubmit = vi.fn<ChatBarProps['onSubmit']>(async () => true)
    const { container } = render(<Harness onSubmit={onSubmit} />)
    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    typeInto(editor, '/')

    await waitFor(() =>
      expect(editor.ownerDocument.querySelector('[data-slot="composer-completion-drawer"]')?.textContent).toContain('/new')
    )

    let notPrevented = false
    act(() => {
      notPrevented = fireEvent.keyDown(editor, { key: 'Enter', shiftKey: true })
    })

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(notPrevented).toBe(true)
    expect(onSubmit).not.toHaveBeenCalled()
  })
})
