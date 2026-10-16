// @vitest-environment jsdom
/**
 * Esc must not reach the chat while a full page covers it.
 *
 * A page renders as a LAYER above the session view, so the chat stays MOUNTED
 * with a live turn running behind it (`app/contrib/surfaces.tsx`). The
 * composer's editor is therefore alive while the page is on screen, and an
 * ungated Esc on the editor halted that turn — swallowing the key the page
 * needs to go back out of itself. The gate is the same covering-surface signal
 * every other surface-sensitive path reads (`composerFocusBlockedBySurface`).
 */
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $workspaceIsPage } from '@/app/routes'
import { I18nProvider } from '@/i18n'
import { mainComposerScope } from '@/store/composer'

import { RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  $workspaceIsPage.set(false)
})

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness({ onCancel }: { onCancel: () => void }) {
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: true,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <ChatBar
            busy
            disabled={false}
            gateway={null}
            onCancel={onCancel}
            onSubmit={vi.fn(async () => true)}
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

/** Mount a busy composer and focus its editor, the way a user's typing leaves it. */
async function openBusyEditor(onCancel: () => void) {
  const { container } = render(<Harness onCancel={onCancel} />)
  const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

  // jsdom does not make contenteditable focusable like Chromium does.
  editor.tabIndex = 0

  await act(async () => {
    editor.focus()
  })

  return editor
}

describe('Esc on the composer while a full page covers it', () => {
  it('halts the run when the chat is the surface on screen', async () => {
    const onCancel = vi.fn()
    const editor = await openBusyEditor(onCancel)

    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Escape' })
    })

    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('leaves the turn alone while a page is showing, so the page owns Esc', async () => {
    $workspaceIsPage.set(true)
    const onCancel = vi.fn()
    const editor = await openBusyEditor(onCancel)

    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Escape' })
    })

    expect(onCancel).not.toHaveBeenCalled()
  })
})
