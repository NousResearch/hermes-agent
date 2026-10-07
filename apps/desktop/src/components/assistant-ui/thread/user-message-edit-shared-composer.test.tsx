// @vitest-environment jsdom
// The inline edit composer opens inside a chat pane whose composer surface is
// registered with the shared floating-composer recipient tracker
// (`floating-target.ts`). That tracker redirects stray focus to the pane's
// composer; the edit editor's mount-time focus and the user's mouse movement
// while editing must not count as stray, or the pane composer steals the caret
// and the edit's blur guard cancels the edit right after it opened (#112935).
import { AssistantRuntimeProvider, ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { FloatingComposerSurface } from '@/app/chat/composer/floating-surface'
import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { PaneGroupContext, PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, userMessage } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

// jsdom does not make contenteditable focusable like Chromium does; give the
// production editor a tab stop the moment production code tries to focus it.
beforeAll(() => {
  const nativeFocus = HTMLElement.prototype.focus

  HTMLElement.prototype.focus = function focus(options?: FocusOptions) {
    if (this.getAttribute('contenteditable') === 'true' && !this.hasAttribute('tabindex')) {
      this.tabIndex = 0
    }

    nativeFocus.call(this, options)
  }
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

const noopAsync = async () => {}
const EDIT_ROOT = '[data-slot="aui_edit-composer-root"]'

// Mirrors chat/index.tsx: the transcript and the pane's composer share one
// `[data-chat-surface]`, and the composer mounts through FloatingComposerSurface.
function Harness() {
  const [repository] = useState(() => ExportedMessageRepository.fromArray([userMessage(), assistantMessage()]))

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: repository,
    isRunning: false,
    setMessages: () => {},
    onNew: noopAsync,
    onEdit: noopAsync,
    onCancel: noopAsync,
    onReload: noopAsync
  })

  return (
    <PaneGroupContext value="group-1">
      <PaneVisibleContext value>
        <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target: 'main' }}>
          <ComposerSurfaceProvider value="surface-1">
            <div data-chat-surface="" data-composer-surface-id="surface-1" data-tree-group="group-1">
              <AssistantRuntimeProvider runtime={runtime}>
                <Thread cwd={null} gateway={null} sessionId="session-1" />
              </AssistantRuntimeProvider>
              <FloatingComposerSurface>
                <div aria-label="Message" contentEditable data-slot="composer-rich-input" role="textbox" tabIndex={0} />
              </FloatingComposerSurface>
            </div>
          </ComposerSurfaceProvider>
        </ComposerScopeProvider>
      </PaneVisibleContext>
    </PaneGroupContext>
  )
}

async function openEdit() {
  const view = render(<Harness />)
  const main = screen.getByRole('textbox', { name: 'Message' })
  main.focus()

  const bubble = await screen.findByRole('button', { name: 'Edit message' })

  await act(async () => {
    fireEvent.pointerDown(bubble, { button: 0 })
    // jsdom does not move focus on a synthetic pointerdown; the explicit blur
    // stands in for Chromium's mousedown focus leaving the pane composer.
    main.blur()
    fireEvent.pointerUp(bubble, { button: 0 })
    fireEvent.click(bubble)
  })

  const editor = await screen.findByRole('textbox', { name: 'Edit message' })

  return { ...view, editor, main }
}

const settleBlurGuard = () =>
  act(async () => {
    await new Promise(resolve => window.setTimeout(resolve, 200))
  })

describe('inline edit inside a shared composer surface', () => {
  it('keeps the edit composer open and focused after the bubble click', async () => {
    const { container, editor } = await openEdit()

    await settleBlurGuard()

    expect(container.querySelector(EDIT_ROOT)).toBeTruthy()
    expect(window.document.activeElement).toBe(editor)
  })

  it('does not hand the caret back to the pane composer when the mouse moves over the pane', async () => {
    const { container, editor } = await openEdit()
    const surface = container.querySelector<HTMLElement>('[data-chat-surface]')!

    await act(async () => {
      fireEvent.pointerMove(surface, { clientX: 120, clientY: 80 })
      fireEvent.pointerMove(surface, { clientX: 140, clientY: 96 })
    })

    await settleBlurGuard()

    expect(container.querySelector(EDIT_ROOT)).toBeTruthy()
    expect(window.document.activeElement).toBe(editor)
  })
})
