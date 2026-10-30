// @vitest-environment jsdom
// Sending a message closes the `@`, `/` or `:` completion list that was open over its text, so the
// next Enter or Tab acts on the new, empty draft instead of picking a row from the stale list.
//
// Native Desktop: pressing Send with the mouse takes focus from the editor, and that blur closes the
// list, as it always has. The browser-hosted Desktop keeps the editor focused through the press (an
// iOS tap would otherwise drop the on-screen keyboard), so there the send itself closes the list.
//
// jsdom moves no focus on mousedown, so `pressSend` plays the browser's part: a mousedown whose
// default was not prevented moves focus to the button, as Chromium does.
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, createEvent, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { mainComposerScope } from '@/store/composer'
import { setReactionsEnabled } from '@/store/reactions-enabled'

import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

// The bundled emoji index the `:` list reads, cut down to one entry.
const EMOJI_DATA = [{ emoji: '😄', hexcode: '1F604', label: 'grinning face with smiling eyes', tags: ['smile'] }]
const EMOJI_CODES = { '1F604': ['smile'] }

function Harness({ busy = false, onSubmit }: { busy?: boolean; onSubmit: (text: string) => Promise<boolean> }) {
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
            busy={busy}
            disabled={false}
            gateway={null}
            onCancel={vi.fn()}
            onSteer={vi.fn(async () => true)}
            onSubmit={onSubmit}
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

const settle = (ms = 0) => act(() => new Promise<void>(resolve => window.setTimeout(resolve, ms)))

function caretToEnd(editor: HTMLElement) {
  const range = window.document.createRange()
  range.selectNodeContents(editor)
  range.collapse(false)
  const selection = window.getSelection()
  selection?.removeAllRanges()
  selection?.addRange(range)
}

/** Type `text` into a focused, empty composer and wait for the completion list it opens. */
async function openListWhileTyping(text: string, { busy = false } = {}) {
  const onSubmit = vi.fn(async (_text: string) => true)
  const { container } = render(<Harness busy={busy} onSubmit={onSubmit} />)
  const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

  // jsdom does not make contenteditable focusable like Chromium does.
  editor.tabIndex = 0

  await act(async () => {
    editor.focus()
    editor.textContent = text
    caretToEnd(editor)
    fireEvent.input(editor)
    fireEvent.keyUp(editor, { key: text.at(-1) })
  })

  // The file's first `:` waits out the real 60 ms debounce, loads the emoji index and paints the list
  // for the first time: 150-630 ms on a runner with 8x more runnable threads than cores, past
  // vi.waitFor's 1 s default in 2 of 20 runs. Later opens hit the cached index and take ~5 ms.
  await vi.waitFor(() => expect(container.querySelector('[role="listbox"] button')).not.toBeNull(), {
    timeout: 10_000
  })

  const send = container.querySelector<HTMLButtonElement>('button[type="submit"]')!

  return { container, editor, onSubmit, send }
}

/** A mouse press on a composer button, with the browser's focus move on an unprevented mousedown. */
async function pressSend(send: HTMLButtonElement) {
  await act(async () => {
    fireEvent.pointerDown(send)
    const mouseDown = createEvent.mouseDown(send)
    fireEvent(send, mouseDown)

    if (!mouseDown.defaultPrevented) {
      send.focus()
    }

    fireEvent.mouseUp(send)
    fireEvent.click(send)
  })

  // A blur closes the list after a short delay; give it time to run.
  await settle(150)
}

function expectNoStalePick(container: HTMLElement, editor: HTMLElement) {
  expect(container.querySelector('[data-slot="composer-completion-drawer"]')).toBeNull()

  fireEvent.keyDown(editor, { key: 'Enter' })
  fireEvent.keyDown(editor, { key: 'Tab' })

  expect(composerPlainText(editor)).toBe('')
}

beforeEach(() => {
  setReactionsEnabled(true)
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)

      return new Response(JSON.stringify(url.includes('shortcodes') ? EMOJI_CODES : EMOJI_DATA), {
        headers: { 'content-type': 'application/json' }
      })
    })
  )
})

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  setReactionsEnabled(false)
  window.document.documentElement.removeAttribute('data-hermes-desktop-host')
  vi.unstubAllGlobals()
})

describe('Send closes the completion list', () => {
  it('native Desktop: after Send no list is open and Enter or Tab picks nothing', async () => {
    const { container, editor, onSubmit, send } = await openListWhileTyping('hello :smi')

    await pressSend(send)

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit.mock.calls[0][0]).toBe('hello :smi')
    expectNoStalePick(container, editor)
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('native Desktop: a mouse press on Send takes focus from the editor, as it always has', async () => {
    const { send } = await openListWhileTyping('hello :smi')
    const mouseDown = createEvent.mouseDown(send)

    fireEvent(send, mouseDown)

    expect(mouseDown.defaultPrevented).toBe(false)
  })

  it('browser-hosted Desktop: Send keeps the editor focused and still closes the list', async () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    const { container, editor, onSubmit, send } = await openListWhileTyping('hello :smi')
    const mouseDown = createEvent.mouseDown(send)

    fireEvent(send, mouseDown)
    expect(mouseDown.defaultPrevented).toBe(true)

    await pressSend(send)

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(window.document.activeElement).toBe(editor)
    expectNoStalePick(container, editor)
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('browser-hosted Desktop: queueing from the Queue button closes the list too', async () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    const { container, editor } = await openListWhileTyping('later :smi', { busy: true })
    const queue = container.querySelector<HTMLButtonElement>('button[aria-label="Queue message"]')!

    await pressSend(queue)

    expect(container.querySelector('[data-slot="composer-completion-drawer"]')).toBeNull()
    fireEvent.keyDown(editor, { key: 'Tab' })
    expect(composerPlainText(editor)).not.toContain('😄')
  })
})
