// @vitest-environment jsdom
// The main chat composer shares useComposerUndo with the edit composer and has
// the same two blind spots: Chromium runs a cut and an intra-editor text drag
// without a React-visible beforeinput (deleteByCut / insertFromDrop never reach
// the onBeforeInput polyfill), so the pre-edit snapshot has to be banked from
// the native `cut` / `drop` events instead — or ⌘Z skips the cut entirely
// (#115462, main-composer sibling of the edit-composer fix).
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

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
})

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness() {
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
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

async function openComposer() {
  const { container } = render(<Harness />)
  const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

  // jsdom does not make contenteditable focusable like Chromium does.
  editor.tabIndex = 0

  await act(async () => {
    editor.focus()
    editor.textContent = 'one two three four'
    fireEvent.input(editor)
  })

  return editor
}

function select(editor: HTMLElement, start: number, end: number) {
  const text = editor.firstChild!

  if (text.nodeType !== Node.TEXT_NODE) {
    throw new Error('expected a text node editor')
  }

  const range = window.document.createRange()
  range.setStart(text, start)
  range.setEnd(text, end)
  const selection = window.getSelection()
  selection?.removeAllRanges()
  selection?.addRange(range)
}

async function undo(editor: HTMLElement) {
  await act(async () => {
    editor.focus()
    fireEvent.keyDown(editor, { key: 'z', metaKey: true })
  })
}

// Chromium's cut: a cancelable `cut` event, then the DOM mutation, then
// `input` deleteByCut (jsdom has no editing pipeline, so the mutation is
// applied where the pipeline would).
async function cutSelection(editor: HTMLElement) {
  await act(async () => {
    fireEvent.cut(editor)
    editor.textContent = editor.textContent!.replace(' three', '')
    fireEvent.input(editor)
  })
}

// Chromium's intra-editor drag-move: a plain-text `drop` (no attachment MIME),
// then the mutation, then `input` insertFromDrop.
async function dropText(editor: HTMLElement, dragged: string) {
  await act(async () => {
    fireEvent.drop(editor, { dataTransfer: { types: ['text/plain'], files: [], items: [], getData: () => dragged } })
    editor.textContent = editor.textContent!.replace(` ${dragged}`, '') + ` ${dragged}`
    fireEvent.input(editor)
  })
}

describe('main composer undo — cut and drag coverage', () => {
  it('undo restores text removed by a cut', async () => {
    const editor = await openComposer()

    select(editor, 8, 14) // " three"
    await cutSelection(editor)
    expect(editor.textContent).toBe('one two four')

    await undo(editor)
    expect(editor.textContent).toBe('one two three four')
  })

  it('undo restores text after a drag-move within the editor', async () => {
    const editor = await openComposer()

    select(editor, 9, 12) // "two"
    await dropText(editor, 'two')
    expect(editor.textContent).toBe('one three four two')

    await undo(editor)
    expect(editor.textContent).toBe('one two three four')
  })
})
