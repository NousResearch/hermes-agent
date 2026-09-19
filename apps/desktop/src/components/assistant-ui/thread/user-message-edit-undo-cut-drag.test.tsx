// The edit composer's undo stack misses the two mutation paths Chromium runs
// without a React-visible beforeinput: cutting selected text and dragging a
// selection within the editor. Chromium fires the native `cut` / `drop`
// clipboard events before mutating the DOM, and React 19's onBeforeInput
// polyfill (keypress/textInput/paste) never sees the deleteByCut /
// insertFromDrop input types at all — so handleBeforeInput cannot bank the
// pre-edit snapshot for them. Undo then skips the cut entirely or steps a
// paste back only to the post-cut state, the two gaps reported in #115462.
import { type AppendMessage, AssistantRuntimeProvider, ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

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
})

function Harness({ onEdit }: { onEdit: (message: AppendMessage) => Promise<void> }) {
  const repository = ExportedMessageRepository.fromArray([userMessage(), assistantMessage()])

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: repository,
    isRunning: false,
    setMessages: () => {},
    onNew: async () => {},
    onEdit,
    onCancel: async () => {},
    onReload: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread cwd={null} gateway={null} sessionId="session-1" />
    </AssistantRuntimeProvider>
  )
}

async function openEditor() {
  render(<Harness onEdit={vi.fn(async () => {})} />)

  fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))
  const editor = await screen.findByRole('textbox', { name: 'Edit message' })

  await act(async () => {
    editor.focus()
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
// `input` deleteByCut (probe-verified on real Chromium; jsdom has no editing
// pipeline, so the mutation is applied where the pipeline would).
async function cutSelection(editor: HTMLElement) {
  await act(async () => {
    fireEvent.cut(editor)
    editor.textContent = editor.textContent!.replace(' three', '')
    fireEvent.input(editor)
  })
}

// Chromium's intra-editor drag-move: a cancelable `drop` event carrying the
// dragged text, then the mutation, then `input` insertFromDrop.
async function dropText(editor: HTMLElement, dragged: string) {
  await act(async () => {
    fireEvent.drop(editor)
    editor.textContent = editor.textContent!.replace(` ${dragged}`, '') + ` ${dragged}`
    fireEvent.input(editor)
  })
}

describe('edit composer undo — cut and drag coverage', () => {
  it('undo restores text removed by a cut', async () => {
    const editor = await openEditor()

    await act(async () => {
      editor.textContent = 'one two three four'
      fireEvent.input(editor)
    })

    select(editor, 8, 14) // " three"
    await cutSelection(editor)
    expect(editor.textContent).toBe('one two four')

    await undo(editor)
    expect(editor.textContent).toBe('one two three four')
  })

  it('undo restores text after a drag-move within the editor', async () => {
    const editor = await openEditor()

    await act(async () => {
      editor.textContent = 'one two three four'
      fireEvent.input(editor)
    })

    select(editor, 9, 12) // "two"
    await dropText(editor, 'two')
    expect(editor.textContent).toBe('one three four two')

    await undo(editor)
    expect(editor.textContent).toBe('one two three four')
  })

  it('undoing a paste after a cut steps back past the cut as well', async () => {
    const editor = await openEditor()

    await act(async () => {
      editor.textContent = 'one two three four'
      fireEvent.input(editor)
    })

    select(editor, 8, 14) // " three"
    await cutSelection(editor)

    // Paste a replacement (the paste path already banks its own undo point).
    await act(async () => {
      fireEvent.paste(editor, { clipboardData: { getData: () => ' five' } })
    })

    await undo(editor) // undo the paste
    expect(editor.textContent).toBe('one two four')

    await undo(editor) // undo the cut — the gap reported in #115462
    expect(editor.textContent).toBe('one two three four')
  })
})
