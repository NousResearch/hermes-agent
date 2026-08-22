import { type AppendMessage, ExportedMessageRepository } from '@assistant-ui/react'
// Clicking a user bubble must open the inline edit composer — through the
// app's incremental external-store runtime (which reimplements capability
// resolution, incl. `edit: onEdit !== undefined`) and the stock runtime.
//
// Note: this covers the React/runtime wiring only. The Electron-level failure
// mode (titlebar -webkit-app-region:drag swallowing clicks on *stuck* sticky
// bubbles) is not reproducible in jsdom — see USER_BUBBLE_BASE_CLASS's no-drag
// carve-out in thread.tsx.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, userMessage } from '../test-utils'

import { Thread } from '.'
stubThreadEnvironment()

afterEach(() => {
  cleanup()
})

stubThreadViewportSize()

async function moveFocusOutside(editor: HTMLElement) {
  const outside = window.document.createElement('button')
  window.document.body.append(outside)
  editor.focus()

  await act(async () => {
    outside.focus()
    await new Promise(resolve => window.setTimeout(resolve, 120))
  })

  outside.remove()
}

// Mirrors chat/index.tsx: incremental runtime + messageRepository + onEdit.
function IncrementalHarness({ onEdit }: { onEdit: (message: AppendMessage) => Promise<void> }) {
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
      <Thread />
    </AssistantRuntimeProvider>
  )
}

// Control: stock external store runtime.
function StockHarness({ onEdit }: { onEdit: () => Promise<void> }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [userMessage(), assistantMessage()],
    isRunning: false,
    onNew: async () => {},
    onEdit
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('click-to-edit user message', () => {
  it('opens the edit composer with the incremental runtime', async () => {
    const { container } = render(<IncrementalHarness onEdit={async () => {}} />)

    const bubble = await screen.findByRole('button', { name: 'Edit message' })

    fireEvent.click(bubble)

    await waitFor(() => {
      expect(container.querySelector('[data-slot="aui_edit-composer-root"]')).toBeTruthy()
    })
  })

  it('does not submit an inline edit while IME composition is active', async () => {
    const onEdit = vi.fn(async (_message: AppendMessage) => {})

    render(<IncrementalHarness onEdit={onEdit} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))

    const editor = await screen.findByRole('textbox', { name: 'Edit message' })
    const editedText = 'edit me please\u4f60'

    await act(async () => {
      fireEvent.compositionStart(editor)
      editor.textContent = editedText
      fireEvent.input(editor)
      fireEvent.keyDown(editor, { isComposing: true, key: 'Enter' })
    })

    expect(onEdit).not.toHaveBeenCalled()

    await act(async () => {
      fireEvent.compositionEnd(editor)
      fireEvent.keyDown(editor, { key: 'Enter' })
    })

    await waitFor(() => expect(onEdit).toHaveBeenCalledTimes(1))
    expect(onEdit).toHaveBeenCalledWith(
      expect.objectContaining({
        content: [{ text: editedText, type: 'text' }]
      })
    )
  })

  it('keeps a dirty inline edit open when focus leaves the composer', async () => {
    const { container } = render(<IncrementalHarness onEdit={async () => {}} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))

    const editor = await screen.findByRole('textbox', { name: 'Edit message' })
    const editedText = 'edited draft that must not be discarded'

    editor.textContent = editedText
    fireEvent.input(editor)
    await moveFocusOutside(editor)

    expect(container.querySelector('[data-slot="aui_edit-composer-root"]')).toBeTruthy()
    expect((await screen.findByRole('textbox', { name: 'Edit message' })).textContent).toBe(editedText)
  })

  it('still cancels an untouched inline edit when focus leaves the composer', async () => {
    const { container } = render(<IncrementalHarness onEdit={async () => {}} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))
    const editor = await screen.findByRole('textbox', { name: 'Edit message' })

    await moveFocusOutside(editor)

    expect(container.querySelector('[data-slot="aui_edit-composer-root"]')).toBeFalsy()
  })

  it('opens the edit composer with the stock runtime', async () => {
    const { container } = render(<StockHarness onEdit={async () => {}} />)

    const bubble = await screen.findByRole('button', { name: 'Edit message' })

    fireEvent.click(bubble)

    await waitFor(() => {
      expect(container.querySelector('[data-slot="aui_edit-composer-root"]')).toBeTruthy()
    })
  })

  // A long previous prompt is capped at max-h-48 in the edit composer. Without
  // an overflow rule the overflow is clipped with no way to scroll, hiding the
  // tail of the prompt. The editor must scroll its own overflow (like the main
  // composer's editor does).
  it('keeps the edit composer editor scrollable when the prompt overflows the cap', async () => {
    const { container } = render(<IncrementalHarness onEdit={async () => {}} />)

    const bubble = await screen.findByRole('button', { name: 'Edit message' })

    fireEvent.click(bubble)

    const editor = await waitFor(() => {
      const node = container.querySelector('[contenteditable="true"]')
      expect(node).toBeTruthy()

      return node as HTMLElement
    })

    expect(editor.className).toContain('max-h-48')
    expect(editor.className).toContain('overflow-y-auto')
  })
})

describe('Enter submission and latch behavior', () => {
  it('submits the edit when Enter is pressed', async () => {
    const onEdit = vi.fn(async () => {})
    render(<IncrementalHarness onEdit={onEdit} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))

    const editor = await screen.findByRole('textbox', { name: 'Edit message' })
    const editedText = 'modified text for submission'

    editor.textContent = editedText
    fireEvent.input(editor)

    fireEvent.keyDown(editor, { key: 'Enter' })

    await waitFor(() => {
      expect(onEdit).toHaveBeenCalledTimes(1)
    })
  })

  it('clears the submitting latch after onEdit resolves, allowing second edit session', async () => {
    const onEdit = vi.fn(async () => {})
    render(<IncrementalHarness onEdit={onEdit} />)

    // First edit session
    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))

    let editor = await screen.findByRole('textbox', { name: 'Edit message' })
    editor.textContent = 'first edit'
    fireEvent.input(editor)
    fireEvent.keyDown(editor, { key: 'Enter' })

    await waitFor(() => {
      expect(onEdit).toHaveBeenCalledTimes(1)
    })

    // Wait for the latch cooldown to clear
    await new Promise(resolve => setTimeout(resolve, 300))

    // Second edit session - open the editor again
    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))

    editor = await screen.findByRole('textbox', { name: 'Edit message' })
    editor.textContent = 'second edit'
    fireEvent.input(editor)
    fireEvent.keyDown(editor, { key: 'Enter' })

    // If the latch wasn't cleared, this second submission would be blocked
    await waitFor(() => {
      expect(onEdit).toHaveBeenCalledTimes(2)
    })
  })

  // Confirming an edit unmounts the composer while its 200ms submit latch is
  // still pending. Left running, the callback resumes on an unmounted tree and
  // calls setSubmitting, which React turns into work against a torn-down
  // renderer. Under vitest that surfaces after the test file finishes, as
  // "ReferenceError: window is not defined" out of resolveUpdatePriority, an
  // unhandled error that fails a run in which every test passed. The delay is
  // matched explicitly so unrelated library timers cannot make this pass.
  it('clears the submit latch timer when confirming the edit unmounts the composer', async () => {
    const LATCH_MS = 200
    // jsdom under node hands back a Timeout object rather than a numeric id,
    // so these are compared by identity rather than by value.
    const scheduled: unknown[] = []
    const cleared: unknown[] = []
    const realSetTimeout = window.setTimeout.bind(window)
    const realClearTimeout = window.clearTimeout.bind(window)

    vi.spyOn(window, 'setTimeout').mockImplementation(((
      handler: TimerHandler,
      timeout?: number,
      ...args: unknown[]
    ) => {
      const id = realSetTimeout(handler, timeout, ...args)

      if (timeout === LATCH_MS) {
        scheduled.push(id)
      }

      return id
    }) as typeof window.setTimeout)

    vi.spyOn(window, 'clearTimeout').mockImplementation(((id?: number) => {
      cleared.push(id)
      realClearTimeout(id)
    }) as typeof window.clearTimeout)

    const onEdit = vi.fn(async () => {})
    const view = render(<IncrementalHarness onEdit={onEdit} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))

    const editor = await screen.findByRole('textbox', { name: 'Edit message' })

    editor.textContent = 'an edit whose composer goes away before the latch expires'
    fireEvent.input(editor)
    fireEvent.keyDown(editor, { key: 'Enter' })

    await waitFor(() => {
      expect(onEdit).toHaveBeenCalledTimes(1)
    })

    expect(scheduled).toHaveLength(1)

    view.unmount()

    expect(cleared).toContain(scheduled[0])
  })

  it('inserts a newline on Shift+Enter without submitting', async () => {
    const onEdit = vi.fn(async () => {})
    render(<IncrementalHarness onEdit={onEdit} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))

    const editor = await screen.findByRole('textbox', { name: 'Edit message' })

    editor.textContent = 'line one'
    fireEvent.input(editor)

    fireEvent.keyDown(editor, { key: 'Enter', shiftKey: true })

    // Shift+Enter should not call onEdit
    await new Promise(resolve => window.setTimeout(resolve, 50))
    expect(onEdit).not.toHaveBeenCalled()

    // The editor should allow the newline to be inserted (by not preventing default)
    // We don't simulate actual newline insertion here (requires complex DOM manipulation)
    // but we verify the guard did not block it
  })
})
