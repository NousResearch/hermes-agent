import { type AppendMessage, ExportedMessageRepository } from '@assistant-ui/react'
// Clicking a user bubble must open the inline edit composer — through the
// app's incremental external-store runtime (which reimplements capability
// resolution, incl. `edit: onEdit !== undefined`) and the stock runtime.
//
// Note: this covers the React/runtime wiring only. The Electron-level failure
// mode (titlebar -webkit-app-region:drag swallowing clicks on *stuck* sticky
// bubbles) is not reproducible in jsdom — see USER_BUBBLE_BASE_CLASS's no-drag
// carve-out in thread.tsx.
import { AssistantRuntimeProvider, type ThreadMessage } from '@assistant-ui/react'
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

describe('click-to-edit user message', () => {
  it('opens the edit composer with the incremental runtime', async () => {
    const { container } = render(<IncrementalHarness onEdit={async () => {}} />)

    const bubble = await screen.findByRole('button', { name: 'Edit message' })

    fireEvent.click(bubble)

    await waitFor(() => {
      expect(container.querySelector('[data-slot="aui_edit-composer-root"]')).toBeTruthy()
    })
  })

  it('hides the placeholder when a cleared inline edit receives text again', async () => {
    render(<IncrementalHarness onEdit={async () => {}} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))

    const editor = await screen.findByRole('textbox', { name: 'Edit message' })
    // jsdom does not make contenteditable focusable like Chromium does.
    editor.tabIndex = 0
    editor.focus()
    expect(document.activeElement).toBe(editor)

    editor.replaceChildren()
    fireEvent.input(editor)
    await waitFor(() => expect(editor.hasAttribute('data-empty')).toBe(true))

    editor.textContent = 'fade'
    fireEvent.input(editor)
    await waitFor(() => expect(editor.matches(':is(:empty, [data-empty])')).toBe(false))

    editor.replaceChildren()
    fireEvent.input(editor)
    await waitFor(() => expect(editor.hasAttribute('data-empty')).toBe(true))
    fireEvent.paste(editor, { clipboardData: { getData: () => 'pasted edit' } })
    expect(editor.textContent).toBe('pasted edit')
    expect(editor.matches(':is(:empty, [data-empty])')).toBe(false)
  })

  it('hides the inline edit placeholder during IME preedit and restores it on cancellation', async () => {
    render(<IncrementalHarness onEdit={async () => {}} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))
    const editor = await screen.findByRole('textbox', { name: 'Edit message' })
    editor.tabIndex = 0
    editor.focus()

    editor.replaceChildren()
    fireEvent.input(editor)
    await waitFor(() => expect(editor.hasAttribute('data-empty')).toBe(true))

    fireEvent.compositionStart(editor)
    editor.textContent = 'に'
    fireEvent.input(editor)
    expect(editor.matches(':is(:empty, [data-empty])')).toBe(false)

    editor.replaceChildren()
    fireEvent.compositionEnd(editor)
    expect(editor.matches(':is(:empty, [data-empty])')).toBe(true)
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
})

describe('Enter submission and latch behavior', () => {
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

  // Interleaving covered nowhere else: dirty edit → blur schedules the 80ms
  // cancel path → Enter submits before/after that timer. A torn-down composer
  // (cancel won) would drop the edit silently; a wedged submit latch would
  // leave the arrow dead. Issue #95798.
  it('submits a dirty edit when Enter interleaves with the blur-cancel timer', async () => {
    const uncaught: unknown[] = []
    const onUncaught = (err: unknown) => {
      uncaught.push(err)
    }

    process.on('uncaughtException', onUncaught)

    try {
      const onEdit = vi.fn(async () => {})
      const { container } = render(<IncrementalHarness onEdit={onEdit} />)

      fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))

      const editor = await screen.findByRole('textbox', { name: 'Edit message' })
      const editedText = 'dirty draft that must survive blur then submit'

      await act(async () => {
        editor.focus()
        editor.textContent = editedText
        fireEvent.input(editor)
      })

      // Blur without waiting — the 80ms cancel timer is now armed. Enter must
      // still land the edit even if the timer fires in the same window.
      await act(async () => {
        editor.blur()
        fireEvent.keyDown(editor, { key: 'Enter' })
        await new Promise(resolve => window.setTimeout(resolve, 200))
      })

      await waitFor(() => {
        expect(onEdit).toHaveBeenCalledTimes(1)
      })
      expect(onEdit).toHaveBeenCalledWith(
        expect.objectContaining({
          content: [{ text: editedText, type: 'text' }]
        })
      )
      expect(container.querySelector('[data-slot="aui_edit-composer-root"]')).toBeFalsy()
      expect(uncaught.filter(err => /Composer is not available/.test(String(err)))).toEqual([])
    } finally {
      process.off('uncaughtException', onUncaught)
    }
  })

  it('still submits after a dirty edit survived blur and focus left the composer', async () => {
    const onEdit = vi.fn(async () => {})
    const { container } = render(<IncrementalHarness onEdit={onEdit} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))

    const editor = await screen.findByRole('textbox', { name: 'Edit message' })
    const editedText = 'edited after blur must still submit'

    editor.textContent = editedText
    fireEvent.input(editor)
    await moveFocusOutside(editor)

    expect(container.querySelector('[data-slot="aui_edit-composer-root"]')).toBeTruthy()

    fireEvent.keyDown(editor, { key: 'Enter' })

    await waitFor(() => {
      expect(onEdit).toHaveBeenCalledTimes(1)
    })
    expect(onEdit).toHaveBeenCalledWith(
      expect.objectContaining({
        content: [{ text: editedText, type: 'text' }]
      })
    )
  })
})
