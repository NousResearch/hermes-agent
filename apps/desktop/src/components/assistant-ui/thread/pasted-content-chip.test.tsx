// The user's report: pasting a long text sends a "Pasted content" chip, and
// clicking it opened the message editor with nothing in it. This pins the
// end-to-end gesture on the real transcript — the chip opens its reader and the
// edit composer never mounts.
import { AssistantRuntimeProvider, ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, userMessage } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

afterEach(() => {
  cleanup()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

const PASTE_PATH = '/tmp/hermes/pasted_content_20261104_091500_ab12cd.txt'
const PASTE_TEXT = 'the text the user actually pasted'

function Harness({ text }: { text: string }) {
  const repository = ExportedMessageRepository.fromArray([userMessage('user-1', text), assistantMessage()])

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: repository,
    isRunning: false,
    setMessages: () => {},
    onNew: async () => {},
    onEdit: async () => {},
    onCancel: async () => {},
    onReload: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('pasted content chip in the transcript', () => {
  it('opens the paste reader instead of the edit composer', async () => {
    const readFileText = vi.fn(async () => ({ path: PASTE_PATH, text: PASTE_TEXT }))
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { readFileText } })

    const { container } = render(<Harness text={`@file:${PASTE_PATH}`} />)

    // The bubble's own edit affordance is present — the click must not take it.
    expect(await screen.findByRole('button', { name: 'Edit message' })).toBeTruthy()

    const chip = container.querySelector<HTMLElement>('[data-slot="aui_directive-chip"]')

    expect(chip?.tagName).toBe('BUTTON')

    fireEvent.pointerDown(chip!)
    fireEvent.click(chip!)

    expect(await screen.findByRole('dialog')).toBeTruthy()
    await waitFor(() => expect(readFileText).toHaveBeenCalledWith(PASTE_PATH))
    expect(container.querySelector('[data-slot="aui_edit-composer-root"]')).toBeNull()
  })
})
