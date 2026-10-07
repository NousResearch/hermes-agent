// The user bubble's hover cluster carries a Copy button — the mirror of the
// assistant bubble's. The prompt is the one thing users re-send elsewhere, and
// until now copying it meant drag-selecting the bubble by hand.
//
// This covers the React/runtime wiring only: that the affordance renders, that
// clicking it writes the prompt text (not the response) to the clipboard, and
// that the click does NOT open the edit composer (the bubble's own onClick
// rewinds the thread, so a copy that leaked through would be destructive).
import { AssistantRuntimeProvider, ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, userMessage } from '../test-utils'

import { Thread } from '.'
stubThreadEnvironment()
stubThreadViewportSize()

const PROMPT = 'copy this prompt, not the reply'

function Harness({ onEdit }: { onEdit?: (message: never) => Promise<void> }) {
  const repository = ExportedMessageRepository.fromArray([userMessage('user-1', PROMPT), assistantMessage()])

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: repository,
    isRunning: false,
    setMessages: () => {},
    onNew: async () => {},
    onEdit: onEdit as unknown as (message: never) => Promise<void>,
    onCancel: async () => {},
    onReload: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

const writeText = vi.fn<(text: string) => Promise<void>>()

beforeEach(() => {
  writeText.mockClear()
  writeText.mockResolvedValue(undefined)
  Object.assign(navigator, { clipboard: { writeText } })
})

afterEach(() => {
  cleanup()
})

// Both bubbles now carry a Copy button (the assistant one already did), so
// scope the query to the user bubble's own hover cluster. The cluster is the
// absolutely-positioned strip anchored to the bubble's bottom-right corner.
async function findUserCopyButton(): Promise<HTMLElement> {
  const userBubble = await screen.findByText(PROMPT)

  const cluster = userBubble.closest('[data-slot="aui_user-bubble-actions"]')

  if (!cluster) {
    throw new Error('user bubble action bar not found')
  }

  // The cluster's own cluster holds the bubble row; query within the bubble.
  const scope = cluster.querySelector('.pointer-events-none.absolute') ?? cluster

  const button = scope.querySelector<HTMLButtonElement>('button[aria-label="Copy"]')

  if (!button) {
    throw new Error('user bubble copy button not found')
  }

  return button
}

describe('user bubble copy affordance', () => {
  it('copies the prompt text to the clipboard', async () => {
    render(<Harness />)

    const copyButton = await findUserCopyButton()

    await act(async () => {
      fireEvent.click(copyButton)
    })

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(PROMPT))
    // The reply must never be what lands on the clipboard.
    expect(writeText).not.toHaveBeenCalledWith('done')
  })

  it('does not open the edit composer when copying', async () => {
    const onEdit = vi.fn().mockResolvedValue(undefined)

    render(<Harness onEdit={onEdit} />)

    const copyButton = await findUserCopyButton()

    await act(async () => {
      fireEvent.click(copyButton)
    })

    await waitFor(() => expect(writeText).toHaveBeenCalled())
    expect(onEdit).not.toHaveBeenCalled()
  })
})