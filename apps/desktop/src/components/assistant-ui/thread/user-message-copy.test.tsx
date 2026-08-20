import { AssistantRuntimeProvider, ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, userMessage } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function Harness({ prompt }: { prompt: string }) {
  const repository = ExportedMessageRepository.fromArray([userMessage('user-1', prompt), assistantMessage()])

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

describe('user message copy action', () => {
  it('copies the exact prompt without opening the edit composer', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    const prompt = 'first line\n`literal code` and 한글'

    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText }
    })

    const { container } = render(<Harness prompt={prompt} />)

    const userMessageRoot = await waitFor(() => {
      const node = container.querySelector('[data-slot="aui_user-message-root"]')
      expect(node).toBeTruthy()

      return node as HTMLElement
    })

    fireEvent.click(within(userMessageRoot).getByRole('button', { name: 'Copy' }))

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(prompt))
    expect(container.querySelector('[data-slot="aui_edit-composer-root"]')).toBeNull()
    expect(screen.getByRole('button', { name: 'Edit message' })).toBeTruthy()
  })
})
