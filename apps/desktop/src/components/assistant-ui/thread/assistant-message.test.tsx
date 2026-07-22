import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { Thread } from '.'

const createdAt = new Date('2026-05-01T12:34:00.000Z')

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0)
)
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))

Element.prototype.scrollTo = function scrollTo() {}

function assistantMessage(): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text: 'done' }],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function Harness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [assistantMessage()],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('assistant message footer', () => {
  it('keeps the message timestamp visible without opening the more-actions menu', async () => {
    const { container } = render(<Harness />)

    await screen.findByText('done')

    const timestamp = container.querySelector('[data-slot="aui_msg-timestamp"]')

    expect(timestamp).toBeTruthy()
    expect(timestamp?.textContent?.trim()).not.toBe('')
  })
})
