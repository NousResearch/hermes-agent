import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { ThreadTimeline } from './timeline'

const { triggerHaptic } = vi.hoisted(() => ({ triggerHaptic: vi.fn() }))

vi.mock('@/lib/haptics', () => ({ triggerHaptic }))

const createdAt = new Date('2026-07-10T00:00:00.000Z')

function userMessage(id: string, text: string): ThreadMessage {
  return {
    id,
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function Harness() {
  const messages = [
    userMessage('user-1', 'First prompt'),
    userMessage('user-2', 'Second prompt'),
    userMessage('user-3', 'Third prompt'),
    userMessage('user:4["quoted"]', 'Fourth prompt')
  ]

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: false,
    messages,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div data-slot="aui_thread-viewport">
        {messages.map(message => (
          <div data-message-id={message.id} key={message.id}>
            {String(message.content)}
          </div>
        ))}
        <ThreadTimeline />
      </div>
    </AssistantRuntimeProvider>
  )
}

describe('ThreadTimeline', () => {
  it('locates message IDs that require CSS selector escaping', () => {
    render(<Harness />)

    const timeline = screen.getByRole('navigation', { name: 'Conversation timeline' })
    const fourthPromptTicks = within(timeline).getAllByRole('button', { name: 'Fourth prompt' })

    fireEvent.click(fourthPromptTicks[0])

    expect(triggerHaptic).toHaveBeenCalledTimes(1)
  })
})
