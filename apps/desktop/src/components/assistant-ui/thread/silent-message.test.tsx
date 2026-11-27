import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { useRuntimeMessageRepository } from '@/app/chat/runtime-repository'
import type { ChatMessage } from '@/lib/chat-messages'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
afterEach(cleanup)

function Harness({ messages }: { messages: ChatMessage[] }) {
  const repository = useRuntimeMessageRepository(messages)

  const [headId, setHeadId] = useState<string | null>(null)

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messageRepository: headId ? { ...repository, headId } : repository,
    setMessages: next => setHeadId(next.at(-1)?.id ?? null),
    isRunning: messages.some(message => Boolean(message.pending)),
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('silent assistant message', () => {
  it('leaves no empty assistant landmark while keeping another branch reachable', async () => {
    const view = render(
      <Harness
        messages={[
          { id: 'user', role: 'user', parts: [{ text: 'Check', type: 'text' }] },
          {
            branchGroupId: 'alternatives',
            id: 'answer',
            role: 'assistant',
            parts: [{ text: 'Previous useful answer', type: 'text' }]
          },
          {
            branchGroupId: 'alternatives',
            id: 'silent',
            role: 'assistant',
            parts: [{ text: '[[SILENT]]', type: 'text' }]
          }
        ]}
      />
    )

    const counter = await view.findByText('2 / 2')

    expect(view.container.textContent).not.toContain('[[SILENT]]')
    expect(view.container.querySelector('[data-role="assistant"]')).toBeNull()
    fireEvent.click(counter.parentElement!.querySelector('button')!)
    await view.findByText('Previous useful answer')
  })
})
