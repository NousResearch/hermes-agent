import { AssistantRuntimeProvider, MessagePrimitive, ThreadPrimitive } from '@assistant-ui/react'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { type ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useRuntimeMessageRepository } from '@/app/chat/runtime-repository'
import type { ChatMessage } from '@/lib/chat-messages'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'

import { MESSAGE_PARTS_COMPONENTS } from './message-parts'

vi.mock('@/components/assistant-ui/transcript-code-block', () => ({
  TranscriptCodeBlockScope: ({ children, partIndex }: { children: ReactNode; partIndex: number }) => (
    <div data-part-index={partIndex} data-testid="transcript-part-index">
      {children}
    </div>
  )
}))

vi.mock('@/components/assistant-ui/markdown-text', () => ({ MarkdownText: () => null }))
vi.mock('@/components/assistant-ui/thread/timeline-timestamp', () => ({ TimelineTimestamp: () => null }))

afterEach(() => cleanup())

function NativeAuiRuntime({ children, messages }: { children: ReactNode; messages: ChatMessage[] }) {
  const messageRepository = useRuntimeMessageRepository(messages)
  const runtime = useIncrementalExternalStoreRuntime({ messageRepository, isRunning: false, onNew: async () => {} })

  return <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>
}

describe('native assistant-ui message part indexes', () => {
  it('scopes assistant text by native part index and leaves user text unscoped', async () => {
    const message: ChatMessage = {
      id: 'assistant-native-parts',
      role: 'assistant',
      parts: [
        { type: 'text', text: 'First transcript segment.' },
        { type: 'reasoning', text: 'Hidden reasoning part.' },
        { type: 'text', text: 'Second transcript segment.' }
      ]
    }

    const AssistantMessage = () => (
      <MessagePrimitive.Root>
        <MessagePrimitive.Parts
          components={{
            ...MESSAGE_PARTS_COMPONENTS,
            Reasoning: () => null
          }}
        />
      </MessagePrimitive.Root>
    )

    render(
      <NativeAuiRuntime
        messages={[
          { id: 'user-example', role: 'user', parts: [{ type: 'text', text: 'A user fence stays code.' }] },
          message
        ]}
      >
        <ThreadPrimitive.Root>
          <ThreadPrimitive.Messages components={{ AssistantMessage, UserMessage: AssistantMessage }} />
        </ThreadPrimitive.Root>
      </NativeAuiRuntime>
    )

    await waitFor(() => expect(screen.getAllByTestId('transcript-part-index')).toHaveLength(2))

    expect(screen.getAllByTestId('transcript-part-index').map(node => node.getAttribute('data-part-index'))).toEqual([
      '0',
      '2'
    ])
  })
})
