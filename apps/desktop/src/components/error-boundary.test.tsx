import {
  AssistantRuntimeProvider,
  ThreadPrimitive,
  type ThreadMessage,
  useAuiState,
  useExternalStoreRuntime
} from '@assistant-ui/react'
import { render, screen, waitFor } from '@testing-library/react'
import { useEffect, useState } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { ErrorBoundary } from './error-boundary'

const createdAt = new Date('2026-05-01T00:00:00.000Z')

function assistantMessage(index: number): ThreadMessage {
  return {
    id: `message-${index}`,
    role: 'assistant',
    content: [{ type: 'text', text: `message-${index}` }],
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

function messageList(count: number) {
  return Array.from({ length: count }, (_, index) => assistantMessage(index))
}

function AssistantMessageProbe() {
  const messageId = useAuiState(s => s.message.id)

  return <div>{messageId}</div>
}

function UserMessageProbe() {
  const messageId = useAuiState(s => s.message.id)

  return <div>{messageId}</div>
}

function TransientTapClientLookupScenario() {
  const [messages, setMessages] = useState<ThreadMessage[]>(() => messageList(24))
  const [visibleIndex, setVisibleIndex] = useState(23)

  useEffect(() => {
    setMessages(messageList(18))

    const timer = window.setTimeout(() => {
      setVisibleIndex(17)
    }, 0)

    return () => {
      window.clearTimeout(timer)
    }
  }, [])

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: false,
    messages,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ErrorBoundary label="root">
        <ThreadPrimitive.MessageByIndex
          components={{ AssistantMessage: AssistantMessageProbe, UserMessage: UserMessageProbe }}
          index={visibleIndex}
        />
      </ErrorBoundary>
    </AssistantRuntimeProvider>
  )
}

describe('ErrorBoundary', () => {
  it('auto-recovers the root boundary from tapClientLookup stale-index races', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})

    render(<TransientTapClientLookupScenario />)

    await waitFor(() => {
      expect(screen.queryByRole('button', { name: 'Reload window' })).toBeNull()
      expect(screen.getByText('message-17')).toBeTruthy()
    })

    const errorLines = errorSpy.mock.calls.flatMap(call =>
      call.map(value => {
        if (value instanceof Error) {
          return value.message
        }

        return typeof value === 'string' ? value : String(value)
      })
    )

    expect(errorLines.some(line => line.includes('tapClientLookup: Index 23 out of bounds (length: 18)'))).toBe(true)
    expect(
      warnSpy.mock.calls.some(call =>
        call.some(value => String(value).includes('auto-recovering from transient render error'))
      )
    ).toBe(true)
  })
})
