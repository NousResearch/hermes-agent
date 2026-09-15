import {
  AssistantRuntimeProvider,
  type ThreadMessage,
  ThreadPrimitive,
  useExternalStoreRuntime
} from '@assistant-ui/react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { SystemMessage } from '@/components/assistant-ui/thread/system-message'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'

import { renderMessageStream } from './test-harness'

const SID = 'recalling-session'

function RecallTranscript({ messages }: { messages: ChatMessage[] }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: messages.map(toRuntimeMessage),
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root>
        <ThreadPrimitive.Messages components={{ Message: SystemMessage }} />
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  )
}

afterEach(cleanup)

describe('memory recall in the Desktop transcript', () => {
  it.each([
    ['memory_recall', 'Provider without a fixed label: context loaded'],
    ['lifecycle', '🧠 Hindsight — recalled 1 memory'],
    ['lifecycle', '👁️ Hindsight — recalled 3 memories'],
    ['lifecycle', '🧠 Honcho — recalled relevant memory'],
    ['lifecycle', '🧠 Notes — recalled 2 memories  🧠 Provider — recalled 5 memories']
  ])('renders %s in the originating chat and keeps it after completion: %s', (kind, text) => {
    // Focus is elsewhere: background sessions must keep their own recall row.
    const stream = renderMessageStream('other-session')

    act(() => {
      stream.handleEvent({ type: 'message.start', session_id: SID })
      stream.handleEvent({ type: 'status.update', session_id: SID, payload: { kind, text, timestamp: 123.5 } })
      stream.handleEvent({ type: 'message.delta', session_id: SID, payload: { text: 'An answer.' } })
      stream.handleEvent({
        type: 'status.update',
        session_id: SID,
        payload: { kind: 'lifecycle', text: 'An unrelated lifecycle diagnostic.' }
      })
      stream.handleEvent({ type: 'message.complete', session_id: SID, payload: { text: 'An answer.' } })
    })

    const messages = stream.state(SID).messages
    expect(messages.map(message => message.role)).toEqual(['system', 'assistant'])
    expect(messages.map(chatMessageText)).toEqual([text, 'An answer.'])
    expect(messages[0].timestamp).toBe(123.5)
    expect(messages[0].parts[0].timestamp).toBe(123.5)
    expect(stream.state('other-session').messages).toEqual([])
    expect(stream.state(SID).busy).toBe(false)

    render(<RecallTranscript messages={messages.filter(message => message.role === 'system')} />)
    expect(screen.getByText(text.replace(/\s+/g, ' ')).closest('[data-role="system"]')).not.toBeNull()
  })

  it('ignores empty/malformed/non-recall statuses without changing turn state', () => {
    const stream = renderMessageStream(SID)
    act(() => stream.handleEvent({ type: 'message.start', session_id: SID }))
    const before = stream.state(SID)

    act(() => {
      for (const payload of [
        { kind: 'memory_recall', text: '   ' },
        { kind: 'memory_recall' },
        { kind: 'memory_recall', text: { text: 'not a status string' } },
        { kind: 'lifecycle', text: 'Loading memory provider...' },
        { kind: 'lifecycle', text: '🧠 Provider — recalled 3 memories\nprivate context' },
        { kind: 'warn', text: '🧠 Provider — recalled 3 memories' }
      ]) {
        stream.handleEvent({ type: 'status.update', session_id: SID, payload })
      }
    })

    expect(stream.state(SID)).toBe(before)
    expect(stream.state(SID).messages).toEqual([])
  })
})
