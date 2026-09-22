// The gap case: the agent is working, the composer's arc border is on and Stop
// is armed, but the tail bubble has settled — a sealed interim row, or a turn
// whose last message completed while the agent kept going. The transcript used
// to show nothing there, and the seconds went uncounted.
import { AssistantRuntimeProvider, type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useRuntimeMessageRepository } from '@/app/chat/runtime-repository'
import { __resetElapsedTimerRegistryForTests } from '@/components/chat/activity-timer'
import type { ChatMessage } from '@/lib/chat-messages'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'
import { $activeSessionId, $busy, $messages, $turnStartedAt } from '@/store/session'

import { stubThreadEnvironment, ThreadRuntime, userMessage } from '../test-utils'

import { Thread } from '.'
stubThreadEnvironment()

const createdAt = new Date('2026-05-01T00:00:00.000Z')
const sessionId = 'session-turn-gap'

function assistant(id: string, content: unknown[], running: boolean): ThreadMessage {
  return {
    id,
    role: 'assistant',
    content,
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  } as unknown as ThreadMessage
}

const toolCall = (toolName: string, settled: boolean) => ({
  type: 'tool-call',
  toolCallId: `${toolName}-1`,
  toolName,
  args: {},
  ...(settled ? { result: 'ok' } : {})
})

const Harness = ({ messages }: { messages: ThreadMessage[] }) => (
  <ThreadRuntime messages={messages}>
    <Thread />
  </ThreadRuntime>
)

// Exercise the production conversion/adapter: reconnect may retire busy
// before a pending transcript row receives its terminal message.complete.
const RuntimeHarness = ({ messages }: { messages: ChatMessage[] }) => {
  const messageRepository = useRuntimeMessageRepository(messages)
  const runtime = useIncrementalExternalStoreRuntime({ messageRepository, isRunning: false, onNew: async () => {} })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

const timerText = (value: string) => screen.getAllByText((_, node) => node?.textContent === value)

describe('the turn timer covers the gaps, not just the streaming', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-01-01T00:00:00.000Z'))
    vi.spyOn(globalThis.document, 'hasFocus').mockReturnValue(true)
    __resetElapsedTimerRegistryForTests()
    $activeSessionId.set(sessionId)
    $messages.set([])
    $turnStartedAt.set(Date.now())
    $busy.set(true)
  })

  afterEach(() => {
    cleanup()
    $activeSessionId.set(null)
    $turnStartedAt.set(null)
    $busy.set(false)
    __resetElapsedTimerRegistryForTests()
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it('times a settled tail bubble while the session is still working', () => {
    // The sealed-bubble gap. Nothing is running at message level; the session
    // is busy, so the transcript owes the user a line and a count — measured
    // from the last thing the turn produced, not from when the row appeared.
    const { container } = render(
      <Harness
        messages={[userMessage('u1', 'do the thing'), assistant('a1', [{ type: 'text', text: 'On it.' }], false)]}
      />
    )

    act(() => vi.advanceTimersByTime(7_000))

    expect(container.querySelector('[data-slot="aui_turn-activity"]')).not.toBeNull()
    expect(timerText('7s').length).toBeGreaterThan(0)
  })

  it('times the gap between a finished tool call and the next thing', () => {
    const { container } = render(
      <Harness messages={[userMessage('u1', 'read it'), assistant('a1', [toolCall('read_file', true)], true)]} />
    )

    act(() => vi.advanceTimersByTime(9_000))

    expect(container.querySelector('[data-slot="aui_turn-activity"]')).not.toBeNull()
    expect(timerText('9s').length).toBeGreaterThan(0)
  })

  it('stays silent under a tool call still in flight — that row has its own timer', () => {
    const { container } = render(
      <Harness messages={[userMessage('u1', 'run it'), assistant('a1', [toolCall('terminal', false)], true)]} />
    )

    act(() => vi.advanceTimersByTime(9_000))

    expect(container.querySelector('[data-slot="aui_turn-activity"]')).toBeNull()
  })

  it('stops when the session stops working', () => {
    $busy.set(false)

    const { container } = render(
      <Harness
        messages={[userMessage('u1', 'do the thing'), assistant('a1', [{ type: 'text', text: 'Done.' }], false)]}
      />
    )

    act(() => vi.advanceTimersByTime(7_000))

    expect(container.querySelector('[data-slot="aui_turn-activity"]')).toBeNull()
  })

  it('does not revive a tail timer from a pending row after the session retires its busy claim', () => {
    $busy.set(false)
    $turnStartedAt.set(null)

    const messages: ChatMessage[] = [
      { id: 'u1', role: 'user', parts: [{ type: 'text', text: 'do the thing' }] },
      { id: 'a1', role: 'assistant', parts: [{ type: 'text', text: 'Done.' }], pending: true }
    ]

    const { container } = render(<RuntimeHarness messages={messages} />)

    act(() => vi.advanceTimersByTime(7_000))

    expect(container.querySelector('[data-slot="aui_turn-activity"]')).toBeNull()
  })

  it('still narrates a pending first bubble before the session busy flush arrives', () => {
    $busy.set(false)
    // Submit has armed the turn clock; the non-critical busy=true view flush
    // can trail the first streamed message by a frame.

    const messages: ChatMessage[] = [
      { id: 'u1', role: 'user', parts: [{ type: 'text', text: 'do the thing' }] },
      { id: 'a1', role: 'assistant', parts: [{ type: 'text', text: 'Working.' }], pending: true }
    ]

    const { container } = render(<RuntimeHarness messages={messages} />)

    act(() => vi.advanceTimersByTime(7_000))

    expect(container.querySelector('[data-slot="aui_turn-activity"]')).not.toBeNull()
  })
})
