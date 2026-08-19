// The gap case: the agent is working, the composer's arc border is on and Stop
// is armed, but the tail bubble has settled — a sealed interim row, or a turn
// whose last message completed while the agent kept going. The transcript used
// to show nothing there, and the seconds went uncounted.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { __resetElapsedTimerRegistryForTests } from '@/components/chat/activity-timer'
import { $activeSessionId, $busy, $messages, $turnStartedAt } from '@/store/session'

import { Thread } from '.'

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
vi.stubGlobal('CSS', { escape: (str: string) => str })

Element.prototype.scrollTo = function scrollTo() {}

Element.prototype.animate = function animate() {
  return { cancel() {}, finished: Promise.resolve() } as unknown as Animation
}

const createdAt = new Date('2026-05-01T00:00:00.000Z')
const sessionId = 'session-turn-gap'

function user(id: string, text: string): ThreadMessage {
  return {
    id,
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

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

function Harness({ messages }: { messages: ThreadMessage[] }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning: messages.at(-1)?.status?.type === 'running',
    onNew: async () => {}
  })

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
      <Harness messages={[user('u1', 'do the thing'), assistant('a1', [{ type: 'text', text: 'On it.' }], false)]} />
    )

    act(() => vi.advanceTimersByTime(7_000))

    expect(container.querySelector('[data-slot="aui_turn-activity"]')).not.toBeNull()
    expect(timerText('7s').length).toBeGreaterThan(0)
  })

  it('times the gap between a finished tool call and the next thing', () => {
    const { container } = render(
      <Harness messages={[user('u1', 'read it'), assistant('a1', [toolCall('read_file', true)], true)]} />
    )

    act(() => vi.advanceTimersByTime(9_000))

    expect(container.querySelector('[data-slot="aui_turn-activity"]')).not.toBeNull()
    expect(timerText('9s').length).toBeGreaterThan(0)
  })

  it('stays silent under a tool call still in flight — that row has its own timer', () => {
    const { container } = render(
      <Harness messages={[user('u1', 'run it'), assistant('a1', [toolCall('terminal', false)], true)]} />
    )

    act(() => vi.advanceTimersByTime(9_000))

    expect(container.querySelector('[data-slot="aui_turn-activity"]')).toBeNull()
  })

  it('stops when the session stops working', () => {
    $busy.set(false)

    const { container } = render(
      <Harness messages={[user('u1', 'do the thing'), assistant('a1', [{ type: 'text', text: 'Done.' }], false)]} />
    )

    act(() => vi.advanceTimersByTime(7_000))

    expect(container.querySelector('[data-slot="aui_turn-activity"]')).toBeNull()
  })
})
