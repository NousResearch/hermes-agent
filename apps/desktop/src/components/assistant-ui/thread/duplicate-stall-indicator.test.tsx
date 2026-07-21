import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { __resetElapsedTimerRegistryForTests } from '@/components/chat/activity-timer'
import { setSessionCompacting } from '@/store/compaction'
import { $activeSessionId, $turnStartedAt } from '@/store/session'

import { Thread } from '.'

// Layout/observer stubs mirrored from streaming.test.tsx — jsdom has no
// ResizeObserver, rAF, or real layout, and the Thread scroll container needs
// non-zero dimensions to mount without throwing.
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
  return {
    cancel: () => {},
    finished: Promise.resolve()
  } as unknown as Animation
}

function stubOffsetDimension(
  prop: 'offsetHeight' | 'offsetWidth',
  clientProp: 'clientHeight' | 'clientWidth',
  fallback: number
) {
  const previous = Object.getOwnPropertyDescriptor(HTMLElement.prototype, prop)

  Object.defineProperty(HTMLElement.prototype, prop, {
    configurable: true,
    get() {
      return previous?.get?.call(this) || (this as HTMLElement)[clientProp] || fallback
    }
  })
}

stubOffsetDimension('offsetWidth', 'clientWidth', 800)
stubOffsetDimension('offsetHeight', 'clientHeight', 600)

// Regression test for #68634: every running assistant bubble mounts
// StreamStallIndicator, so a reconnect that leaves a queued prompt behind can
// produce TWO simultaneously "running" assistant bubbles — each rendering its
// own "Summarizing thread" row. The indicator is documented as tail-only, so
// only the LAST bubble should ever show it.

const createdAt = new Date('2026-05-01T00:00:00.000Z')
const sessionId = 'session-68634'

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

function runningAssistantMessage(id: string, text: string): ThreadMessage {
  return {
    id,
    role: 'assistant',
    content: [{ type: 'text', text }],
    status: { type: 'running' },
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

function TwoRunningBubblesHarness() {
  const messages: ThreadMessage[] = [
    userMessage('user-1', 'Summarize this thread for me'),
    runningAssistantMessage('assistant-1', 'Working on it'),
    userMessage('user-2', 'hola?'),
    runningAssistantMessage('assistant-2', '')
  ]

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning: true,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('StreamStallIndicator tail gating (#68634)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-01-01T00:00:00.000Z'))
    __resetElapsedTimerRegistryForTests()
    $activeSessionId.set(sessionId)
    $turnStartedAt.set(Date.now())
    setSessionCompacting(sessionId, true)
  })

  afterEach(() => {
    cleanup()
    setSessionCompacting(sessionId, false)
    $activeSessionId.set(null)
    $turnStartedAt.set(null)
    __resetElapsedTimerRegistryForTests()
    vi.useRealTimers()
  })

  it('renders exactly one "Summarizing thread" indicator when two bubbles are running', () => {
    render(<TwoRunningBubblesHarness />)

    act(() => {
      vi.advanceTimersByTime(5_000)
    })

    const indicators = screen.getAllByRole('status', { name: 'Summarizing thread' })
    expect(indicators.length).toBe(1)
  })
})
