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

// Regression tests for #68634: every running assistant bubble mounts
// StreamStallIndicator, so a reconnect that leaves a queued prompt behind can
// produce TWO simultaneously "running" assistant bubbles — each rendering its
// own "Summarizing thread" row. The indicator is documented as tail-only, so
// only the LAST ASSISTANT-ROLE bubble should ever show it — a trailing
// user/system row (queued prompt, `/steer` note) must NOT suppress it, since
// those are appended while the assistant is still running and are not
// assistant messages themselves.

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

// Shape mirrors the `/steer` note appended by appendSessionTextMessage
// (apps/desktop/src/app/session/hooks/use-prompt-actions/index.ts:623),
// rendered as a codicon row by SystemMessage.
function systemMessage(id: string, text: string): ThreadMessage {
  return {
    id,
    role: 'system',
    content: [{ type: 'text', text }],
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

function Harness({ messages, isRunning = false }: { messages: ThreadMessage[]; isRunning?: boolean }) {
  // isRunning: false at the runtime level — per-message `status: {type:
  // 'running'}` is what drives StreamStallIndicator mounting/gating.
  // Passing isRunning: true makes useExternalStoreRuntime auto-append a
  // synthetic empty trailing assistant placeholder whenever the last message
  // isn't already a running assistant (e.g. the "trailing user prompt" /
  // "trailing system row" cases below). That is the real production flow —
  // exercised explicitly by the isRunning:true regression below to prove the
  // gate skips that placeholder rather than latching onto it.
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning,
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

  it('renders exactly one indicator, on the later bubble, when two assistant bubbles are running with content', () => {
    const { container } = render(
      <Harness
        messages={[
          userMessage('user-1', 'Summarize this thread for me'),
          runningAssistantMessage('assistant-1', 'Working on it'),
          userMessage('user-2', 'hola?'),
          runningAssistantMessage('assistant-2', 'On it too')
        ]}
      />
    )

    act(() => {
      vi.advanceTimersByTime(5_000)
    })

    const indicators = screen.getAllByRole('status', { name: 'Summarizing thread' })
    expect(indicators.length).toBe(1)

    const roots = container.querySelectorAll('[data-slot="aui_assistant-message-root"]')
    expect(roots.length).toBe(2)
    // The indicator must live under the LAST assistant root, not the first.
    expect(roots[0]?.querySelector('[data-slot="aui_stream-stall"]')).toBeNull()
    expect(roots[1]?.querySelector('[data-slot="aui_stream-stall"]')).not.toBeNull()
  })

  it('still renders the indicator when a queued user prompt trails the running assistant', () => {
    render(
      <Harness
        messages={[
          userMessage('user-1', 'Summarize this thread for me'),
          runningAssistantMessage('assistant-1', 'Working on it'),
          userMessage('user-2', 'hola?')
        ]}
      />
    )

    act(() => {
      vi.advanceTimersByTime(5_000)
    })

    const indicators = screen.getAllByRole('status', { name: 'Summarizing thread' })
    expect(indicators.length).toBe(1)
  })

  it('still renders the indicator when a /steer system note trails the running assistant', () => {
    render(
      <Harness
        messages={[
          userMessage('user-1', 'Summarize this thread for me'),
          runningAssistantMessage('assistant-1', 'Working on it'),
          systemMessage('system-steer-1', 'steer:focus on the errors')
        ]}
      />
    )

    act(() => {
      vi.advanceTimersByTime(5_000)
    })

    const indicators = screen.getAllByRole('status', { name: 'Summarizing thread' })
    expect(indicators.length).toBe(1)
  })

  // Production flow (isRunning: true): a trailing queued user prompt makes the
  // runtime auto-append an empty optimistic assistant placeholder AFTER the
  // real running bubble. The placeholder is the true last assistant-role
  // message but renders null (isPlaceholder). The gate must skip it and keep
  // the indicator on the real running bubble — without the skip this renders
  // ZERO indicators. This is the case the isRunning:false tests above cannot
  // reach.
  it('still renders the indicator when the runtime appends an optimistic placeholder (isRunning:true)', () => {
    render(
      <Harness
        isRunning
        messages={[
          userMessage('user-1', 'Summarize this thread for me'),
          runningAssistantMessage('assistant-1', 'Working on it'),
          userMessage('user-2', 'hola?')
        ]}
      />
    )

    act(() => {
      vi.advanceTimersByTime(5_000)
    })

    const indicators = screen.getAllByRole('status', { name: 'Summarizing thread' })
    expect(indicators.length).toBe(1)
  })
})
