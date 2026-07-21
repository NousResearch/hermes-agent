import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { __resetElapsedTimerRegistryForTests } from '@/components/chat/activity-timer'
import { I18nProvider } from '@/i18n'
import { setSessionCompacting } from '@/store/compaction'
import { $activeSessionId, $turnStartedAt } from '@/store/session'

import { Thread } from '.'

// Regression cover for #68634. The compaction/stall indicator reflects
// SESSION-global state, so mounting it per running assistant bubble rendered
// one identical "Summarizing thread" row and timer per bubble. A reconnect or a
// queued prompt can legitimately leave an earlier bubble marked running, so the
// count depended on transport timing rather than on anything the user did.

const createdAt = new Date('2026-05-01T00:00:00.000Z')
const SESSION_ID = 'session-68634'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)
vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => window.setTimeout(() => cb(0), 0))
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
vi.stubGlobal('CSS', { escape: (value: string) => value })

Element.prototype.scrollTo = function scrollTo() {}

Element.prototype.animate = function animate() {
  return { cancel: () => {}, finished: Promise.resolve() } as unknown as Animation
}

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

function runningAssistant(id: string, text: string): ThreadMessage {
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

function Harness({ messages }: { messages: ThreadMessage[] }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning: true,
    onNew: async () => {}
  })

  return (
    <I18nProvider configClient={null} initialLocale="en">
      <AssistantRuntimeProvider runtime={runtime}>
        <Thread />
      </AssistantRuntimeProvider>
    </I18nProvider>
  )
}

describe('compaction indicator (#68634)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-01-01T00:00:00.000Z'))
    __resetElapsedTimerRegistryForTests()
    $activeSessionId.set(SESSION_ID)
    $turnStartedAt.set(Date.now())
    setSessionCompacting(SESSION_ID, true)
  })

  afterEach(() => {
    cleanup()
    setSessionCompacting(SESSION_ID, false)
    $activeSessionId.set(null)
    $turnStartedAt.set(null)
    __resetElapsedTimerRegistryForTests()
    vi.useRealTimers()
  })

  it('renders one indicator when several assistant bubbles are still running', () => {
    // The reported shape: an earlier assistant bubble never left `running`
    // before the user's next prompt was inserted.
    render(
      <Harness
        messages={[
          userMessage('user-1', 'first question'),
          runningAssistant('assistant-1', 'partial answer'),
          userMessage('user-2', 'hola?'),
          runningAssistant('assistant-2', '')
        ]}
      />
    )

    act(() => vi.advanceTimersByTime(17_000))

    expect(screen.getAllByRole('status', { name: 'Summarizing thread' })).toHaveLength(1)
  })

  it('still renders the indicator for a single running bubble', () => {
    render(<Harness messages={[userMessage('user-1', 'question'), runningAssistant('assistant-1', 'answer')]} />)

    act(() => vi.advanceTimersByTime(17_000))

    expect(screen.getAllByRole('status', { name: 'Summarizing thread' })).toHaveLength(1)
  })
})
