import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { Thread } from './thread'

class NoopResizeObserver {
  observe() {}

  unobserve() {}

  disconnect() {}
}

vi.stubGlobal('ResizeObserver', NoopResizeObserver)
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0)
)
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))

Element.prototype.scrollTo = function scrollTo() {}

Element.prototype.animate = function animate() {
  return {
    cancel: () => {},
    finished: Promise.resolve()
  } as unknown as Animation
}

// jsdom returns 0 for offset*; the virtualizer reads those to size its
// viewport. Fall through to client* or a sane default so virtualized
// items render (same stub as streaming.test.tsx).
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

const createdAt = new Date('2026-05-01T00:00:00.000Z')

const MESSAGES: ThreadMessage[] = [
  {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'hello from the user' }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage,
  {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text: 'stable assistant reply' }],
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
]

function Harness({
  onBranchInNewChat,
  onCancel
}: {
  onBranchInNewChat: (messageId: string) => void
  onCancel: () => void
}) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: MESSAGES,
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread onBranchInNewChat={onBranchInNewChat} onCancel={onCancel} />
    </AssistantRuntimeProvider>
  )
}

describe('thread message mount stability', () => {
  // Regression: the desktop controller re-renders every 15s (status
  // snapshot poll) and used to pass freshly-created callbacks down to
  // <Thread/>. Those callbacks were deps of the `messageComponents`
  // useMemo, so new component *types* were created each poll and React
  // unmounted/remounted every visible message — shiki re-highlighted
  // code blocks and the whole thread visibly jumped.
  it('keeps message DOM nodes mounted when callback props get new identities', async () => {
    const { rerender } = render(<Harness onBranchInNewChat={() => {}} onCancel={() => {}} />)

    await waitFor(() => {
      expect(screen.getByText('stable assistant reply')).toBeTruthy()
      expect(screen.getByText('hello from the user')).toBeTruthy()
    })

    const assistantBefore = screen.getByText('stable assistant reply')
    const userBefore = screen.getByText('hello from the user')

    // Same data, new callback identities — exactly what a parent
    // re-render driven by an unrelated state update produces.
    await act(async () => {
      rerender(<Harness onBranchInNewChat={() => {}} onCancel={() => {}} />)
    })

    expect(screen.getByText('stable assistant reply')).toBe(assistantBefore)
    expect(screen.getByText('hello from the user')).toBe(userBefore)
  })
})
