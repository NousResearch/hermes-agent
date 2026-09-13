// The invalidation-scoping property, as a render-count contract.
//
// A streaming turn flips its message status many times a second, and at
// stream breadth N that is N status flips per flush. The whole point of this
// work is that a flip re-renders only the small leaves that actually display
// status — never the message ROOT, whose subtree is the entire rendered
// message and whose re-render is what widened style recalculation to document
// scope.
//
// That property is invisible to a DOM assertion: the transcript looks
// identical either way. So this counts renders instead. AssistantMessageBody
// is the root component, and `useTapbackDoubleClick` is called by it and by
// nothing else in the tree, which makes it an exact render counter for the
// root without needing to export or wrap an internal component.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as messageReactionsModule from '@/components/assistant-ui/thread/use-message-reactions'

import { Thread } from '.'

let rootRenders = 0

vi.mock('@/components/assistant-ui/thread/use-message-reactions', async importActual => {
  const actual = await importActual<typeof messageReactionsModule>()

  return {
    ...actual,
    useTapbackDoubleClick: (messageId: string, role: 'assistant' | 'user') => {
      if (role === 'assistant') {
        rootRenders += 1
      }

      return actual.useTapbackDoubleClick(messageId, role)
    }
  }
})

const createdAt = new Date('2026-05-01T00:00:00.000Z')

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

beforeEach(() => {
  rootRenders = 0
})

afterEach(() => {
  cleanup()
})

const assistantMetadata = { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }

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

function assistant(id: string, text: string, running: boolean): ThreadMessage {
  return {
    id,
    role: 'assistant',
    content: text ? [{ type: 'text', text }] : [],
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: assistantMetadata
  } as ThreadMessage
}

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

describe('streaming-status invalidation scope', () => {
  it('does not re-render the message root when the turn settles', async () => {
    const messages = [user('u1', 'question'), assistant('a1', 'partial answer', true)]
    const { container, findByText, rerender } = render(<Harness messages={messages} />)

    await findByText('partial answer')
    // The leaf carries the streaming flag while the turn is in flight.
    expect(container.querySelector('[data-message-streaming="true"]')).toBeTruthy()

    const rendersWhileStreaming = rootRenders

    rerender(<Harness messages={[messages[0], assistant('a1', 'partial answer', false)]} />)

    // The leaf saw the flip...
    await waitFor(() => {
      expect(container.querySelector('[data-message-streaming="true"]')).toBeNull()
    })
    // ...on the same permanently-mounted node (attribute toggle, no remount)...
    expect(container.querySelector('[data-slot="aui_message-streaming-marker"]')).toBeTruthy()
    // ...and the root did not re-render for it.
    expect(rootRenders).toBe(rendersWhileStreaming)
  })

  it('does not re-render the message root when streaming text arrives', async () => {
    const messages = [user('u1', 'question'), assistant('a1', 'one', true)]
    const { findByText, rerender } = render(<Harness messages={messages} />)

    await findByText('one')
    const rendersAfterFirstToken = rootRenders

    // A delta flush: same status, more text. The root must not subscribe to
    // the streaming text either — only the markdown part re-renders.
    rerender(<Harness messages={[messages[0], assistant('a1', 'one two', true)]} />)
    await findByText('one two')

    expect(rootRenders).toBe(rendersAfterFirstToken)
  })

  it('keeps the same root DOM node across the settle transition', async () => {
    // A direct bot-to-bot reply now renders EXPANDED (visible) on settle, not
    // collapsed behind a "Replied to" toggle. Rendering that as a competing
    // root previously swapped the element type at this position, so React
    // unmounted the row and mounted a fresh one — discarding the DOM the scroll
    // anchor was holding. Same root, updated in place: settle is now a prop
    // change React applies without a remount.
    const delivery = 'Message from 🤖 Hermes (@hermes): please check the build'
    const messages = [user('u1', delivery), assistant('a1', 'working on it', true)]
    const { container, findByText, rerender } = render(<Harness messages={messages} />)

    await findByText('working on it')
    const before = container.querySelector('[data-slot="aui_assistant-message-root"]')

    expect(before).toBeTruthy()

    rerender(<Harness messages={[messages[0], assistant('a1', 'working on it', false)]} />)
    // Settled reply stays visible inline; no collapse / no remount.
    await findByText('working on it')
    expect(container.querySelector('[data-slot="aui_assistant-message-root"] [data-slot="aui_assistant-message-content"]')).toBeTruthy()

    const after = container.querySelector('[data-slot="aui_assistant-message-root"]')
    // Same element, updated in place — not a remount.
    expect(after).toBe(before)
  })
})
