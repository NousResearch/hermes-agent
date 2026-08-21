// Link previews moved off the message root into the AssistantPreviewEmbeds
// leaf, because the selector behind them (`'' while running`, the full
// `messageContentText(content)` join once settled) flipped on every
// running <-> settled transition and re-rendered the root with it.
//
// Two things are pinned here. That the embed still renders at all — the move
// was verbatim JSX and had no coverage before. And that it renders ONLY once
// the turn settles: the '' branch is a deliberate streaming optimization (it
// keeps the selector referentially stable so per-token flushes skip the regex
// scan), so a rewrite that drops it would make previews flicker in mid-stream
// with nothing to catch it.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Thread } from '.'

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

const TARGET = 'https://example.com/docs'
const WITH_PREVIEW = `Serving now: [Preview: example](#preview/${TARGET})`

describe('settled-turn link previews', () => {
  it('renders the embed once the turn has settled', async () => {
    const { container } = render(<Harness messages={[user('u1', 'start it'), assistant('a1', WITH_PREVIEW, false)]} />)

    await screen.findByText('Serving now:', { exact: false })

    expect(container.querySelector(`[title="${TARGET}"]`)).toBeTruthy()
  })

  it('does not render the embed while the turn is still running', async () => {
    const { container } = render(<Harness messages={[user('u1', 'start it'), assistant('a1', WITH_PREVIEW, true)]} />)

    await screen.findByText('Serving now:', { exact: false })

    expect(container.querySelector(`[title="${TARGET}"]`)).toBeNull()
  })
})
