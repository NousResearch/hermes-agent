// The thinking indicator (dither block) may only ever render at the TAIL of
// the thread. A message stuck status:running mid-transcript — however it got
// there (missed settle event, steer race, upstream state bug) — must render
// its content with no spinner: a live indicator above a later user message
// reads as the agent answering out of order.
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
// Enter animation fires for running messages; jsdom has no WAAPI.
Element.prototype.animate = function animate() {
  return { cancel() {}, finished: Promise.resolve() } as unknown as Animation
}

afterEach(() => {
  cleanup()
})

const assistantMetadata = {
  unstable_state: null,
  unstable_annotations: [],
  unstable_data: [],
  steps: [],
  custom: {}
}

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

describe('thinking indicator is tail-only', () => {
  it('shows the loading indicator on a running placeholder at the tail', async () => {
    const { container } = render(<Harness messages={[user('u1', 'question'), assistant('a1', '', true)]} />)

    expect(await screen.findByRole('status', { name: 'Hermes is loading a response' })).toBeTruthy()
    expect(container.querySelector('[data-slot="aui_response-loading"]')).toBeTruthy()
  })

  it('never shows an indicator on a stale running message mid-transcript', async () => {
    // A stranded pending bubble from an earlier turn, then a newer exchange.
    const { container } = render(
      <Harness
        messages={[
          user('u1', 'first question'),
          assistant('a1', '', true),
          user('u2', 'second question'),
          assistant('a2', 'answered', false)
        ]}
      />
    )

    await screen.findByText('answered')

    expect(container.querySelector('[data-slot="aui_response-loading"]')).toBeNull()
    expect(container.querySelector('[data-slot="aui_stream-stall"]')).toBeNull()
  })
})
