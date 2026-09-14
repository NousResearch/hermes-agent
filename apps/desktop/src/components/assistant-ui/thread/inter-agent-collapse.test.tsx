// Two contracts with no coverage before the invalidation-scoping work split
// AssistantMessage into InterAgentAssistantMessage + AssistantMessageBody:
//
// 1. The collapse gate. A reply to an inter-agent delivery renders collapsed
//    ("Replied to <sender>", expandable) ONLY once it settles — never while it
//    streams, because the user should see progress. That gate is the sole
//    remaining root-level `isRunning` subscription, so it is the thing most
//    likely to break if the split is revisited.
// 2. The streaming marker. `data-message-streaming` moved off the message root
//    onto a permanently-mounted hidden leaf, and
//    scripts/run-short-session-hang-repro.mjs derives its settled-row count by
//    subtracting `[data-message-streaming="true"]` markers from message roots.
//    Nothing in the app itself reads it, so without this test a delete would
//    look free and would silently regress that repro's response gate.
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

function user(id: string, text: string, isHuman = true): ThreadMessage {
  return {
    id,
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt,
    metadata: { custom: { isHuman } }
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

const DELIVERY = 'Message from 🤖 Hermes (@hermes): please check the build'

describe('inter-agent collapse gate', () => {
  it('collapses a settled reply to an inter-agent delivery', async () => {
    render(<Harness messages={[user('u1', DELIVERY), assistant('a1', 'build is green', false)]} />)

    expect(await screen.findByText(/Replied to/)).toBeTruthy()
    expect(screen.getByText('show reply')).toBeTruthy()
  })

  it('does NOT collapse while that reply is still streaming', async () => {
    const { container } = render(<Harness messages={[user('u1', DELIVERY), assistant('a1', 'working on it', true)]} />)

    await screen.findByText('working on it')
    expect(screen.queryByText('show reply')).toBeNull()
    // Expanded => the full body root, which carries the streaming marker.
    expect(container.querySelector('[data-message-streaming="true"]')).toBeTruthy()
  })

  it('leaves an ordinary reply expanded', async () => {
    render(<Harness messages={[user('u1', 'ordinary question'), assistant('a1', 'ordinary answer', false)]} />)

    await screen.findByText('ordinary answer')
    expect(screen.queryByText(/Replied to/)).toBeNull()
  })

  it('clears the streaming marker once the turn settles', async () => {
    const { container } = render(<Harness messages={[user('u1', 'q'), assistant('a1', 'done', false)]} />)

    await screen.findByText('done')
    expect(container.querySelector('[data-message-streaming="true"]')).toBeNull()
    // The marker element itself stays mounted (attribute toggles, no remount).
    expect(
      container.querySelector('[data-slot="aui_assistant-message-root"] [data-slot="aui_message-streaming-marker"]')
    ).toBeTruthy()
  })

  it('exempts a reply to an agent delivery when an earlier user message is human', async () => {
    render(
      <Harness
        messages={[
          user('u1', 'human: please do it', true),
          user('u2', DELIVERY, false),
          assistant('a1', 'Done — here is the report for you.', false)
        ]}
      />
    )

    // The reply follows an agent delivery, but an earlier row is a REAL
    // human prompt, so this is an owner-directed report — expanded, not
    // collapsed. The human flag is the runtime-stamped authoritative signal,
    // not a text regex at render time.
    expect(await screen.findByText(/Done — here is the report for you/)).toBeTruthy()
    expect(screen.queryByText(/Replied to/)).toBeNull()
    expect(screen.queryByText('show reply')).toBeNull()
  })

  it('does not treat an agent delivery as a human message (no false exemption)', async () => {
    render(
      <Harness
        messages={[
          user('u1', DELIVERY, false),
          user('u2', 'Message from 🤖 Hermes (@hermes): one more', false),
          assistant('a1', 'ack', false)
        ]}
      />
    )

    // All earlier user rows are bot deliveries (isHuman=false) even though
    // they do not match the collapse regex at render time — the authoritative
    // flag keeps the Grok-bots collapse for a pure bot exchange.
    expect(await screen.findByText(/Replied to/)).toBeTruthy()
    expect(screen.getByText('show reply')).toBeTruthy()
  })

  it('does not treat a background-process notice as a human message', async () => {
    render(
      <Harness
        messages={[
          user('u1', '[IMPORTANT: Background process 123 finished]', false),
          user('u2', DELIVERY, false),
          assistant('a1', 'ack', false)
        ]}
      />
    )

    // The injected notice is a synthetic user-role row — the converter stamps
    // isHuman=false, so the bot exchange stays collapsed.
    expect(await screen.findByText(/Replied to/)).toBeTruthy()
    expect(screen.getByText('show reply')).toBeTruthy()
  })

  it('collapses when the only prior user row is the delivery (pure bot exchange)', async () => {
    render(
      <Harness
        messages={[
          user('u1', DELIVERY, false),
          assistant('a1', 'build is green', false)
        ]}
      />
    )

    expect(await screen.findByText(/Replied to/)).toBeTruthy()
    expect(screen.getByText('show reply')).toBeTruthy()
  })

  it('does NOT lift an already-collapsed reply when a human message arrives later', async () => {
    render(
      <Harness
        messages={[
          user('u1', DELIVERY, false),
          assistant('a1', 'build is green', false),
          user('u2', 'human: thanks', true)
        ]}
      />
    )

    // The exemption is decided from PRIOR evidence only: the reply follows
    // the bot delivery with no earlier human row, so it renders collapsed.
    // A later human message must not retroactively expand it (that would
    // shift the transcript layout under the reader).
    expect(await screen.findByText(/Replied to/)).toBeTruthy()
    expect(screen.getByText('show reply')).toBeTruthy()
  })
})
