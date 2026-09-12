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

  it('exempts a reply to an agent delivery when the thread contains a real user message', async () => {
    render(
      <Harness
        messages={[
          user('u1', 'human: please do it'),
          user('u2', DELIVERY),
          assistant('a1', 'Done — here is the report for you.', false)
        ]}
      />
    )

    // The reply follows an agent delivery, but the thread is not pure
    // bot-to-bot: a real human message exists, so this is an owner-directed
    // report and must render expanded.
    expect(await screen.findByText(/Done — here is the report for you/)).toBeTruthy()
    expect(screen.queryByText(/Replied to/)).toBeNull()
    expect(screen.queryByText('show reply')).toBeNull()
  })

  it('still collapses when the reply is explicitly marked as an inter-agent answer', async () => {
    const marked = {
      ...assistant('a1', 'ACK, reviewer.', false),
      metadata: {
        ...assistantMetadata,
        custom: { agentReply: true }
      }
    } as unknown as ThreadMessage

    render(<Harness messages={[user('u1', 'human: please do it'), user('u2', DELIVERY), marked]} />)

    // Even with a real user message present, an explicit agent-reply marker
    // keeps the Grok-bots collapse behaviour for genuine bot exchanges.
    expect(await screen.findByText(/Replied to/)).toBeTruthy()
    expect(screen.getByText('show reply')).toBeTruthy()
  })

  it('still collapses when the thread has no real user message (pure bot exchange)', async () => {
    render(
      <Harness
        messages={[
          user('u1', DELIVERY),
          user('u2', 'Message from 🤖 Hermes (@hermes): one more'),
          assistant('a1', 'ack', false)
        ]}
      />
    )

    expect(await screen.findByText(/Replied to/)).toBeTruthy()
    expect(screen.getByText('show reply')).toBeTruthy()
  })

  it('exempts thread-wide: a human message after the reply also lifts the collapse', async () => {
    render(
      <Harness
        messages={[user('u1', DELIVERY), assistant('a1', 'build is green', false), user('u2', 'human: thanks')]}
      />
    )

    // The exemption scans the whole thread, not just the messages preceding
    // the reply: any real human message means the thread is not pure
    // bot-to-bot, so the reply renders expanded.
    expect(await screen.findByText('build is green')).toBeTruthy()
    expect(screen.queryByText(/Replied to/)).toBeNull()
  })
})
