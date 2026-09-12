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
  it('shows a settled reply to an inter-agent delivery inline (not collapsed)', async () => {
    render(<Harness messages={[user('u1', DELIVERY), assistant('a1', 'build is green', false)]} />)

    // The response is visible without expanding anything — a direct DM has no
    // other place to surface it (regression: collapsing hid the reply).
    expect(await screen.findByText('build is green')).toBeTruthy()
    expect(screen.queryByText(/Replied to/)).toBeNull()
    expect(screen.queryByText('show reply')).toBeNull()
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

  // Regression: a DIRECT bot-to-bot DM (the sender is a bot, not the human,
  // and the two are NOT in a shared group) must show the responding bot's
  // reply inline — collapsing it behind a closed "show reply" <details>
  // hides the answer entirely, which is the bug report. Only in-group or
  // human-visible conversation collapses for Grok-bots parity.
  it('shows a direct bot-to-bot reply inline (not collapsed/hidden)', async () => {
    render(<Harness messages={[user('u1', DELIVERY), assistant('a1', 'the build is green', false)]} />)

    // The reply text itself must be visible without expanding anything.
    const reply = await screen.findByText('the build is green')
    expect(reply).toBeTruthy()
    // No collapse affordance => the response is not hidden behind a toggle.
    expect(screen.queryByText(/Replied to/)).toBeNull()
    expect(screen.queryByText('show reply')).toBeNull()
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
})
