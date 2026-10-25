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

import { dispatchedTo } from '@/components/assistant-ui/thread/agent-delivery'

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

/** This bot's own `message_agent` dispatch to a teammate (a Bot Chat turn). */
function dispatch(id: string, target: string): ThreadMessage {
  return {
    id,
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: `${id}-call`,
        toolName: 'message_agent',
        args: { target, message: 'send me the list' },
        argsText: JSON.stringify({ target, message: 'send me the list' })
      }
    ],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: assistantMetadata
  } as ThreadMessage
}

describe('inter-agent collapse gate', () => {
  it('keeps the report after a relayed answer to this bot\u2019s own dispatch expanded', async () => {
    // Bot Mode round trip seen from the DISPATCHING bot: it messaged @hermes,
    // the answer came back as an inbound "Message from" row, and the next
    // assistant message is its report to the human — not a reply to hermes.
    render(
      <Harness
        messages={[
          user('u1', 'ask hermes for the list'),
          dispatch('a0', '@Hermes'),
          user('u2', DELIVERY),
          assistant('a1', 'here is the list hermes sent', false)
        ]}
      />
    )

    await screen.findByText('here is the list hermes sent')
    expect(screen.queryByText(/Replied to/)).toBeNull()
    expect(screen.queryByText('show reply')).toBeNull()
  })

  it('still collapses the reply to an unsolicited delivery (dispatch went to another teammate)', async () => {
    render(
      <Harness messages={[dispatch('a0', 'scribe'), user('u2', DELIVERY), assistant('a1', 'build is green', false)]} />
    )

    expect(await screen.findByText(/Replied to/)).toBeTruthy()
    expect(screen.getByText('show reply')).toBeTruthy()
  })

  it('collapses the reply to a LATER unsolicited delivery from a teammate dispatched to earlier', async () => {
    // One dispatch exempts only the answer that follows it. Turns later, the
    // same teammate messages in on its own — that exchange had no dispatch,
    // so the deliberate fold (#85884) must still apply.
    render(
      <Harness
        messages={[
          user('u1', 'ask hermes for the list'),
          dispatch('a0', '@Hermes'),
          user('u2', DELIVERY),
          assistant('a1', 'here is the list hermes sent', false),
          user('u3', 'ok thanks'),
          assistant('a2', 'anytime', false),
          user('u4', 'Message from 🤖 Hermes (@hermes): unsolicited: build broke'),
          assistant('a3', 'on it, checking the build', false)
        ]}
      />
    )

    expect(await screen.findByText(/Replied to/)).toBeTruthy()
    expect(screen.getAllByText(/Replied to/)).toHaveLength(1)
    expect(screen.getByText('show reply')).toBeTruthy()
  })

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
})

const SCRIBE = 'Message from 🤖 Scribe (@scribe): restore done'
const SCRIBE_AGAIN = 'Message from 🤖 Scribe (@scribe): header fix is in too'
const COMPLETION = '[IMPORTANT: Background process proc_1 completed normally (exit code 0).\nOutput:\n{"reply": "ok"}]'

describe('inter-agent collapse gate: only dispatches made for the human count', () => {
  it('collapses the reply when the bot answered the teammate via message_agent, as Bot Mode instructs', async () => {
    // The built-in Bot Mode prompt tells a bot receiving "Message from <name>"
    // to "reply concisely via message_agent to their handle". The human asked
    // nothing here, so the teammate's next message is a new exchange and the
    // bot's reply to it must fold, not render as a report to the human.
    render(
      <Harness
        messages={[
          user('u1', SCRIBE),
          dispatch('a0', '@scribe'),
          user('u2', SCRIBE_AGAIN),
          assistant('a1', 'thanks, noted the header fix', false)
        ]}
      />
    )

    // Both replies to Scribe fold; before the fix only the first one did.
    expect(await screen.findAllByText(/Replied to Scribe/)).toHaveLength(2)
    expect(screen.getByText('thanks, noted the header fix').closest('details')).toBeTruthy()
  })

  it('collapses the reply when the dispatch to the sender was made while answering another teammate', async () => {
    // Scribe wrote in, the bot messaged hermes while answering scribe, then
    // hermes wrote in. That exchange is bot-to-bot, so the reply to it folds.
    // (The dispatch turn answering scribe folds as "Replied to Scribe".)
    render(
      <Harness
        messages={[
          user('u1', SCRIBE),
          dispatch('a0', '@hermes'),
          user('u2', DELIVERY),
          assistant('a1', 'thanks, noted the build', false)
        ]}
      />
    )

    expect(await screen.findByText(/Replied to Hermes/)).toBeTruthy()
  })

  it('keeps the report expanded when the dispatch was made while answering the human (#114629)', async () => {
    render(
      <Harness
        messages={[
          user('u1', 'ask hermes for the list'),
          dispatch('a0', '@hermes'),
          user('u2', DELIVERY),
          assistant('a1', 'here is the list hermes sent', false)
        ]}
      />
    )

    await screen.findByText('here is the list hermes sent')
    expect(screen.queryByText(/Replied to/)).toBeNull()
  })

  it('keeps the final report expanded for a chain the human asked for', async () => {
    // human -> dispatch to scribe -> scribe answers -> dispatch to hermes ->
    // hermes answers. The hermes dispatch answered a row that was itself the
    // answer to a dispatch made for the human, so it counts too.
    render(
      <Harness
        messages={[
          user('u1', 'get the restore status, then ask hermes for the list'),
          dispatch('a0', '@scribe'),
          user('u2', SCRIBE),
          dispatch('a1', '@hermes'),
          user('u3', DELIVERY),
          assistant('a2', 'restore is done and here is the list hermes sent', false)
        ]}
      />
    )

    await screen.findByText('restore is done and here is the list hermes sent')
    expect(screen.queryByText(/Replied to/)).toBeNull()
  })

  it('still collapses the end of a chain that began with an unsolicited teammate message', async () => {
    // Same shape as the chain above, but Scribe wrote in unprompted: scribe ->
    // dispatch to hermes -> hermes answers -> dispatch to scribe -> scribe
    // answers. No link of this chain was made for the human, so it folds.
    render(
      <Harness
        messages={[
          user('u1', SCRIBE),
          dispatch('a0', '@hermes'),
          user('u2', DELIVERY),
          dispatch('a1', '@scribe'),
          user('u3', SCRIBE_AGAIN),
          assistant('a2', 'thanks, noted the header fix', false)
        ]}
      />
    )

    await screen.findByText('thanks, noted the header fix')
    expect(screen.getByText('thanks, noted the header fix').closest('details')).toBeTruthy()
  })

  it('looks through a background-process completion to the exchange that started the turn', async () => {
    render(
      <Harness
        messages={[
          user('u1', SCRIBE),
          dispatch('a0', 'scribe'),
          user('u2', COMPLETION),
          dispatch('a1', '@hermes'),
          user('u3', DELIVERY),
          assistant('a2', 'thanks, noted the build', false)
        ]}
      />
    )

    expect(await screen.findByText(/Replied to Hermes/)).toBeTruthy()
  })

  it('keeps the report expanded when the human asked for the dispatch, even after a completion', async () => {
    render(
      <Harness
        messages={[
          user('u1', 'ask hermes for the list'),
          dispatch('a0', 'scribe'),
          user('u2', COMPLETION),
          dispatch('a1', '@hermes'),
          user('u3', DELIVERY),
          assistant('a2', 'here is the list hermes sent', false)
        ]}
      />
    )

    await screen.findByText('here is the list hermes sent')
    expect(screen.queryByText(/Replied to/)).toBeNull()
  })

  it('still honours an earlier dispatch the human asked for when a later one answered a teammate', async () => {
    render(
      <Harness
        messages={[
          user('u1', 'ask hermes for the list'),
          dispatch('a0', '@hermes'),
          user('u2', SCRIBE),
          dispatch('a1', '@hermes'),
          user('u3', DELIVERY),
          assistant('a2', 'here is the list hermes sent', false)
        ]}
      />
    )

    await screen.findByText('here is the list hermes sent')
    expect(screen.queryByText(/Replied to Hermes/)).toBeNull()
  })

  it('treats a dispatch with no earlier prompt as made for the human (truncated history)', async () => {
    // Imported or truncated transcripts can start mid-turn. Showing beats
    // hiding, so a dispatch whose prompt is not in view still counts.
    render(
      <Harness
        messages={[
          dispatch('a0', '@hermes'),
          user('u1', DELIVERY),
          assistant('a1', 'here is the list hermes sent', false)
        ]}
      />
    )

    await screen.findByText('here is the list hermes sent')
    expect(screen.queryByText(/Replied to/)).toBeNull()
  })

  it('consumes the dispatch with its answer even without a human row in between', async () => {
    // human -> dispatch to scribe -> scribe answers -> scribe writes again.
    // The second Scribe row is unsolicited (#85884), so the reply to it folds.
    render(
      <Harness
        messages={[
          user('u1', 'get the restore status'),
          dispatch('a0', '@scribe'),
          user('u2', SCRIBE),
          assistant('a1', 'the restore is done', false),
          user('u3', SCRIBE_AGAIN),
          assistant('a2', 'thanks, noted the header fix', false)
        ]}
      />
    )

    expect(await screen.findAllByText(/Replied to Scribe/)).toHaveLength(1)
    expect(screen.getByText('the restore is done').closest('details')).toBeNull()
  })
})

describe('dispatchedTo: bounded work', () => {
  it('classifies a long bot-to-bot exchange with repeated dispatches in linear content reads', () => {
    // Two teammates alternate with no human row, and the bot dispatches to the
    // next sender twice per hop, each in its own row. Classifying every hop from
    // scratch doubles the work per hop; one forward pass reads each row a
    // bounded number of times.
    const hops = 16
    let reads = 0

    const counted = (row: { content: unknown; role: string }) => ({
      role: row.role,
      get content() {
        reads++

        return row.content
      }
    })

    const call = (target: string) => [{ type: 'tool-call', toolName: 'message_agent', args: { target, message: 'ok' } }]

    const rows: { content: unknown; role: string }[] = [{ role: 'user', content: [{ type: 'text', text: SCRIBE }] }]

    for (let h = 0; h < hops; h++) {
      const next = h % 2 === 0 ? '@hermes' : '@scribe'
      rows.push({ role: 'assistant', content: call(next) }, { role: 'assistant', content: call(next) })
      rows.push({ role: 'user', content: [{ type: 'text', text: h % 2 === 0 ? DELIVERY : SCRIBE_AGAIN }] })
    }

    // The caller passes the thread before the newest inbound row (Scribe here).
    const thread = rows.slice(0, -1).map(counted)

    expect(dispatchedTo(thread, ['Scribe', 'scribe', undefined])).toBe(false)
    expect(reads).toBeLessThan(20 * thread.length)
  })
})
