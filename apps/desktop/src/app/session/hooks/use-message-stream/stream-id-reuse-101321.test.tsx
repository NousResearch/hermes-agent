import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { playCompletionSound } from '@/lib/completion-sound'
import { $turnStartedAt } from '@/store/session'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

vi.mock('@/lib/completion-sound', () => ({ playCompletionSound: vi.fn() }))

/**
 * #101321: a provider stream drop (Grok) can leave the previous turn's
 * in-flight bubble id (`state.streamId`) live across the turn boundary, so
 * the next turn's deltas merge into the PREVIOUS answer below the new
 * prompt, and its terminal frame replaces the new answer with the old one.
 *
 * The boundary here is a user row, the way a real second prompt makes one —
 * the harness's own submit path would clear streamId itself (seedOptimistic)
 * and never exercise the seam. Each test seeds the state the drop leaves
 * behind and asserts the new turn paints only the new answer.
 */

const SID = 'stream-id-reuse-101321'

let stream: MessageStreamHarness

const ev = (type: string, payload: Record<string, unknown> = {}): GatewayEvent =>
  ({ payload, session_id: SID, type }) as GatewayEvent

const start = () => act(() => stream.handleEvent(ev('message.start')))
const delta = (text: string) => act(() => stream.handleEvent(ev('message.delta', { text })))

const complete = (payload: Record<string, unknown>) => act(() => stream.handleEvent(ev('message.complete', payload)))

/** Deltas flush on a coalescing timer — drain it so a delta has painted. */
const flush = () =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(100)
  })

const userRow = (text: string): ChatMessage => ({
  id: `user-${text.replace(/\W+/g, '-')}`,
  role: 'user',
  parts: [{ type: 'text', text }]
})

const assistantRow = (message: Partial<ChatMessage> & { id: string }): ChatMessage =>
  ({
    role: 'assistant',
    parts: [{ type: 'text', text: '' }],
    ...message
  }) as ChatMessage

/** Turn A streamed but never settled (the drop), then prompt B landed. */
function seedDroppedTurn({ aText, bText }: { aText: string; bText: string }) {
  const a = assistantRow({
    id: 'assistant-stream-a',
    parts: [{ type: 'text', text: aText }],
    pending: true,
    timestamp: 1
  })

  stream.states.set(SID, {
    ...stream.state(SID),
    busy: true,
    awaitingResponse: true,
    turnLive: true,
    streamId: 'assistant-stream-a',
    messages: [userRow('prompt A'), a, userRow(bText)]
  } as never)

  return a
}

function assistantTexts(): string[] {
  return stream
    .state(SID)
    .messages.filter(m => m.role === 'assistant' && !m.hidden)
    .map(m => chatMessageText(m))
    .filter(Boolean)
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  cleanup()
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('#101321 streamId reuse across turns', () => {
  it('turn B seeds its own bubble: deltas append below the prompt, not into A', async () => {
    stream = renderMessageStream(SID)
    seedDroppedTurn({ aText: 'AAA — the previous answer', bText: 'prompt B' })

    await start()
    await delta('BBB')
    await complete({ text: 'BBB — the new answer' })

    const messages = stream.state(SID).messages

    // B's answer is its own bubble below B's prompt, in its own words —
    // the sticky id did not append it into A's row above the prompt.
    expect(assistantTexts()).toEqual(['AAA — the previous answer', 'BBB — the new answer'])
    // A's row keeps A's text, untouched.
    expect(messages[1]).toMatchObject({ id: 'assistant-stream-a' })
    expect(chatMessageText(messages[1])).toBe('AAA — the previous answer')
    // B's answer sits at the tail, below B's prompt.
    expect(messages.at(-1)?.role).toBe('assistant')
    expect(chatMessageText(messages.at(-1)!)).toBe('BBB — the new answer')
    // The turn settled: bookkeeping released.
    expect(stream.state(SID)).toMatchObject({ busy: false, streamId: null })
  })

  it('a late terminal frame from turn A settles its own orphan instead of replacing B', async () => {
    stream = renderMessageStream(SID)
    seedDroppedTurn({ aText: 'AAA partial', bText: 'prompt B' })

    // B starts and streams its own answer.
    await start()
    await delta('BBB so far')
    await flush()
    // Then turn A's terminal frame finally arrives — after B started.
    await complete({ text: 'AAA partial — the dropped turn completed' })

    const messages = stream.state(SID).messages
    const byId = new Map(messages.map(m => [m.id, m]))

    // A's orphan settled with A's text, above B's prompt.
    expect(byId.get('assistant-stream-a')).toMatchObject({ pending: false, interim: false })
    expect(chatMessageText(byId.get('assistant-stream-a')!)).toBe('AAA partial — the dropped turn completed')
    // B's live bubble keeps its streamed text — the late frame did not
    // replace it, clear it, or stamp B's turn state.
    const live = [...messages].reverse().find(m => m.id !== 'assistant-stream-a' && m.role === 'assistant')
    expect(chatMessageText(live!)).toBe('BBB so far')
    expect(live?.pending).toBe(true)
    // The live turn's bookkeeping is untouched — B's own terminal owns it.
    expect(stream.state(SID)).toMatchObject({ busy: true, turnLive: true, streamId: expect.any(String) })

    // B's own complete then settles B normally.
    await complete({ text: 'BBB so far — done' })
    expect(assistantTexts()).toEqual(['AAA partial — the dropped turn completed', 'BBB so far — done'])
    expect(stream.state(SID)).toMatchObject({ busy: false, streamId: null })
  })

  it("a no-delta B completion repeating a pending A reply is B's own occurrence", async () => {
    stream = renderMessageStream(SID)
    seedDroppedTurn({ aText: 'The answer is unchanged.', bText: 'prompt B' })

    // B starts and completes without deltas, with the same valid reply.
    await start()
    await complete({ text: 'The answer is unchanged.' })

    // B appended its own bubble below its prompt instead of re-completing
    // A's row, and B's turn settled.
    const messages = stream.state(SID).messages
    expect(messages.filter(m => m.role === 'assistant')).toHaveLength(2)
    expect(messages.at(-1)?.role).toBe('assistant')
    expect(chatMessageText(messages.at(-1)!)).toBe('The answer is unchanged.')
    expect(stream.state(SID)).toMatchObject({ busy: false, turnLive: false })
  })

  it("a late A terminal frame does not run B's turn-end effects", async () => {
    stream = renderMessageStream(SID)
    seedDroppedTurn({ aText: 'AAA partial', bText: 'prompt B' })

    await start()
    await delta('BBB so far')
    await flush()

    const bStartedAt = $turnStartedAt.get()
    expect(bStartedAt).not.toBeNull()
    vi.mocked(playCompletionSound).mockClear()

    await complete({ text: 'AAA partial — the dropped turn completed' })

    // B is still live: its clock, completion sound and busy state are B's.
    expect($turnStartedAt.get()).toBe(bStartedAt)
    expect(playCompletionSound).not.toHaveBeenCalled()
    expect(stream.state(SID)).toMatchObject({ busy: true, turnLive: true })

    // B's own terminal frame still runs them.
    await complete({ text: 'BBB so far — done' })
    expect($turnStartedAt.get()).toBeNull()
    expect(playCompletionSound).toHaveBeenCalledTimes(1)
  })

  it('ten sequential turns keep one bubble per turn (no cross-turn growth)', async () => {
    stream = renderMessageStream(SID)

    stream.states.set(SID, {
      ...stream.state(SID),
      messages: [userRow('prompt 1')]
    } as never)

    for (let turn = 1; turn <= 10; turn += 1) {
      const answer = `answer ${turn}`
      await start()
      await delta(answer)
      await complete({ text: answer })
      // The next prompt lands as a user row between turns — the Grok
      // cadence of prompt → stream → settle, ten times over.
      stream.states.set(SID, {
        ...stream.state(SID),
        messages: [...stream.state(SID).messages, userRow(`prompt ${turn + 1}`)]
      } as never)
    }

    const messages = stream.state(SID).messages
    // Exactly one bubble per turn, each holding only its own turn's text —
    // deltas never appended into the previous turn's row across a boundary.
    expect(messages.filter(m => m.role === 'assistant').map(m => chatMessageText(m))).toEqual(
      Array.from({ length: 10 }, (_, i) => `answer ${i + 1}`)
    )
    // The transcript stays linear in turn count, not quadratic.
    expect(messages.filter(m => m.role === 'user')).toHaveLength(11)
  })

  it('a same-turn restart keeps streaming into the live bubble (chained turns, #74560 guard)', async () => {
    stream = renderMessageStream(SID)

    // Turn with a user row, streaming a partial answer.
    stream.states.set(SID, {
      ...stream.state(SID),
      messages: [userRow('prompt A')]
    } as never)
    await start()
    await delta('partial')
    await flush()
    const liveId = stream.state(SID).streamId
    expect(liveId).toBeTruthy()

    // A chained message.start for the SAME turn (goal follow-up, queue
    // drain): the live bubble sits BELOW the user row, so the id must be
    // kept and the stream must not split into two bubbles.
    await start()
    await delta(' answer continued')
    await complete({ text: 'partial answer continued' })

    expect(assistantTexts()).toEqual(['partial answer continued'])
  })
})
