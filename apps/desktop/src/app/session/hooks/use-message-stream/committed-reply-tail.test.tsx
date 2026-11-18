import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { overlayConcurrentMessageChanges } from '@/app/session/hooks/use-session-actions/utils'
import { type ChatMessage, chatMessageText, renderMediaTags, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'

import { renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

// transcript-integrity "ws drop + reconnect": the re-resume page stores the
// whole reply and the live row adopts it while the stream's tail is still on
// the wire (one delta already queued, one in flight). Those deltas are already
// in the adopted body; the completion that would repair a doubled tail can be
// lost to the same reconnect.
const SID = 'committed-reply-tail'
const TURN = 'turn-a8'
const REPLY = 'A8 survives the drop'

const prompt: ChatMessage = { id: 'user-optimistic', role: 'user', parts: [{ type: 'text', text: 'U8 across a drop' }] }

const page: ChatMessage[] = [
  { id: '16-user', role: 'user', rowId: 16, parts: [{ type: 'text', text: 'U8 across a drop' }] } as ChatMessage,
  { id: '17-assistant', role: 'assistant', rowId: 17, parts: [{ type: 'text', text: REPLY, sourceRowId: 17 }] }
]

describe('a live row that adopted the stored reply', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  const harness = () => {
    vi.useFakeTimers()
    const states = new Map([[SID, { ...createClientSessionState(), messages: [prompt] }]])
    const stream = renderMessageStream(SID, { states })

    const send = (type: string, payload: Record<string, unknown>) =>
      act(() => stream.handleEvent({ payload, session_id: SID, type: type as never }))

    const delta = (text: string, offset: number) => send('message.delta', { offset, segment: 0, text, turn_id: TURN })
    const flush = () => act(async () => void (await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS + 10)))

    const reconnect = (pageMessages: ChatMessage[], baseline: ChatMessage[]) =>
      states.set(SID, {
        ...stream.state(),
        messages: overlayConcurrentMessageChanges(pageMessages, baseline, stream.state().messages)
      })

    const replies = () =>
      stream
        .state()
        .messages.filter(message => message.role === 'assistant')
        .map(chatMessageText)

    return { delta, flush, reconnect, replies, send, stream }
  }

  it('draws its late tail deltas once when the completion never arrives', async () => {
    const { delta, flush, reconnect, replies, send, stream } = harness()

    await send('message.start', { turn_id: TURN })
    await delta('A8 ', 0)
    await flush()
    const baseline = stream.state().messages
    await delta('survives the', 3)
    await flush()
    await delta(' dr', 15)

    reconnect(page, baseline)

    await delta('op', 18)
    await flush()

    expect(replies()).toEqual([REPLY])
  })

  // The reconnect can lose a chunk the stored body already holds: the next one
  // lands past the gap, inside the adopted text, and must not be drawn again.
  it('drops a late chunk inside the adopted reply after a lost one', async () => {
    const { delta, flush, reconnect, replies, send, stream } = harness()

    await send('message.start', { turn_id: TURN })
    await delta('A8 ', 0)
    await flush()
    const baseline = stream.state().messages
    await delta('survives the', 3)
    await flush()

    reconnect(page, baseline)

    // ' dr' at offset 15 is lost to the reconnect.
    await delta('op', 18)
    await flush()

    expect(replies()).toEqual([REPLY])

    await delta(' now', 20)
    await flush()

    expect(replies()).toEqual([`${REPLY} now`])
  })

  // The stored reply renders its MEDIA tag as a file card; the stream still
  // sends the raw tag, which must be recognised as the adopted text.
  it('draws an adopted attachment once when its raw MEDIA tag arrives late', async () => {
    const { delta, flush, reconnect, replies, send, stream } = harness()
    const raw = 'Here is MEDIA:/tmp/report.pdf'

    const mediaPage = toChatMessages([
      { role: 'user', content: 'U8 across a drop', row_id: 16, timestamp: 1 },
      { role: 'assistant', content: raw, row_id: 17, timestamp: 2 }
    ] as never)

    await send('message.start', { turn_id: TURN })
    await delta('Here ', 0)
    await flush()
    const baseline = stream.state().messages
    await delta('is ', 5)
    await flush()

    reconnect(mediaPage, baseline)

    await delta('MEDIA:/tmp/report.pdf', 8)
    await flush()

    expect(replies()).toEqual([renderMediaTags(raw)])
  })
})
